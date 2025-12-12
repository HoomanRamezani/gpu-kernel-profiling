
import os
import tempfile
import time
from contextlib import nullcontext, contextmanager
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from transformers import AutoModelForCausalLM
import gc
import threading

from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers import AttentionInterface
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from math import sqrt

try:
    import flash_attn_interface
    _HAVE_FA3 = True
except:
    _HAVE_FA3 = False

try:
    from flash_attn.cute import flash_attn_func as fa4_func
    _HAVE_FA4 = True
except:
    _HAVE_FA4 = False


ATTN_ALIASES = {
    "fa2": "flash_attention_2",
    "fa3": "flash_attention_3",
    "fa4": "fa4_custom",  # Custom implementation using CUTE-DSL
    "eager": "eager",
    "sdpa_cudnn": "sdpa",
    "sdpa_flash": "sdpa",
    "sdpa_mem": "sdpa",
    "sdpa_math": "sdpa",
    "flex": "flex_attention",
}

SDPA_IMPL = {
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_mem": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_math": SDPBackend.MATH,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
}

def cleanup_torch_compile():
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor_")
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_")


def build_additive_causal_mask_4d(attn2d: torch.Tensor, *, device, dtype=torch.float32):
    """
    attn2d: [B, S] boolean (1=keep, 0=pad); can be None if no padding.
    Returns additive 4D mask [B, 1, S, S] with 0 on valid positions and
    large negative on masked ones (causal + optional padding).
    """
    assert attn2d is not None, "attn2d cannot be None for eager mask construction"

    B, S = attn2d.shape

    tri = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
    mask = torch.zeros((B, 1, S, S), device=device, dtype=dtype)
    neg = torch.finfo(dtype).min
    mask = mask.masked_fill(~tri, neg)

    key_pad = (~attn2d).unsqueeze(1).unsqueeze(2).to(mask.dtype) * neg  # [B,1,1,S]
    mask = mask + key_pad
    return mask


class FA4AttentionWrapper:
    """Wrapper to replace HF attention with FlashAttention-4 CUTE."""
    def __init__(self, original_attn):
        self.original_attn = original_attn
        self._original_forward = original_attn.forward

    def __enter__(self):
        """Replace forward method with FA4 implementation."""
        def fa4_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            if not _HAVE_FA4:
                raise RuntimeError("FlashAttention-4 (CUTE) is not installed. Install with: pip install flash-attn-cute")

            bsz, q_len, _ = hidden_states.size()

            # QKV projection
            query_states = self.original_attn.q_proj(hidden_states)
            key_states = self.original_attn.k_proj(hidden_states)
            value_states = self.original_attn.v_proj(hidden_states)

            # Reshape for multi-head attention
            num_heads = self.original_attn.config.num_attention_heads
            num_kv_heads = self.original_attn.config.num_key_value_heads
            head_dim = self.original_attn.head_dim

            query_states = query_states.view(bsz, q_len, num_heads, head_dim)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim)

            # Apply RoPE
            if position_embeddings is None:
                cos, sin = self.original_attn.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings

            # apply_rotary_pos_emb expects [batch, heads, seq, head_dim], so transpose
            query_states = query_states.transpose(1, 2)  # [B, H, S, D]
            key_states = key_states.transpose(1, 2)      # [B, H, S, D]

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Transpose back to [B, S, H, D] for FA4
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

            # FA4 expects [B, S, H, D] format (already correct after transpose)

            # Use FA4 for attention computation
            # FA4 expects (batch, seqlen, nheads, headdim)
            attn_output, _ = fa4_func(
                query_states,
                key_states,
                value_states,
                causal=True,  # Causal for autoregressive LLMs
                softmax_scale=1.0 / sqrt(head_dim),
            )

            # Reshape and project output
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = self.original_attn.o_proj(attn_output)

            # Return format: (attn_output, present_key_value)
            return attn_output, past_key_value

        self.original_attn.forward = fa4_forward
        return self

    def __exit__(self, *args):
        """Restore original forward method."""
        self.original_attn.forward = self._original_forward


class Qwen2Bench:
    def __init__(
        self,
        model_name: str, 
        layer_idx: int = 0,
    ):
        self.layer_idx = layer_idx
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # load with a safe defaulwt, we'll switch later
        ).to("cuda")

        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads


    def run(
        self, 
        mode: str,
        attn_backend: str, 
        seq_len: int, 
        enable_backward: bool,
        enable_compile: bool,
        batch_size: int = 1,
    ):
        assert mode in ["layer", "attention"]

        cleanup_torch_compile()
        self.enable_backward = enable_backward
        self.attn_backend = attn_backend
        self.batch_size = batch_size
        self.seq_len = seq_len

        tag=f"attn[{attn_backend}:compiled:{enable_compile}]"
        
        if not self.enable_backward:
            torch.set_grad_enabled(False)
            self.model.eval()

        with self.enable_attn_backend():
            layer = self.model.model.layers[self.layer_idx]



            to_compile = layer
            if mode == "attention":
                to_compile = layer._modules["self_attn"]
            
            if enable_compile:
                self.compile_module(
                    module=to_compile,
                    bench_mode=mode,
                )

            # annotate attn.forward()
            _orig = layer._modules["self_attn"].forward
            def _wrapped(*args, **kwargs):
                with record_function(tag):
                    return _orig(*args, **kwargs)
            layer._modules["self_attn"].forward = _wrapped

            


            inputs = self.prepare_fake_inputs("layer")
            def run_once():
                if not enable_backward:
                    ctx = torch.no_grad()
                else:
                    ctx = nullcontext()
                
                # forward pass
                with ctx:
                    with record_function("profile::layer_forward"):
                        out = layer(**inputs)
                if enable_backward:
                    # backward pass
                    grad_out = torch.randn_like(out)
                    with torch.autograd.profiler.record_function("profile::layer_backward"):
                        out.backward(grad_out)
                    del out
                    layer.zero_grad(set_to_none=True)
                    # delete grad for input as well to start anew
                    # this allows to reuse `hidden_states` for next iteration
                    inputs["hidden_states"].grad = None


            outdir = "traces_attn"
            os.makedirs(outdir, exist_ok=True)
            trace_path = os.path.join(outdir, f"trace_{self.attn_backend}_L{self.layer_idx}_B{self.batch_size}_S{self.seq_len}")
            if enable_backward:
                trace_path += "_backward"
            if enable_compile:
                trace_path += "_compiled"
            
            trace_path += ".json"

            # profiler setup
            warmup_steps = 30
            active_steps = 10
            repeat = 1
            sched = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=repeat)
            total_steps = (warmup_steps + active_steps) * repeat

            cycle = warmup_steps + active_steps
            active_time_ms = 0.0
            active_iters = 0

            evt_start = torch.cuda.Event(enable_timing=True)
            evt_end   = torch.cuda.Event(enable_timing=True)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=sched,
                record_shapes=True,
            ) as prof:
                torch.cuda.synchronize()
                for step in range(total_steps):
                    is_active = (step % cycle) >= warmup_steps

                    if is_active:
                        evt_start.record()

                    run_once()

                    if is_active:
                        evt_end.record()
                        torch.cuda.synchronize()
                        active_time_ms += evt_start.elapsed_time(evt_end)
                        active_iters += 1
                    else:
                        torch.cuda.synchronize()


                    prof.step()

            per_step_ms = active_time_ms / max(1, active_iters)

            # table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
            # print(table)

            prof.export_chrome_trace(trace_path)
            print(f"[saved] Chrome trace -> {trace_path}")

        return {
            "impl": self.attn_backend,
            "seq_len": self.seq_len,
            "per_step_ms": per_step_ms / 1000.0,  # seconds
            "trace": trace_path,
        }

                    

    def prepare_fake_inputs(self, mode="layer"):
        dtype_t = torch.bfloat16
        device = "cuda"

        B, S = self.batch_size, self.seq_len
        head_dim = self.hidden_size // self.num_heads


        # Inputs
        hidden_states = torch.randn(B, S, self.hidden_size, device=device, dtype=dtype_t)
        position_ids = torch.arange(S, device=device).unsqueeze(0)

        # RoPE
        dummy_q = torch.empty(B, S, self.num_heads, head_dim, device=device, dtype=dtype_t)
        cos, sin = self.model.model.rotary_emb(dummy_q, position_ids)


        # Attention mask (need to construct for eager mode)
        if mode != "attention" and self.attn_backend == "eager":
            attention_mask = build_additive_causal_mask_4d(
                torch.ones(B, S, device=device, dtype=torch.bool), 
                device=device, 
                dtype=dtype_t
            )
        else:
            attention_mask = None 

        if self.enable_backward:
            # we will be calling backward on it
            hidden_states = hidden_states.detach().requires_grad_(True)

        return dict(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            position_embeddings=(cos, sin)
        )
        
    def compile_module(self, module, bench_mode):

        # full CUDA graph does not work for vanilla fa2 (HF <3), fa3, and fa4 backends
        fullgraph = self.attn_backend not in ["fa2", "fa3", "fa4"]
        
        module.forward = torch.compile(
            module.forward, 
            mode="max-autotune", 
            fullgraph=fullgraph,
        )

        # run with fake inputs to trigger JIT / graph capture
        for _ in range(3):
            out = module.forward(**self.prepare_fake_inputs(bench_mode))

            if self.enable_backward:
                torch._dynamo.config.compiled_autograd = True

            if bench_mode == "attention":
                out = out[0] #HF impl returns a tuple       
            
            out = out.clone()
            grad_out = torch.randn_like(out) #fake grad to trigger backward

            torch.compile(
                lambda: out.backward(grad_out),
                mode="max-autotune", 
                fullgraph=fullgraph,
            )

            out.backward(grad_out)

            del out
            module.zero_grad(set_to_none=True)

    @contextmanager
    def enable_attn_backend(self):
        # Special handling for FA4
        if self.attn_backend == "fa4":
            if not _HAVE_FA4:
                raise RuntimeError("FlashAttention-4 (CUTE) is not installed. Install with: pip install flash-attn-cute")

            # Keep model in eager mode but use FA4 wrapper
            self.model.set_attn_implementation("eager")
            layer = self.model.model.layers[self.layer_idx]
            fa4_wrapper = FA4AttentionWrapper(layer.self_attn)

            try:
                with fa4_wrapper:
                    yield None
            finally:
                self.model.set_attn_implementation("eager")
        else:
            self.model.set_attn_implementation(ATTN_ALIASES[self.attn_backend])
            if self.attn_backend.startswith("sdpa"):
                ctx = sdpa_kernel(SDPA_IMPL[self.attn_backend])
            else:
                ctx = nullcontext()

            try:
                with ctx as c:
                    yield c
            finally:
                self.model.set_attn_implementation("eager")


