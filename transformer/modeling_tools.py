"""
GPU kernel profiling utilities for attention backends on Hugging Face causal LMs.

This module provides:
- ATTN_ALIASES / SDPA_IMPL: name mappings to switch HF attention backends and SDPA kernels
- cleanup_torch_compile: resets Dynamo/Inductor state and forces fresh cache dirs per run
- build_additive_causal_mask_4d: constructs a causal + padding additive mask for eager attention
- Qwen2Bench: a focused benchmark harness for a single Transformer layer in Qwen models

High-level flow in Qwen2Bench.run:
1) Reset compile state and set test parameters (backend, sequence length, etc.)
2) Optionally wrap the whole layer or just its attention in torch.compile
3) Prepare synthetic inputs (including RoPE and optional additive mask for eager)
4) Run a warmup/active schedule while:
   - Measuring active iterations with CUDA events (for stable wall-clock per-step ms)
   - Capturing a rich CPU+CUDA trace with torch.profiler for Chrome/Perfetto
5) Export the trace JSON and return the per-implementation timing summary

Notes:
- We time only the "active" phase to exclude warmup/JIT/graph capture costs.
- We tag attention.forward with record_function so kernels are grouped in traces.
- For some backends (fa2/fa3) we avoid fullgraph=True because of known issues.
"""
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
from math import sqrt

try:
    import flash_attn_interface
    _HAVE_FA3 = True
except:
    _HAVE_FA3 = False


# Friendly CLI/backend names mapped to HF internal attention implementation identifiers.
# "sdpa_*" all route through HF "sdpa" implementation, and we pick a concrete kernel below.
ATTN_ALIASES = {
    "fa2": "flash_attention_2",
    "fa3": "flash_attention_3",
    "eager": "eager",
    "sdpa_cudnn": "sdpa",
    "sdpa_flash": "sdpa",
    "sdpa_mem": "sdpa",
    "sdpa_math": "sdpa",
    "flex": "flex_attention",
}

# Concrete SDPA kernel selection for torch.nn.attention.sdpa_kernel context manager.
SDPA_IMPL = {
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_mem": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_math": SDPBackend.MATH,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
}

def cleanup_torch_compile():
    """Reset compile/allocator state and point Inductor/Triton caches to fresh temp dirs.

    Rationale:
    - torch._dynamo.reset clears Dynamo state between runs to avoid cross-run caching effects
    - gc + empty_cache free Python/CUDA memory so subsequent runs start from a clean slate
    - Setting cache dirs to new temp folders avoids reusing previously compiled kernels,
      forcing re-compile when comparing different code paths or flags.
    """
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor_")
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_")


def build_additive_causal_mask_4d(attn2d: torch.Tensor, *, device, dtype=torch.float32):
    """Construct a 4D additive attention mask with causal structure and key padding.

    Args:
        attn2d: Boolean tensor of shape [batch, seq_len] where True keeps a token and
                False denotes padding. Must not be None for eager mask construction.
        device: Target CUDA/CPU device for the mask tensors.
        dtype:  Floating dtype used for the additive mask values (e.g., bf16/fp32).

    Returns:
        Tensor of shape [batch, 1, seq_len, seq_len] where valid positions contain 0 and
        masked positions contain a large negative number (finfo(dtype).min). The mask is
        composed of:
        - a causal lower-triangular component (prevents attending to future tokens)
        - an optional key padding component for padded positions in the sequence
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


class Qwen2Bench:
    """Benchmark a single Qwen Transformer layer under different attention backends.

    This class loads a Hugging Face Qwen model, switches its attention implementation
    and SDPA kernel as requested, optionally compiles a module with torch.compile,
    and profiles a single layer with synthetic inputs to produce stable per-step timings
    and a Chrome trace for deep dive analysis.
    """
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
        """Execute the benchmark for a specific backend and configuration.

        Args:
            mode: "layer" to profile the entire Transformer layer, or "attention" to
                  focus on the attention submodule only (also affects compile scope).
            attn_backend: Backend alias from ATTN_ALIASES (eager, fa2, sdpa_flash, ...).
            seq_len: Sequence length for the synthetic inputs.
            enable_backward: If True, run backward() with a random grad to include grads.
            enable_compile: If True, wrap the target module's forward in torch.compile.
            batch_size: Micro-batch size for synthetic inputs.

        Returns:
            dict with keys:
                - impl: backend alias string
                - per_step_ms: average active-iteration time in seconds
                - trace: path to the exported Chrome trace JSON
        """
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

        # Switch HF attention implementation and, for SDPA, the concrete kernel.
        with self.enable_attn_backend():
            layer = self.model.model.layers[self.layer_idx]



            to_compile = layer
            if mode == "attention":
                to_compile = layer._modules["self_attn"]
            
            if enable_compile:
                # Optionally compile either the Layer forward() or only Attention.forward().
                self.compile_module(
                    module=to_compile,
                    bench_mode=mode,
                )

            # annotate attn.forward()
            # We wrap attention.forward with a record_function so traces group related ops
            # under a readable tag (e.g., attn[sdpa_flash:compiled:True]).
            _orig = layer._modules["self_attn"].forward
            def _wrapped(*args, **kwargs):
                with record_function(tag):
                    return _orig(*args, **kwargs)
            layer._modules["self_attn"].forward = _wrapped

            


            # Prepare synthetic inputs (hidden states, RoPE, and additive mask for eager)
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
                    # Use a random grad_out to exercise backward kernels and memory traffic.
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
            # We separate warmup (to stabilize kernels and compilation) from active
            # iterations that are measured with CUDA events for stable per-step latency.
            warmup_steps = 30
            active_steps = 10
            repeat = 1
            sched = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=repeat)
            total_steps = (warmup_steps + active_steps) * repeat

            cycle = warmup_steps + active_steps
            active_time_ms = 0.0
            active_iters = 0

            # Use CUDA events for precise on-device timing during "active" iterations only.
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
            "per_step_ms": per_step_ms / 1000.0,  # seconds
            "trace": trace_path,
        }

                    

    def prepare_fake_inputs(self, mode="layer"):
        """Create synthetic inputs for the selected module and backend.

        Behavior notes:
        - Always uses bf16 tensors on CUDA to match typical Qwen deployment dtype
        - Builds a [B,1,S,S] additive mask only when running the full layer in eager mode
        - Reuses hidden_states buffer across iterations; when backward is enabled we
          clear its .grad between steps
        """
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
        """Wrap module.forward with torch.compile and prime both fwd and bwd.

        We try to compile the forward path and, when requested, compiled autograd
        for backward as well. Some attention backends (fa2/fa3) do not support
        full CUDA graphs reliably in older stacks; for those we set fullgraph=False.
        """

        # full CUDA graph does not work for vanilla fa2 (HF <3) and fa3 backends
        fullgraph = self.attn_backend not in ["fa2", "fa3"]
        
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
        """Context manager that switches HF attention impl and SDPA kernel for a run.

        On entry:
        - Sets model.set_attn_implementation(...) according to ATTN_ALIASES
        - If an SDPA backend is used, selects the concrete SDPBackend via sdpa_kernel

        On exit:
        - Restores attention implementation to "eager" for safety
        """
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


