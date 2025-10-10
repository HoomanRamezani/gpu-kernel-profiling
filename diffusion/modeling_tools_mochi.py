"""
GPU kernel profiling utilities for Mochi diffusion transformer layers.

This module provides a single-layer benchmark harness for Mochi transformer blocks,
parallel to the Qwen2Bench structure used for causal transformers.

Key differences from causal LM benchmarks:
- Mochi uses joint self+cross attention in a single operation
- Inputs are 3D latent tensors + text embeddings (not token sequences)
- Uses custom attention processors instead of HF's standard attention implementations

High-level flow in MochiBench.run:
1) Load full Mochi model and extract a single transformer block
2) Set custom attention processor (eager, flash_attn2, sdpa variants)
3) Optionally wrap the block in torch.compile
4) Run warmup + active iterations with synthetic inputs
5) Capture Chrome trace and return per-step timing
"""

import os
import tempfile
from contextlib import nullcontext, contextmanager
import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from diffusers import MochiTransformer3DModel
import gc

from custom_mochi_processors import set_mochi_attention_processor


def cleanup_torch_compile():
    """Reset compile/allocator state and point caches to fresh temp dirs."""
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor_")
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_")


class MochiBench:
    """Benchmark a single Mochi transformer block under different attention backends.

    Unlike the full model benchmark (bench_diffusion_custom.py), this profiles only
    one transformer block to provide apples-to-apples comparison with single-layer
    causal transformer benchmarks.
    """

    def __init__(
        self,
        model_name: str = "genmo/mochi-1-preview",
        block_idx: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_name: HuggingFace model ID for Mochi
            block_idx: Which transformer block to profile (0-47 for Mochi)
            dtype: Model dtype (bf16 recommended)
        """
        self.block_idx = block_idx
        self.dtype = dtype

        print(f"Loading Mochi model: {model_name}...")
        self.model = MochiTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=dtype,
        ).to("cuda")

        # Extract config info
        self.num_layers = len(self.model.transformer_blocks)
        assert 0 <= block_idx < self.num_layers, f"block_idx must be in [0, {self.num_layers})"

        # Calculate hidden size from attention config
        self.hidden_size = self.model.config.num_attention_heads * self.model.config.attention_head_dim
        self.text_embed_dim = self.model.config.text_embed_dim

        print(f"Model loaded. Total blocks: {self.num_layers}, profiling block {block_idx}")
        print(f"Hidden size: {self.hidden_size}, Text embed dim: {self.text_embed_dim}")

    def run(
        self,
        attn_backend: str,
        frames: int = 8,
        height: int = 64,
        width: int = 64,
        text_seq_len: int = 256,
        enable_compile: bool = False,
        enable_backward: bool = False,
    ):
        """Execute the benchmark for a specific backend and configuration.

        Args:
            attn_backend: Attention backend (eager, flash_attn2, cudnn, efficient, math)
            frames: Number of video frames (temporal dimension)
            height: Spatial height in latent space (//2 of pixel space)
            width: Spatial width in latent space
            text_seq_len: Text encoder sequence length
            enable_compile: Whether to wrap block in torch.compile
            enable_backward: Whether to include backward pass (not typical for diffusion)

        Returns:
            dict with keys:
                - impl: backend alias string
                - per_step_ms: average active-iteration time in seconds
                - trace: path to the exported Chrome trace JSON
        """
        cleanup_torch_compile()

        self.attn_backend = attn_backend
        self.enable_backward = enable_backward

        tag = f"block[{attn_backend}:compiled:{enable_compile}]"

        if not enable_backward:
            torch.set_grad_enabled(False)
            self.model.eval()

        # Set attention processor
        self.model = set_mochi_attention_processor(self.model, attn_backend)

        # Extract single block
        block = self.model.transformer_blocks[self.block_idx]

        if enable_compile:
            print("[INFO] Compiling block (this may take several minutes)...")
            block.forward = torch.compile(
                block.forward,
                mode="max-autotune",
                fullgraph=False,
                dynamic=True,
            )

            # Warmup compile
            test_inputs = self.prepare_fake_inputs(frames, height, width, text_seq_len)
            for _ in range(3):
                with torch.no_grad():
                    _ = block(**test_inputs)
            torch.cuda.synchronize()

        # Prepare inputs
        inputs = self.prepare_fake_inputs(frames, height, width, text_seq_len)

        def run_once():
            if not enable_backward:
                ctx = torch.no_grad()
            else:
                ctx = nullcontext()

            with ctx:
                with record_function(tag):
                    out = block(**inputs)

            if enable_backward:
                grad_out = torch.randn_like(out)
                out.backward(grad_out)
                block.zero_grad(set_to_none=True)
                inputs["hidden_states"].grad = None

        # Setup trace path
        outdir = "traces_mochi_single_block"
        os.makedirs(outdir, exist_ok=True)
        trace_path = os.path.join(
            outdir,
            f"trace_{attn_backend}_block{self.block_idx}_F{frames}_H{height}_W{width}.json"
        )

        # Profiler setup
        warmup_steps = 30
        active_steps = 10
        repeat = 1
        sched = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=repeat)
        total_steps = (warmup_steps + active_steps) * repeat

        cycle = warmup_steps + active_steps
        active_time_ms = 0.0
        active_iters = 0

        # CUDA events for precise timing
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)

        print(f"[INFO] Running benchmark: {warmup_steps} warmup + {active_steps} active iterations...")

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

        prof.export_chrome_trace(trace_path)
        print(f"[saved] Chrome trace -> {trace_path}")

        return {
            "impl": attn_backend,
            "per_step_ms": per_step_ms / 1000.0,  # convert to seconds
            "trace": trace_path,
        }

    def prepare_fake_inputs(self, frames, height, width, text_seq_len):
        """Create synthetic inputs for a single Mochi transformer block.

        This method runs the preprocessing steps that the full model does before
        calling transformer blocks:
        1. Patch embedding of latents
        2. Time embedding processing
        3. RoPE positional encoding generation

        Blocks expect:
        - hidden_states: [B, seq_len, hidden_dim] - already patched and flattened
        - encoder_hidden_states: [B, text_seq_len, text_dim] text embeddings
        - encoder_attention_mask: [B, text_seq_len] boolean mask
        - temb: [B, hidden_dim] timestep embedding (already processed)
        - image_rotary_emb: RoPE positional encodings
        """
        device = "cuda"
        dtype = self.dtype
        batch_size = 1
        in_channels = 12
        patch_size = self.model.config.patch_size

        # Create raw latent tensor
        raw_latents = torch.randn(
            batch_size, in_channels, frames, height, width,
            device=device, dtype=dtype
        )

        # Create raw timestep
        timestep = torch.tensor([500], device=device, dtype=torch.long)

        # Text embeddings
        encoder_hidden_states_raw = torch.randn(
            batch_size, text_seq_len, self.text_embed_dim,
            device=device, dtype=dtype
        )

        # Text attention mask
        encoder_attention_mask = torch.ones(
            batch_size, text_seq_len,
            device=device, dtype=torch.bool
        )

        # Run through model's preprocessing layers
        with torch.no_grad():
            # Time embedding
            temb, encoder_hidden_states = self.model.time_embed(
                timestep,
                encoder_hidden_states_raw,
                encoder_attention_mask,
                hidden_dtype=dtype,
            )

            # Patch embedding
            post_patch_height = height // patch_size
            post_patch_width = width // patch_size

            hidden_states = raw_latents.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.model.patch_embed(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

            # RoPE embeddings
            image_rotary_emb = self.model.rope(
                self.model.pos_frequencies,
                frames,
                post_patch_height,
                post_patch_width,
                device=device,
                dtype=torch.float32,
            )

        if self.enable_backward:
            hidden_states = hidden_states.detach().requires_grad_(True)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "temb": temb,
            "image_rotary_emb": image_rotary_emb,
        }
