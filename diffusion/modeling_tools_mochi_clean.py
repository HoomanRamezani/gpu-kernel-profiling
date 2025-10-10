"""
GPU kernel profiling utilities for Mochi diffusion transformer blocks.

This module provides a clean, class-based benchmark harness for single Mochi transformer
blocks, mirroring the structure of transformer/modeling_tools.py.

Key components:
- BACKEND_PROCESSORS: Mapping of backend names to attention processor classes
- cleanup_torch_compile: Reset compilation state between runs
- MochiBench: Main benchmark class for profiling single Mochi blocks

High-level flow in MochiBench.run:
1) Reset compile state and set backend
2) Extract single block from model
3) Optionally compile the block
4) Prepare synthetic inputs (with preprocessing: patch embed, time embed, RoPE)
5) Run warmup/active schedule with CUDA events for precise timing
6) Capture Chrome trace
7) Return per-backend timing summary
"""

import os
import tempfile
from contextlib import nullcontext, contextmanager
import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from diffusers import MochiTransformer3DModel
import gc

from custom_mochi_processors import set_mochi_attention_processor, BACKEND_PROCESSORS


def cleanup_torch_compile():
    """Reset compile/allocator state and point caches to fresh temp dirs.

    Rationale:
    - torch._dynamo.reset clears Dynamo state between runs
    - gc + empty_cache free Python/CUDA memory
    - Setting cache dirs to new temp folders forces re-compilation
    """
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor_")
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_")


class MochiBench:
    """Benchmark a single Mochi transformer block under different attention backends.

    This class loads a Mochi model, extracts a single block, switches its attention
    processor, optionally compiles it, and profiles with synthetic inputs to produce
    stable per-step timings and Chrome traces.

    Mirrors the structure of transformer/modeling_tools.py::Qwen2Bench
    """

    def __init__(
        self,
        model_name: str = "genmo/mochi-1-preview",
        block_idx: int = 0,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize benchmark with Mochi model.

        Args:
            model_name: HuggingFace model ID
            block_idx: Which transformer block to profile (0-47)
            dtype: Model dtype (fp16 recommended for consistency)
        """
        self.block_idx = block_idx
        self.dtype = dtype

        print(f"[INFO] Loading Mochi model: {model_name}")
        self.model = MochiTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("cuda")

        # Extract config
        self.num_blocks = len(self.model.transformer_blocks)
        self.hidden_size = self.model.config.num_attention_heads * self.model.config.attention_head_dim
        self.text_embed_dim = self.model.config.text_embed_dim
        self.patch_size = self.model.config.patch_size

        assert 0 <= block_idx < self.num_blocks, f"block_idx must be in [0, {self.num_blocks})"

        print(f"[INFO] Model loaded: {self.num_blocks} blocks, profiling block {block_idx}")
        print(f"[INFO] Hidden size: {self.hidden_size}, Text embed dim: {self.text_embed_dim}")

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
            attn_backend: Backend name from BACKEND_PROCESSORS (cudnn, flash, efficient, math, flash_attn2)
            frames: Number of video frames (temporal dimension)
            height: Spatial height in PIXEL space (will be //2 for latent space)
            width: Spatial width in PIXEL space
            text_seq_len: Text encoder sequence length
            enable_compile: If True, wrap block in torch.compile
            enable_backward: If True, include backward pass (not typical for diffusion)

        Returns:
            dict with keys:
                - impl: backend name
                - per_step_ms: average active-iteration time in seconds
                - trace: path to exported Chrome trace JSON
        """
        cleanup_torch_compile()

        self.attn_backend = attn_backend
        self.enable_backward = enable_backward
        self.frames = frames
        self.height = height
        self.width = width
        self.text_seq_len = text_seq_len

        tag = f"block[{attn_backend}:compiled:{enable_compile}]"

        if not enable_backward:
            torch.set_grad_enabled(False)
            self.model.eval()

        # Set attention processor on model
        with self.enable_attn_backend():
            # Extract single block
            block = self.model.transformer_blocks[self.block_idx]

            # Optionally compile
            if enable_compile:
                self.compile_module(block)

            # Prepare inputs
            inputs = self.prepare_fake_inputs()

            def run_once():
                if not enable_backward:
                    ctx = torch.no_grad()
                else:
                    ctx = nullcontext()

                with ctx:
                    with record_function(tag):
                        out = block(**inputs)

                if enable_backward:
                    # Mochi block returns tuple: (hidden_states, encoder_hidden_states)
                    hidden_out, encoder_out = out

                    # Create gradients for both outputs
                    grad_hidden = torch.randn_like(hidden_out)
                    grad_encoder = torch.randn_like(encoder_out)

                    # Backward through both outputs
                    torch.autograd.backward(
                        [hidden_out, encoder_out],
                        [grad_hidden, grad_encoder]
                    )

                    # Clear gradients for next iteration
                    block.zero_grad(set_to_none=True)
                    inputs["hidden_states"].grad = None

            # Setup trace path
            outdir = "traces_mochi_clean"
            os.makedirs(outdir, exist_ok=True)

            seq_len = inputs["hidden_states"].shape[1]
            trace_path = os.path.join(
                outdir,
                f"trace_{attn_backend}_block{self.block_idx}_seq{seq_len}.json"
            )

            # Profiler setup (matching transformer benchmark)
            warmup_steps = 30
            active_steps = 10
            repeat = 1
            sched = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=repeat)
            total_steps = (warmup_steps + active_steps) * repeat

            cycle = warmup_steps + active_steps
            active_time_ms = 0.0
            active_iters = 0

            # CUDA events for precise on-device timing
            evt_start = torch.cuda.Event(enable_timing=True)
            evt_end = torch.cuda.Event(enable_timing=True)

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

    def prepare_fake_inputs(self):
        """Create synthetic inputs for a single Mochi transformer block.

        Unlike transformer benchmark, Mochi requires running preprocessing layers
        (patch_embed, time_embed, rope) to create inputs in the correct format for blocks.

        Returns:
            dict with keys:
                - hidden_states: [B, seq_len, hidden_dim] - already patched/flattened
                - encoder_hidden_states: [B, text_seq_len, text_dim] - text embeddings
                - encoder_attention_mask: [B, text_seq_len] - boolean mask
                - temb: [B, hidden_dim] - timestep embedding
                - image_rotary_emb: tuple of RoPE tensors (3D: spatial+temporal)
        """
        device = "cuda"
        dtype = self.dtype
        batch_size = 1
        in_channels = 12

        # Create raw 5D latent tensor
        raw_latents = torch.randn(
            batch_size, in_channels, self.frames,
            self.height // 2, self.width // 2,
            device=device, dtype=dtype
        )

        # Create raw timestep
        timestep = torch.tensor([500], device=device, dtype=torch.long)

        # Text embeddings
        encoder_hidden_states_raw = torch.randn(
            batch_size, self.text_seq_len, self.text_embed_dim,
            device=device, dtype=dtype
        )

        # Text attention mask
        encoder_attention_mask = torch.ones(
            batch_size, self.text_seq_len,
            device=device, dtype=torch.bool
        )

        # Run through preprocessing layers (NOT part of block forward)
        with torch.no_grad():
            # Time embedding
            temb, encoder_hidden_states = self.model.time_embed(
                timestep,
                encoder_hidden_states_raw,
                encoder_attention_mask,
                hidden_dtype=dtype,
            )

            # Patch embedding
            post_patch_height = (self.height // 2) // self.patch_size
            post_patch_width = (self.width // 2) // self.patch_size

            hidden_states = raw_latents.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.model.patch_embed(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

            # RoPE embeddings (3D: spatial + temporal)
            image_rotary_emb = self.model.rope(
                self.model.pos_frequencies,
                self.frames,
                post_patch_height,
                post_patch_width,
                device=device,
                dtype=torch.float32,
            )

        if self.enable_backward:
            hidden_states = hidden_states.detach().requires_grad_(True)
            encoder_hidden_states = encoder_hidden_states.detach().requires_grad_(True)
            temb = temb.detach().requires_grad_(True)
            # Note: image_rotary_emb is fp32 positional encoding, doesn't need gradients
            # encoder_attention_mask is boolean, doesn't need gradients

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "temb": temb,
            "image_rotary_emb": image_rotary_emb,
        }

    def compile_module(self, block):
        """Wrap block.forward with torch.compile and prime both forward and backward.

        Args:
            block: The MochiTransformerBlock to compile
        """
        print("[INFO] Compiling block (this may take several minutes)...")

        # Always use fullgraph=False for Mochi (complex architecture)
        block.forward = torch.compile(
            block.forward,
            mode="max-autotune",
            fullgraph=False,
            dynamic=True,
        )

        # Prime forward pass
        for _ in range(3):
            with torch.no_grad():
                _ = block(**self.prepare_fake_inputs())

        # Prime backward pass if enabled
        if self.enable_backward:
            torch._dynamo.config.compiled_autograd = True
            print("[INFO] Priming backward pass...")

            for _ in range(3):
                inputs = self.prepare_fake_inputs()
                out = block(**inputs)

                # Unpack tuple output
                hidden_out, encoder_out = out

                # Create gradients
                grad_hidden = torch.randn_like(hidden_out)
                grad_encoder = torch.randn_like(encoder_out)

                # Backward pass
                torch.autograd.backward(
                    [hidden_out, encoder_out],
                    [grad_hidden, grad_encoder]
                )

                # Clear gradients
                block.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        print("[INFO] Compilation complete")

    @contextmanager
    def enable_attn_backend(self):
        """Context manager that switches attention processor for a run.

        On entry:
        - Sets custom attention processor via set_mochi_attention_processor

        On exit:
        - No-op (Mochi doesn't have built-in restoration like HF models)

        Note: Unlike transformer benchmark which uses HuggingFace's built-in
        set_attn_implementation(), Mochi requires manual processor assignment.
        """
        self.model = set_mochi_attention_processor(self.model, self.attn_backend)

        try:
            yield
        finally:
            # No automatic restore for Mochi (would need to save original processor)
            pass
