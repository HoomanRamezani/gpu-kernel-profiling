#!/usr/bin/env python3
"""Benchmark Mochi diffusion transformer - SINGLE BLOCK ONLY (like transformer benchmark)."""

import argparse
import torch
import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from diffusers import MochiTransformer3DModel
from custom_mochi_processors import set_mochi_attention_processor, get_available_backends
import time

# Import profiler if available
try:
    from torch.profiler import profile, ProfilerActivity, record_function
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False


def prepare_single_block_inputs(model, frames, height, width, text_seq_len=256, device="cuda", dtype=torch.float16):
    """Create inputs for a SINGLE Mochi transformer block (not full model).

    Similar to transformer benchmark, we run preprocessing then benchmark only one block.
    """
    batch_size = 1
    in_channels = 12
    patch_size = model.config.patch_size
    text_embed_dim = model.config.text_embed_dim

    # Create raw inputs
    raw_latents = torch.randn(
        batch_size, in_channels, frames, height // 2, width // 2,
        device=device, dtype=dtype
    )

    timestep = torch.tensor([500], device=device, dtype=torch.long)

    encoder_hidden_states_raw = torch.randn(
        batch_size, text_seq_len, text_embed_dim,
        device=device, dtype=dtype
    )

    encoder_attention_mask = torch.ones(
        batch_size, text_seq_len,
        device=device, dtype=torch.bool
    )

    # Run through preprocessing (time embed, patch embed, rope)
    with torch.no_grad():
        # Time embedding
        temb, encoder_hidden_states = model.time_embed(
            timestep,
            encoder_hidden_states_raw,
            encoder_attention_mask,
            hidden_dtype=dtype,
        )

        # Patch embedding
        post_patch_height = (height // 2) // patch_size
        post_patch_width = (width // 2) // patch_size

        hidden_states = raw_latents.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = model.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        # RoPE embeddings
        image_rotary_emb = model.rope(
            model.pos_frequencies,
            frames,
            post_patch_height,
            post_patch_width,
            device=device,
            dtype=torch.float32,
        )

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "temb": temb,
        "image_rotary_emb": image_rotary_emb,
    }


def benchmark_backend(
    block,
    backend,
    inputs,
    block_idx,
    warmup=3,
    active=5,
    use_compile=False,
    save_trace=False,
    trace_dir="traces_diffusion_single_block",
):
    """Benchmark a specific attention backend on a SINGLE transformer block."""

    print(f"\n{'='*80}")
    print(f"Benchmarking backend: {backend.upper()} (Block {block_idx})")
    print(f"{'='*80}")

    # Compile if requested
    if use_compile:
        print("[INFO] Compiling block (this may take several minutes)...")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        block.forward = torch.compile(
            block.forward,
            mode="max-autotune",
            fullgraph=False,
            dynamic=True,
        )

        # Warmup compile
        for _ in range(3):
            with torch.no_grad():
                _ = block(**inputs)
        torch.cuda.synchronize()

    # Warmup
    print(f"[INFO] Warmup: {warmup} iterations...")
    for i in range(warmup):
        with torch.no_grad():
            output = block(**inputs)
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{warmup}")

    # Benchmark
    print(f"[INFO] Benchmarking: {active} iterations...")
    times = []
    trace_path = None

    if save_trace and HAS_PROFILER:
        os.makedirs(trace_dir, exist_ok=True)
        seq_len = inputs["hidden_states"].shape[1]
        trace_name = f"trace_mochi_{backend}_block{block_idx}_seq{seq_len}_custom"
        if use_compile:
            trace_name += "_compiled"
        trace_path = os.path.join(trace_dir, f"{trace_name}.json")

        print(f"[INFO] Saving trace to: {trace_path}")

        # Create a new profiler instance for each backend to avoid state accumulation
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )

        prof.start()
        try:
            for i in range(active):
                with torch.no_grad():
                    start = time.time()
                    output = block(**inputs)
                    torch.cuda.synchronize()
                    elapsed = (time.time() - start) * 1000
                    times.append(elapsed)
                print(f"  Iter {i+1}/{active}: {elapsed:.2f} ms")
        finally:
            prof.stop()
            prof.export_chrome_trace(trace_path)
            print(f"[saved] Chrome trace -> {trace_path}")
            # Explicitly clean up profiler
            del prof

    else:
        for i in range(active):
            with torch.no_grad():
                start = time.time()
                output = block(**inputs)
                torch.cuda.synchronize()
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            print(f"  Iter {i+1}/{active}: {elapsed:.2f} ms")

    # Stats
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n[RESULTS]")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")

    # Memory
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  Memory:  {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

    result = {
        "backend": backend,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "memory_gb": memory_allocated,
        "trace_path": trace_path if save_trace and HAS_PROFILER else None,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark SINGLE Mochi transformer block (like transformer benchmark)")
    parser.add_argument("--block", type=int, default=0, help="Which block to benchmark (0-47)")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--height", type=int, default=64, help="Height (pixel space)")
    parser.add_argument("--width", type=int, default=64, help="Width (pixel space)")
    parser.add_argument("--backends", nargs="+", default=None,
                        help="Backends to test (default: all available)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--active", type=int, default=5, help="Active iterations")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--save-traces", action="store_true", help="Save profiler traces")
    parser.add_argument("--trace-dir", type=str, default="traces_diffusion_single_block",
                        help="Directory for traces")
    parser.add_argument("--device", type=int, default=None, help="CUDA device ID (auto-selects free GPU if not specified)")

    args = parser.parse_args()

    # Auto-select GPU with most free memory if not specified
    if args.device is None:
        max_free = 0
        best_device = 0
        for i in range(torch.cuda.device_count()):
            free_mem = torch.cuda.mem_get_info(i)[0]
            if free_mem > max_free:
                max_free = free_mem
                best_device = i
        args.device = best_device
        print(f"[INFO] Auto-selected GPU {args.device} with {max_free / 1024**3:.2f} GB free")

    # Set CUDA device
    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"

    print("="*80)
    print("MOCHI SINGLE BLOCK BENCHMARK (Like transformer/bench.py)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Block index: {args.block}")
    print(f"  Frames: {args.frames}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Active: {args.active}")
    print(f"  Compile: {args.compile}")
    print(f"  Save traces: {args.save_traces}")

    # Get available backends
    available = get_available_backends()
    print(f"\nAvailable backends: {', '.join(available)}")

    # Determine which backends to test
    if args.backends is None:
        backends = available
    else:
        backends = [b for b in args.backends if b in available]
        unavailable = [b for b in args.backends if b not in available]
        if unavailable:
            print(f"WARNING: Skipping unavailable backends: {', '.join(unavailable)}")

    print(f"Testing backends: {', '.join(backends)}")

    # Initial cleanup to ensure clean state
    print(f"\n[INFO] Performing initial memory cleanup...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()

    initial_mem = torch.cuda.memory_allocated() / 1024**3
    initial_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[MEMORY] Initial state: {initial_mem:.2f} GB allocated, {initial_reserved:.2f} GB reserved")

    # Load model ONCE (we'll extract the block and reuse it)
    print(f"\n[INFO] Loading Mochi model...")
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024**3

    model = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview",
        subfolder="transformer",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1024**3
    print(f"[MEMORY] Model loaded: {mem_after:.2f} GB (+{mem_after - mem_before:.2f} GB)")

    num_blocks = len(model.transformer_blocks)
    print(f"[INFO] Model has {num_blocks} transformer blocks, benchmarking block {args.block}")

    if args.block >= num_blocks:
        raise ValueError(f"Block index {args.block} out of range (model has {num_blocks} blocks)")

    # Create inputs once (can be reused across backends)
    print(f"\nPreparing inputs for single block...")
    inputs = prepare_single_block_inputs(model, args.frames, args.height, args.width, device=device)

    print(f"Input shapes:")
    for key, val in inputs.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: {val.shape}")
        elif isinstance(val, tuple):
            print(f"  {key}: tuple of {len(val)} tensors")

    # Benchmark each backend
    results = []
    for i, backend in enumerate(backends):
        print(f"\n{'='*80}")
        print(f"Backend {i+1}/{len(backends)}: {backend.upper()}")
        print(f"{'='*80}")

        # Set processor on model
        model = set_mochi_attention_processor(model, backend)

        # Extract the specific block to benchmark
        block = model.transformer_blocks[args.block]

        result = benchmark_backend(
            block,
            backend,
            inputs,
            block_idx=args.block,
            warmup=args.warmup,
            active=args.active,
            use_compile=args.compile,
            save_trace=args.save_traces,
            trace_dir=args.trace_dir,
        )
        results.append(result)

        # Clear compilation cache between backends
        if args.compile:
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    # Cleanup model
    print(f"\n[INFO] Cleaning up model...")
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Backend':<15} {'Avg Time (ms)':<15} {'Memory (GB)':<12} {'Trace':<50}")
    print("-" * 92)

    for result in results:
        trace_str = result["trace_path"] if result["trace_path"] else "N/A"
        print(f"{result['backend']:<15} {result['avg_time']:<15.2f} {result['memory_gb']:<12.2f} {trace_str:<50}")

    # Find best and worst
    best = min(results, key=lambda x: x["avg_time"])
    worst = max(results, key=lambda x: x["avg_time"])
    speedup = worst["avg_time"] / best["avg_time"]

    print(f"\n{'='*80}")
    print("PERFORMANCE")
    print(f"{'='*80}")
    print(f"Best:    {best['backend']:<15} {best['avg_time']:.2f} ms")
    print(f"Worst:   {worst['backend']:<15} {worst['avg_time']:.2f} ms")
    print(f"Speedup: {speedup:.2f}x ({worst['backend']} → {best['backend']})")

    # Show differences from best
    print(f"\nRelative to best ({best['backend']}):")
    for result in sorted(results, key=lambda x: x["avg_time"]):
        if result['backend'] == best['backend']:
            print(f"  {result['backend']:<15} {result['avg_time']:>8.2f} ms (baseline)")
        else:
            slowdown = result['avg_time'] / best['avg_time']
            diff_ms = result['avg_time'] - best['avg_time']
            print(f"  {result['backend']:<15} {result['avg_time']:>8.2f} ms (+{diff_ms:6.2f} ms, {slowdown:.2f}x slower)")


if __name__ == "__main__":
    main()
