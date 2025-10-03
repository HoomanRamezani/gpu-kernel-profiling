#!/usr/bin/env python3
"""Benchmark Mochi diffusion transformer with REAL backend switching using custom processors."""

import argparse
import torch
from diffusers import MochiTransformer3DModel
from custom_mochi_processors import set_mochi_attention_processor, get_available_backends
import time
import os

# Import profiler if available
try:
    from torch.profiler import profile, ProfilerActivity, record_function
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False


def create_dummy_inputs(frames, height, width, device="cuda", dtype=torch.float16):
    """Create dummy inputs for Mochi transformer."""
    # Mochi uses patch_size=2, so spatial dims are //2
    latent_height = height // 2
    latent_width = width // 2
    latent_frames = frames

    # Mochi has 12 latent channels
    batch_size = 1
    latent_channels = 12

    hidden_states = torch.randn(
        batch_size,
        latent_channels,
        latent_frames,
        latent_height,
        latent_width,
        device=device,
        dtype=dtype,
    )

    # Text embeddings: max 256 tokens, 4096 dim
    encoder_hidden_states = torch.randn(
        batch_size, 256, 4096, device=device, dtype=dtype
    )

    # Attention mask for text
    encoder_attention_mask = torch.ones(
        batch_size, 256, device=device, dtype=torch.bool
    )

    # Timestep (LongTensor)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "timestep": timestep,
    }


def benchmark_backend(
    model,
    backend,
    inputs,
    warmup=3,
    active=5,
    use_compile=False,
    save_trace=False,
    trace_dir="traces_diffusion_custom",
):
    """Benchmark a specific attention backend."""

    print(f"\n{'='*80}")
    print(f"Benchmarking backend: {backend.upper()}")
    print(f"{'='*80}")

    # Set the custom processor
    model = set_mochi_attention_processor(model, backend)

    # Compile if requested
    if use_compile:
        print("[INFO] Compiling model (this may take several minutes)...")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        model.forward = torch.compile(
            model.forward,
            mode="max-autotune",
            fullgraph=False,
            dynamic=True,
        )

    # Warmup
    print(f"[INFO] Warmup: {warmup} iterations...")
    for i in range(warmup):
        with torch.no_grad():
            output = model(**inputs)
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{warmup}")

    # Benchmark
    print(f"[INFO] Benchmarking: {active} iterations...")
    times = []

    if save_trace and HAS_PROFILER:
        os.makedirs(trace_dir, exist_ok=True)
        frames = inputs["hidden_states"].shape[2]
        height = inputs["hidden_states"].shape[3] * 2
        width = inputs["hidden_states"].shape[4] * 2
        trace_name = f"trace_mochi_{backend}_F{frames}_H{height}_W{width}_B1_custom"
        if use_compile:
            trace_name += "_compiled"
        trace_path = os.path.join(trace_dir, f"{trace_name}.json")

        print(f"[INFO] Saving trace to: {trace_path}")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for i in range(active):
                with torch.no_grad():
                    start = time.time()
                    output = model(**inputs)
                    torch.cuda.synchronize()
                    elapsed = (time.time() - start) * 1000
                    times.append(elapsed)
                print(f"  Iter {i+1}/{active}: {elapsed:.2f} ms")

        prof.export_chrome_trace(trace_path)
        print(f"[saved] Chrome trace -> {trace_path}")

    else:
        for i in range(active):
            with torch.no_grad():
                start = time.time()
                output = model(**inputs)
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
    parser = argparse.ArgumentParser(description="Benchmark Mochi with real backend switching")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--height", type=int, default=64, help="Height")
    parser.add_argument("--width", type=int, default=64, help="Width")
    parser.add_argument("--backends", nargs="+", default=None,
                        help="Backends to test (default: all available)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--active", type=int, default=5, help="Active iterations")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--save-traces", action="store_true", help="Save profiler traces")
    parser.add_argument("--trace-dir", type=str, default="traces_diffusion_custom",
                        help="Directory for traces")

    args = parser.parse_args()

    print("="*80)
    print("MOCHI DIFFUSION TRANSFORMER BENCHMARK (CUSTOM PROCESSORS)")
    print("="*80)
    print(f"Configuration:")
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

    # Load model
    print(f"\n{'='*80}")
    print("Loading Mochi model...")
    print(f"{'='*80}")

    model = MochiTransformer3DModel.from_pretrained(
        "genmo/mochi-1-preview",
        subfolder="transformer",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")

    print(f"Model loaded: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Create inputs
    print(f"\nCreating dummy inputs...")
    inputs = create_dummy_inputs(args.frames, args.height, args.width)

    print(f"Input shapes:")
    for key, val in inputs.items():
        print(f"  {key}: {val.shape if hasattr(val, 'shape') else val}")

    # Benchmark each backend
    results = []
    for backend in backends:
        # Reload model for each backend to avoid cross-contamination
        torch.cuda.empty_cache()

        result = benchmark_backend(
            model,
            backend,
            inputs,
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
