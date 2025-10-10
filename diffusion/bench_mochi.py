#!/usr/bin/env python3
"""
Single-layer Mochi transformer block benchmark.

Similar to transformer/bench.py but for Mochi diffusion transformer blocks.
Profiles one block at a time for direct comparison with causal LM benchmarks.
"""

import argparse
from modeling_tools_mochi import MochiBench


def format_ms(t_s):
    """Convert seconds to milliseconds string."""
    return f"{t_s * 1000.0:.2f} ms"


def main():
    parser = argparse.ArgumentParser(
        description="Profile attention implementations on a single Mochi transformer block."
    )
    parser.add_argument("--model", type=str, default="genmo/mochi-1-preview")
    parser.add_argument("--block", type=int, default=0, help="Which block to profile (0-47)")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--frames", type=int, default=8, help="Number of video frames")
    parser.add_argument("--height", type=int, default=64, help="Latent height")
    parser.add_argument("--width", type=int, default=64, help="Latent width")
    parser.add_argument("--text-seq-len", type=int, default=256, help="Text sequence length")
    parser.add_argument("--compile", action="store_true", help="Wrap block with torch.compile")
    parser.add_argument("--backward", action="store_true", help="Include backward pass")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["flash_attn2"],
        help="Attention backends to test: flash_attn2, cudnn, flash, efficient, math",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    import torch
    dtype = getattr(torch, dtype_map[args.dtype])

    results = []

    bench = MochiBench(
        model_name=args.model,
        block_idx=args.block,
        dtype=dtype,
    )

    for attn_backend in args.modes:
        results.append(
            bench.run(
                attn_backend=attn_backend,
                frames=args.frames,
                height=args.height,
                width=args.width,
                text_seq_len=args.text_seq_len,
                enable_compile=args.compile,
                enable_backward=args.backward,
            )
        )

    # Pretty summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Block: {args.block}, Frames: {args.frames}, H: {args.height}, W: {args.width}")
    print(f"Text seq len: {args.text_seq_len}, Compile: {args.compile}, Backward: {args.backward}")
    print("-" * 80)

    width = max(len(r["impl"]) for r in results) + 2
    print(f"{'Implementation'.ljust(width)} | Per-step time     | Trace file")
    print("-" * 80)

    for r in results:
        trace_name = r['trace'].split('/')[-1]
        print(f"{r['impl'].ljust(width)} | {format_ms(r['per_step_ms']):>16} | {trace_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
