#!/usr/bin/env python3
"""
Clean Mochi transformer block benchmark.

Mirrors the structure of transformer/bench.py - simple CLI wrapper around MochiBench class.
"""

import argparse

from modeling_tools_mochi_clean import MochiBench


def format_ms(t_s):
    """Convert seconds to milliseconds string."""
    return f"{t_s * 1000.0:.2f} ms"


def main():
    parser = argparse.ArgumentParser(
        description="Profile attention implementations on a single Mochi transformer block."
    )
    parser.add_argument("--model", type=str, default="genmo/mochi-1-preview")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--block", type=int, default=0, help="Which block to profile (0-47)")
    parser.add_argument("--frames", type=int, default=8, help="Number of video frames")
    parser.add_argument("--height", type=int, default=64, help="Spatial height (pixel space)")
    parser.add_argument("--width", type=int, default=64, help="Spatial width (pixel space)")
    parser.add_argument("--compile", action="store_true", help="Wrap the block with torch.compile")
    parser.add_argument("--backward", action="store_true", help="Include backward pass")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["flash_attn2", "cudnn"],
        help=(
            "Which implementations to test. "
            "Supported: cudnn, flash, efficient, math, flash_attn2"
        ),
    )

    args = parser.parse_args()

    # Parse dtype
    import torch
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    results = []

    # Instantiate benchmark class
    bench = MochiBench(
        model_name=args.model,
        block_idx=args.block,
        dtype=dtype,
    )

    # Run benchmarks for each backend
    for attn_backend in args.modes:
        results.append(
            bench.run(
                attn_backend=attn_backend,
                frames=args.frames,
                height=args.height,
                width=args.width,
                enable_compile=args.compile,
                enable_backward=args.backward,
            )
        )

    # Pretty summary (matching transformer benchmark style)
    print("\n=== Per-impl timing (lower is better) ===")
    width = max(len(r["impl"]) for r in results) + 2
    print(f"{'impl'.ljust(width)} | per-step (active) | trace")
    print("-" * (width + 36 + 8))
    for r in results:
        print(f"{r['impl'].ljust(width)} | {format_ms(r['per_step_ms']):>16} | {r['trace']}")


if __name__ == "__main__":
    main()
