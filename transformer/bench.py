import argparse

from modeling_tools import Qwen2Bench

def format_ms(t_s):
    return f"{t_s * 1000.0:.2f} ms"

def main():
    parser = argparse.ArgumentParser(description="Profile attention implementations on a single layer.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, nargs="+", default=[512, ])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--active", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--outdir", type=str, default="traces_attn")
    parser.add_argument("--compile", action="store_true", help="Wrap the layer with torch.compile.")
    parser.add_argument("--compile-scope", type=str, choices=["attention", "layer"], default="attention")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["eager", "sdpa", "fa2"],
        help=(
            "Which implementations to test. "
            "Supported: eager, sdpa_cudnn, sdpa_flash, sdpa_mem, sdpa_math, fa2, flex, fa3 (fa3 needs to be installed)"
        ),
    )
    args = parser.parse_args()

    results = []

    bench = Qwen2Bench(args.model)

    for seq_len in args.seq:
        for attn_backend in args.modes:
            results.append(bench.run(
                mode=args.compile_scope,
                attn_backend=attn_backend,
                seq_len=seq_len,
                enable_compile=args.compile,
                enable_backward=args.backward,
            ))

    # Pretty summary
    print("\n=== Per-impl timing (lower is better) ===")
    width = max(len(r["impl"]) for r in results) + 2
    print(f"{'impl'.ljust(width)} | per-step (active) | trace")
    print("-" * (width + 36 + 8))
    for r in results:
        print(f"{r['impl'].ljust(width)} | {format_ms(r['per_step_ms']):>16} | {r['trace']}")


if __name__ == "__main__":
    main()
