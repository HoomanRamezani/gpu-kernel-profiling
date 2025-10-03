#!/usr/bin/env python3
"""Analyze Chrome trace to break down operation timing."""

import json
import sys
from collections import defaultdict
import re

def analyze_trace(trace_path):
    """Parse Chrome trace and categorize operations by type."""

    with open(trace_path, 'r') as f:
        trace_data = json.load(f)

    # Extract events
    events = trace_data.get('traceEvents', [])

    # Categories for grouping
    categories = {
        'attention': defaultdict(float),
        'convolution': defaultdict(float),
        'normalization': defaultdict(float),
        'activation': defaultdict(float),
        'linear': defaultdict(float),
        'embedding': defaultdict(float),
        'other': defaultdict(float),
    }

    total_time = defaultdict(float)
    kernel_counts = defaultdict(int)

    # Categorize each kernel
    for event in events:
        if event.get('ph') != 'X':  # Only duration events
            continue

        name = event.get('name', '')
        dur = event.get('dur', 0) / 1000.0  # Convert to ms

        if dur == 0:
            continue

        # Categorize based on kernel name
        name_lower = name.lower()

        if any(x in name_lower for x in ['attention', 'attn', 'scaled_dot_product', 'sdpa', 'bmm', 'baddbmm']):
            categories['attention'][name] += dur
            total_time['attention'] += dur
            kernel_counts['attention'] += 1
        elif any(x in name_lower for x in ['conv', 'convolution']):
            categories['convolution'][name] += dur
            total_time['convolution'] += dur
            kernel_counts['convolution'] += 1
        elif any(x in name_lower for x in ['norm', 'layer_norm', 'group_norm', 'rms_norm']):
            categories['normalization'][name] += dur
            total_time['normalization'] += dur
            kernel_counts['normalization'] += 1
        elif any(x in name_lower for x in ['silu', 'gelu', 'relu', 'swish', 'activation']):
            categories['activation'][name] += dur
            total_time['activation'] += dur
            kernel_counts['activation'] += 1
        elif any(x in name_lower for x in ['linear', 'addmm', 'matmul', 'gemm', 'mm']):
            categories['linear'][name] += dur
            total_time['linear'] += dur
            kernel_counts['linear'] += 1
        elif any(x in name_lower for x in ['embed', 'embedding']):
            categories['embedding'][name] += dur
            total_time['embedding'] += dur
            kernel_counts['embedding'] += 1
        else:
            if 'cudnn' in name_lower or 'kernel' in name_lower or 'elementwise' in name_lower:
                categories['other'][name] += dur
                total_time['other'] += dur
                kernel_counts['other'] += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"Trace Analysis: {trace_path}")
    print(f"{'='*80}\n")

    total_gpu_time = sum(total_time.values())

    print(f"{'Operation Type':<20} {'Time (ms)':<15} {'% of Total':<12} {'Count':<10}")
    print(f"{'-'*70}")

    for cat in ['convolution', 'linear', 'attention', 'normalization', 'activation', 'embedding', 'other']:
        time_ms = total_time[cat]
        pct = (time_ms / total_gpu_time * 100) if total_gpu_time > 0 else 0
        count = kernel_counts[cat]
        print(f"{cat:<20} {time_ms:>12.2f} ms {pct:>10.1f}% {count:>10}")

    print(f"{'-'*70}")
    print(f"{'TOTAL':<20} {total_gpu_time:>12.2f} ms {'100.0':>10}%")

    # Print top kernels for each category
    print(f"\n{'='*80}")
    print("Top 5 kernels by category:")
    print(f"{'='*80}\n")

    for cat in ['attention', 'convolution', 'linear', 'normalization']:
        if total_time[cat] > 0:
            print(f"\n{cat.upper()}:")
            print(f"{'-'*70}")
            sorted_kernels = sorted(categories[cat].items(), key=lambda x: x[1], reverse=True)
            for i, (kernel, time) in enumerate(sorted_kernels[:5], 1):
                pct = (time / total_time[cat] * 100)
                # Truncate long kernel names
                kernel_short = kernel[:60] + "..." if len(kernel) > 60 else kernel
                print(f"  {i}. {kernel_short}")
                print(f"     {time:>10.2f} ms ({pct:.1f}% of {cat})")

    return total_time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace.py <trace.json>")
        sys.exit(1)

    analyze_trace(sys.argv[1])
