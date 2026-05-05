"""
Experiment 1: Unified CSR Framework vs Specialized Implementations

This experiment compares our unified CSR-based approach against specialized
implementations for different sparse patterns.
"""
import os
import sys
import time
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.config import ExperimentConfig
from models.data_loader import make_qkv
from utils.csr_utils import make_csr
from utils.memory_utils import memory_mb, reset_peak_memory
from kernels.triton_block_sparse import block_sparse_attention


def benchmark(fn, warmup: int, iters: int, device: str):
    """Benchmark a function with warmup and multiple iterations."""
    for _ in range(warmup):
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        result = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters

    return result, elapsed_ms


def measure_pattern_switching_overhead(patterns, config, q, k, v, device, warmup=2, iters=5):
    """
    Measure the overhead of switching between different sparse patterns.

    Our unified kernel should have zero switching overhead since it's the same kernel.
    Specialized implementations would need to switch kernels.
    """
    results = []

    # Pre-generate all CSR masks
    csrs = {pattern: make_csr(pattern, config, device) for pattern in patterns}

    # Measure time for each pattern individually
    for pattern in patterns:
        csr = csrs[pattern]
        _, time_ms = benchmark(
            lambda: block_sparse_attention(q, k, v, csr, use_triton=True),
            warmup, iters, device
        )
        results.append({'pattern': pattern, 'time_ms': time_ms})

    # Measure time when switching between patterns
    # (simulates a model using different patterns in different layers)
    def switch_patterns():
        outputs = []
        for pattern in patterns:
            out = block_sparse_attention(q, k, v, csrs[pattern], use_triton=True)
            outputs.append(out)
        return outputs

    _, switch_time_ms = benchmark(switch_patterns, warmup, iters, device)
    avg_time_per_pattern = switch_time_ms / len(patterns)

    return results, avg_time_per_pattern


def compare_unified_vs_specialized():
    """
    Compare unified CSR framework vs specialized implementations.

    Note: Since we don't have actual specialized implementations (FlashAttention, xFormers),
    we compare Triton vs PyTorch reference as a proxy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patterns = ['local', 'bigbird', 'longformer', 'grouped']
    seq_lens = [1024, 2048, 4096, 8192]
    warmup, iters = 2, 5

    results = []

    for seq_len in seq_lens:
        print(f"\nTesting seq_len={seq_len}...")

        config = ExperimentConfig(
            batch_size=1,
            num_heads=8,
            seq_len=seq_len,
            head_dim=64,
            block_size=64,
        )

        q, k, v = make_qkv(config, device)

        for pattern in patterns:
            print(f"  Pattern: {pattern}")

            csr = make_csr(pattern, config, device)

            # Our unified Triton implementation
            reset_peak_memory(device)
            out_triton, time_triton = benchmark(
                lambda: block_sparse_attention(q, k, v, csr, use_triton=True),
                warmup, iters, device
            )
            mem_triton = memory_mb(device)

            # PyTorch reference (proxy for specialized implementation)
            reset_peak_memory(device)
            out_pytorch, time_pytorch = benchmark(
                lambda: block_sparse_attention(q, k, v, csr, use_triton=False),
                warmup, iters, device
            )
            mem_pytorch = memory_mb(device)

            # Verify correctness
            max_error = (out_triton.float() - out_pytorch.float()).abs().max().item()

            results.append({
                'seq_len': seq_len,
                'pattern': pattern,
                'time_triton_ms': time_triton,
                'time_pytorch_ms': time_pytorch,
                'speedup': time_pytorch / time_triton,
                'mem_triton_mb': mem_triton,
                'mem_pytorch_mb': mem_pytorch,
                'max_error': max_error,
                'sparsity': csr.density,
            })

        # Measure pattern switching overhead
        print(f"  Measuring pattern switching overhead...")
        switch_results, avg_switch_time = measure_pattern_switching_overhead(
            patterns, config, q, k, v, device, warmup, iters
        )

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Experiment 1: Unified CSR Framework vs Specialized Implementations")
    print("=" * 80)

    results_df = compare_unified_vs_specialized()

    # Save results
    os.makedirs(ROOT / "results" / "exp1_unified_framework", exist_ok=True)
    output_path = ROOT / "results" / "exp1_unified_framework" / "results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

