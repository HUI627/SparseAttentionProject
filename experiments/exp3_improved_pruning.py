"""
Experiment 3 Improved: Optimized Two-Stage Pruning Evaluation

This experiment evaluates improved pruning strategies:
1. norm: Original ||Q||_max * ||K||_max (baseline)
2. exact: Actual max(Q @ K.T) computation
3. adaptive: Keep blocks contributing to 95% score mass
4. entropy: Mean-based scoring for better distribution
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
from kernels.triton_block_sparse import block_sparse_attention
from kernels.block_sparse_attention_reference import (
    dense_attention,
    prune_csr_by_block_score,
)


def benchmark(fn, warmup: int, iters: int, device: str):
    """Benchmark a function."""
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


def evaluate_improved_pruning():
    """
    Evaluate improved pruning strategies.

    Compare different scoring methods:
    - norm: Original ||Q||_max * ||K||_max
    - exact: Actual max(Q @ K.T)
    - adaptive: Keep blocks contributing to 95% score mass
    - entropy: Mean-based scoring
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_patterns = ['bigbird', 'longformer', 'local']
    pruning_methods = ['norm', 'exact', 'adaptive', 'entropy']
    stage2_topks = [4, 8, 16]
    seq_lens = [2048, 4096]
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

        # Dense baseline
        print("  Computing dense baseline...")
        out_dense, time_dense = benchmark(
            lambda: dense_attention(q, k, v, causal=False),
            warmup, iters, device
        )

        for pattern in stage1_patterns:
            print(f"  Pattern: {pattern}")

            # Stage 1: Static pattern
            csr_stage1 = make_csr(pattern, config, device)
            out_stage1, time_stage1 = benchmark(
                lambda: block_sparse_attention(q, k, v, csr_stage1, use_triton=True),
                warmup, iters, device
            )
            error_stage1 = (out_stage1.float() - out_dense.float()).abs().max().item()

            for topk in stage2_topks:
                print(f"    Top-k: {topk}")

                for method in pruning_methods:
                    print(f"      Method: {method}")

                    # Stage 2: Dynamic pruning with different methods
                    csr_stage2 = prune_csr_by_block_score(
                        q, k, csr_stage1, topk, threshold=None, method=method
                    )
                    out_stage2, time_stage2 = benchmark(
                        lambda: block_sparse_attention(q, k, v, csr_stage2, use_triton=True),
                        warmup, iters, device
                    )
                    error_stage2 = (out_stage2.float() - out_dense.float()).abs().max().item()

                    results.append({
                        'seq_len': seq_len,
                        'stage1_pattern': pattern,
                        'stage2_topk': topk,
                        'pruning_method': method,
                        'sparsity_stage1': csr_stage1.density,
                        'sparsity_stage2': csr_stage2.density,
                        'time_dense_ms': time_dense,
                        'time_stage1_ms': time_stage1,
                        'time_stage2_ms': time_stage2,
                        'error_stage1': error_stage1,
                        'error_stage2': error_stage2,
                        'speedup_stage1': time_dense / time_stage1,
                        'speedup_stage2': time_dense / time_stage2,
                        'accuracy_improvement': error_stage1 - error_stage2,
                    })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Experiment 3 Improved: Optimized Two-Stage Pruning")
    print("=" * 80)

    results_df = evaluate_improved_pruning()

    # Save results
    os.makedirs(ROOT / "results" / "exp3_improved_pruning", exist_ok=True)
    output_path = ROOT / "results" / "exp3_improved_pruning" / "results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Analyze results by method
    print("\n" + "=" * 80)
    print("Performance by Pruning Method:")
    print("=" * 80)

    for method in ['norm', 'exact', 'adaptive', 'entropy']:
        method_df = results_df[results_df['pruning_method'] == method]
        avg_speedup = method_df['speedup_stage2'].mean()
        avg_error = method_df['error_stage2'].mean()
        avg_sparsity = method_df['sparsity_stage2'].mean()
        print(f"\n{method.upper()}:")
        print(f"  Avg Speedup: {avg_speedup:.2f}x")
        print(f"  Avg Error: {avg_error:.6f}")
        print(f"  Avg Sparsity: {avg_sparsity:.4f}")

    # Find best method for each configuration
    print("\n" + "=" * 80)
    print("Best Method by Configuration:")
    print("=" * 80)

    for seq_len in results_df['seq_len'].unique():
        for pattern in results_df['stage1_pattern'].unique():
            for topk in results_df['stage2_topk'].unique():
                subset = results_df[
                    (results_df['seq_len'] == seq_len) &
                    (results_df['stage1_pattern'] == pattern) &
                    (results_df['stage2_topk'] == topk)
                ]
                if len(subset) > 0:
                    # Best = lowest error at similar speedup
                    best = subset.loc[subset['error_stage2'].idxmin()]
                    print(f"seq={seq_len}, pattern={pattern}, topk={topk}: "
                          f"{best['pruning_method']} "
                          f"(error={best['error_stage2']:.4f}, "
                          f"speedup={best['speedup_stage2']:.1f}x)")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

