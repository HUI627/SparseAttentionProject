"""
Experiment 3: Two-Stage Pruning Evaluation

This experiment evaluates the effectiveness of two-stage pruning:
Stage 1: Static sparse pattern (e.g., bigbird)
Stage 2: Dynamic pruning based on QK scores
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


def evaluate_two_stage_pruning():
    """
    Evaluate two-stage pruning strategy.

    Compare:
    1. Dense baseline
    2. Stage 1 only (static pattern)
    3. Stage 2 only (direct pruning from dense)
    4. Two-stage (static + dynamic)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1_patterns = ['bigbird', 'longformer', 'local']
    stage2_topks = [4, 8, 16, 32]
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

                # Stage 2: Dynamic pruning from stage 1
                csr_stage2 = prune_csr_by_block_score(q, k, csr_stage1, topk, threshold=None)
                out_stage2, time_stage2 = benchmark(
                    lambda: block_sparse_attention(q, k, v, csr_stage2, use_triton=True),
                    warmup, iters, device
                )
                error_stage2 = (out_stage2.float() - out_dense.float()).abs().max().item()

                # For comparison: direct pruning from dense
                # (create a dense CSR and prune it)
                csr_dense = make_csr('dense', config, device)
                csr_direct = prune_csr_by_block_score(q, k, csr_dense, topk, threshold=None)
                out_direct, time_direct = benchmark(
                    lambda: block_sparse_attention(q, k, v, csr_direct, use_triton=True),
                    warmup, iters, device
                )
                error_direct = (out_direct.float() - out_dense.float()).abs().max().item()

                results.append({
                    'seq_len': seq_len,
                    'stage1_pattern': pattern,
                    'stage2_topk': topk,
                    'sparsity_stage1': csr_stage1.density,
                    'sparsity_stage2': csr_stage2.density,
                    'sparsity_direct': csr_direct.density,
                    'time_dense_ms': time_dense,
                    'time_stage1_ms': time_stage1,
                    'time_stage2_ms': time_stage2,
                    'time_direct_ms': time_direct,
                    'error_stage1': error_stage1,
                    'error_stage2': error_stage2,
                    'error_direct': error_direct,
                    'two_stage_better': error_stage2 < error_direct,
                    'speedup_stage1': time_dense / time_stage1,
                    'speedup_stage2': time_dense / time_stage2,
                    'speedup_direct': time_dense / time_direct,
                })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Experiment 3: Two-Stage Pruning Evaluation")
    print("=" * 80)

    results_df = evaluate_two_stage_pruning()

    # Save results
    os.makedirs(ROOT / "results" / "exp3_two_stage_pruning", exist_ok=True)
    output_path = ROOT / "results" / "exp3_two_stage_pruning" / "results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Analyze results
    print("\n" + "=" * 80)
    print("Key Findings:")
    print("=" * 80)
    two_stage_wins = results_df['two_stage_better'].sum()
    total = len(results_df)
    print(f"Two-stage better than direct pruning: {two_stage_wins}/{total} ({two_stage_wins/total*100:.1f}%)")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

