"""
Quick test script for improved two-stage pruning
Tests all 4 methods on a small example to verify correctness
"""
import sys
from pathlib import Path
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


def test_improved_pruning():
    """Quick test of all pruning methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Small test configuration
    config = ExperimentConfig(
        batch_size=1,
        num_heads=4,
        seq_len=1024,
        head_dim=64,
        block_size=64,
    )

    print(f"\nTest configuration:")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Block size: {config.block_size}")

    # Generate test data
    q, k, v = make_qkv(config, device)

    # Dense baseline
    print("\n" + "="*60)
    print("Computing dense baseline...")
    out_dense = dense_attention(q, k, v, causal=False)
    print(f"Dense output shape: {out_dense.shape}")

    # Stage 1: Create bigbird pattern
    print("\n" + "="*60)
    print("Stage 1: Creating bigbird pattern...")
    csr_stage1 = make_csr('bigbird', config, device)
    print(f"Stage 1 sparsity: {csr_stage1.density:.4f}")
    print(f"Stage 1 NNZ blocks: {len(csr_stage1.col_ind)}")

    out_stage1 = block_sparse_attention(q, k, v, csr_stage1, use_triton=True)
    error_stage1 = (out_stage1.float() - out_dense.float()).abs().max().item()
    print(f"Stage 1 error vs dense: {error_stage1:.6f}")

    # Test all pruning methods
    methods = ['norm', 'exact', 'adaptive', 'entropy']
    topk = 8

    print("\n" + "="*60)
    print(f"Stage 2: Testing pruning methods (topk={topk})...")
    print("="*60)

    results = []
    for method in methods:
        print(f"\nMethod: {method.upper()}")
        print("-" * 40)

        try:
            # Prune with this method
            csr_stage2 = prune_csr_by_block_score(
                q, k, csr_stage1, topk, threshold=None, method=method
            )

            # Compute attention
            out_stage2 = block_sparse_attention(q, k, v, csr_stage2, use_triton=True)

            # Measure error
            error_stage2 = (out_stage2.float() - out_dense.float()).abs().max().item()

            print(f"  Sparsity: {csr_stage2.density:.4f}")
            print(f"  NNZ blocks: {len(csr_stage2.col_ind)}")
            print(f"  Error vs dense: {error_stage2:.6f}")
            print(f"  Error reduction: {error_stage1 - error_stage2:.6f}")

            results.append({
                'method': method,
                'sparsity': csr_stage2.density,
                'nnz_blocks': len(csr_stage2.col_ind),
                'error': error_stage2,
                'error_reduction': error_stage1 - error_stage2,
            })

            print(f"  ✓ Success")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<12} {'Sparsity':<12} {'NNZ Blocks':<12} {'Error':<12} {'Δ Error':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<12} {r['sparsity']:<12.4f} {r['nnz_blocks']:<12} "
              f"{r['error']:<12.6f} {r['error_reduction']:<12.6f}")

    # Find best method
    if results:
        best = min(results, key=lambda x: x['error'])
        print("\n" + "="*60)
        print(f"Best method: {best['method'].upper()}")
        print(f"  Lowest error: {best['error']:.6f}")
        print(f"  Sparsity: {best['sparsity']:.4f}")
        print("="*60)

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    test_improved_pruning()
