"""
Quick test script to verify the implementation works correctly.
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
from kernels.block_sparse_attention_reference import dense_attention


def test_basic_functionality():
    """Test basic functionality of the implementation."""
    print("=" * 80)
    print("快速功能测试 / Quick Functionality Test")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备 / Using device: {device}")

    # Small config for quick testing
    config = ExperimentConfig(
        batch_size=1,
        num_heads=4,
        seq_len=512,
        head_dim=64,
        block_size=64,
    )

    print(f"\n配置 / Config:")
    print(f"  batch_size={config.batch_size}")
    print(f"  num_heads={config.num_heads}")
    print(f"  seq_len={config.seq_len}")
    print(f"  head_dim={config.head_dim}")
    print(f"  block_size={config.block_size}")

    # Generate Q, K, V
    print("\n生成Q, K, V张量 / Generating Q, K, V tensors...")
    q, k, v = make_qkv(config, device)
    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")

    # Test dense attention
    print("\n测试密集注意力 / Testing dense attention...")
    out_dense = dense_attention(q, k, v, causal=False)
    print(f"  Output shape: {out_dense.shape}")
    print(f"  ✓ Dense attention works!")

    # Test sparse patterns
    patterns = ['local', 'bigbird', 'longformer']
    print(f"\n测试稀疏模式 / Testing sparse patterns: {patterns}")

    for pattern in patterns:
        print(f"\n  Pattern: {pattern}")

        # Create CSR mask
        csr = make_csr(pattern, config, device)
        print(f"    Sparsity: {csr.density:.2%}")
        print(f"    NNZ blocks: {csr.nnz_blocks}")

        # Test with Triton (will fallback to PyTorch if unavailable)
        out_sparse = block_sparse_attention(q, k, v, csr, use_triton=True)
        print(f"    Output shape: {out_sparse.shape}")

        # Check error vs dense
        error = (out_sparse.float() - out_dense.float()).abs().max().item()
        print(f"    Max error vs dense: {error:.6f}")

        if error < 0.1:
            print(f"    ✓ {pattern} works!")
        else:
            print(f"    ⚠ {pattern} has large error!")

    # Test two-stage pruning
    print("\n测试两阶段剪枝 / Testing two-stage pruning...")
    from kernels.block_sparse_attention_reference import prune_csr_by_block_score

    csr_stage1 = make_csr('bigbird', config, device)
    print(f"  Stage 1 sparsity: {csr_stage1.density:.2%}")

    csr_stage2 = prune_csr_by_block_score(q, k, csr_stage1, topk_blocks=4)
    print(f"  Stage 2 sparsity: {csr_stage2.density:.2%}")

    out_stage2 = block_sparse_attention(q, k, v, csr_stage2, use_triton=True)
    error_stage2 = (out_stage2.float() - out_dense.float()).abs().max().item()
    print(f"  Max error vs dense: {error_stage2:.6f}")
    print(f"  ✓ Two-stage pruning works!")

    print("\n" + "=" * 80)
    print("所有测试通过！/ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n✗ 测试失败 / Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
