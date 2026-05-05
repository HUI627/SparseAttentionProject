"""
Run all experiments in the unified CSR block sparse attention project.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 80)
    print("统一CSR块稀疏注意力实验套件")
    print("Unified CSR Block Sparse Attention Experiment Suite")
    print("=" * 80)

    experiments = [
        ("exp1_unified_framework", "统一框架对比 / Unified Framework Comparison"),
        ("exp2_online_softmax", "在线Softmax验证 / Online Softmax Verification"),
        ("exp3_two_stage_pruning", "两阶段稀疏评估 / Two-Stage Pruning Evaluation"),
        ("exp4_csr_encoding_overhead", "CSR编码开销分析 / CSR Encoding Overhead Analysis"),
    ]

    for i, (exp_name, exp_desc) in enumerate(experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(experiments)}] {exp_desc}")
        print(f"{'=' * 80}")

        try:
            # Import and run the experiment
            exp_module = __import__(exp_name)
            exp_module.main()
            print(f"\n✓ {exp_name} completed successfully")
        except Exception as e:
            print(f"\n✗ {exp_name} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("所有实验完成！/ All experiments completed!")
    print("结果已保存到 results/ 目录 / Results saved to results/ directory")
    print("=" * 80)


if __name__ == "__main__":
    main()
