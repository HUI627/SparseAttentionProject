"""
Visualization utilities for experiment results.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_unified_framework_comparison(results_df, output_dir):
    """Plot results from Experiment 1: Unified Framework Comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Latency comparison
    patterns = results_df['pattern'].unique()
    for pattern in patterns:
        data = results_df[results_df['pattern'] == pattern]
        axes[0, 0].plot(data['seq_len'], data['time_triton_ms'],
                        label=f'{pattern} (Triton)', marker='o')
        axes[0, 0].plot(data['seq_len'], data['time_pytorch_ms'],
                        label=f'{pattern} (PyTorch)', marker='x', linestyle='--', alpha=0.7)

    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Latency (ms)')
    axes[0, 0].set_title('Latency: Triton vs PyTorch')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Speedup
    speedup_data = results_df.groupby('pattern')['speedup'].mean()
    bars = axes[0, 1].bar(range(len(speedup_data)), speedup_data.values)
    axes[0, 1].set_xticks(range(len(speedup_data)))
    axes[0, 1].set_xticklabels(speedup_data.index, rotation=45)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Baseline', linewidth=2)
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].set_title('Average Speedup (Triton vs PyTorch)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Color bars based on speedup
    for bar, val in zip(bars, speedup_data.values):
        bar.set_color('green' if val > 1.0 else 'orange')

    # 3. Memory usage
    for pattern in patterns:
        data = results_df[results_df['pattern'] == pattern]
        axes[1, 0].plot(data['seq_len'], data['mem_triton_mb'],
                        label=f'{pattern} (Triton)', marker='o')

    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].set_title('Memory Usage')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Sparsity vs Speedup
    axes[1, 1].scatter(results_df['sparsity'], results_df['speedup'],
                       c=results_df['seq_len'], cmap='viridis', s=100, alpha=0.6)
    axes[1, 1].set_xlabel('Sparsity (density)')
    axes[1, 1].set_ylabel('Speedup')
    axes[1, 1].set_title('Sparsity vs Speedup')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Sequence Length')

    plt.tight_layout()
    plt.savefig(output_dir / 'unified_framework_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'unified_framework_comparison.png'}")
    plt.close()


def plot_online_softmax_results(correctness_df, memory_df, compute_df, output_dir):
    """Plot results from Experiment 2: Online Softmax Verification."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Correctness (max error)
    patterns = correctness_df['pattern'].unique()
    x = np.arange(len(correctness_df['seq_len'].unique()))
    width = 0.25

    for i, pattern in enumerate(patterns):
        data = correctness_df[correctness_df['pattern'] == pattern]
        axes[0, 0].bar(x + i * width, data['error_triton_vs_pytorch'],
                       width, label=pattern, alpha=0.8)

    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Max Absolute Error')
    axes[0, 0].set_title('Numerical Accuracy (Triton vs PyTorch)')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(correctness_df['seq_len'].unique())
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Memory efficiency
    axes[0, 1].plot(memory_df['seq_len'], memory_df['mem_triton_mb'],
                    label='Triton (Online)', marker='o', linewidth=2)
    axes[0, 1].plot(memory_df['seq_len'], memory_df['mem_pytorch_mb'],
                    label='PyTorch (Materialized)', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Memory saving percentage
    axes[1, 0].plot(memory_df['seq_len'], memory_df['memory_saving_pct'],
                    marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Memory Saving (%)')
    axes[1, 0].set_title('Memory Saving (Online vs Materialized)')
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 4. Throughput
    axes[1, 1].plot(compute_df['seq_len'], compute_df['tokens_per_sec_triton'],
                    label='Triton', marker='o', linewidth=2)
    axes[1, 1].plot(compute_df['seq_len'], compute_df['tokens_per_sec_pytorch'],
                    label='PyTorch', marker='s', linewidth=2)
    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].set_ylabel('Throughput (tokens/s)')
    axes[1, 1].set_title('Compute Throughput')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'online_softmax_results.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'online_softmax_results.png'}")
    plt.close()


def plot_two_stage_analysis(results_df, output_dir):
    """Plot results from Experiment 3: Two-Stage Pruning."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy: Two-stage vs Direct
    patterns = results_df['stage1_pattern'].unique()
    for pattern in patterns:
        data = results_df[results_df['stage1_pattern'] == pattern]
        axes[0, 0].plot(data['sparsity_stage2'], data['error_stage2'],
                        label=f'{pattern} (Two-stage)', marker='o')
        axes[0, 0].plot(data['sparsity_direct'], data['error_direct'],
                        label=f'{pattern} (Direct)', marker='x', linestyle='--', alpha=0.7)

    axes[0, 0].set_xlabel('Final Sparsity')
    axes[0, 0].set_ylabel('Max Absolute Error')
    axes[0, 0].set_title('Accuracy: Two-Stage vs Direct Pruning')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Speedup comparison
    speedup_comparison = results_df.groupby('stage1_pattern')[['speedup_stage1', 'speedup_stage2']].mean()
    x = np.arange(len(speedup_comparison))
    width = 0.35

    axes[0, 1].bar(x - width/2, speedup_comparison['speedup_stage1'],
                   width, label='Stage 1 only', alpha=0.8)
    axes[0, 1].bar(x + width/2, speedup_comparison['speedup_stage2'],
                   width, label='Two-stage', alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(speedup_comparison.index, rotation=45)
    axes[0, 1].set_ylabel('Speedup vs Dense')
    axes[0, 1].set_title('Speedup Comparison')
    axes[0, 1].legend()
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Win rate of two-stage
    win_rate = results_df.groupby('stage1_pattern')['two_stage_better'].mean() * 100
    bars = axes[1, 0].bar(range(len(win_rate)), win_rate.values)
    axes[1, 0].set_xticks(range(len(win_rate)))
    axes[1, 0].set_xticklabels(win_rate.index, rotation=45)
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].set_title('Two-Stage Better Than Direct (%)')
    axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for bar in bars:
        bar.set_color('green' if bar.get_height() > 50 else 'orange')

    # 4. Sparsity reduction
    axes[1, 1].scatter(results_df['sparsity_stage1'], results_df['sparsity_stage2'],
                       c=results_df['error_stage2'], cmap='viridis', s=100, alpha=0.6)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='No pruning')
    axes[1, 1].set_xlabel('Stage 1 Sparsity')
    axes[1, 1].set_ylabel('Stage 2 Sparsity')
    axes[1, 1].set_title('Sparsity Reduction (Stage 1 → Stage 2)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Error')

    plt.tight_layout()
    plt.savefig(output_dir / 'two_stage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'two_stage_analysis.png'}")
    plt.close()


def plot_csr_encoding_overhead(results_df, output_dir):
    """Plot results from Experiment 4: CSR Encoding Overhead."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    patterns = results_df['pattern'].unique()

    # 1. Encoding time vs sequence length
    for pattern in patterns:
        data = results_df[results_df['pattern'] == pattern]
        axes[0, 0].plot(data['seq_len'], data['encode_ms'],
                        label=pattern, marker='o')

    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Encoding Time (ms)')
    axes[0, 0].set_title('CSR Encoding Time')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Overhead percentage
    for pattern in patterns:
        data = results_df[results_df['pattern'] == pattern]
        axes[0, 1].plot(data['seq_len'], data['overhead_pct'],
                        label=pattern, marker='o')

    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Overhead (%)')
    axes[0, 1].set_title('Encoding Overhead (% of Attention Time)')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Average overhead by pattern
    avg_overhead = results_df.groupby('pattern')['overhead_pct'].mean()
    bars = axes[1, 0].bar(range(len(avg_overhead)), avg_overhead.values)
    axes[1, 0].set_xticks(range(len(avg_overhead)))
    axes[1, 0].set_xticklabels(avg_overhead.index, rotation=45)
    axes[1, 0].set_ylabel('Average Overhead (%)')
    axes[1, 0].set_title('Average Encoding Overhead by Pattern')
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for bar in bars:
        bar.set_color('green' if bar.get_height() < 1.0 else 'orange')

    # 4. Encoding time vs NNZ blocks
    axes[1, 1].scatter(results_df['nnz_blocks'], results_df['encode_ms'],
                       c=results_df['seq_len'], cmap='viridis', s=100, alpha=0.6)
    axes[1, 1].set_xlabel('Number of Non-Zero Blocks')
    axes[1, 1].set_ylabel('Encoding Time (ms)')
    axes[1, 1].set_title('Encoding Time vs NNZ Blocks')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Sequence Length')

    plt.tight_layout()
    plt.savefig(output_dir / 'csr_encoding_overhead.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'csr_encoding_overhead.png'}")
    plt.close()


def main():
    """Generate all plots from experiment results."""
    print("=" * 80)
    print("生成实验结果可视化 / Generating Experiment Visualizations")
    print("=" * 80)

    results_dir = ROOT / "results"

    # Experiment 1
    exp1_dir = results_dir / "exp1_unified_framework"
    if (exp1_dir / "results.csv").exists():
        print("\n[1/4] Plotting Experiment 1...")
        df = pd.read_csv(exp1_dir / "results.csv")
        plot_unified_framework_comparison(df, exp1_dir)

    # Experiment 2
    exp2_dir = results_dir / "exp2_online_softmax"
    if all((exp2_dir / f).exists() for f in ["correctness.csv", "memory.csv", "compute.csv"]):
        print("\n[2/4] Plotting Experiment 2...")
        correctness_df = pd.read_csv(exp2_dir / "correctness.csv")
        memory_df = pd.read_csv(exp2_dir / "memory.csv")
        compute_df = pd.read_csv(exp2_dir / "compute.csv")
        plot_online_softmax_results(correctness_df, memory_df, compute_df, exp2_dir)

    # Experiment 3
    exp3_dir = results_dir / "exp3_two_stage_pruning"
    if (exp3_dir / "results.csv").exists():
        print("\n[3/4] Plotting Experiment 3...")
        df = pd.read_csv(exp3_dir / "results.csv")
        plot_two_stage_analysis(df, exp3_dir)

    # Experiment 4
    exp4_dir = results_dir / "exp4_csr_encoding"
    if (exp4_dir / "results.csv").exists():
        print("\n[4/4] Plotting Experiment 4...")
        df = pd.read_csv(exp4_dir / "results.csv")
        plot_csr_encoding_overhead(df, exp4_dir)

    print("\n" + "=" * 80)
    print("所有可视化完成！/ All visualizations completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

