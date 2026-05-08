"""
Visualization for Improved Two-Stage Pruning Results
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_improved_pruning_results():
    """Generate visualizations for improved pruning experiment."""

    # Load results
    results_path = ROOT / "results" / "exp3_improved_pruning" / "results.csv"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Improved Two-Stage Pruning Analysis', fontsize=16, fontweight='bold')

    methods = ['norm', 'exact', 'adaptive', 'entropy']
    colors = {'norm': '#1f77b4', 'exact': '#ff7f0e', 'adaptive': '#2ca02c', 'entropy': '#d62728'}

    # Plot 1: Speedup vs Error Trade-off
    ax = axes[0, 0]
    for method in methods:
        method_df = df[df['pruning_method'] == method]
        ax.scatter(method_df['error_stage2'], method_df['speedup_stage2'],
                  label=method, alpha=0.6, s=100, color=colors[method])
    ax.set_xlabel('Error vs Dense', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup vs Error Trade-off', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Average Performance by Method
    ax = axes[0, 1]
    method_stats = []
    for method in methods:
        method_df = df[df['pruning_method'] == method]
        method_stats.append({
            'method': method,
            'speedup': method_df['speedup_stage2'].mean(),
            'error': method_df['error_stage2'].mean()
        })

    x = np.arange(len(methods))
    speedups = [s['speedup'] for s in method_stats]
    ax.bar(x, speedups, color=[colors[m] for m in methods], alpha=0.7)
    ax.set_xlabel('Pruning Method', fontsize=12)
    ax.set_ylabel('Average Speedup', fontsize=12)
    ax.set_title('Average Speedup by Method', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Average Error by Method
    ax = axes[0, 2]
    errors = [s['error'] for s in method_stats]
    ax.bar(x, errors, color=[colors[m] for m in methods], alpha=0.7)
    ax.set_xlabel('Pruning Method', fontsize=12)
    ax.set_ylabel('Average Error', fontsize=12)
    ax.set_title('Average Error by Method', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Speedup by Top-k
    ax = axes[1, 0]
    for method in methods:
        topk_speedups = []
        topks = sorted(df['stage2_topk'].unique())
        for topk in topks:
            subset = df[(df['pruning_method'] == method) & (df['stage2_topk'] == topk)]
            topk_speedups.append(subset['speedup_stage2'].mean())
        ax.plot(topks, topk_speedups, marker='o', label=method,
               color=colors[method], linewidth=2, markersize=8)
    ax.set_xlabel('Top-k Blocks', fontsize=12)
    ax.set_ylabel('Average Speedup', fontsize=12)
    ax.set_title('Speedup vs Top-k', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Error by Top-k
    ax = axes[1, 1]
    for method in methods:
        topk_errors = []
        topks = sorted(df['stage2_topk'].unique())
        for topk in topks:
            subset = df[(df['pruning_method'] == method) & (df['stage2_topk'] == topk)]
            topk_errors.append(subset['error_stage2'].mean())
        ax.plot(topks, topk_errors, marker='o', label=method,
               color=colors[method], linewidth=2, markersize=8)
    ax.set_xlabel('Top-k Blocks', fontsize=12)
    ax.set_ylabel('Average Error', fontsize=12)
    ax.set_title('Error vs Top-k', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Sparsity Comparison
    ax = axes[1, 2]
    for method in methods:
        method_df = df[df['pruning_method'] == method]
        sparsities = method_df['sparsity_stage2'].values
        ax.hist(sparsities, alpha=0.5, label=method, bins=15, color=colors[method])
    ax.set_xlabel('Sparsity (Stage 2)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Sparsity Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = ROOT / "results" / "exp3_improved_pruning" / "improved_pruning_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_improved_pruning_results()
