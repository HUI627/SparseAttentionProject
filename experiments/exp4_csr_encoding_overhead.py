"""
Experiment 4: CSR Encoding Overhead Analysis

This experiment measures the time cost of CSR encoding and compares it
to the attention computation time.
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


def analyze_csr_encoding_overhead():
    """Analyze CSR encoding overhead."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patterns = ['dense', 'local', 'bigbird', 'longformer', 'grouped']
    seq_lens = [1024, 2048, 4096, 8192, 16384]
    warmup, iters = 2, 10

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

            # Measure encoding time
            _, encode_time = benchmark(
                lambda: make_csr(pattern, config, device),
                warmup, iters, device
            )

            # Measure attention computation time
            csr = make_csr(pattern, config, device)
            _, attention_time = benchmark(
                lambda: block_sparse_attention(q, k, v, csr, use_triton=True),
                warmup, iters, device
            )

            overhead_pct = (encode_time / attention_time * 100) if attention_time > 0 else 0

            results.append({
                'seq_len': seq_len,
                'pattern': pattern,
                'encode_ms': encode_time,
                'attention_ms': attention_time,
                'overhead_pct': overhead_pct,
                'sparsity': csr.density,
                'nnz_blocks': csr.nnz_blocks,
            })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Experiment 4: CSR Encoding Overhead Analysis")
    print("=" * 80)

    results_df = analyze_csr_encoding_overhead()

    # Save results
    os.makedirs(ROOT / "results" / "exp4_csr_encoding", exist_ok=True)
    output_path = ROOT / "results" / "exp4_csr_encoding" / "results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Results Summary:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Key Findings:")
    print("=" * 80)
    avg_overhead = results_df['overhead_pct'].mean()
    max_overhead = results_df['overhead_pct'].max()
    print(f"Average encoding overhead: {avg_overhead:.2f}%")
    print(f"Maximum encoding overhead: {max_overhead:.2f}%")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
