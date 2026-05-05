"""
Experiment 2: Online Softmax Verification

This experiment verifies the correctness and efficiency of the online softmax algorithm.
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
from kernels.block_sparse_attention_reference import dense_attention


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


def verify_correctness():
    """Verify numerical correctness of online softmax vs reference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patterns = ['local', 'bigbird', 'longformer']
    seq_lens = [1024, 2048, 4096]

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
        out_dense = dense_attention(q, k, v, causal=False)

        for pattern in patterns:
            print(f"  Pattern: {pattern}")

            csr = make_csr(pattern, config, device)

            # Triton online softmax
            out_triton = block_sparse_attention(q, k, v, csr, use_triton=True)

            # PyTorch reference
            out_pytorch = block_sparse_attention(q, k, v, csr, use_triton=False)

            # Compute errors
            error_triton_vs_pytorch = (out_triton.float() - out_pytorch.float()).abs().max().item()
            error_triton_vs_dense = (out_triton.float() - out_dense.float()).abs().max().item()
            error_pytorch_vs_dense = (out_pytorch.float() - out_dense.float()).abs().max().item()

            results.append({
                'seq_len': seq_len,
                'pattern': pattern,
                'sparsity': csr.density,
                'error_triton_vs_pytorch': error_triton_vs_pytorch,
                'error_triton_vs_dense': error_triton_vs_dense,
                'error_pytorch_vs_dense': error_pytorch_vs_dense,
            })

    return pd.DataFrame(results)


def measure_memory_efficiency():
    """Measure memory efficiency of online softmax."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_lens = [1024, 2048, 4096, 8192, 16384]
    pattern = 'bigbird'

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
        csr = make_csr(pattern, config, device)

        # Measure memory for Triton (online softmax)
        reset_peak_memory(device)
        _ = block_sparse_attention(q, k, v, csr, use_triton=True)
        mem_triton = memory_mb(device)

        # Measure memory for PyTorch reference (materializes intermediate results)
        reset_peak_memory(device)
        _ = block_sparse_attention(q, k, v, csr, use_triton=False)
        mem_pytorch = memory_mb(device)

        memory_saving = (mem_pytorch - mem_triton) / mem_pytorch if mem_pytorch > 0 else 0

        results.append({
            'seq_len': seq_len,
            'mem_triton_mb': mem_triton,
            'mem_pytorch_mb': mem_pytorch,
            'memory_saving_pct': memory_saving * 100,
        })

    return pd.DataFrame(results)


def measure_compute_efficiency():
    """Measure compute efficiency (throughput)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_lens = [1024, 2048, 4096, 8192]
    pattern = 'bigbird'
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
        csr = make_csr(pattern, config, device)

        # Benchmark Triton
        _, time_triton = benchmark(
            lambda: block_sparse_attention(q, k, v, csr, use_triton=True),
            warmup, iters, device
        )

        # Benchmark PyTorch
        _, time_pytorch = benchmark(
            lambda: block_sparse_attention(q, k, v, csr, use_triton=False),
            warmup, iters, device
        )

        tokens_per_sec_triton = config.batch_size * seq_len / (time_triton / 1000)
        tokens_per_sec_pytorch = config.batch_size * seq_len / (time_pytorch / 1000)

        results.append({
            'seq_len': seq_len,
            'time_triton_ms': time_triton,
            'time_pytorch_ms': time_pytorch,
            'tokens_per_sec_triton': tokens_per_sec_triton,
            'tokens_per_sec_pytorch': tokens_per_sec_pytorch,
            'speedup': time_pytorch / time_triton,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Experiment 2: Online Softmax Verification")
    print("=" * 80)

    # Test 1: Correctness
    print("\n[1/3] Verifying correctness...")
    correctness_df = verify_correctness()

    # Test 2: Memory efficiency
    print("\n[2/3] Measuring memory efficiency...")
    memory_df = measure_memory_efficiency()

    # Test 3: Compute efficiency
    print("\n[3/3] Measuring compute efficiency...")
    compute_df = measure_compute_efficiency()

    # Save results
    os.makedirs(ROOT / "results" / "exp2_online_softmax", exist_ok=True)
    correctness_df.to_csv(ROOT / "results" / "exp2_online_softmax" / "correctness.csv", index=False)
    memory_df.to_csv(ROOT / "results" / "exp2_online_softmax" / "memory.csv", index=False)
    compute_df.to_csv(ROOT / "results" / "exp2_online_softmax" / "compute.csv", index=False)

    print("\n" + "=" * 80)
    print("Correctness Results:")
    print("=" * 80)
    print(correctness_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Memory Efficiency Results:")
    print("=" * 80)
    print(memory_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Compute Efficiency Results:")
    print("=" * 80)
    print(compute_df.to_string(index=False))


if __name__ == "__main__":
    main()

