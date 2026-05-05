from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.experiment_config import parse_args, make_config
from models.data_loader import make_qkv
from utils.csr_utils import make_csr
from utils.memory_utils import memory_mb, reset_peak_memory
from kernels.block_sparse_attention_reference import dense_attention, block_sparse_attention_reference, prune_csr_by_block_score


def synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, warmup: int, iters: int, device: str):
    for _ in range(warmup):
        fn(); synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(); synchronize(device)
    ms = (time.perf_counter() - t0) * 1000 / iters
    return out, ms


def main():
    args = parse_args()
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    os.makedirs(ROOT / "results", exist_ok=True)
    rows = []

    for seq_len in args.seq_lens:
        cfg = make_config(args, seq_len)
        q, k, v = make_qkv(cfg, device)
        reset_peak_memory(device)
        dense_out, dense_ms = bench(lambda: dense_attention(q, k, v, cfg.causal), args.warmup, args.iters, device)
        dense_tokens = cfg.batch_size * seq_len / (dense_ms / 1000)
        dense_mem = memory_mb(device)
        rows.append(dict(seq_len=seq_len, pattern="dense_flash_ref", stage2=False, density=1.0, encode_ms=0.0,
                         latency_ms=dense_ms, tokens_per_s=dense_tokens, memory_mb=dense_mem, max_abs_error=0.0,
                         kept_blocks_ratio=1.0))

        for pattern in args.patterns:
            if pattern == "dense":
                continue
            csr = make_csr(pattern, cfg, device)
            reset_peak_memory(device)
            out, ms = bench(lambda: block_sparse_attention_reference(q, k, v, csr, cfg.causal), args.warmup, args.iters, device)
            err = (out.float() - dense_out.float()).abs().max().item()
            rows.append(dict(seq_len=seq_len, pattern=pattern, stage2=False, density=csr.density, encode_ms=csr.encode_ms,
                             latency_ms=ms, tokens_per_s=cfg.batch_size * seq_len / (ms / 1000), memory_mb=memory_mb(device),
                             max_abs_error=err, kept_blocks_ratio=1.0))

            if args.enable_two_stage:
                pruned = prune_csr_by_block_score(q, k, csr, cfg.topk_blocks, cfg.threshold)
                reset_peak_memory(device)
                out2, ms2 = bench(lambda: block_sparse_attention_reference(q, k, v, pruned, cfg.causal), args.warmup, args.iters, device)
                err2 = (out2.float() - dense_out.float()).abs().max().item()
                rows.append(dict(seq_len=seq_len, pattern=pattern + "+two_stage", stage2=True, density=pruned.density,
                                 encode_ms=csr.encode_ms + pruned.encode_ms, latency_ms=ms2,
                                 tokens_per_s=cfg.batch_size * seq_len / (ms2 / 1000), memory_mb=memory_mb(device),
                                 max_abs_error=err2, kept_blocks_ratio=pruned.nnz_blocks / max(1, csr.nnz_blocks)))

    df = pd.DataFrame(rows)
    out1 = ROOT / "experiments" / "benchmark_results.csv"
    out2 = ROOT / "results" / "latency_results.csv"
    out3 = ROOT / "results" / "accuracy_results.csv"
    df.to_csv(out1, index=False)
    df[["seq_len","pattern","stage2","latency_ms","tokens_per_s","memory_mb","density","kept_blocks_ratio"]].to_csv(out2, index=False)
    df[["seq_len","pattern","stage2","max_abs_error"]].to_csv(out3, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote:\n- {out1}\n- {out2}\n- {out3}")


if __name__ == "__main__":
    main()
