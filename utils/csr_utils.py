from __future__ import annotations
import time
from dataclasses import dataclass
import torch


@dataclass
class CSRMask:
    row_ptr: torch.Tensor
    col_ind: torch.Tensor
    num_q_blocks: int
    num_k_blocks: int
    block_size: int
    encode_ms: float
    pattern: str

    @property
    def nnz_blocks(self) -> int:
        return int(self.col_ind.numel())

    @property
    def density(self) -> float:
        denom = max(1, self.num_q_blocks * self.num_k_blocks)
        return self.nnz_blocks / denom


def _finish(rows: list[list[int]], n_k: int, block_size: int, pattern: str, start: float, device: str) -> CSRMask:
    row_ptr = [0]
    cols = []
    for r in rows:
        clean = sorted(set([c for c in r if 0 <= c < n_k]))
        cols.extend(clean)
        row_ptr.append(len(cols))
    return CSRMask(
        row_ptr=torch.tensor(row_ptr, dtype=torch.int32, device=device),
        col_ind=torch.tensor(cols, dtype=torch.int32, device=device),
        num_q_blocks=len(rows),
        num_k_blocks=n_k,
        block_size=block_size,
        encode_ms=(time.perf_counter() - start) * 1000,
        pattern=pattern,
    )


def dense_csr(seq_len: int, block_size: int, device: str = "cpu") -> CSRMask:
    st = time.perf_counter(); n = (seq_len + block_size - 1) // block_size
    return _finish([list(range(n)) for _ in range(n)], n, block_size, "dense", st, device)


def local_window_csr(seq_len: int, block_size: int, window_blocks: int, causal: bool = False, device: str = "cpu") -> CSRMask:
    st = time.perf_counter(); n = (seq_len + block_size - 1) // block_size
    rows = []
    for i in range(n):
        lo = max(0, i - window_blocks)
        hi = i + 1 if causal else min(n, i + window_blocks + 1)
        rows.append(list(range(lo, hi)))
    return _finish(rows, n, block_size, "local", st, device)


def bigbird_csr(seq_len: int, block_size: int, window_blocks: int, global_blocks: int, random_blocks: int, seed: int, device: str = "cpu") -> CSRMask:
    st = time.perf_counter(); n = (seq_len + block_size - 1) // block_size
    g = list(range(min(global_blocks, n)))
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        local = list(range(max(0, i-window_blocks), min(n, i+window_blocks+1)))
        rand = torch.randperm(n, generator=gen)[:min(random_blocks, n)].tolist()
        rows.append(g + local + rand)
    return _finish(rows, n, block_size, "bigbird", st, device)


def longformer_csr(seq_len: int, block_size: int, window_blocks: int, global_blocks: int, device: str = "cpu") -> CSRMask:
    st = time.perf_counter(); n = (seq_len + block_size - 1) // block_size
    globals_ = list(range(min(global_blocks, n)))
    rows = []
    for i in range(n):
        local = list(range(max(0, i-window_blocks), min(n, i+window_blocks+1)))
        if i in globals_:
            rows.append(list(range(n)))
        else:
            rows.append(globals_ + local)
    return _finish(rows, n, block_size, "longformer", st, device)


def grouped_csr(seq_len: int, block_size: int, groups: int, pick_per_group: int, seed: int, device: str = "cpu") -> CSRMask:
    st = time.perf_counter(); n = (seq_len + block_size - 1) // block_size
    gen = torch.Generator().manual_seed(seed)
    group_ids = torch.arange(n).chunk(max(1, min(groups, n)))
    rows = []
    for _ in range(n):
        picked = []
        for group in group_ids:
            perm = group[torch.randperm(len(group), generator=gen)[:min(pick_per_group, len(group))]]
            picked.extend(perm.tolist())
        rows.append(picked)
    return _finish(rows, n, block_size, "grouped", st, device)


def make_csr(pattern: str, cfg, device: str = "cpu") -> CSRMask:
    if pattern == "dense":
        return dense_csr(cfg.seq_len, cfg.block_size, device)
    if pattern == "local":
        return local_window_csr(cfg.seq_len, cfg.block_size, cfg.local_window_blocks, cfg.causal, device)
    if pattern == "bigbird":
        return bigbird_csr(cfg.seq_len, cfg.block_size, cfg.local_window_blocks, cfg.global_blocks, cfg.random_blocks, cfg.seed, device)
    if pattern == "longformer":
        return longformer_csr(cfg.seq_len, cfg.block_size, cfg.local_window_blocks, cfg.global_blocks, device)
    if pattern == "grouped":
        return grouped_csr(cfg.seq_len, cfg.block_size, cfg.grouped_num_groups, cfg.grouped_pick_per_group, cfg.seed, device)
    raise ValueError(f"unknown pattern: {pattern}")
