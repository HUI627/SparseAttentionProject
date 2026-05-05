from __future__ import annotations
import math
import torch
from utils.csr_utils import CSRMask


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def csr_to_bool_mask(csr: CSRMask, seq_len: int, device: str) -> torch.Tensor:
    bs = csr.block_size
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    row_ptr = csr.row_ptr.cpu().tolist(); cols = csr.col_ind.cpu().tolist()
    for qb in range(csr.num_q_blocks):
        q0, q1 = qb * bs, min(seq_len, (qb + 1) * bs)
        for pos in range(row_ptr[qb], row_ptr[qb + 1]):
            kb = cols[pos]
            k0, k1 = kb * bs, min(seq_len, (kb + 1) * bs)
            mask[q0:q1, k0:k1] = True
    return mask


def block_sparse_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, csr: CSRMask, causal: bool = False) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = csr_to_bool_mask(csr, q.shape[-2], q.device)
    if causal:
        n = q.shape[-2]
        mask = mask & ~torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def prune_csr_by_block_score(q: torch.Tensor, k: torch.Tensor, csr: CSRMask, topk_blocks: int, threshold: float | None = None) -> CSRMask:
    """Second-stage dynamic pruning. Scores each candidate block by max QK score upper proxy.

    For each query block, compute ||Q_block||_max * ||K_block||_max as a cheap upper-bound-style score,
    then keep top-k or blocks above threshold. This function is intentionally PyTorch-level so the
    experiment can validate the pruning policy before moving the logic into Triton.
    """
    bs = csr.block_size
    row_ptr = csr.row_ptr.cpu().tolist(); cols = csr.col_ind.cpu().tolist()
    q_norm = q.float().norm(dim=-1).amax(dim=(0, 1))  # [S]
    k_norm = k.float().norm(dim=-1).amax(dim=(0, 1))  # [S]
    rows = []
    for qb in range(csr.num_q_blocks):
        q0, q1 = qb * bs, min(q.shape[-2], (qb + 1) * bs)
        q_score = q_norm[q0:q1].max().item()
        scored = []
        for pos in range(row_ptr[qb], row_ptr[qb + 1]):
            kb = cols[pos]
            k0, k1 = kb * bs, min(k.shape[-2], (kb + 1) * bs)
            score = q_score * k_norm[k0:k1].max().item()
            scored.append((score, kb))
        if threshold is not None:
            kept = [kb for score, kb in scored if score >= threshold]
        else:
            kept = [kb for _, kb in sorted(scored, reverse=True)[:max(1, min(topk_blocks, len(scored)))]]
        rows.append(kept or ([scored[0][1]] if scored else []))
    from utils.csr_utils import _finish
    import time
    return _finish(rows, csr.num_k_blocks, csr.block_size, csr.pattern + "+stage2", time.perf_counter(), str(q.device))
