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


def prune_csr_by_block_score(q: torch.Tensor, k: torch.Tensor, csr: CSRMask, topk_blocks: int, threshold: float | None = None, method: str = "norm") -> CSRMask:
    """Second-stage dynamic pruning with multiple scoring methods.

    Args:
        q, k: Query and key tensors [B, H, S, D]
        csr: Stage 1 CSR mask
        topk_blocks: Number of top blocks to keep per query block
        threshold: Optional score threshold (overrides topk if set)
        method: Scoring method - 'norm', 'exact', 'adaptive', or 'entropy'
            - norm: ||Q_block||_max * ||K_block||_max (fast upper bound)
            - exact: max(Q_block @ K_block.T) (accurate but slower)
            - adaptive: Keep blocks contributing to 95% score mass
            - entropy: Mean-based scoring for better distribution

    Returns:
        Pruned CSR mask
    """
    bs = csr.block_size
    row_ptr = csr.row_ptr.cpu().tolist(); cols = csr.col_ind.cpu().tolist()
    scale = 1.0 / math.sqrt(q.shape[-1])

    # Precompute norms for 'norm' method
    if method == "norm":
        q_norm = q.float().norm(dim=-1).amax(dim=(0, 1))  # [S]
        k_norm = k.float().norm(dim=-1).amax(dim=(0, 1))  # [S]

    rows = []
    for qb in range(csr.num_q_blocks):
        q0, q1 = qb * bs, min(q.shape[-2], (qb + 1) * bs)
        q_block = q[:, :, q0:q1, :]  # [B, H, block_len, D]

        scored = []
        for pos in range(row_ptr[qb], row_ptr[qb + 1]):
            kb = cols[pos]
            k0, k1 = kb * bs, min(k.shape[-2], (kb + 1) * bs)
            k_block = k[:, :, k0:k1, :]  # [B, H, block_len, D]

            if method == "norm":
                # Fast: ||Q||_max * ||K||_max upper bound
                q_score = q_norm[q0:q1].max().item()
                k_score = k_norm[k0:k1].max().item()
                score = q_score * k_score
            elif method == "exact":
                # Accurate: actual max(Q @ K.T)
                qk = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
                score = qk.max().item()
            elif method == "entropy":
                # Mean-based: better distribution awareness
                qk = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
                score = qk.mean().item()
            else:
                raise ValueError(f"Unknown method: {method}")

            scored.append((score, kb))

        # Select blocks based on method
        if method == "adaptive":
            # Adaptive: keep blocks contributing to 95% score mass
            if scored:
                scores_only = torch.tensor([s for s, _ in scored])
                scores_sorted, indices = scores_only.sort(descending=True)
                cumsum = scores_sorted.cumsum(0)
                total = cumsum[-1]
                cutoff_idx = (cumsum >= 0.95 * total).nonzero()[0].item() if total > 0 else 0
                kept_indices = indices[:cutoff_idx + 1].tolist()
                kept = [scored[i][1] for i in kept_indices]
            else:
                kept = []
        elif threshold is not None:
            # Threshold-based selection
            kept = [kb for score, kb in scored if score >= threshold]
        else:
            # Top-k selection
            kept = [kb for _, kb in sorted(scored, reverse=True)[:max(1, min(topk_blocks, len(scored)))]]

        # Ensure at least one block per query block
        rows.append(kept or ([scored[0][1]] if scored else []))

    from utils.csr_utils import _finish
    import time
    return _finish(rows, csr.num_k_blocks, csr.block_size, csr.pattern + f"+{method}", time.perf_counter(), str(q.device))
