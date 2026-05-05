"""
Triton implementation of block sparse attention with online softmax.

Key features:
- CSR format traversal for arbitrary sparse patterns
- Online softmax algorithm for memory efficiency
- Fused QK computation, softmax, and attention output
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _block_sparse_attention_kernel(
    Q, K, V, O,
    csr_row_ptr, csr_col_ind,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    batch_size, num_heads, seq_len, head_dim,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Block sparse attention kernel with online softmax.

    Each program processes one query block for one head in one batch.
    Uses online softmax to accumulate attention output block by block.
    """
    # Program IDs
    pid_m = tl.program_id(0)  # query block index
    pid_bh = tl.program_id(1)  # batch * num_heads

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Offsets for Q block (load once, reuse for all K blocks)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block into SRAM
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # output accumulator

    # Get CSR row range for this query block
    row_start = tl.load(csr_row_ptr + pid_m)
    row_end = tl.load(csr_row_ptr + pid_m + 1)

    # Iterate over non-zero K blocks in CSR format
    for pos in range(row_start, row_end):
        # Get K block index from CSR
        k_block_idx = tl.load(csr_col_ind + pos)

        # Offsets for K block
        offs_n = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        # Load K block
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= 1.0 / tl.sqrt(head_dim.to(tl.float32))

        # Update online softmax statistics
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale previous accumulator
        scale_old = tl.exp(m_i - m_new)
        acc = acc * scale_old[:, None]
        l_i = l_i * scale_old

        # Compute attention weights for current block
        p = tl.exp(qk - m_new[:, None])

        # Load V block
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Accumulate attention output
        acc += tl.dot(p.to(v.dtype), v)

        # Update running sum
        l_i += tl.sum(p, axis=1)

        # Update running max
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Write output
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=mask_m[:, None])


def triton_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Block sparse attention using Triton kernel with online softmax.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]
        csr_row_ptr: CSR row pointers [num_q_blocks + 1]
        csr_col_ind: CSR column indices [nnz_blocks]
        block_size: Size of each block

    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    assert k.shape == v.shape == q.shape

    # Allocate output
    o = torch.empty_like(q)

    # Block dimensions
    BLOCK_M = BLOCK_N = block_size
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Number of query blocks
    num_q_blocks = (seq_len + block_size - 1) // block_size

    # Grid dimensions
    grid = (num_q_blocks, batch_size * num_heads)

    # Launch kernel
    _block_sparse_attention_kernel[grid](
        q, k, v, o,
        csr_row_ptr, csr_col_ind,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        block_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return o

