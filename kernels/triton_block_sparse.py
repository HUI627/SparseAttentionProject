"""Triton block sparse attention implementation.

This module provides the main entry point for block sparse attention.
It automatically falls back to PyTorch reference implementation if Triton is unavailable.
"""
import torch

try:
    import triton
    from .triton_online_softmax import triton_block_sparse_attention
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton_block_sparse_attention = None

from .block_sparse_attention_reference import block_sparse_attention_reference
from utils.csr_utils import CSRMask


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    csr: CSRMask,
    causal: bool = False,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Block sparse attention with automatic fallback.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]
        csr: CSRMask object containing sparse pattern
        causal: Whether to apply causal masking
        use_triton: Whether to use Triton kernel (if available)

    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    # Use Triton if available and requested
    if use_triton and TRITON_AVAILABLE and torch.cuda.is_available() and q.is_cuda:
        if causal:
            # TODO: Implement causal masking in Triton kernel
            # For now, fall back to reference implementation
            return block_sparse_attention_reference(q, k, v, csr, causal)

        return triton_block_sparse_attention(
            q, k, v,
            csr.row_ptr,
            csr.col_ind,
            csr.block_size,
        )
    else:
        # Fall back to PyTorch reference implementation
        return block_sparse_attention_reference(q, k, v, csr, causal)

