import torch
from .config import ExperimentConfig


def make_qkv(cfg: ExperimentConfig, device: str):
    torch.manual_seed(cfg.seed)
    dtype = torch.float16 if cfg.dtype == "float16" and device.startswith("cuda") else torch.float32
    shape = (cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v
