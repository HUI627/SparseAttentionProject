import os
import psutil
import torch


def memory_mb(device: str = "cpu") -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def reset_peak_memory(device: str = "cpu") -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
