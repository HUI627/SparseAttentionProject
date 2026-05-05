from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    batch_size: int = 1
    num_heads: int = 8
    seq_len: int = 2048
    head_dim: int = 64
    block_size: int = 64
    local_window_blocks: int = 2
    global_blocks: int = 2
    random_blocks: int = 2
    grouped_num_groups: int = 8
    grouped_pick_per_group: int = 1
    topk_blocks: int = 8
    threshold: float | None = None
    dtype: str = "float16"
    causal: bool = False
    seed: int = 1234
