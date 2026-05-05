import argparse
from models.config import ExperimentConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--patterns", nargs="+", default=["dense", "local", "bigbird", "longformer", "grouped"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--topk-blocks", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--enable-two-stage", action="store_true")
    return p.parse_args()


def make_config(args, seq_len: int) -> ExperimentConfig:
    return ExperimentConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=seq_len,
        head_dim=args.head_dim,
        block_size=args.block_size,
        topk_blocks=args.topk_blocks,
    )
