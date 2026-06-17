"""Autoresearch 112 — trt: Lookahead Optimizer Wrapper (Zhang et al. 2019).

A/B vs the plain tiny1m3m baseline. Wraps the inner Muon+AdamW optimizers
in a Lookahead wrapper: every k inner optimizer steps, the slow weights
pull halfway toward the fast weights (slow ← slow + α·(fast − slow))
and the fast weights are reset to slow. Inner optimizer state is
cleared at the outer step to avoid stale momentum carrying across the
slow reset. k=5, α=0.5 (paper defaults).
"""
import sys
from configs.llm_config import Tiny1M3MLookaheadConfig


class C(Tiny1M3MLookaheadConfig):
    pass


if __name__ == "__main__":
    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ]
    train_llm.main()
