"""Autoresearch 141 — trt: AdaBelief optimizer (Zhuang et al. 2020, arXiv:2010.07468).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `AdaBelief`: same first moment `m = β1·m + (1−β1)·g`
as AdamW, but second moment becomes the variance of the residual:
`s = β2·s + (1−β2)·(g − m)² + ε`. The 2-D Muon path is unchanged.
"""
import sys
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_adabelief: bool = True


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