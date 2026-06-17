"""Autoresearch 139 — trt: Lion sign-based optimizer (Chen et al. 2023, arXiv:2302.06675).

A/B vs the plain tiny1m3m baseline. Replaces the 2-D non-embed/non-norm
optimization slot with `Lion`: `update = sign(β1·m + (1−β1)·g)`, with β2·m
EMA on the gradient. The 1-D / embedding / norm / head path stays on AdamW.
Defaults: lr=3e-4, β1=0.9, β2=0.98. Decoupled weight decay in float32.
"""
import sys
from configs.llm_config import Tiny1M3MLionConfig


class C(Tiny1M3MLionConfig):
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