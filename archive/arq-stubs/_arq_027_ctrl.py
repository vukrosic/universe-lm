"""Autoresearch 027 — ctrl: plain Tiny1M3M (matches 015's plain-Muon ctrl bar).

A/B vs `Tiny1M3MConfig` — both Moonlight (015) and QK-Norm (016) closed
against the plain ctrl. Stacking 015+016 ON, comparing against plain
isolates the joint effect of the stack (additivity test per plan.md).
"""
import sys
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
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
