"""Autoresearch 120 — trt: D-Adaptation Automatic LR Discovery (Defazio 2023).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `DAdaptAdamW`: a thin AdamW subclass that maintains
a per-group scalar `D` and derives the effective LR as `lr_t = D / ‖g_t‖`.
Muon 2-D path is unchanged.
"""
import sys
from configs.llm_config import Tiny1M3MDAdaptConfig


class C(Tiny1M3MDAdaptConfig):
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
