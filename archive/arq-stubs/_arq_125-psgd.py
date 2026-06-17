"""Autoresearch 125 — trt: PSGD Preconditioned Stochastic Gradient Descent (Li et al. 2024).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `PSGD`: an online-learned preconditioner that
whitens the gradient. The 2-D Muon path is unchanged. PSGD learns Q
(or Q, P for rectangular matrices) per param via a coupled update.
"""
import sys
from configs.llm_config import Tiny1M3MPSGDConfig


class C(Tiny1M3MPSGDConfig):
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
