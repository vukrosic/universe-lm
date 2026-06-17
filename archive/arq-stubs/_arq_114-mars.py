"""Autoresearch 114 — trt: MARS Variance-Reduced AdamW (Yuan et al. 2024).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `MARSAdamW`: a thin AdamW subclass that adds a
lag-based variance-reduction correction `g̃_t = g_t + mix_coef *
(m_{t-lag} − m_{t-2*lag})` to the gradient input. The 2-D Muon path is
unchanged. `lag=10`, `mix_coef=0.5` per paper.
"""
import sys
from configs.llm_config import Tiny1M3MMARSConfig


class C(Tiny1M3MMARSConfig):
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
