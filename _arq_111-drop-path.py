"""Autoresearch 111 — trt: DropPath / Stochastic Depth (Huang et al. 2016).

A/B vs the plain tiny1m3m baseline. Per-block Bernoulli gate during
training: with probability 1 - p_l skip the whole block; with
probability p_l keep and rescale the residual contribution by 1/p_l.
p_l = 1 - drop_path_max * l / (n_layers - 1), 0-indexed layer
position. drop_path_max=0.1 (paper default).
"""
import sys
from configs.llm_config import Tiny1M3MDropPathConfig


class C(Tiny1M3MDropPathConfig):
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