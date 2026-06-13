"""Autoresearch 135 — trt: Adan Adaptive Nesterov Momentum with N-Step Lookback (Xie et al. 2022).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `Adan`: an AdamW variant that combines (1) a
1-step first moment, (2) an N-step lookback variance estimate, and
(3) a Nesterov-style extrapolated gradient. The 2-D Muon path is
unchanged. `adan_n_lookback=4` and `adan_lookahead_beta=0.5` are the
paper's defaults.
"""
import sys
from configs.llm_config import Tiny1M3MAdanConfig


class C(Tiny1M3MAdanConfig):
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
