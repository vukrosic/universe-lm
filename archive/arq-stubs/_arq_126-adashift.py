"""Autoresearch 126 — trt: AdaShift Decorrelated Adam via Delayed Gradients (Zhou et al. 2019).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `AdaShift`: an AdamW variant where the 2nd moment
uses a delayed gradient `g_{t-n}²` for decorrelation. The 2-D Muon path
is unchanged. `adashift_n=3` paper default.
"""
import sys
from configs.llm_config import Tiny1M3MAdaShiftConfig


class C(Tiny1M3MAdaShiftConfig):
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
