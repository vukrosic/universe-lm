"""Autoresearch 128 — trt: Spectral Decoupling (Yong et al. 2022,
arXiv:2202.05380, NeurIPS 2022).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `SDAdamW` — a thin subclass of `torch.optim.AdamW`
that projects each per-param gradient off the weight direction before
delegating to AdamW's `.step()`. The 2-D Muon path is unchanged.
"""
import sys
from configs.llm_config import Tiny1M3MSDConfig


class C(Tiny1M3MSDConfig):
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
