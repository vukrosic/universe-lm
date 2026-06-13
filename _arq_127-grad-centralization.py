"""Autoresearch 127 — trt: Gradient Centralization (Yong et al. 2020,
arXiv:2004.01461).

A/B vs the plain tiny1m3m baseline. Routes AdamW-eligible params through
`GCAdamW`, a thin subclass of `torch.optim.AdamW` that subtracts the
per-row mean from each 2-D gradient matrix before the AdamW update runs.
The 2-D Muon path is unchanged. The forward graph is unchanged, so step-0
`val_loss` is bit-identical to baseline.
"""
import sys
from configs.llm_config import Tiny1M3MGCConfig


class C(Tiny1M3MGCConfig):
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
