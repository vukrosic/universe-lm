"""Autoresearch 122 — ctrl: plain tiny1m3m baseline (Muon + AdamW).

A/B ctrl for 122-tiger. Identical to the treatment in everything
except the optimizer: Muon on the 2-D non-embed slot, AdamW on
1-D / embedding / norm. This is the standard ctrl recipe — pair
with `_arq_122-tiger.py` (Tiger on the same slot) for the A/B.
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
