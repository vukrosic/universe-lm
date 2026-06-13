"""Autoresearch 121 — trt: Prodigy Parameter-Free AdamW (Mishchenko & Defazio 2023).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `Prodigy`: a D-Adaptation successor that uses a
continuous Adam-style gradient similarity for the ramp-up. The 2-D Muon
path is unchanged.
"""
import sys
from configs.llm_config import Tiny1M3MProdigyConfig


class C(Tiny1M3MProdigyConfig):
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
