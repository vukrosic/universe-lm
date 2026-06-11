"""Autoresearch 022 — ctrl: FIRE-equipped 009 WIN signature (no softpick).

NOTE: the @dataclass decorator IS required — without it,
`use_fire_pe: bool = True` is a type annotation only and the parent
dataclass's __init__ overrides it with the parent default (False).
Verified after the 2026-06-10 wiring bug.
"""
import sys
from dataclasses import dataclass
from configs.llm_config import Tiny1M3MVQGainSWAHighRoPE250KConfig


@dataclass
class C(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    use_fire_pe: bool = True


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
