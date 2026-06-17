"""Autoresearch 030 — ctrl: FIRE-equipped 009 WIN signature (no U-Net).

NOTE: the @dataclass decorator IS required — without it, the
`use_fire_pe: bool = True` class-attribute is treated as a type
annotation and the parent dataclass's `__init__` sets
`self.use_fire_pe = False` (the parent default). Verified with
`dataclasses.fields(C)`. The 020/023/026 ctrl scripts use the
undecorated form and are silently running FIRE-OFF — flagged in
plan.md as a cross-idea coordination issue for the code-reviewer.
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
