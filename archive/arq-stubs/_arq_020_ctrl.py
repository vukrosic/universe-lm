"""Autoresearch 020 — ctrl: FIRE-equipped 009 WIN signature."""
import sys
from configs.llm_config import Tiny1M3MVQGainSWAHighRoPE250KConfig


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
