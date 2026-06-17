"""Autoresearch 030 — trt: FIRE + U-Net sigmoid skips (init -1.5)."""
import sys
from configs.llm_config import Tiny1M3MUNetSigmoidOnFireConfig


class C(Tiny1M3MUNetSigmoidOnFireConfig):
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
