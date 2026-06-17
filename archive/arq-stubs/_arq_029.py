"""Autoresearch 029 — trt: QK-Norm + V-Norm (per-head LayerNorm on V)."""
import sys
from configs.llm_config import Tiny1M3MVNormOnQKNormConfig


class C(Tiny1M3MVNormOnQKNormConfig):
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
