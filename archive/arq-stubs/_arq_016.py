"""Autoresearch 016 — QK-Norm (LayerNorm on Q,K head-dim)."""
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_qk_layernorm: bool = True


if __name__ == "__main__":
    import sys
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
