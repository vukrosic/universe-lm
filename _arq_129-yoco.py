"""Autoresearch 129 — trt: YOCO: You Only Cache Once
(Sun et al. 2024, arXiv:2405.05254, ICLR 2024 workshop).

A/B vs the plain tiny1m3m baseline. Splits the 12L model at layer 6: the
lower half (layers 0..5) runs standard self-attention; the upper half
(layers 6..11) uses `YOCOLlamaBlock` whose attention reads a SHARED
`(K_g, V_g)` cache projected from the lower half's final residual stream.
"""
import sys
from configs.llm_config import Tiny1M3MYOCOConfig


class C(Tiny1M3MYOCOConfig):
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
