"""Autoresearch 131 — trt: LayerDrop: Stochastic Depth for Whole Layers
(Fan, Grave, Joulin 2019, arXiv:1904.09728, ICLR 2020).

A/B vs the plain tiny1m3m baseline. Each `TransformerBlock` forward
applies a Bernoulli(1-p_l) per-batch gate; if the coin is 0, the block
is skipped (identity). Kept blocks are rescaled by `1/p_l` so the
expected residual matches baseline. The 2-D Muon path is unchanged.
The `use_layerdrop` flag is on the base `Tiny1M3MConfig`; we flip it
True here.
"""
import sys
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_layerdrop: bool = True
    layerdrop_p: float = 0.2
    layerdrop_schedule: str = "constant"


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
