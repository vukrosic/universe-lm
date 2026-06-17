"""Autoresearch 122 — trt: Tiger: Adaptive Sign-Based Momentum with
EMA Magnitude (Chen et al. 2024, arXiv:2401.16691).

A/B vs the plain tiny1m3m baseline. Tiger replaces Muon on the 2-D
non-embedding, non-norm routing slot. State per param: exp_avg (m,
gradient EMA β1=0.9) and exp_avg_mag (v, magnitude EMA on |g|
β2=0.999). Update: `m / (√v + ε)` — sign-stable but per-parameter
magnitude-adaptive (vs Lion's unit-magnitude sign update). 1-D /
embedding / norm stay on AdamW. Cold-start `m_0=0, v_0=0` ⇒ first
update = 0 ⇒ step-0 val_loss is bit-identical to baseline.
"""
import sys
from configs.llm_config import Tiny1M3MTigerConfig


class C(Tiny1M3MTigerConfig):
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
