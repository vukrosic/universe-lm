"""340 — ShortConv (Hyena-style causal depthwise conv) stacked on champion 323.

Hypothesis: a short identity-init causal depthwise Conv1d on the residual stream BEFORE
the attention layer acts as a token-mixing pre-filter — giving each position access to
a small local context window before the global attention computation. This is the SSM/Hyena
locality prior applied cheaply. Idea 143-shortconv was "borderline NULL" on an older champion
(beat all 4 same-day controls but didn't formally pass). The current champion (323) has a
×2.0 LR which might give the scalar gate (init 0) enough gradient to activate earlier,
making this mechanism more effective than the borderline pre-323 result.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + ×2.0 LR) + short_conv.
Screen bar: 6.1700 (= champion 6.1720 − 0.02 band).
"""
import os

os.environ["ALIBI_SLOPE_INIT"] = "geometric"
os.environ["ALIBI_SLOPE_DIST"] = "uniform"
os.environ["ALIBI_SLOPE_SCALE"] = "3.0"
os.environ["ALIBI_SLOPE_LEARNABLE"] = "1"
os.environ["POLY_ALIBI_C_INIT"] = "geometric"
os.environ["POLY_ALIBI_C_SCALE"] = "3.0"

from dataclasses import dataclass

from configs.llm_config import Tiny1M3MAlibiConfig


@dataclass
class C(Tiny1M3MAlibiConfig):
    use_deepnet_alpha: bool = True
    use_poly_alibi: bool = True
    muon_momentum: float = 0.90
    muon_lr: float = 0.048
    adamw_lr: float = 0.012
    use_short_conv: bool = True


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
