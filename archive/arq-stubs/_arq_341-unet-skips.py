"""341 — U-Net skip connections stacked on champion 323.

Hypothesis: U-Net-style skip connections from early transformer layers to late symmetric
layers (via learnable per-channel gates, init 0 = no-op) allow early representational
features to bypass the middle stack and directly influence final-layer outputs. This is
analogous to how U-Net encoder features skip to the decoder, giving the model a path to
preserve low-level features that might otherwise be overwritten. The per-channel gate
structure means the model learns which features to carry forward. With ×2.0 LR the gates
can activate faster from their zero init.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + ×2.0 LR) + unet_skips.
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
    use_unet_skips: bool = True


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
