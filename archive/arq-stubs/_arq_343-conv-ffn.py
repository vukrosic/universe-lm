"""343 — Conv-FFN (depthwise conv inside FFN) stacked on champion 323.

Hypothesis: inserting a depthwise Conv1d inside the FFN (post-FFN, pre-residual add,
identity-init, symmetric) adds a causal sequential mixing step at the point where the
model is processing individual token representations. This is different from pre-attention
conv (short-conv, which operates on the residual stream) — the conv-ffn applies AFTER
the FFN nonlinearity, so it mixes the post-activation features. Idea 157-conv-ffn was NULL
on an older champion; the current 323 champion has ×2.0 LR and could more effectively
learn the conv weights. The post-FFN position is also a different computational slot from
the pre-attention or post-attention mixing positions tested in 329/339/340.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + ×2.0 LR) + conv_ffn.
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
    use_conv_ffn: bool = True


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
