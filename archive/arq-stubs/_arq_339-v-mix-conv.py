"""339 — V-Mix depthwise Conv stacked on champion 323.

Hypothesis: applying a causal depthwise Conv1d on V before the AV weighted sum allows each
attention head to mix V information across nearby token positions before projecting to output.
This is a token-mixing mechanism in V-space, complementary to gMLP-SGU (329, Δ-0.0123 best
novel result) which applies spatial gating to the post-attention output. The V-mix operation
is lighter (identity-init conv, no gating overhead) and operates earlier in the attention
pipeline — potentially a different compute pathway from gMLP-SGU.

Idea 163-v-mix-conv ran in a pre-machine-format era with no recorded verdict; this is a
fresh controlled test on the 323 champion baseline.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + ×2.0 LR) + v_mix_conv.
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
    use_v_mix_conv: bool = True


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
