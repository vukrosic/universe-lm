"""344 — Stack: gMLP-SGU + tied-output-mlp on champion 323.

Motivation: the two strongest single-mechanism results on the 323 champion are:
  329-gmlp-sgu         Δ-0.0123  (post-attention spatial gating / token mixing)
  334-tied-output-mlp  Δ-0.0164  (autoencoder-tied output MLP around block stack)
Both are structurally independent (different computational positions, different axes):
gMLP-SGU operates inside the attention block on post-AV features; tied-output-mlp wraps
the entire block stack with an encoder/decoder bottleneck. No parameter overlap.

If they stack even at 60% efficiency: 0.6×(0.0123+0.0164) = 0.0172 — still sub-band.
At 70%: 0.0201 → crosses the 0.0200 screen bar.
At 100% (additive): 0.0287 → strong screen-win.

The stacking science from 323 (mom0.90 + ×2.0 LR super-additive at -0.0278) confirms that
independent mechanisms on the same model CAN stack super-additively. This combo is the
highest-confidence screen-win candidate in the current batch.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + ×2.0 LR)
       + gmlp_sgu + tied_output_mlp.
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
    use_gmlp_sgu: bool = True
    use_tied_output_mlp: bool = True


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
