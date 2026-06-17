"""338 — MishGLU FFN activation stacked on champion 323.

Hypothesis: Mish activation (x·tanh(softplus(x))) inside a GLU gate gives a smoother,
more expressive FFN than the default GELU. Orthogonal to SwiGLU (171-swiglu-alibi NULL)
because SwiGLU uses SiLU (σ(x)·x) as the gate; Mish has a different curvature profile
(non-monotone, negative values for x<~-0.3). The step-0 probe showed a 0.0881 logit diff
on the champion path — genuinely fires. Gate proj zero-inits mean the mechanism is dormant
at step 0 and must be learned; with ×2.0 LR the model has a better chance of learning it.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + ×2.0 LR) + mish_glu.
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
    use_mish_glu: bool = True


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
