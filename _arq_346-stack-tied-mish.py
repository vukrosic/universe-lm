"""346-stack-tied-mish — stacking matrix on champion 323 (best single (tied-output) + FFN-activation).

Record path = stack independent right-sign sub-band levers (323 playbook). Three
independent-axis survivors on the 323 champion: tied-output-mlp -0.0164 (output
bottleneck), gmlp-sgu -0.0123 (token-mix), mish-glu -0.0114 (FFN activation). If
~additive a 3-way clears the 0.02 band even at ~50% efficiency. This stub = use_tied_output_mlp+use_mish_glu.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + x2.0 LR) + the flags.
Screen bar: 6.1700 (= champion 6.1720 - 0.02 band).
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
    use_tied_output_mlp: bool = True
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
