"""349-tied-wo-across-blocks — NEW structural axis on champion 323 (tie the attention output projection W_O across blocks (soft blend)).

Stacking has topped out (best stack 345 = -0.0154, below best single lever -0.0164;
344 anti-stacked). Per RULE 0 + loop rule 4, opening a genuinely NEW structural axis:
cross-block weight sharing (regularization via tying, 0-to-few new params). Not in
closed.md, wired in models/llm.py. EV honestly LOW (champion well-optimized, tiny tier
near-saturated) but RULE = keep mining new structural levers, never stall GPU.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + x2.0 LR) + use_tied_wo_across_blocks.
Screen bar: 6.1700 (= champion 6.1720 - 0.02 band).
"""
import os
os.environ["ALIBI_SLOPE_INIT"]="geometric"
os.environ["ALIBI_SLOPE_DIST"]="uniform"
os.environ["ALIBI_SLOPE_SCALE"]="3.0"
os.environ["ALIBI_SLOPE_LEARNABLE"]="1"
os.environ["POLY_ALIBI_C_INIT"]="geometric"
os.environ["POLY_ALIBI_C_SCALE"]="3.0"
from dataclasses import dataclass
from configs.llm_config import Tiny1M3MAlibiConfig
@dataclass
class C(Tiny1M3MAlibiConfig):
    use_deepnet_alpha: bool = True
    use_poly_alibi: bool = True
    muon_momentum: float = 0.90
    muon_lr: float = 0.048
    adamw_lr: float = 0.012
    use_tied_wo_across_blocks: bool = True
if __name__ == "__main__":
    import sys, train_llm
    sys.modules["__main__"].C = C
    sys.argv=["train_llm.py","--config_class","__main__.C","--seed","42","--dataset_path","processed_data/pretrain_1B","--warmup","false"]
    train_llm.main()
