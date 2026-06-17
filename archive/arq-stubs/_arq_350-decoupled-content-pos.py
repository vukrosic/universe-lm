"""350-decoupled-content-pos — NEW attention-structure axis on champion 323 (DeBERTa-style disentangled content-vs-position attention paths).

Stacking axis CLOSED (344 anti-stacked, 345 best -0.0154 sub-band, 346 worse than solo).
Pivoting to genuinely-new structural attention mechanisms. Not in closed.md, wired in
models/llm.py. RULE 0 (structural attention mechanism). EV uncertain (tiny tier looks
near-saturated) but RULE = keep mining new structural levers, never stall GPU.

Stack: 323 champion (combo env + deepnet_alpha + poly_alibi + mom0.90 + x2.0 LR) + use_decoupled_content_pos.
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
    use_decoupled_content_pos: bool = True
if __name__ == "__main__":
    import sys, train_llm
    sys.modules["__main__"].C = C
    sys.argv=["train_llm.py","--config_class","__main__.C","--seed","42","--dataset_path","processed_data/pretrain_1B","--warmup","false"]
    train_llm.main()
