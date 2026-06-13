"""Autoresearch 123 — trt: CAME Confidence-guided Adaptive Memory Efficient (Luo et al. 2023).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `CAME`: a confidence-rescaled AdamW variant.
The 2-D Muon path is unchanged. Cold-start `m_0=0, v_0=0` ⇒ first-step
update ≈ 0 ⇒ baseline path byte-identical at step 0.
"""
import sys
from configs.llm_config import Tiny1M3MCAMEConfig


class C(Tiny1M3MCAMEConfig):
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
