"""Autoresearch 138 — trt: LookSAM Periodic Sharpness-Aware Minimization
(Du, Yan, Feng, Zhu, Yang, Sui, ICLR 2023, arXiv:2205.13539).

Compute-efficient variant of SAM (119). Standard SAM does a 2-backward
ascent-descent dance on *every* step (2x compute). LookSAM does the
SAM-style 2-backward step only every K steps; the K-1 steps in between
are plain AdamW. With paper default K=5, the effective compute is ~1.2x
of plain AdamW (vs. SAM's 2x), at ~80% of the flatness benefit.

Wraps the AdamW path (1-D / embedding / norm) with `LookSAM`. The 2-D
Muon path is unchanged. `looksam_k=5, looksam_rho=0.05` (paper defaults).
Mutex with `use_sam` (119): if both are on, `use_sam` wins.

Identity at step 0: with K=5 the first 4 steps are plain AdamW
(`step_count=0..3`, `next_is_sam=False`), so the first-step gradient is
bit-identical to AdamW. The first SAM ascent fires at `step_count=4`
(the 5th step). This is *more* bit-identical at step 0 than full SAM
(119), which runs the ascent on the first step.
"""
import sys
from configs.llm_config import Tiny1M3MLookSAMConfig


class C(Tiny1M3MLookSAMConfig):
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
