#!/usr/bin/env python
"""Bootstrap for 174-xpos-decay.

xPos exponential decay on RoPE (Sun et al. 2022, arXiv:2212.10554).
One learnable per-layer scalar `xpos_gamma` (init 0 ⇒ bit-identical to
500k-base RoPE at step 0) multiplies the rotated K by
`exp(-xpos_gamma · t)` per position. See
`autoresearch/ideas/174-xpos-decay/idea.md`.
"""
from configs.llm_config import Tiny1M3MXPosConfig


# Top-level `C` — the daemon's build-smoke target. We alias to the
# canonical treatment subclass (a re-declared `use_xpos: bool = True`
# in a fresh `class C(Tiny1M3MConfig)` would NOT override the parent
# dataclass default under dynamic exec — the field default stays at
# `False` because dataclass field defaults aren't inherited as
# instance defaults; the `class C` body just re-annotates, the field
# table is built only from the leaf class's annotation if it's a
# *new* annotation. Aliasing to the canonical `Tiny1M3MXPosConfig`
# sidesteps that entirely.).
C = Tiny1M3MXPosConfig


if __name__ == "__main__":
    import sys
    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py", "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ]
    train_llm.main()
