#!/usr/bin/env python
"""Bootstrap for 173-entmax-15.

Entmax-1.5 sparse attention (Peters/Niculae/Martins, ACL 2019,
arXiv:1905.09018). Tsallis α-entmax with α=1.5 replaces softmax in
the manual attention path. Per-head learnable α_h is parameterized
as `α_h = 1 + 0.5·(1 + tanh(α_raw_h))`, init `α_raw_h = 0` ⇒ `α_h = 1`
⇒ the helper short-circuits to `torch.softmax` for byte-identity at
step 0. As training proceeds the optimizer can push `α_raw_h`
positive to make the attention sparser (approaching sparsemax at
α=2). See `autoresearch/ideas/173-entmax-15/idea.md`.

We alias to the canonical `Tiny1M3MEntmaxConfig` because the
dataclass-inheritance pitfall means a fresh
`class C(Tiny1M3MConfig): use_entmax: bool = True` re-declaration
does NOT override the parent's `False` default (the field table is
built only from the leaf class's *new* annotations; bare re-annotation
in a non-`@dataclass` subclass is silently ignored). This is the same
trick `_arq_174-xpos-decay.py` uses for xPos — see the long comment
in `models/layers.py:3844` and the `Tiny1M3MTalkingHeadsConfig` doc
in `configs/llm_config.py` (the latter explicitly references the
pitfall).
"""
from configs.llm_config import Tiny1M3MEntmaxConfig


# Top-level `C` — the daemon's build-smoke target.
C = Tiny1M3MEntmaxConfig


if __name__ == "__main__":
    import sys
    import train_llm

    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class",
        "__main__.C",
        "--seed",
        "42",
        "--dataset_path",
        "processed_data/pretrain_1B",
        "--warmup",
        "false",
    ]
    train_llm.main()