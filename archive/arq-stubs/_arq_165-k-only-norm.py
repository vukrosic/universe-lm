#!/usr/bin/env python
"""Bootstrap for 165-k-only-norm. Subclass Tiny1M3MConfig with
use_k_only_norm=True.

Apply RMSNorm to the key vectors only (not the queries) before the
attention score is computed — start with the standard scale so step-0
is byte-identical to the baseline (within the spec's fp32 max-abs-diff
< 1e-3 tolerance, same trade-off as 159-emb-layernorm, 162-q-only-norm).

The wiring already exists in `models/layers.py` (the
`MultiHeadAttention.use_k_only_norm` kwarg + `self.k_only_norm` module
registration + the three forward branches at nope/cope, post-RoPE, and
pre-RoPE + the MoA `extra_K` branch) and is read by
`MinimalLLM.__init__` in `models/llm.py`. Default off ⇒ baseline path
bit-identical. This stub only toggles the flag.

NB: imports `Tiny1M3MKOnlyNormConfig` directly rather than declaring
its own `class C(Tiny1M3MConfig): use_k_only_norm: bool = True`,
because the latter would NOT override the parent's dataclass field
default (the parent's `False` is inherited verbatim without a
`@dataclass` re-decoration on the subclass, so the annotation is
ignored and `C().use_k_only_norm` resolves to `False`). The
canonical `Tiny1M3MKOnlyNormConfig` is `@dataclass`-decorated in
`configs/llm_config.py` and correctly sets `True`.

The K-mirror of 162-q-only-norm. Together with 162 (Q-only) and
016-qk-norm (symmetric QK), the three levers form a clean 3-way
orthogonal attribution test for the 016 WIN at tiny1m3m.

Caches are at /root/universe-lm/_arq_165-k-only-norm.py on the box.
Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MKOnlyNormConfig as C


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
