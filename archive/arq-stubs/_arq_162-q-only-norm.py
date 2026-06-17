#!/usr/bin/env python
"""Bootstrap for 162-q-only-norm. Subclass Tiny1M3MConfig with
use_q_only_norm=True.

Apply RMSNorm to the query vectors only (not the keys) before the
attention score is computed — start with the standard scale so
step-0 is byte-identical to the baseline (within the spec's fp32
max-abs-diff < 1e-3 tolerance, same trade-off as 159-emb-layernorm).

The wiring already exists in `models/layers.py` (the
`MultiHeadAttention.use_q_only_norm` kwarg and the three forward
branches at nope/cope, post-RoPE, and pre-RoPE) and is read by
`MinimalLLM.__init__` in `models/llm.py`. Default off ⇒ baseline path
bit-identical. This stub only toggles the flag.

NB: imports `Tiny1M3MQOnlyNormConfig` directly rather than declaring
its own `class C(Tiny1M3MConfig): use_q_only_norm: bool = True`,
because the latter would NOT override the parent's dataclass field
default (the parent's `False` is inherited verbatim without a
`@dataclass` re-decoration on the subclass, so the annotation is
ignored and `C().use_q_only_norm` resolves to `False`). The
canonical `Tiny1M3MQOnlyNormConfig` is `@dataclass`-decorated in
`configs/llm_config.py` and correctly sets `True`.

Caches are at /root/universe-lm/_arq_162-q-only-norm.py on the box.
Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MQOnlyNormConfig as C


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
