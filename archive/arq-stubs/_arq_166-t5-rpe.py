#!/usr/bin/env python
"""Bootstrap for 166-t5-rpe. Subclass Tiny1M3MConfig with
use_t5_rpe=True, t5_rpe_buckets=32.

Add a learnable per-head logit bias indexed by
`bucket(|i-j|) = floor(log2(|i-j|+1)).clamp_max(B-1)` (T5-style
bucketed relative position bias; Raffel et al. JMLR 2020,
arXiv:1910.10683; re-used in BigBird, REALM, LongT5).

`rpe_bias = zeros(H, B)` init ⇒ `scores + 0` is bit-identical
to the no-RPE baseline at step 0. The lever composes
additively with RoPE / FIRE / CoPE / per-head-temp / per-head-
logit-bias (all live on the score side). Forces the manual
attention path so the bucket-indexed bias can't go through
SDPA's flash kernel. Cost: +1,536 params (+0.16% — negligible).

NB: imports `Tiny1M3MT5RPEConfig` directly rather than
declaring its own `class C(Tiny1M3MConfig): use_t5_rpe: bool =
True`, because the latter would NOT override the parent's
dataclass field default (the parent's `False` is inherited
verbatim without a `@dataclass` re-decoration on the
subclass, so the annotation is ignored and `C().use_t5_rpe`
resolves to `False`). The canonical `Tiny1M3MT5RPEConfig` is
`@dataclass`-decorated in `configs/llm_config.py` and
correctly sets `True`.

Caches are at /root/universe-lm/_arq_166-t5-rpe.py on the box.
Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MT5RPEConfig as C


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
