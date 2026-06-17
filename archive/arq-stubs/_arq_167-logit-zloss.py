#!/usr/bin/env python
"""Bootstrap for 167-logit-zloss. Subclass Tiny1M3MConfig with
use_z_loss=True and z_loss_lambda=1e-4.

Add the PaLM-style auxiliary training loss
    L_z = λ · mean(log(Z)²),  Z = sum_v exp(logits_v)
to the total train loss via `logits.logsumexp(dim=-1).pow(2).mean()`.
Penalises logit magnitude so the largest logit cannot grow without
bound and the softmax cannot collapse to a near-delta. The wiring
is already in `training/trainer.py:1464-1465, 1548-1552, 1654-1658,
1599, 1702, 1706` — the trainer reads `use_z_loss` and `z_loss_lambda`
via `getattr(config, ..., False / 0.0)` and adds the term when both
are positive. At λ=0 the term is exactly 0 ⇒ baseline path bit-
identical. This stub only toggles the flag.

NB: imports `Tiny1M3MZLossConfig` directly rather than declaring
its own `class C(Tiny1M3MConfig): use_z_loss: bool = True`, because
the latter would NOT override the parent's dataclass field
default (the parent's `False` is inherited verbatim without a
`@dataclass` re-decoration on the subclass, so the annotation is
ignored and `C().use_z_loss` resolves to `False`). The
canonical `Tiny1M3MZLossConfig` is `@dataclass`-decorated in
`configs/llm_config.py` and correctly sets `True` / `1e-4`.

Chowdhery et al. 2022, "PaLM: Scaling Language Modeling with
Pathways" (arXiv:2204.02311, §3.3) — PaLM 540B uses z-loss with
λ=1e-4. The lever is a logit-magnitude penalty, structurally
distinct from the closed 066-070 loss-shape axes (label smoothing,
confidence penalty, unlikelihood, focal loss, MTP head), which
all target *target/prediction* softening. Z-loss targets *logit
magnitude*. Either outcome is informative at 0.94M.

Caches are at /root/universe-lm/_arq_167-logit-zloss.py on the box.
Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MZLossConfig as C


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