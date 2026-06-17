#!/usr/bin/env python
"""Bootstrap for 176-v-pre-av-norm. Subclass Tiny1M3MConfig with
use_v_rmsnorm=True.

Apply RMSNorm to the value vectors (V) along the `d_k` axis BEFORE
the AV product, with a per-head scalar gate `α_h = relu(α_raw_h)`
(init 0 ⇒ identity blend at step 0) and a per-head gain
`γ_h ∈ R^{d_k}` (init 1.0 ⇒ identity gain at step 0). Output
`V_out = (1 − α_h)·V + α_h·RMSNorm(V)·γ_h`. At init α=0,γ=1 ⇒
`V_out = V` exactly ⇒ forward is byte-identical to baseline at
step 0 (max-abs-diff = 0.0, not fp32 tolerance — algebraic identity).

Wiring lives in `models/layers.py` (the `MultiHeadAttention.use_v_rmsnorm`
kwarg at `:1043`, the parallel `elif use_v_rmsnorm:` arm at `:1344`
that registers `v_rmsnorm_alpha ∈ R^H` and `v_rmsnorm_gain ∈ R^{H × d_k}`,
the gated-RMSNorm apply site at `:2883`, and the mutual-exclusion
asserts in `MultiHeadAttention.forward`) and is read by
`MinimalLLM.__init__` in `models/llm.py` (`:530` read + `:805`/`:1099`
thread into the standard `TransformerBlock` constructor sites).
Default off ⇒ baseline path bit-identical. This stub only toggles the
flag.

NB: imports `Tiny1M3MVPreAVNormConfig` directly rather than declaring
its own `class C(Tiny1M3MConfig): use_v_rmsnorm: bool = True`,
because the latter would NOT override the parent's dataclass field
default (the parent's `False` is inherited verbatim without a
`@dataclass` re-decoration on the subclass, so the annotation is
ignored and `C().use_v_rmsnorm` resolves to `False`). The canonical
`Tiny1M3MVPreAVNormConfig` is `@dataclass`-decorated in
`configs/llm_config.py` and correctly sets `True` (per the
162/165/155/161 precedent that bare-class annotation breaks
dataclass field inheritance).

Caches are at /root/universe-lm/_arq_176-v-pre-av-norm.py on the box.
Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MVPreAVNormConfig as C


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
