---
id: 058-fixup
status: needs-run
round: 1
updated: 2026-06-11T01:16:36Z
transfer-risk: med
---

# 058 — Fixup (zero-init residual projections) at 6L

## Source
Zhang, Dauphin, and Ma, "Fixup Initialization: Residual Learning Without
Normalization" (arXiv:1901.09321). The cheapest Fixup ingredient — *zero-init
the last linear in each residual branch* — is already implemented in this repo
as the `zero_init_resid` config flag (`configs/llm_config.py:47`,
`models/llm.py:531-538`, commit `4be65bb` "residual-stream levers"), but no
idea has been filed for it and the flag has never been ablated. This idea
files the missing experiment.

## Mechanism
Flip `zero_init_resid=True` (zero new LoC). After the global 0.02 init pass,
re-zero two weights per TransformerBlock: the attention output projection
(O-slice of the fused `qkvo_proj`) and the FFN `down_proj`. Every block
becomes an exact identity at step 0 — `x_{l+1} = x_l + 0 = x_l` — so the
residual stream at step 0 is the embedding output, regardless of depth.
RMSNorm, RoPE, and everything else stays exactly as the baseline. The
mechanism is the *subset* of Fixup that survives when you keep normalization:
the identity-at-init property, not the no-norm regime.

## Scale evidence
The full Fixup recipe was demonstrated at 10,000-layer ResNets. The single
ingredient tested here (zero-init branch outputs) is the *most decisive*
Fixup component — the others (L^(-1/(2m-2)) branch scaling, scalar bias
initialization) are corrections for *without-norm* training. With RMSNorm
in place, only the identity-at-init half of Fixup matters, and that half is
depth-invariant: the residual stream at step 0 is the input whether the
stack is 6 layers or 10,000. transfer-risk: med (mechanism validated at
scale by the source, but the specific "keep norm, only zero-init branches"
ablation has no published precedent at any scale).

## Why it's worth a slot
The `zero_init_resid` flag is **dead code** — implemented, committed, never
tested. Filing the experiment retires that surface area: WIN = ship the
flag (a free init change with no param cost, no architectural cost, no
schedule change), NULL = delete the flag and free the design space. The
bet at 6L: "step-0 identity removes the first ~50 warmup steps of gradient
shock where uncalibrated branch outputs fight the residual stream, so we
expect a small but consistent val-loss improvement that survives the
seed-42/no-multi-seed protocol." A null is informative in a specific way
the previous r1 framing wasn't — it tells us norm is doing work *beyond*
init-rescaling at 6L, which is a sharper finding than "norm helps at 6L"
(already known and closed at `closed.md:25`). Either outcome, this slot
closes the norm-axis residual stream question and retires a flag that
shouldn't ship untested.
