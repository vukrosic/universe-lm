---
id: 017-sub-ln-sandwich
status: done
round: 1
updated: 2026-06-09T16:13:26Z
---

# 017 — Sub-LN / Sandwich block (additional LN after each sublayer)

## Source
"DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., Microsoft,
2022 — arXiv:2203.00555, §3.1 introduces the sub-LN pattern). Related:
"NormFormer" (Shleifer et al., Microsoft, 2021). Pre-norm-only
(closed) and post-norm (closed) are the *corner* cases; sub-LN is the
interior of that axis.

## Mechanism
Replace the standard pre-norm sublayer

```
y = x + Sublayer(LN_pre(x))
```

with a sub-LN / sandwiched sublayer

```
y = x + LN_post(Sublayer(LN_pre(x)))
```

where `LN_post` is a fresh `nn.LayerNorm(d_model)` whose γ is init to 1
and β to 0 (identity at step 0 — baseline preserved). One new LN per
attention block and one per FFN block in `TransformerBlock.forward`,
wired around the existing `self.attn` / `self.mlp` calls. < 20 LoC
of net new code on top of `models/layers.py`. The two LNs collapse
algebraically when γ=1, β=0 so step-0 is bit-identical to the
pre-norm baseline; no special init needed for any other parameter.

## Why it's worth a slot
The bet: pre-norm (closed but baseline) lets the residual stream's scale
drift across depth, which is the dominant failure mode when scaling a
transformer past ~12–24 layers. Sub-LN constrains each sublayer's
*contribution* to the residual stream to be unit-RMS, fighting that
drift locally instead of relying on the post-norm global re-centering
(closed: post-norm). It is a *mechanism* (norm placement on the
sublayer output) and is a single global flag — one boolean on
`TransformerBlock` — so it composes cleanly with every other ablation
already in the queue. A null at 6 layers would be expected and
informative (sub-LN pays off in deeper stacks per the DeepNet
ablations); a small win at 6 layers would imply the drift mechanic is
active even at our depth and would license promoting it as a default
for any future larger-tier re-runs.
