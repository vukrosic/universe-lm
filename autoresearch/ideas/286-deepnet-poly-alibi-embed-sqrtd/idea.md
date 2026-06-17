---
id: 286-deepnet-poly-alibi-embed-sqrtd
status: done
round: 1
updated: 2026-06-16T09:50:58Z
transfer-risk: low
plain: Keep the champion (alibi + DeepNet-α + poly-alibi) and scale the token embeddings by √d_model so the residual stream enters block 1 at the magnitude DeepNet-α's residual budget is conditioned for. One fixed scalar multiply — no extra params, no distribution reshape.
---

# 286 — Embedding √d scaling on the deepnet+poly-alibi champion

## Why this, why now
Same thesis as 285: only **step-0 conditioning** levers bind at 0.94M/92 steps;
capacity/FFN/attention-internal levers are all NULL (see closed.md). The champion
owns positional bias + residual scaling; the **embedding-magnitude** axis is
untouched on the champion.

## Source
Vaswani et al. 2017 §3.4 (embeddings × √d_model); the modern μ-scaling lineage
(Gemma, OLMo). In-repo: idea 194-embed-sqrt-d.

## Mechanism
`use_embed_sqrt_d_scaling` multiplies the token embeddings by √d_model (a single
fixed scalar) before the first block. DeepNet-α scales the residual *increments*;
this sets the residual *baseline* magnitude the increments add onto, so the two
compose rather than fight.

## Distinct from the NULL prior
159-emb-layernorm was NULL (DRIFT, +0.071) — but that lever LayerNorm-reshapes the
per-token embedding distribution and pays an implicit LR/warmup re-tune. This is a
**scalar multiply**: no reshape, no learned affine, no distribution change — just
the entry magnitude. Different mechanism, far cheaper.

## Hypothesis
Right-sign Δ if the tiny1m3m residual stream is entering under-scaled relative to
what DeepNet-α + the LM-head tie expect; NULL if the existing emb_scale path
already lands the stream in the right band.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm before any promotion). Single seed (42).
