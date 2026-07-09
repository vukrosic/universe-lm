---
id: 285-deepnet-poly-alibi-mup
status: done
round: 1
updated: 2026-06-16T09:29:40Z
transfer-risk: low
plain: Keep the champion (alibi + DeepNet-α + poly-alibi) and add μP joint init — rescale the embedding learning rate and output logit scale the Maximal-Update-Parametrization way, so the optimizer sees a width-consistent trust region. Step-0 byte-identical to the champion; it only changes how fast the embeddings/logits move.
---

# 285 — μP joint init on the deepnet+poly-alibi champion

## Why this, why now
Every lever that has WON at tiny1m3m (0.94M / 12L / 4H / 92 update steps, 3M
tokens) is a **step-0-active conditioning** lever: alibi (positional bias),
DeepNet-α (fixed residual scaling), poly-alibi (distance curvature). Every
**capacity / FFN / attention-internal / regularizer** lever has been NULL —
SwiGLU (170/211), ReLU² (153), GAU (158), sparse/soft MoE (146/117), qk-norm
variants (265/268/272/273), value-residual (275), drop-path (277), logit-scale
(216). The closed.md refrain is "re-evaluate at ≥135M Phase-2." So the only live
axis at this tier is conditioning, and the champion already owns position +
residual scaling. μP joint init is the cleanest **optimization-conditioning**
lever left.

## Source
Yang & Hu, "Feature Learning in Infinite-Width Networks" (μP, arXiv:2011.14522);
Tensor Programs V (arXiv:2203.03466). μP sets `lr_emb ∝ d_model` and a matched
logit/output scale so updates are width-consistent. In-repo: idea 193-mup-init.

## Mechanism
`use_mup_joint_init` rescales the embedding LR and logit scale (the `1/0.02 = 50×`
emb-LR rule) without adding a parameter or a forward-graph branch — step-0 output
is byte-identical to the champion. It re-tunes the effective trust region the
optimizer works in.

## Hypothesis
At 92 update steps the embedding table is the slowest-moving, most under-trained
matrix in the model; a width-consistent emb-LR lets it move enough to matter
inside the 3M-token budget. Predict a small right-sign Δ if it binds; NULL if the
existing global LR already saturates the embedding update.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm before any promotion). Single seed (42).
