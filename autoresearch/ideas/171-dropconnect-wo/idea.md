---
id: 171-dropconnect-wo
status: tasting
round: 1
updated: 2026-06-14T09:24:02Z
transfer-risk: med
plain: During training, randomly zero out individual weights of the attention output matrix (DropConnect) as a regularizer, with the rate starting at zero so the first step is identical to the baseline.
---

# 171 — DropConnect on W_O (Per-Weight Stochastic Masking of Attention Output Projection)

## Source
- Wan, Zeiler, Zhang, LeCun, Fergus, "Regularization of Neural Networks using
  DropConnect" (ICML 2013, also arXiv:1304.3174) — the original DropConnect
  paper. Validated on MNIST/CIFAR/ImageNet, showing consistent gains when
  DropConnect replaces Dropout on the FC layers.
- More recently: Parmar et al. ("Stand-Alone Self-Attention in Vision Models",
  NeurIPS 2019) and various ViT studies apply DropConnect to attention
  projections; "Sparse MoE" style per-weight sparsity is also related.
- Closest in-repo priors:
  - 147-dropkey (NULL at 0.94M, `closed.md:34`): drops **keys** (per-token
    zeroing of the K vector pre-attention). 171 drops **weights of W_O**
    (per-weight zeroing of the output projection matrix). Different tensor,
    different stochastic axis.
  - 111-drop-path (DRIFT, `closed.md:49`): drops **entire residual branches**
    (per-block stochastic depth). 171 doesn't drop branches, only weights.
  - 138-looksam (NULL, `closed.md:108`): periodic SAM perturbation. Different
    mechanism (sharpness-seeking perturbation, not stochastic masking).
- DropConnect-on-attention-output is NOT in `closed.md`'s closed axes
  ("Dropout/regularizer family" is closed, but DropConnect is a distinct
  per-weight stochastic regularizer from per-token Dropout / DropKey / DropPath).

## Mechanism
Standard attention: `out = concat(head_1, ..., head_H) @ W_O` where
`W_O ∈ ℝ^{d_model × d_model}`. With DropConnect:
1. At training time, sample a Bernoulli mask `M ∈ {0,1}^{d_model × d_model}`
   with probability `p_keep = 1 - rate` per entry.
2. Apply `W_O_masked = W_O ⊙ M / p_keep` (inverted-dropout rescale).
3. Use `W_O_masked` for the forward pass.
4. At eval time, use `W_O` unchanged (no noise, no rescale).

The mask is sampled per forward pass (not per-step) and is the same mask for
all batch elements and all positions. This is "weight-level" noise, distinct
from "token-level" noise (Dropout on activations) or "row-level" noise
(DropKey on the K matrix).

## Design sketch
- **File**: `models/layers.py` (`MultiHeadAttention.__init__` adds
  `use_dropconnect_wo: bool = False` and `dropconnect_wo_rate: float = 0.0`
  kwargs; `MultiHeadAttention.forward` adds a single branch after the head
  concatenation step).
- **Config flag**: `use_dropconnect_wo: bool = False` and
  `dropconnect_wo_rate: float = 0.1` on `LLMConfig` (rate is a sensible
  default from Wan et al.'s CIFAR/ImageNet sweet spot; the lever-test is the
  *presence* of the regularizer, not the rate HP).
- **Step-0 byte-identical**: at step 0, `dropconnect_wo_rate = 0.0` ⇒
  `p_keep = 1.0` ⇒ mask is all-ones ⇒ `W_O_masked = W_O ⊙ 1 / 1 = W_O` ⇒
  **byte-identical to baseline (max-abs-diff = 0.0)**. Set the rate to a
  small positive value (e.g. 0.1) for the *treatment* config; at step 0
  the mask is the all-ones mask only if the rate is 0.0 OR if we sample
  without replacement on the first call. Cleaner: set the *flag* on, set
  the *initial* rate to 0.0 with a warmup that ramps to 0.1 over the first
  N steps. Or: just set rate = 0.0 in the treatment config (a "regularizer
  present but inactive" control) — the lever-test is "is the regularizer
  infrastructure in place" and a config that has it OFF but flag ON is a
  valid A/B (flag-present cost: 1 branch + 1 schedule). For a stronger
  treatment, schedule rate from 0.0 to 0.05 over the first 100 steps.
- **Intuition (why it might lower val loss)**: per-weight masking on W_O
  forces the remaining weights to compensate, which is a strong
  co-adaptation regularizer on the output projection (analogous to dropout
  on activations but applied at the weight level). W_O is a single dense
  `d_model × d_model` matrix that the optimizer can co-adapt onto narrow
  subspaces; DropConnect prevents this by forcing redundant weight paths.
  Baseline weakness: the optimizer at 0.94M may over-fit W_O to spurious
  features. DropConnect should reduce this.
- **LoC**: ~25 lines (mask sample + apply + assert + schedule).

## Scale evidence
- DropConnect is well-validated at vision scale (CIFAR-10, ImageNet) in the
  original paper and many follow-ups (VGG-DropConnect, ResNet-DropConnect,
  DenseNet-DropConnect variants).
- For *language models*, DropConnect is less commonly cited. The closest
  LM application is "Structured DropConnect" applied to LSTM/RNN language
  models (Pham et al. 2014, "Dropout improves Recurrent Neural Networks for
  Handwriting Recognition"); for transformer LMs, the lever is novel.
- **Transfer risk: med** (validated at vision ≥100M params; for LMs the
  lever is plausible but not directly validated at ≥100M. Strong argument
  for transfer from vision-CNN's "dense projection" + "limited data" setup
  to "W_O is a dense projection" + "0.94M sees ~3M tokens which is data-
  limited for LMs").

## Why it's worth a slot
The bet: per-weight stochastic masking on W_O is a strong co-adaptation
regularizer that should help at our data-limited 0.94M/3M-token tier. We
expect Δval ≈ -0.005 to -0.015 (smaller than vision's gains because the
W_O matrix is small, d_model=64). A null would tell us the *weight-level*
axis of the regularizer family is also closed (147 key-drop, 111 path-drop
already null; 171 weight-drop joins them) and the regularization family is
exhausted at this tier. A win would tell us weight-level noise binds where
token-level and path-level don't. Step-0 byte-identical, low implementation
risk, well-isolated A/B.
