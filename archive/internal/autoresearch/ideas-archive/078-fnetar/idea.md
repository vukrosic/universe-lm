---
id: 078-fnetar
status: needs-plan
round: 1
updated: 2026-06-11T01:21:40Z
transfer-risk: med
---

# 078 — FNetAR Causal Fourier Residual Mixer

## Source
FNetAR: Mixing Tokens with Autoregressive Fourier Transforms
(arXiv:2107.10932, Lee-Thorp & Ainslie, 2021). Causal counterpart to
FNet (arXiv:2105.03824); shown competitive with Transformer-XL on
WikiText-103 with 75% of attention layers replaced.

## Mechanism
Add **one gated, causal Fourier residual mixer per block**, on the
residual stream just before attention's pre-LN — `x = x + tanh(g) ·
FNetAR_mix(LN(x))`, with scalar gate `g` init = 0 so step-0 ≡ baseline
(zero-init / identity, transferable).

The FNetAR mixer is the paper's lower-triangular causal DFT — *not* a
naive `tril`-masked vanilla FFT. Per token position `t` it computes
`y[t] = Re( F_t · x[:t+1] )` over the channel dim, where `F_t` is the
causal Fourier matrix restricted to keys `≤ t` (equivalent to a
length-`t+1` DFT with the upper-triangular outputs masked out, so no
future leakage). Implemented as a single matmul against a precomputed
causal-DFT basis (`shape = [T, T]` projected per channel) or, for
speed, as a per-position `rfft` over the causal prefix; either way
< 80 LoC and no new learnable params except `g` (one scalar per
block). Hybrid placement (paper §3.2) is kept: the existing attention
sublayer stays untouched, FNetAR is purely additive global mixing.

## Scale evidence
WikiText-103 causal LM — within ~0.4 ppl of Transformer-XL at the same
param budget while replacing 75% of attention layers (paper Table 2,
~150M params). transfer-risk: **med** — paper reports at 100M+ causal
LM, but the lever is an *additive* gated residual here (not a
replacement), which is a strictly weaker, safer use than the paper's
attention-substitution; the additive use has a higher chance of
clearing than the swap-out form.

## Why it's worth a slot
**Bet:** at tiny1m3m's `max_seq_len=2048` and 6 layers, vanilla softmax
attention provably flattens at late positions (the exact mechanism
025-SSMax exploits — denominator scales with `n`, logit variance
fixed). A parameter-free, content-independent **causal frequency-
domain mixer** injects a global-mixing inductive bias attention can't
trivially recover at this depth, predicting **val-loss Δ ≤ −0.01** vs
the FIRE-equipped ctrl (the same PASS bar 025/023 use). The lever is
**zero-init gated**, so a null is bounded: drift > +0.01 only if FNetAR
*actively* hurts — which itself is publishable info (refutes
"orthogonal global mixing always helps" lore at small scale).

**Portfolio note:** 078 is the *sole* Fourier-mixer slot in the queue
— there is no 077-fnet (the only Fourier-related `idea.md` matching
`fnet|fourier|fft` is this one). The taste-r1 overlap concern is
therefore moot; the non-causal FNet variant is excluded by
construction (we train causal LM, and non-causal FFT leaks future
tokens).

**Leverage vs alternatives in flight:** distinct axis from 023-canon-
conv (local depthwise *time-domain* mixing, kernel=3) and from
025-SSMax (length-dependent *temperature* on logits). FNetAR is global
*frequency-domain* mixing — orthogonal to both, so it can stack later
if it clears solo. A null still teaches us that the global-mixing slot
is already filled by attention at this scale, which directly informs
whether to file Mamba-style SSM ideas (also global, also content-
independent core).
