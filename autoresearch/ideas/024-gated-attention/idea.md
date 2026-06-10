---
id: 024-gated-attention
status: running
round: 1
updated: 2026-06-10T07:59:38Z
---

# 024 — Gated Attention (per-head sigmoid output gate)

## Source
Qiu et al., "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free" (arXiv:2505.06708), May 2025. The same head-output gate appears in modded-nanogpt speedrun variants. (Not Qwen's "gated attention" — Qwen gates **Q** pre-softmax, a *different* site on a different axis. This idea is on the o_h site.)

## Mechanism
After the per-head attention output `o_h = A_h V_h` (before the output projection that mixes heads), multiply it element-wise by a head-specific input-dependent sigmoid gate: `o_h ← o_h · 2·σ(W_g·x + b)`, with `W_g` a `nn.Linear(d_model, H)` (per-head **scalar** gate, not per-head vector — see Gate shape).

**Gate input:** the sublayer input residual `x` (pre-LN/attn), NOT `o_h` itself — avoids circularity and matches Qiu's pre-attention residual site.

**Identity-init (pinned form, no improvisation):** `2·σ(W·x + b)` with `W=0`, `b=0` → constant 1.0 at step 0, so step-0 ≡ baseline to floating-point precision with no `o_proj` init gymnastics. This is the (b) form from the original spec; chosen over (a) `σ` w/ 2× `o_proj` init and (c) `1+σ` w/ (2/3)× `o_proj` init because the constant is exactly 1 and `o_proj` init is unchanged.

**Gate shape (pinned):** per-head **scalar**, `nn.Linear(d_model, H)` (1 gate per head). Not the per-head vector form `nn.Linear(d_model, H·d_k)` — at tiny1m3m (d_k=32, H=8, 6L) the vector form is 65,536 params/layer × 6L = **393,216 params → 42% of the 0.94M model**, a *parameter* lever disguised as a structural one. Scalar is 256·8 = 2,048 params/layer × 6L = 12,288 → **1.3% of the model**. The Qiu paper primarily tests the vector form at d_k=128; the cheap scalar variant is the right ablation at this tier (the paper's d_k=128 result does not transfer directly).

## Why it's worth a slot
We expect a val-loss drop because the gate injects input-conditional **non-linearity and sparsity** between attention and the output projection — the paper attributes its gains to letting each head *suppress its own output* when uninformative, which also kills attention sinks and massive activations. This is categorically distinct from every filed/closed lever: it gates the **head output value** (post-`AV`), whereas 020-FoX gates the attention-*matrix* decay, 021-value-residual mixes a cross-layer *value*, 022-softpick changes the *normalizer*, and logit-softcap (closed) clamps *logits*. It fires every step (no EMA/schedule trap), is parameter-cheap, and gives the few heads of a tiny model a learned per-token on/off switch. A null tells us head-output gating is redundant with SwiGLU's gating at this scale; a win is a sub-30-LoC transferable lever stackable on FIRE.

## Pass bar
Box noise at tiny1m3m is ~±0.01 val loss; smaller deltas are unresolvable at seed 42. `Δ := trt_val − ctrl_val`; **pass iff `Δ ≤ −0.01`**. Sub-noise → log null and close (per the seed-42 rule; do **not** add seeds to chase sub-noise).
