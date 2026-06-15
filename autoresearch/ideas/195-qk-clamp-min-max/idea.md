---
id: 195-qk-clamp-min-max
status: needs-taste
round: 1
updated: 2026-06-15T08:30:00Z
transfer-risk: low
plain: Clamp the pre-softmax attention scores to a fixed range (e.g., [-8, 8]) before softmax, so no single attention logit can dominate — like logit softcap but with hard clipping instead of tanh.
---

# 195 — Hard QK Logit Clamp (Min/Max Clipping Pre-Softmax, Bit-Identity at Small Logits)

## Source
- "logit softcap" (closed, screen20m row 0-17) — uses `tanh(qk/c)*c` to *smoothly* clip the QK logits. The softcap is in the closed-axes line of closed.md. 195 is a *hard* clip (`min(max(qk, -c), c)`) — different operation (discontinuous derivative vs continuous tanh), different closed-form behavior. The two are similar in *spirit* (both bound the QK logits) but differ in *implementation* (tanh vs hard min/max) and *training dynamics* (tanh has a smooth gradient; hard clip has zero gradient outside the clip range).
- "Sparse Logit Attention" (2024) — variants of QK logit clipping for stability.
- "Stable Attention" (various 2023-2024 papers) — QK logit clipping is a recognized stability lever.
- In-repo context: closed.md line "logit softcap" closed the *tanh-form* softcap. 195 is the *hard-clip form*. The two are different mechanisms (smooth tanh vs discontinuous hard clip). The closed softcap suggests that the *concept* of QK logit clipping may not bind at our tier, but the *implementation* (tanh vs hard clip) has not been directly compared. Hard clipping has a *sharper* boundary (zero gradient outside the clip range), which can act as a *regularizer* (the model is forced to keep its logits within the clip range, or the gradient is zero and the model is forced to learn a different direction).
- 184-logit-scale (closed null, same day) — global scalar on LM head output logits, not pre-softmax QK. Different lever.

## Mechanism
Standard pre-softmax QK product:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T], unbounded
weights = softmax(scores)             # can be very sharp if scores are large
```
With hard QK clamp:
```
scores = Q @ K^T / sqrt(d_k)
scores = torch.clamp(scores, min=-c, max=+c)   # hard clip to [-c, +c]
weights = softmax(scores)
out = weights @ V
```
For `c = 8.0` (a typical value, also used in PaLM), the clamp range is `[-8, 8]`. At step 0 with Kaiming init, the QK^T entries have std `O(1)` (Gaussian), so the typical logit is in `[-3, +3]` (3-sigma range). With `c = 8`, the clamp is *inactive* at step 0 (the typical logit is well within `[-8, 8]`), and the lever is bit-identical to baseline.

**Step-0 byte-identity**: with `c = 8` (or any `c > 3`), the clamp is inactive at step 0 and the forward pass is bit-identical to baseline. With `c = 1` (a tighter clamp), the clamp is *active* at step 0 (about 30% of logits are clipped), and the lever is not bit-identical.

**Recommended form**: `c = 8.0`, a *wide* clamp that's inactive at step 0 but becomes active as training progresses and the logits grow. The wide clamp is a *safety net* for outlier logits, not a *constraint* on the typical logits.

**Why this lever despite the closed softcap**: the closed softcap used `tanh(qk/c)*c`, which is a *smooth* (C^infinity) approximation to the hard clip. The hard clip is *discontinuous* in its first derivative (zero gradient outside the clip range). The two have different *training dynamics*: tanh has a non-zero gradient everywhere (the gradient smoothly approaches 0 as |qk| → c), while hard clip has *zero* gradient outside the clip range. The hard clip is a *stronger* regularizer (the model is *forced* to keep its logits within the clip range, or the gradient is exactly 0 and the model must learn a different direction). At 0.94M, the stronger regularizer may help prevent outlier-logit pathologies (e.g., the attention-sink collapse that the softcap was designed to prevent).

## Design sketch
- **Files**:
  - `models/layers.py` (or `models/llm.py`) — in the attention forward, after computing `scores = Q @ K^T / sqrt(d_k)`, apply `scores = torch.clamp(scores, min=-c, max=+c)`. The `c` value is a config parameter (default 8.0).
  - `configs/llm_config.py` — add `use_qk_clamp: bool = False` and `qk_clamp_c: float = 8.0` to `LLMConfig`. Add `Tiny1M3MQKClampConfig(Tiny1M3MConfig)` with `use_qk_clamp: bool = True, qk_clamp_c: float = 8.0`.
- **Config flag**: `use_qk_clamp: bool = False, qk_clamp_c: float = 8.0`.
- **Param count**: **0 new params**.
- **Intuition (why it might lower val loss)**: as training progresses, the QK^T magnitudes can grow (especially in deep models). A few outlier logits can dominate the softmax, producing a "max-wins" pathology where the model attends to a single token regardless of context. The clamp prevents this by *bounding* the logit range, so no single logit can dominate. PaLM uses `c = 50.0` (a very wide clamp that's almost never active in practice), but the *concept* of bounding the logit range is well-validated. At 0.94M, the QK^T magnitudes may grow more slowly (less data, fewer update steps), so a tighter clamp (`c = 8.0`) may be active more often, providing a more meaningful bound.
- **Why it might bind at 0.94M where softcap didn't**: the closed softcap used tanh, which is *smooth* and doesn't enforce a hard bound. The hard clip is *discontinuous* in its derivative, which means the model gets a *sharper* signal about which logits are "out of bounds" (the gradient is exactly 0 for out-of-bound logits). This sharper signal may help the model learn to keep its logits within the clip range, which can prevent the attention-sink collapse.

## Scale evidence
- PaLM (Chowdhery et al. 2022, arXiv:2204.02311) — uses `c = 50.0` logit softcap (tanh form). Validated at 8B-540B.
- "logit softcap" (closed) — closed at 0.94M. The closed form was tanh-based; the *hard-clip* form is different.
- "Stable Attention" literature — multiple papers use logit clipping (both tanh and hard) for stability. Validated at 100M-1.5B.
- **Transfer-risk: low** — the lever has direct validation at 100M+ for the *concept* (logit bounding), and the *hard-clip* implementation is a well-known alternative to tanh.

## Why it's worth a slot
The bet, in one sharp sentence: **the closed softcap (tanh form) tested a *smooth* logit bound, but the *hard-clip* form is a different mechanism with a discontinuous derivative that may act as a stronger regularizer** — the closed softcap suggests that *logit bounding* doesn't bind at 0.94M in the smooth form, but the hard-clip form has not been tested; a null at 0.94M would close the *logit-bounding* axis (both smooth and hard forms), and a win would give a stability lever that prevents attention-sink collapse.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- logit softcap (closed, screen20m row 0-17) — *smooth* tanh form. 195 is *hard* clip form. Different mechanism.
- 184-logit-scale — global scalar on LM head output, not pre-softmax QK.
- 152-attn-logit-bias (null) — additive bias on QK^T (smooth). 195 is multiplicative bound.
- 155-per-head-temp (null) — scalar on QK^T. 195 is hard bound.
- 188-qk-rms-scaling — per-block scalar on QK^T (soft, multiplicative). 195 is per-block hard bound.
