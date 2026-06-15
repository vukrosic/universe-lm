---
id: 195-qk-clamp-min-max
status: needs-plan
round: 1
updated: 2026-06-15T08:57:47Z
transfer-risk: low
plain: Hard-clip the pre-softmax QK scores to a tight range (c=2.0) so the discontinuous-gradient boundary is exercised at step 0 — distinct from the closed tanh softcap which was inactive at c=8.
---

# 195 — Tight Hard QK Logit Clamp (c=2.0, Active at Step 0)

## Source
- "logit softcap" (closed, screen20m row 0-17) — uses `tanh(qk/c)*c` to *smoothly* clip the QK logits. **r0.5 note: r0 of this idea proposed c=8, which the taste reviewer correctly flagged as bit-identical to the closed softcap at the same c** (smooth tanh and hard min/max agree when the argument is well inside the clamp range; at c=8 the range is never reached from Kaiming init). This repitch reshapes the bet to c=2.0 — *active* at step 0 — so the lever is *not* bit-identical to the closed softcap and the discontinuous-derivative claim is testable.
- "Sparse Logit Attention" (2024) — variants of QK logit clipping for stability.
- "Stable Attention" (various 2023-2024 papers) — QK logit clipping is a recognized stability lever.
- PaLM (Chowdhery et al. 2022, arXiv:2204.02311) — uses `c = 50.0` logit softcap (tanh form). Validated at 8B-540B.

## Mechanism
Standard pre-softmax QK product:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T], unbounded
weights = softmax(scores)             # can be very sharp if scores are large
```
With tight hard QK clamp (c=2.0):
```
scores = Q @ K^T / sqrt(d_k)
scores = torch.clamp(scores, min=-2.0, max=+2.0)   # hard clip to [-2, +2]
weights = softmax(scores)
out = weights @ V
```

**Why c=2.0 and not c=8.0 (r0's choice)?** At Kaiming init, the QK^T entries are O(1) Gaussian, so the typical logit is in the 3-sigma range ≈ [-3, +3]. A clamp at c=8 is *inactive* across that range, so for the whole A/B the lever is mathematically a no-op (the only way it can fire is if some QK^T magnitude grows past 8, which is rare at 0.94M/12L/4H/3M-tokens). The closed softcap at c=8 tested the *smooth* logit-bounding axis; a hard clip at c=8 is bit-identical to that closed softcap. To test the *hard* form, c must be *active* at step 0.

At c=2.0:
- The clamp range is `[-2, +2]`. About 5% of the Gaussian-distributed QK^T entries at init will exceed |2| (2-sigma tail), so the clamp is *active* on ~5% of logits at step 0.
- The lever is **not** bit-identical to baseline at step 0.
- The lever is **not** bit-identical to the closed softcap (which was inactive at step 0).
- The discontinuous-derivative property is *exercised*: for the 5% of logits that hit the boundary, the gradient is exactly 0 (no pull toward smaller magnitude), so the model is forced to learn a different direction to reduce attention to clipped tokens.

**Step-0 byte-identity caveat:** at c=2.0, the lever is *not* bit-identical at step 0 (the forward pass differs on ~5% of logits). This is an explicit departure from r0's wide-form niche check. The bet is that the *regularizer effect* of an active hard clamp — even at init — is the mechanism being tested, not a null activation. The taste reviewer's r1 note explicitly allowed this: "Pick a c that's *active* at step 0 (so the lever is not bit-identical to baseline and not bit-identical to the closed softcap)."

## Design sketch
- **Files**:
  - `models/layers.py` (or `models/llm.py`) — in the attention forward, after computing `scores = Q @ K^T / sqrt(d_k)`, apply `scores = torch.clamp(scores, min=-c, max=+c)`. The `c` value is a config parameter.
  - `configs/llm_config.py` — add `use_qk_clamp: bool = False` and `qk_clamp_c: float = 2.0` to `LLMConfig`. Add `Tiny1M3MQKClampConfig(Tiny1M3MConfig)` with `use_qk_clamp: bool = True, qk_clamp_c: float = 2.0`.
- **Config flag**: `use_qk_clamp: bool = False, qk_clamp_c: float = 2.0`.
- **Param count**: **0 new params**.
- **Intuition (why it might lower val loss)**: as training progresses, the QK^T magnitudes can grow (especially in deep models). A few outlier logits can dominate the softmax, producing a "max-wins" pathology where the model attends to a single token regardless of context. The clamp prevents this by *bounding* the logit range, so no single logit can dominate. At c=2.0, the bound is *tight* — typical training-time QK^T magnitudes will frequently hit the boundary, forcing the model to *spread* attention rather than collapse onto a single token. The discontinuous gradient (zero outside the clamp range) means the model gets a *sharp* signal: "this logit is out of bounds; learn a different direction" — not a smooth gradient that the model can fight by tiny updates.
- **Why it might bind at 0.94M where softcap didn't**: the closed softcap used tanh at c=8 (inactive at step 0, smooth gradient everywhere). The tight hard clamp at c=2.0 is *active* at step 0 with a *discontinuous* gradient. The combined effect — active + discontinuous — has not been tested in this repo. If the regularizer effect is real, it should show up at 0.94M as a measurable val-loss difference.

## Scale evidence
- PaLM (Chowdhery et al. 2022, arXiv:2204.02311) — uses `c = 50.0` logit softcap (tanh form). Validated at 8B-540B. The *concept* of bounding QK logits is well-validated at scale; the *tight* clamp (c=2.0) and *hard* form are the untested parts.
- "logit softcap" (closed) — closed at 0.94M. The closed form was tanh at c=8 (inactive at step 0). The *tight hard clamp* at c=2.0 is a different mechanism.
- "Stable Attention" literature — multiple papers use logit clipping (both tanh and hard) for stability. Validated at 100M-1.5B.
- **Transfer-risk: low** — the lever has direct validation at 100M+ for the *concept* (logit bounding), and the *tight hard-clamp* implementation is a well-known alternative to tanh (e.g., sparse logit attention variants).

## Why it's worth a slot
The bet, in one sharp sentence: **a *tight* hard clamp at c=2.0 is active at step 0 with a discontinuous gradient, testing whether the combined "active + discontinuous" regularizer effect of hard QK clamping lowers val loss — a different bet from the closed softcap (inactive + smooth), and a null would close the *hard-clamp sub-axis* of the logit-bounding family.**

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`. A null closes the *hard-clamp sub-axis* of the logit-bounding family (smooth form already closed, tight hard form to follow).
- **DRIFT**: `trt_val > ctrl_val + 0.01` — would suggest the discontinuous gradient at c=2.0 is too aggressive for our scale.

## Distinct from closed axes (defensive)
- logit softcap (closed, screen20m row 0-17) — *smooth* tanh at c=8 (inactive at step 0). 195 is *hard* clip at c=2.0 (active at step 0). Different mechanism, different bet.
- 184-logit-scale — global scalar on LM head output, not pre-softmax QK.
- 152-attn-logit-bias (null) — additive bias on QK^T (smooth). 195 is hard bound.
- 155-per-head-temp (null) — scalar on QK^T. 195 is hard bound.
- 188-qk-rms-scaling — per-block scalar on QK^T (soft, multiplicative). 195 is per-block hard bound.
