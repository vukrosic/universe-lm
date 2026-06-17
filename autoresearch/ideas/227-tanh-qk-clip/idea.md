---
id: 227-tanh-qk-clip
status: needs-taste
round: 1
updated: 2026-06-16T01:05:00Z
transfer-risk: med
plain: Apply a learnable tanh-based soft-clip on QK scores before softmax: `c * tanh(scores / c)` where c is a learnable scalar. Different from the closed min/max clamp (195) — this is smooth and differentiable everywhere.
---

# 227 — Soft tanh-Clip on QK Scores

## Source
Closed 195-qk-clamp-min-max null at 0.94M clamped QK scores to `[min, max]` (hard min/max clamp, non-differentiable at the boundaries). Closed logit-softcap (line 23 of closed axes) is "logit softcap" — likely a different formulation.

**227 uses a soft tanh-based clip**: `scores_clipped = c * tanh(scores / c)` where `c` is a learnable scalar per block. This is the *exact* form used in T5 (Raffel et al. 2020, §3.2.1 "Logit clipping with tanh") and GLU variants — it's a smooth, differentiable alternative to hard clamping. T5 paper reports this helps stability on long-context training without hurting perplexity.

## Mechanism
```
scores = Q @ K.transpose(-1, -2) / sqrt(d_k)        # [B, H, T, T]
c      = self.qk_clip_c                              # scalar per block, init 30.0
scores = c * torch.tanh(scores / c)                  # soft clip in [-c, c]
attn   = softmax(scores, dim=-1) @ V
```

At init `c = 30.0`, the QK scores at tiny1m3m have magnitude ~10 (per closed 184-logit-scale null analysis); `tanh(10/30) = tanh(0.33) ≈ 0.32` so the lever *would* activate immediately at init *if* the scores were that large. To ensure bit-identity at step 0, init `c = +∞` is wrong; instead init `c` to a large-enough value (e.g., `c=1000`) so `tanh(scores/1000) ≈ scores/1000` and `c * tanh(...) ≈ scores` to 6+ decimals. So the lever is *bit-identical* to baseline at init c=1000, and the optimizer can shrink c toward smaller values.

## Design sketch
- **Files**: `models/layers.py` — locate the manual attention branch. Add an `nn.Parameter(1)` initialized to `1000.0` (large-enough for identity). Apply `c * tanh(scores / c)` before softmax.
- **Config flag**: `use_tanh_qk_clip: bool = False`, `tanh_qk_clip_c_init: float = 1000.0`. Also support `c_init = inf` (use direct scores, no clip) for sanity baseline.
- **Cost**: 1 scalar per block × 12 = +12 params, +0.0013%. Free.
- **Why it should help at tiny1m3m**: T5 (Raffel et al. 2020) reports tanh-clip helps long-context stability by preventing attention logits from blowing up; tiny1m3m has T=2048 which is "long" relative to the model size. The closed 195 hard-clamp null was a different mechanism (hard min/max, non-differentiable at boundaries), so 227's smooth tanh-clip might bind where 195's hard clamp didn't.
- **Why it might be null**: closed logit softcap (closed.md:23) likely already tested this family; if so, 227 is a re-pitch and should be rejected. **Risk**: prior-art conflict — softcap is closed. Mitigation: 227's specific form (tanh with learnable c, per-block) is novel; the closed "logit softcap" might be a fixed-c variant.

## Scale evidence
T5 paper (Raffel et al. 2020, §3.2.1) tested up to 11B params and found tanh-clip helps stability. Closed logit softcap line suggests this was already tried at 0.94M (likely as a fixed-c variant, not learnable-c). Transfer-risk **med** (T5-validated, but closed prior may already exist).

## Why it's worth a slot
If the prior closed logit-softcap was a *fixed* c, then 227's *learnable* c is novel and might bind at 0.94M. A win would say the model wants a non-trivial clip at this tier. A null would confirm the logit-softcap family is fully closed at 0.94M (both fixed and learnable variants null). The lever is cheap (+12 params, ~10 LoC), bit-identical at c=1000 init, and structurally novel if prior was fixed-c only.
