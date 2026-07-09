---
id: 226-per-channel-attn-temp
status: needs-taste
round: 1
updated: 2026-06-16T01:05:00Z
transfer-risk: med
plain: Before softmax, multiply QK scores by a per-channel learnable temperature vector. Different from per-head temperature (closed null 155) — this varies the temperature across feature dimensions within each head, not across heads.
---

# 226 — Per-Channel Attention Temperature

## Source
Closed 155-per-head-temp null at 0.94M applied a per-head learnable scalar τ_h to the QK logits (`scores * τ_h` before softmax). 184-logit-scale null was a global scalar logit scale. 205-per-head-mult-logit-scale null was per-head multiplicative. All three closed null at 0.94M.

**226 is structurally different**: a per-channel (per-feature-dim within the head) temperature vector applied to the QK scores *before* softmax. Mathematically: `scores = (Q @ K^T) * τ_vec` where `τ_vec ∈ R^{d_k}` is a learnable vector of size d_k=16, *shared across heads and tokens*. Different from per-head (varies across heads), this varies across the d_k feature dims within a head.

Mechanistically, this is a "soft max-then-min per feature dim" axis: it amplifies logits where τ_i > 1 (focus on that feature dim's contribution) and damps where τ_i < 1 (ignore that feature). At init τ_i = 1 so the lever is bit-identical to baseline; the optimizer can re-weight feature dims within each head.

## Mechanism
```
scores = Q @ K.transpose(-1, -2) / sqrt(d_k)   # [B, H, T, T]
tau    = self.attention_channel_temp            # [d_k], init 1.0
scores = scores * tau.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # broadcast
attn   = softmax(scores, dim=-1) @ V
```

At `tau = 1.0` (init), `scores * 1 = scores` exactly ⇒ step-0 bit-identical to baseline.

## Design sketch
- **Files**: `models/layers.py` — locate the attention forward (manual branch). Add an `nn.Parameter(d_k)` initialized to 1.0, applied to scores before softmax. The `d_k=16` per-block vector × 12 blocks = +192 params, +0.020% of 0.94M.
- **Config flag**: `use_per_channel_attn_temp: bool = False`, `per_channel_temp_init: float = 1.0`.
- **Cost**: 192 params total. Negligible.
- **Why it should help at tiny1m3m**: at d_model=64, d_k=16 (4 heads × 16-dim each), the QK dot product sums 16 terms per score. If some dims dominate the magnitude of the dot product (e.g., the first 2 dims have magnitude 5× mean while others are noise), the attention is effectively a 2-dim function. Per-channel τ lets the optimizer *flatten* this distribution: learn τ=0.5 for noisy dims and τ=2.0 for informative dims. The closed 016-qk-norm WIN normalizes per-feature across the whole token (different axis), so 226's per-feature-on-QK-scores is a different lever.
- **Why it might be null**: the closed 155-per-head-temp null showed per-head granularity doesn't bind at 0.94M. Per-channel has 4× more knobs but is also finer-grained — the optimizer may not have enough signal to learn useful τ values in 92 update steps.

## Scale evidence
Per-channel (vs per-head) temperature scaling on attention is novel. The closest analog is ALiBi's per-head slope (175 WIN), which is *additive*; 226 is *multiplicative per feature dim*. Transfer-risk **med** (per-feature scaling is well-validated in many normalization contexts, but per-feature attention temperature is novel).

## Why it's worth a slot
A win would say the model finds a useful per-feature temperature schedule on attention, giving a complementary axis to 016-qk-norm WIN (per-token feature normalization). A null would close the per-axis-temperature family at 0.94M (alongside per-head 155 null and global 184 null). The lever is cheap (+192 params, ~5 LoC), bit-identical step 0 at init=1.0, and structurally novel from all closed attention-shape levers.
