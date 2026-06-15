---
id: 181-cross-head-rmsnorm
status: running
round: 2
updated: 2026-06-15T07:48:52Z
transfer-risk: med
plain: Normalize each token's attention output across heads (so all four heads land on the same scale) before mixing back into the residual stream, starting with the per-head gain at 1 so step-0 is byte-identical.
---

# 181 — Cross-Head Channel RMSNorm (Normalize Attention Output Across Head Dim, Pre-W_O)

## Source
- The closest published analog is **NormFormer** (Shleifer et al. 2021, arXiv:2110.09423) which adds an extra LayerNorm on the attention output before the residual — validated on ViT-class models and small LMs.
- **RMSNorm** (Zhang & Sennrich 2019, arXiv:1910.07467) is the well-known "drop the mean-subtraction" simplification; RMSNorm on attention output is a known variant in Qwen-2 and Gemma-2.
- The **specific cross-head axis** (normalize across H dim, treating the H×d_k as a single H·d_k channel) is novel at this tier. Standard RMSNorm on attention output normalizes over the d_model axis (concatenated across heads). Cross-head RMSNorm instead treats each head's d_k channels as a single channel and normalizes *across heads within each d_k slice*.
- In-repo context: norm zoo closed (pnorm, manhattan, center, squash, clip, channelscale). 160-rms-gain-per-head null is post-AV per-head gain (different axis — gain not normalization). 142-layerscale null is per-channel diagonal gain (different axis). 176-v-pre-av-norm is V-normalization pre-AV. The cross-head normalization axis is fresh.

## Mechanism
Standard attention output:
- `attn_w = softmax(QK^T / √d_k)` — `[B, H, T, T]`.
- `out = attn_w @ V` — `[B, H, T, d_k]`.
- `out_concat = out.transpose(1, 2).reshape(B, T, H · d_k)` — `[B, T, d_model]`.
- `output = out_concat @ W_O` — `[B, T, d_model]`.

With cross-head channel RMSNorm (applied on `out` before reshape/concat/W_O):

For each (b, t) position and each d_k index k (in 0..d_k-1):
- Compute RMS across the H heads: `rms[b, t, k] = sqrt(mean(out[b, :, t, k]^2) + ε)`.
- Normalize: `out_norm[b, h, t, k] = out[b, h, t, k] / rms[b, t, k]`.

**Canonical parameterization (gate-α + tanh-gain):**
- Per-head gate `α_h = relu(α_raw_h)` with `α_raw_h` init `−1e-3` ⇒ `α_h ≈ 0` at step 0 (sub-epsilon; effectively α=0 ⇒ output equals `out` byte-identically within fp32 noise).
- Per-(h,k) gain `γ_h[k] = 1 + tanh(γ_raw_h[k])` with `γ_raw_h[k]` init 0 ⇒ `γ=1` at step 0.
- Mixed output:
  `out_final[b, h, t, k] = (1 − α_h) · out[b, h, t, k] + α_h · (out[b, h, t, k] / rms[b, t, k]) · (1 + tanh(γ_raw_h[k]))`

Step-0 byte-identity: `α_h ≈ 0` ⇒ `out_final ≈ out` exactly (within sub-epsilon fp32 noise from the relu near zero). The `(1−α) · out + α · ...` form makes the byte-identity clean and recoverable: the runner should assert `trt_step0_logits == ctrl_step0_logits` byte-exact (max-abs-diff = 0.0).

The lever: `α_h` is a per-head scalar gate that turns the cross-head RMSNorm on; `γ_h[k]` is the per-(h,k) post-normalization gain. The optimizer can:
- scale any individual head's coupling on/off independently (`α_h`),
- reshape the post-normalization output magnitude per head, per channel (`γ_h[k]`).

Bit-identity at step 0: `α_h = 0` for all h, `γ_h[k] = 1` for all h,k ⇒ `out_final = out` exactly ⇒ **byte-identical to baseline at step 0** (within fp32 noise from the relu near zero).

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_cross_head_rmsnorm: bool = False` to `MultiHeadAttention.__init__`. Allocate:
    - `self.cross_head_rmsnorm_alpha_raw = nn.Parameter(torch.full((n_heads,), -1e-3))` (gate-α init ⇒ α≈0 ⇒ byte-identity)
    - `self.cross_head_rmsnorm_gain_raw = nn.Parameter(torch.zeros(n_heads, d_k))` (tanh-gain init ⇒ γ=1)
    After computing `out = attn_w @ V` (shape `[B, H, T, d_k]`), apply:
    `rms = (out.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()`,
    `alpha = relu(self.cross_head_rmsnorm_alpha_raw).view(1, H, 1, 1)`,
    `gain = 1 + torch.tanh(self.cross_head_rmsnorm_gain_raw).view(1, H, 1, d_k)`,
    `out = (1 - alpha) * out + alpha * (out / rms) * gain`.
    The mean is over the H axis.
  - `configs/llm_config.py` — add `use_cross_head_rmsnorm: bool = False`. Add `@dataclass`-decorated `Tiny1M3MCrossHeadRMSNormConfig(Tiny1M3MConfig)` subclass with `use_cross_head_rmsnorm: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_cross_head_rmsnorm: bool = False` on `LLMConfig` (sibling of `use_head_gain` at `configs/llm_config.py:176`).
- **Param count**: H=4, d_k=16, n_layers=12.
  - Per block: H × 1 (alpha_raw) + H × d_k (gain_raw) = 4 + 64 = **68 params**.
  - Total: 68 × 12 = **816 params (+0.087% of 0.94M)**.
- **Intuition (why it might lower val loss)**: standard post-AV gain (160 closed null) operates on each head independently. Cross-head RMSNorm explicitly couples the heads by normalizing their magnitudes against each other before they enter W_O. The motivation: if one head has much larger output magnitude than the others, the W_O projection sees an imbalanced input and may have to rebalance. Cross-head normalization makes the input to W_O more uniform, which could be easier to learn. This is a *coupling* lever, fundamentally different from per-head independent gains.

## Scale evidence
- NormFormer at 100M-300M (encoder-decoder) showed modest gains.
- RMSNorm on attention output is well-validated (Qwen-2, Gemma-2, LLaMA-3 at 7B-70B).
- The **cross-head** axis is novel at 100M+. Transfer-risk is **med**.

## Plan
- **Field name**: `use_cross_head_rmsnorm: bool = False` on `LLMConfig` (sibling of `use_head_gain` at `configs/llm_config.py:176`).
- **Config subclass**: `@dataclass`-decorated `Tiny1M3MCrossHeadRMSNormConfig(Tiny1M3MConfig)` with `use_cross_head_rmsnorm: bool = True` (per the 162/165/155/161/176 precedent that bare-class annotation breaks dataclass field inheritance).
- **MHA kwarg plumbing**: add `use_cross_head_rmsnorm: bool = False` kwarg to `MultiHeadAttention.__init__` (sibling of `use_head_gain` at line 809 in current `models/layers.py`). Register parameters when flag is on:
  - `self.cross_head_rmsnorm_alpha_raw = nn.Parameter(torch.full((n_heads,), -1e-3))` — per-head gate-α, init `−1e-3` ⇒ `relu(α_raw) ≈ 0` ⇒ byte-identity at step 0.
  - `self.cross_head_rmsnorm_gain_raw = nn.Parameter(torch.zeros(n_heads, d_k))` — per-(h,k) gain, init 0 ⇒ `γ = 1 + tanh(0) = 1`.
- **Apply site**: after computing `out = attn_w @ V` (shape `[B, H, T, d_k]`) at the AV-product step in MHA.forward. Compute `rms = (out.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()` (mean over H axis). Apply:
  `out = (1 - alpha) * out + alpha * (out / rms) * (1 + tanh(gain_raw))`
  where `alpha = relu(self.cross_head_rmsnorm_alpha_raw).view(1, H, 1, 1)` and `gain_raw` reshaped to `(1, H, 1, d_k)`.
  Apply BEFORE the existing `use_head_gain` site at `models/layers.py:1668` (so the 160 post-AV per-head scalar gain composes on top, not underneath; if both are off, no branch taken, baseline graph byte-identical). The mutual-exclusion asserts below forbid combining, so the composition is theoretical only.
- **Mutual exclusion asserts** (top of MHA.forward, mirror the `use_cope ∧ use_qk_norm_post_rope` assertion pattern at `models/layers.py:2468`):
  - `assert not (self.use_cross_head_rmsnorm and self.use_head_gain)` — the two compose mathematically, but isolating the cross-head axis requires turning 160 OFF; assert prevents the implementer from accidentally turning both on.
  - `assert not (self.use_cross_head_rmsnorm and self.use_attn_output_gate)` — same reasoning.
  - `assert not (self.use_cross_head_rmsnorm and self.use_gated_attn)` — same reasoning.
- **TransformerBlock pass-through**: add `use_cross_head_rmsnorm: bool = False` to `TransformerBlock.__init__` (sibling of `use_head_gain`), pass into the MHA constructor.
- **llm.py read+thread**: add `self.use_cross_head_rmsnorm = getattr(config, "use_cross_head_rmsnorm", False)` in `MinimalLLM.__init__` (sibling of `self.use_head_gain`), thread into both `TransformerBlock(...)` constructor sites.
- **Param count**: per block = H × 1 (alpha_raw) + H × d_k (gain_raw) = 4 + 64 = **68 params × 12 blocks = 816 params (+0.087% of 0.94M)**. Mirrors 176-v-pre-av-norm's exact param count (also H × (1 α + d_k γ) per block = 68 × 12 = 816).
- **Runner stub**: `_arq_181-cross-head-rmsnorm.py` mirroring the 162/165/169/170/176 pattern (`build Tiny1M3MCrossHeadRMSNormConfig`, `config_class`, `/venv/main/bin/python _arq_181-cross-head-rmsnorm.py`). Must include a step-0 byte-identity check (`trt_step0_logits == ctrl_step0_logits`, expect max-abs-diff = 0.0).
- **Identity-init tolerance**: with the gate-α approach, step-0 is byte-identical (max-abs-diff = 0.0) within fp32 epsilon. The runner should assert byte-exact match in `trt_step0_logits == ctrl_step0_logits`.

## Pass / fail bar
- **tier**: `tiny1m3m` (0.94M params · 3M tok, **seed 42 only** — one-seed-only per pipeline rule).
- **control**: unmodded `Tiny1M3MConfig` (no cross-head RMSNorm), against cached baseline `val_mean = 6.3988 ± 0.0088`, `noise_band = 0.04` (per `autoresearch/baseline-cache.json`).
- **WIN**: `Δval = treatment_val − control_val ≤ −0.005` AND clears the cached `noise_band` (`treatment_val < val_mean − noise_band = 6.3588`).
  - Mirrors 016-qk-norm's bar; clears the ±0.04 box noise at tiny1m3m by ≥8×.
- **NULL**: `|Δval| < 0.005` (inside noise band).
- **DRIFT**: `Δval ≥ +0.005` (cross-head coupling breaks W_O's pre-trained magnitude assumptions ⇒ expected to be small but not catastrophic; +0.005 catches any wrong-sign axis collapse).
- **Sub-noise is inconclusive**: per one-seed-only rule, a `|Δ| < 0.005` outcome does not get a re-run with extra seeds — it is logged NULL and moves on.
- **Box check**: control val must be within `val_mean ± noise_band` of `LEADERBOARD.md`; if control drifts > ~0.01 from the cached baseline, the box is bad and the idea stays `needs-run`.

## Why it's worth a slot
The bet, in one sharp sentence: **the closed post-AV per-head gain (160 null) tested per-head independent rescaling and confirmed the per-head axis is closed; 181 tests a cross-head coupling that explicitly ties head magnitudes together — a different lever axis that 160 did not cover.** A null at 0.94M would close the cross-head-normalization family at our tier and confirm that head-magnitude coupling is not the binding constraint; a win would unlock the cross-head coupling axis for Phase-2 ≥135M where the head-magnitude balance matters more.

## Reviser note (r1)
- Promoted the gate-α to the canonical parameterization per review cleanup finding #1; chose `α_h = relu(α_raw_h)` init `−1e-3` over `sigmoid(α_raw)` to keep α exactly 0 at init (rather than sigmoid(-5)≈0.0067 non-zero). The tanh-gain on γ_h[k] stays as in the original sketch (init 0 ⇒ γ=1).
- Re-derived param count with gate-α included (cleanup finding #2): 4 (α) + 64 (γ) = 68 per block × 12 = 816 total (+0.087%).
- Added `## Plan` section locking field name, config subclass, MHA plumbing, apply site, mutual-exclusion asserts, param count, runner stub.
- Added `## Pass / fail bar` section with numerical thresholds (WIN ≤ -0.005, NULL |Δ| < 0.005, DRIFT ≥ +0.005). Used cached `val_mean = 6.3988` from `baseline-cache.json` (the reviewer's draft said ≈6.4394 — that was an older number; corrected to current cache).
- No disagreements with the reviewer's findings; all four were applied as proposed.
