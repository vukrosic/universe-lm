---
id: 203-pre-wo-se-channel-attn
status: needs-review
round: 3
updated: 2026-06-15T16:48:14Z
transfer-risk: med
plain: Insert a tiny Squeeze-Excitation channel-attention block right before the W_O projection (γ-gate init ≈ 0 ⇒ step-0 max-abs-diff < 1e-5 vs baseline; internal se_weight ≈ 0.5 but γ-gate silences the branch), letting each token softly up- or down-weight its own channels.
---

# 203 — Pre-W_O Squeeze-Excitation Channel Attention (Per-Token Channel Reweighting)

## Source
- Hu et al., "Squeeze-and-Excitation Networks" (SE, TPAMI 2019, arXiv:1709.01507) — channel attention in CNNs; validated at ImageNet across all scales (ResNet, EfficientNet).
- 181-cross-head-rmsnorm (closed null Δ=+0.1722 wrong-sign above band) — cross-head RMSNorm on attention output pre-W_O. Different axis (cross-head normalization vs channel attention).
- 142-layerscale (closed null Δ=+0.0172 wrong-sign) — per-channel diagonal gain on residual stream; SE is per-token channel attention (different placement and different shape).
- 160-rms-gain-per-head (closed null Δ=−0.0023 inside band) — per-head gain on attention output. Different axis (head-wise vs channel-wise).
- 191-token-attn-gain (in-repo idea, status needs-taste) — per-token *scalar* on attention output. 203 is per-token *channel vector* on attention output (broader).
- Woo et al., "CBAM" (ECCV 2018, arXiv:1807.06521) — channel + spatial attention; SE is the channel-only branch.

## Mechanism
Standard pre-W_O: `attn_out_post = attn_out` (shape `[B, T, d_model]`), then `W_O(attn_out_post)`.

Per-token SE channel attention: insert a per-token channel attention block that reweights each token's channels. The MLP is applied **per-token to the channel vector** (no T-axis pooling — the original SE-Net CNN pattern pools over the spatial axis, but here the lever is the per-token content-dependent cell, not the original CNN cell):
```
se_weight_t(x_t) = sigmoid(W_2 · gelu(W_1 · x_t))    # per-token, x_t ∈ R^{d_model}
attn_out_se   = attn_out ⊙ se_weight_t(attn_out)     # elementwise along channel axis
attn_out_post = (1 − γ) · attn_out + γ · attn_out_se # γ = sigmoid(γ_raw), init γ_raw=-10
out           = W_O(attn_out_post)
```
Where `W_1 ∈ R^{d_model × d_model/r}` and `W_2 ∈ R^{d_model/r × d_model}` with reduction ratio `r=4` (so `W_1` is `64 × 16`, `W_2` is `16 × 64`). Same W_1, W_2 applied to every token/position; no `[B, d_model/r]` pooled intermediate.

At init γ=sigmoid(-10) ≈ 4.54e-5, `attn_out_post ≈ attn_out` (clean A/B vs same-seed baseline; see `## Pass bar` for the numeric bit-identity bar). The internal `se_weight_t` at step 0 is ~0.5 per channel (not 1.0), but the γ-gate silences the whole branch, so the blend's contribution to `attn_out_post` is at the fp32 floor regardless. As γ grows, the SE branch is *added* to the residual via γ-weighted blend.

## Design sketch
- **File**: `models/layers.py` — add an `se_channel_attn` module to attention block.
- **Config flag**: `use_se_pre_wo: bool = False`, `se_reduction_ratio: int = 4`, `se_alpha_init: float = -10.0` (sigmoid ≈ 0).
- **Compute**: `attn_out_post = (1 − sigmoid(γ_raw)) · attn_out + sigmoid(γ_raw) · (attn_out ⊙ sigmoid(W_2 · gelu(W_1 · attn_out)))`.
- **Step-0 bit-identity (clean A/B)**: γ_raw=-10 ⇒ sigmoid(γ_raw) ≈ 4.54e-5, so `attn_out_post = (1 − 4.54e-5)·attn_out + 4.54e-5·(attn_out ⊙ se_weight_t)` ⇒ `max-abs-diff(attn_out_post, attn_out) < 1e-5` vs the same-seed baseline run at fp32. The internal `se_weight_t` is ~0.5 (not 1.0), but the γ-gate silences the whole branch — implementer's self-check is `max-abs-diff < 1e-5`, not "exactly bit-identical."
- **Params**: SE block: 2 × (d_model × d_model/r) = 2 × (64 × 16) = 2048 params per block × 12 blocks = 24,576 params (+2.6% of 0.94M); plus 12 γ_raw scalars.
- **Param group for γ_raw**: route to the **Muon** param group (not AdamW). 1-D gain scalars benefit from Muon's LR scale; AdamW at peak LR 0.024 is ~10× too hot for a scalar. The default `is_muon_candidate` in `training/trainer.py` requires `ndim==2`, so the implementer should either (a) name the param with a `norm`-suffixed key (so `muon_for_1d_norm=True` catches it) or (b) add a small explicit `if 'se_gamma' in name` branch in the Muon routing. Per 021/207 reviewer precedent (1-D gains → Muon).
- **Intuition**: SE channel attention reweights each token's channels based on the *content* of that token. The bet: at 0.94M, attention already mixes cross-token information, but within-token channel mixing is implicit through W_O. An explicit per-token channel attention gives the optimizer a clean axis to up-weight informative channels and down-weight noise channels — distinct from LayerScale (per-channel diagonal gain) and cross-head RMSNorm (cross-head normalization).

## Pass bar
- **Numeric bar (WIN)**: `Δval ≤ -0.01` vs the 4-ctrl cluster mean (6.4394) at tiny1m3m, seed 42. The cache band at tiny1m3m is ±0.04 and the box-noise band is ±0.01; -0.01 clears the noise band and must also beat **both** individual ctrls in the cluster per the §2 two-ctrl rule.
- **NULL (informative)**: `Δval` inside `6.4394 ± 0.04` (cache band) closes the post-AV axis family at 0.94M — paired with the static cells (142, 160, 181, 191) this completes the {static, content} × {per-channel, per-token} 2×2 on the post-AV axis family.
- **Above band (informative)**: `Δval ≥ 6.4394 + 0.04` and no param-group mistake → consider the lever actively harmful and abandon.
- **Bit-identity check (gate for the WIN/NULL read)**: the implementer must report `max-abs-diff(attn_out_post, attn_out) < 1e-5` at step 0 vs the same-seed baseline. If the diff is much larger, suspect a config-flag wiring bug (γ not on the residual, W_1/W_2 init non-default) — not a real signal.

## Scale evidence
SE-Net validated at ImageNet (all scales); CBAM validated at ImageNet. No published "pre-W_O SE channel attention for LMs" paper I'm aware of. Transfer-risk: med (lever is a well-known CNN primitive applied to a fresh placement in attention output).

## Why it's worth a slot
**Pattern**: per-channel gain (142 LayerScale) and cross-head RMSNorm (181) both closed null at 0.94M. 203 is *per-token channel attention* — a learned *content-dependent* channel reweighting, distinct from per-channel static gain (LayerScale) and cross-head normalization. The bet: at 0.94M, the *content* of the token matters for which channels are informative, and a per-token SE block exploits this. A 203 WIN would mean content-dependent channel reweighting binds; a 203 NULL would mean channel reweighting is generally redundant with W_O at 0.94M.
