---
id: 199-spectral-attn-output
status: needs-repitch
round: 1
updated: 2026-06-15T08:31:36Z
transfer-risk: med
plain: Constrain each token's attention output to have bounded Lipschitz constant (a soft spectral-norm cap), implemented as a learnable per-token scalar that scales the output so its norm matches a target — starting at 1 so step-0 is byte-identical.
---

# 199 — Spectral-Norm-Bounded Attention Output (Per-Token Spectral Cap)

## Source
- Miyato et al., "Spectral Normalization for GANs" (ICLR 2018, arXiv:1802.05957) — spectral normalization as Lipschitz constraint on discriminator weights; canonical reference.
- Gouk et al., "Regularisation of Neural Networks by Enforcing Lipschitz Continuity" (2021, arXiv:1804.04368) — Lipschitz bounding strategies.
- 142-layerscale (closed null) — per-channel diagonal gain on residual stream; *Lipschitz* bounding is different from *scale* bounding.
- 016-qk-norm (in-repo WIN) — QK normalization. 199 is *attention-output* normalization, different placement.
- 181-cross-head-rmsnorm (closed null Δ=+0.1722) — cross-head RMSNorm on attention output. Different axis (cross-head, not per-token spectral).
- 176-v-pre-av-norm (closed null Δ=+0.0303) — V normalization pre-AV product. Different placement (V, not output).

## Mechanism
Standard attention output: `out = softmax(QK^T/√d) @ V`, shape `[B, H, T, d_k]`.

Spectral-norm-bounded attention output: enforce that the per-token *spectral norm* of the attention output (treating the H × d_k matrix for each token as a linear operator) is bounded by 1:
```
out_post = out / (spectral_norm(out) + eps)    # spectral_norm = max singular value
out_final = out_post · γ                       # γ learnable, init 1
```
At init γ=1, the output is *exactly* the standard attention output (no normalization applied because the multiplier is `1 / (something) · 1`). Hmm, that's not right — the lever must have a non-trivial effect at step 0 to be byte-identical.

**Cleaner formulation (proposed for 199)**: apply a learnable per-token scalar `γ_t` that *multiplies* the attention output, where γ_t is computed from the spectral norm:
```
out_post = out · min(1, γ_target / (spectral_norm(out_per_token) + eps))
out_final = out_post · γ_h_global              # γ_h_global: per-head learnable scalar, init 1
```
At init `γ_h_global = 1`, no scaling applied (γ_target ≈ typical spectral norm of attention output, so `min(1, γ_target/spectral_norm) ≈ 1` most of the time). The lever is *learning to tighten the cap* by setting γ_h_global < 1 (forcing the spectral norm to be lower).

Actually, the simplest lever: apply `out_post = out · γ_h_global` where γ_h_global is per-head learnable scalar init 1. At γ_h_global = 1, byte-identical. The "spectral" framing is interpretation, not mechanism.

## Design sketch
- **File**: `models/layers.py` — add a per-head spectral-cap scalar to the attention output.
- **Config flag**: `use_attn_output_spectral_cap: bool = False`, `attn_output_cap_init: float = 1.0`.
- **Compute**: per head h, compute the per-token spectral norm `||out[h, t, :]||_2` (a vector norm, not spectral norm of H × d_k matrix — that's too expensive). Apply: `out[h, t, :] = out[h, t, :] · γ_h_global / (||out[h, t, :]||_2 + eps)`. So `out_post = out · γ_h_global · out / (||out||_2 + eps)` — this is unit-vector scaled output multiplied by γ_h_global.
- **Bit-identical at step 0**: γ_h_global = 1 ⇒ `out_post = out · 1 / (||out||_2 + eps) · out = out · out / (||out||_2 + eps)` — NOT byte-identical! This formulation always normalizes.
- **Better formulation**: just `out_post = out · γ_h_global` with γ_h_global init 1. At γ=1, byte-identical. The "spectral cap" is the *interpretation*; the lever is a per-head scalar on attention output. But this is essentially 160-rms-gain-per-head closed null.
- **Different framing**: apply the spectral norm computation to the **W_O projection** (Lipschitz control on the projection). Spectral-norm-regularize W_O: `W_O_eff = W_O · min(1, γ_target / σ_max(W_O))`. At γ_target = ∞, no regularization. Init at γ_target = some large value matching typical W_O spectral norm (e.g., σ_max(W_O_init) ≈ 2.0 at d_model=64). Bit-identical at step 0 if γ_target matches.

## Design sketch (revised — spectral-norm W_O regularization)
- **File**: `models/layers.py` — apply spectral normalization to W_O projection.
- **Config flag**: `use_wo_spectral_norm: bool = False`, `wo_spectral_cap: float = "auto"` (default = σ_max(W_O_init), no regularization).
- **Compute**: `W_O_eff = W_O · min(1, wo_spectral_cap / σ_max(W_O))` (no learnable params; the cap is a fixed HP). Power-iteration to compute σ_max(W_O).
- **Bit-identical at step 0**: if `wo_spectral_cap = σ_max(W_O_init)`, then `min(1, 1.0) = 1.0` ⇒ `W_O_eff = W_O` exactly.
- **Params**: 0 (cap is a fixed HP).
- **Intuition**: spectral-norm-regularized W_O is a Lipschitz constraint on what attention can write back to the residual stream. Different from 160-rms-gain-per-head (which is a *post-attention-output* gain) — 199 is a *W_O Lipschitz constraint*, applied at the projection level.

## Scale evidence
Spectral normalization validated at GAN training (Miyato et al. 2018, image generation at all scales); Gouk et al. 2021 validates Lipschitz regularization for general NNs. No published "spectral-norm W_O regularization" win for LMs that I'm aware of. Transfer-risk: med.

## Why it's worth a slot
**Pattern**: per-channel gain on attention output (160) closed null; per-channel gain on residual stream (142 LayerScale) closed null at 12L. 199 is a *different* shape: spectral-norm constraint on W_O (a Lipschitz control on the projection). The bet: at 0.94M, W_O is unconstrained, and *clipping* its spectral norm to its init value is regularization in a fresh axis. A 199 WIN would mean Lipschitz control on W_O binds at 0.94M (different from per-channel gain which doesn't bind); a 199 NULL would mean Lipschitz control is redundant with W_O's own training dynamics.
