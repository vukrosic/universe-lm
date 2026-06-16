---
id: 199-spectral-attn-output
status: needs-run
round: 2
updated: 2026-06-16T00:34:26Z
transfer-risk: med
plain: Per-block learnable scalar γ_l multiplies σ_max(W_O_init) to form the W_O spectral cap; γ_l init 0 ⇒ byte-identical baseline; optimizer can loosen (γ_l>0) or tighten (γ_l<0) the Lipschitz bound on the projection itself, opening a regularization axis on the projection rather than a magnitude axis on its output.
---

# 199 — Spectral-Norm-Bounded W_O Projection (Per-Block Learnable Lipschitz Cap)

## Source
- Miyato et al., "Spectral Normalization for GANs" (ICLR 2018, arXiv:1802.05957) — spectral normalization as Lipschitz constraint on discriminator weights; canonical reference.
- Gouk et al., "Regularisation of Neural Networks by Enforcing Lipschitz Continuity" (2021, arXiv:1804.04368) — Lipschitz bounding strategies.
- 128-spectral-decoupling (closed null Δ=+0.10 wrong-sign) — gradient-space twin of 199's W_O Lipschitz framing. The two are NOT the same axis: 128 modulates the *update direction* (gradient orthogonalization in optimizer space), 199 modulates the *forward Lipschitz constant* of a specific projection. Different signal, different placement, different backprop path.
- 160-rms-gain-per-head (closed null Δ=-0.0023 inside band) — per-head scalar on attention *output* (post-AV, post-W_O). 199 is a scalar on W_O's *spectral norm* (pre-AV output, intra-projection). Different placement, different gradient signal — see "Distinction from 160" below.
- 142-layerscale (closed null Δ=-0.0012) — per-channel diagonal gain on residual stream.
- 176-v-pre-av-norm (closed null Δ=+0.0303) — V normalization pre-AV product.
- 181-cross-head-rmsnorm (closed null Δ=+0.1722 wrong-sign) — cross-head RMSNorm on attention output.

## Mechanism (revised r2 — single branch, learnable axis)

Standard attention output: `out = (softmax(QK^T/√d) @ V) @ W_O`, shape `[B, T, d_model]` where W_O is `[d_model, d_model]` (single matmul) or `[H, d_k, d_model]` depending on impl.

Spectral-norm-bounded W_O projection: per block *l*, apply an *asymmetric* Lipschitz cap on W_O's spectral norm σ_max(W_O^[l]):

```
cap_l       = σ_max(W_O_init^[l]) · exp(γ_l)     # γ_l: per-block learnable scalar, init 0
W_O_eff^[l] = W_O^[l] · min(1, cap_l / σ_max(W_O^[l]))
            = W_O^[l] · min(1, σ_max_init / σ_max_current · exp(γ_l))
```

At init `γ_l = 0` and `σ_max(W_O^[l]) = σ_max(W_O_init^[l])`, so the ratio `σ_max_init/σ_max_current = 1` and `exp(0) = 1`, giving `min(1, 1·1) = 1` and `W_O_eff^[l] = W_O^[l]` **byte-identically**. The lever is dormant.

As training proceeds, σ_max(W_O) drifts up or down. The optimizer can:
- **Loosen** by pushing `γ_l > 0`: cap grows above current σ_max, factor stays 1, W_O unchanged.
- **Tighten** by pushing `γ_l < 0`: cap drops below current σ_max, factor `<1`, W_O scaled down (Lipschitz constraint binds).

The asymmetry (clip-only) is the bet: σ_max typically only grows under SGD at init scale 1, so the *informative* direction is `γ_l < 0` (tighten). γ_l > 0 is wasted optimizer signal that the model must discover and ignore. This is **damage-only-with-learnable-axis** — a deliberate trade: byte-identical baseline (good), but the optimizer must learn the right tightness from scratch (no inductive prior).

## Design sketch
- **File**: `models/layers.py` — wrap the W_O matmul with the spectral-cap factor; cache σ_max(W_O) via power iteration (1 step per block per forward pass is sufficient for tracking; σ_max updates slowly at 0.94M/12L).
- **Config flag**: `use_wo_spectral_cap: bool = False`, `wo_spectral_cap_pi_iters: int = 1` (power-iteration steps per forward).
- **Params**: `n_layers` (one γ_l per block) = 12 at tiny1m3m · +0.001% overhead.
- **Bit-identical at step 0**: γ_l = 0, σ_max matches init ⇒ factor = 1 ⇒ output is byte-identical. Confirmed by construction.
- **Compute overhead**: `n_blocks · d_model · d_k` power-iteration vector (u) maintained; one matmul per block per forward to update u ← W_O·v / ||·||₂. At d_model=64, d_k=16, n_blocks=12 this is ~12K FLOPs per forward — negligible vs the 0.94M body.

## Distinction from 160 (load-bearing)

| axis | 160-rms-gain-per-head | 199-wo-spectral-cap |
|---|---|---|
| placement | post-AV, post-W_O (attention *output*) | intra-W_O (projection *itself*) |
| granularity | per-head (H·n_blocks = 4·12 = 48) | per-block (1 per block = 12) |
| what changes | magnitude of attention contribution to residual stream | Lipschitz constant of W_O projection |
| forward effect | `out_post = out · γ_h` | `out_post = out · min(1, cap_l/σ_max(W_O))` |
| gradient signal | dL/dγ_h flows through post-attention add | dL/dγ_l flows through projection parameter; only fires when cap binds |
| inductive prior | symmetric (can grow or shrink) | asymmetric (only shrinks); learnable but single-sided |

160 modifies the *downstream* gain (magnitude axis on attention output, easily invertible — the model can just compensate by raising W_O). 199 modifies the *projection*'s Lipschitz constant (regularization axis on the projection itself — once W_O is clipped, downstream layers must adapt to the clipped projection). These are different *signals* the optimizer must solve for: 160 is a soft gain on a residual add (magnitude prior), 199 is a hard constraint on a weight (capacity regularization). 160 nulled at Δ=-0.0023 (inside band, optimizer had no pressure to push gains off 1). 199's bet is that the projection constraint *does* bind at 0.94M — because σ_max(W_O) grows under SGD (the standard Lipschitz-growth dynamic), and clipping it is information the model can't recover elsewhere cheaply.

## Sharp prediction (Δval window)

**Hypothesis**: at tiny1m3m, σ_max(W_O^[l]) grows ~1.3-1.5× over the 92-step run (typical for a 12-layer transformer with init σ ≈ √(2/fan_in) ≈ 1.0 and He/Kaiming forward). Power-iteration tracks this growth, and the optimal γ_l drifts to ≈ -log(1.3..1.5) ≈ -0.26..-0.40. The cap binds softly through the late second half of training.

**Predicted Δval window**: `[-0.005, -0.012]` vs fresh ctrl mean. Justification:
- *Lower bound* -0.005: matches the best 160/142 null magnitudes; the lever is sub-noise but not actively harmful.
- *Upper bound* -0.012: the 016-qk-norm WIN was -0.012 (norm on a projection inside attention); 199 is a *softer* constraint on a *larger* projection (W_O is the bigger weight), so its leverage should be at most equal. Realistically half of 016's leverage.
- **Wrong-sign risk**: σ_max might *shrink* under AdamW's per-parameter scaling at 0.94M (He forward + AdamW second-moment damping). If σ_max(W_O) decreases, the cap never binds, and 199 reduces to a no-op spending power-iteration overhead. Track σ_max(W_O) during the run; if it never exceeds init, the lever is mechanically a null (informative: closes the W_O-Lipschitz axis, distinct from 160's null).

**Information-value floor**: a clean null at 0.94M (|Δ|<0.005) would close the post-attention-shape null family at **5 deep** (160, 142, 181, 176, 199). That is itself useful — the family has burned 4 slots for nulls inside the noise band, and a 5th lets us retire the family as "axis exhausted at 0.94M" with high confidence.

## Scale evidence
Spectral normalization validated at GAN training (Miyato et al. 2018, image generation at all scales); Gouk et al. 2021 validates Lipschitz regularization for general NNs. No published "spectral-norm W_O regularization" win for LMs. The closest LM analogue is spectral-decoupling (128) which nulled at 0.94M. Transfer-risk: **med** (strong mechanistic argument: σ_max growth under SGD is a textbook dynamic; the question is whether the cap binds informatively, not whether the mechanism is sound).

## Why it's worth a slot (revised)
**Bet**: 199 is *structurally distinct* from 160 (regularization on projection vs magnitude on output). The 4-deep null family burned on output-side gains (160, 142) and output-side normalizations (181, 176) — none touched the projection's Lipschitz constant. W_O is the single largest projection in the model (d_model·d_model = 4K params at d=64) and σ_max(W_O) is unconstrained at 0.94M. A 199 WIN would mean Lipschitz control on the projection is the missing axis the post-AV gains missed. A 199 NULL would close the family at 5 deep and retire the post-attention-shape null axis with high confidence. **Either outcome is informative** — the lever costs 12 params and ~12K FLOPs/step to test. Null is the more likely outcome (mechanism is sub-noise at 0.94M per the 4-deep family prior), but the win/loss asymmetry is favorable because the null closes a family.
