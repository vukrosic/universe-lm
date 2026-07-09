---
id: 054-polycom
status: needs-plan
round: 2
updated: 2026-06-11T01:23:34Z
transfer-risk: low
---

# 054 — PolyCom (PolyNorm FFN activation)

## Source
Zhuo et al., "Polynomial Composition Activations: Unleashing the Dynamics of
Large Language Models" (arXiv:2411.03884). The paper proposes two FFN
activation modules: **PolyReLU** `Σ aᵢ·ReLU(x)^i` (a fixed sum of polynomial
powers of `ReLU(x)`, with learnable scalar coefs) and **PolyNorm** (the same
form with per-power RMSNorm inserted before each power). The paper's headline
result is PolyNorm-LLaMA outperforming SwiGLU/GELU/ReLU baselines at the 1B
dense and 1B-active-MoE scales.

## Mechanism
**Pin: PolyNorm-3** (degree-3 PolyNorm, the paper's headline variant per §3.2).
For the FFN hidden activation `h = up_proj(x)`, replace the standard activation
with:

```
PolyNorm-3(h) = a0·RMS(h)
              + a1·RMS(h) * h
              + a2·RMS(h) * h^2
              + a3·RMS(h) * h^3
```

where `RMS(y) = y / sqrt(mean(y^2) + eps)` (per-power RMSNorm, eps=1e-6,
weight-free — matches the paper's "no learned gain on the per-power norm"
choice). The four `aᵢ` are learnable scalar coefs.

**Identity init (mandatory):** `a0=0, a1=1, a2=0, a3=0` so step-0 reduces to
`h` (a linear pass-through — bit-equal to the no-activation baseline at init).
This is the zero-init standard already used by 020/023/024.

## Control variant
**Pin: `Tiny1M3MGELUConfig` (new)** as control, with `ffn_variant="gelu"`.
Rationale: the paper's own baselines are GELU/SwiGLU/ReLU, not squared_relu.
Replicating the paper's comparison first (A/B vs GELU) lets a win or null
reference the paper's reported delta directly. A second-round A/B vs
`squared_relu` (the leaderboard baseline) is a follow-up if the first round
wins — but it's not the primary ablation. `GELUFeedForward` already exists in
`models/components.py:55`, so the control needs only a new tiny config class.

**Treatment variant:** `Tiny1M3MPolyComConfig` (new), inheriting `Tiny1M3MConfig`
and switching `ffn_variant="polycom"`. Code side will wire `polycom` to a
`PolyComFeedForward` placed next to `GELUFeedForward` in `models/components.py`.

## Pass/fail bar
A/B at tiny1m3m, seed 42, identical data/optim/HP — only the FFN block swaps.

- **WIN**: treatment val_loss < GELU-control val_loss **by ≥ 0.01** (matches
  PIPELINE.md's ~±0.01 val-loss noise floor; this is the bracket convention
  used by 015/016). The treatment must also clear the **two-ctrl bracket**:
  run `Tiny1M3MConfig` (squared_relu) and `Tiny1M3MGELUConfig` (gelu) in
  parallel and verify the GELU-control sits within noise of the squared_relu
  leaderboard value (DRIFT check). If GELU-ctrl is itself a quiet winner
  (< squared_relu by ≥ 0.01) we still accept the result — PolyNorm beat the
  paper's baseline, which is the question.
- **NULL (inconclusive)**: `|Δ| < 0.005` between treatment and GELU-control.
  Logged as a null — polynomial curvature is a large-model trick.
- **LOSS**: treatment val_loss > GELU-control by ≥ 0.005 (i.e. the
  polynomial terms hurt at this scale).

Two-ctrl protocol: the runner launches three jobs on the box (squared_relu
ctrl, gelu ctrl, polynorm treatment) and reports the val-loss triple.

## Numerical stability
Polynomial powers ≥2 in bf16 have repeatedly NaN'd this repo (cf. 020
forgetting-attn and 022 softpick episodes). Mitigations, all mandatory:

1. **Per-power RMSNorm is non-optional.** Without it, `h^3` runs away for
   any outlier channel and the FFN output blows up. This is the whole point
   of PolyNorm vs PolyReLU; do not regress to PolyReLU.
2. **Polynomial powers computed in fp32 then cast back to bf16.** The FFN's
   `up_proj` runs in bf16 (matches the rest of the model), then
   `h.float().pow(k)` for k=2,3 with the final sum cast back to bf16 before
   `down_proj`. This costs ~2× the activation memory transiently but kills
   the NaN tail.
3. **Init coef magnitudes stay bounded.** `a0=0, a1=1, a2=a3=0` at step 0.
   During training, `a2, a3` are clamped to `[-2, 2]` after each optimizer
   step (cheap soft-clip) to prevent runaway growth — the paper's reported
   converged coefs are O(1), so the clamp is non-binding in the win regime.
4. **Gradient clip stays at the trainer default** (the 1.0 global-norm clip
   used by every other tiny1m3m run). No special clip.
5. **Forward-pass NaN guard.** If `torch.isnan(h).any()` triggers, the
   existing trainer NaN-detection path aborts cleanly (same handling as
   020/022).

## Param-count parity
- Baseline `Tiny1M3MConfig` (squared_relu, d_ff=256, 12 layers): ~0.94M params,
  of which the FFN pair is `2 · d_model · d_ff · n_layers = 2 · 64 · 256 · 12
  = 393,216` params (≈42% of the model).
- PolyNorm-3 adds 4 scalar coefs per layer (4 · 12 = 48 params) — **negligible**
  (0.005% of model).
- Decision: **unmatched-with-note.** No `d_ff` shrink; the overhead is well
  inside the 1% leaderboard reporting noise and the paper's reported gain is
  at full-width FFN. Document the +48 scalar params in the run record.

## Scale evidence
Zhuo et al. (2024) report a **0.02 average validation-loss improvement** for
PolyNorm over SwiGLU on the MoE table (Table 2: 2.39 vs 2.41 avg), plus
better downstream averages in Table 3 and scaling curves that favor PolyNorm
from 110M to 1.3B. transfer-risk: low because the paper's evidence is 1B dense
LLM pretraining loss/PPL plus 1B-active MoE, i.e. direct LM transfer, not
sequence-length or long-context driven.

## Why it's worth a slot
If PolyNorm wins at tiny1m3m, the FFN is under-expressing local function
class at the small scale too — i.e. the polynomial curvature is a *general*
lever, not a 1B+ artifact. If it nulls, we learn polynomial curvature is a
large-model trick and the FFN swap slot stays open for cheaper activation
tests (GELU/SwiGLU at our scale). Both directions inform.
