---
id: 188-qk-rms-scaling
status: planning
round: 1
updated: 2026-06-15T12:03:33Z
transfer-risk: low
plain: On top of the WIN 016-qk-norm, add one learnable per-block scalar on pre-softmax scores (init 1, exp-parameterized), so the optimizer can compensate for any sub-optimal sharpness that the symmetric QK norm induces.
---

# 188 — Per-Block QK-Logit Scalar Stacked on the WIN 016-qk-norm (init=1)

## Source
- 016-qk-norm (closed WIN, 2026-06-09, closed.md) — symmetric QK RMSNorm lowers val_loss from 6.4044/6.4091 to 6.3906 at tiny1m3m. The norm **scales** the magnitude of Q and K separately (gain per-head), then **divides** by sqrt(d_k); it does not introduce a free per-block scalar on the post-norm QK^T product. Stacking a per-block scalar is the natural follow-up.
- 161-dyt-temp (closed DRIFT 2026-06-14, Δ=+0.0830) — same parameterization, but stacked on the **plain baseline**, not on 016. r1 missed this; 161 closed on the *plain ctrl mean*, where the lever fights the canonical attention-scale prior. Stacking on 016 changes the gradient landscape because the QK norm already re-scales |Q| and |K|, so the magnitude of QK^T is no longer pinned to d_k and a per-block scalar has a *different* axis to exploit (re-sharpening whatever the norm flattened, or sharpening past what the norm left alone).
- μP / μTransfer (Yang et al. 2022, arXiv:2203.03466) — per-layer logit-temperature scaling is the standard μP axis at 10M-13B; stacking it on a normalization layer is a published µP recipe (init=1, exp-parameterized).
- Pythia per-layer LR multiplier (Biderman et al. 2023, arXiv:2304.01373) — establishes per-block scalar re-tuning as a recognized axis at 70M-12B (different mechanism — LR, not temperature — but the *per-block* parameter count and signal-to-noise story is analogous).

## Mechanism
Standard pre-softmax QK product:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T]
weights = softmax(scores)
out = weights @ V
```

016-qk-norm (the closed WIN) **already** does:
```
Q = RMSNorm(Q) * gamma_h_q   # gamma_h_q: per-head learnable gain
K = RMSNorm(K) * gamma_h_k
scores = Q @ K^T / sqrt(d_k)
```

This re-pitch stacks **one extra per-block scalar** on top:
```
scores = (Q @ K^T / sqrt(d_k)) * s_l   # s_l: [n_layers] learnable, init 1.0
weights = softmax(scores)
out = weights @ V
```

`l ∈ {0, ..., n_layers-1}` indexes the transformer block. `s_l` is a per-block scalar that sharpens (`s_l > 1`) or flattens (`s_l < 1`) the post-QK-norm attention distribution *uniformly across all heads in the block*.

**Parameterization**: `s_l = exp(s_param_l)` with `s_param_l = 0` init ⇒ `s_l = exp(0) = 1.0` exactly in IEEE 754 ⇒ **byte-identical to 016-alone at step 0** (no epsilon leak; the post-norm scores are unchanged).

**Why this is a different axis from 161 (and not just init=1 vs init=1/sqrt(d_k))**:
- 161 sat on the plain baseline where `|Q|` and `|K|` are uncontrolled, so the per-block scalar fights the canonical `1/sqrt(d_k)` prior *and* the Q/K gradient magnitude noise. At 0.94M the gradient signal on `s_l` was swamped.
- On top of 016, `|Q|` and `|K|` are RMS-normalized *per head* — the magnitude of `QK^T` is bounded and centered around `1` per dim, so `s_l` operates on a *signal* (post-norm) rather than on a *noise prior* (uncontrolled norm magnitude). The gradient signal on `s_l` is now: "this block's softmax is too sharp or too flat *relative to* what 016 leaves behind" — a cleaner axis than 161 had.
- The init difference (1.0 vs 1/sqrt(d_k)) is no longer the bet — the bet is that the **post-norm** signal-to-noise ratio is high enough at 0.94M for `s_l` to find a useful different-per-layer schedule that 161 could not.

**Why this isn't the "5th variant in a row"**: 016 (WIN) is a *normalization* lever, 152/155/161 are *temperature/bias* levers, 169 is a *placement* lever, 184 is an *output-side* lever. 188-stacked-on-016 is the first lever that *combines* a normalization (closed WIN) with a temperature; the null is informative ("norm + temp doesn't bind at 0.94M"), and the win amplifies the WIN.

## Design sketch
- **Files**:
  - `configs/llm_config.py` — add `use_qk_rms_scaling: bool = False` on `LLMConfig` (next to `use_per_layer_temp` at line ~157). Add `Tiny1M3MQKRMSConfig(Tiny1M3MQKNormConfig)` (subclass the **016 config** — `use_qk_norm: bool = True`, `use_qk_rms_scaling: bool = True`), mirroring the `Tiny1M3MLogitScaleConfig` pattern.
  - `models/layers.py` — `MultiHeadAttention.__init__`: add `use_qk_rms_scaling: bool = False` kwarg and store `self.qk_rms_param = nn.Parameter(torch.zeros(1))` when flag is on (one scalar per MHA = one per block). `MultiHeadAttention.forward`: after the existing 016 `Q @ K^T / sqrt(d_k)`, multiply `scores = scores * self.qk_rms_param.exp()` before the causal mask and softmax. Add `or self.use_qk_rms_scaling` to the manual-path forcing list (so SDPA flash doesn't perturb step-0 numerics).
- **Config flag**: `use_qk_rms_scaling: bool = False` (off by default; baseline path bit-identical).
- **Param count**: 1 scalar per MHA × 12 blocks = 12 scalars total (+0.0013% of 0.94M).
- **Step-0 byte-identity**: `nn.Parameter(torch.zeros(1))` → `s = exp(0) = 1.0` exactly in IEEE 754 → `scores * 1.0 = scores` exactly. The exp form guarantees positivity (a negative scalar would invert the softmax — meaningless zero-init).
- **Intuition**: 016 makes each block's Q/K magnitude uniform; it does not pick a *temperature* per block. Different depths plausibly want different temperatures even after norm: late-layer post-norm logits may saturate (the closed 161 DRIFT is consistent with over-sharpening on the baseline ctrl), early-layer post-norm logits may under-saturate. The per-block `s_l` is the simplest depth-conditional re-temperature that operates on the post-norm signal.

## Scale evidence
- 016-qk-norm (closed WIN) — Δ=−0.0138/−0.0185 at tiny1m3m; 188-stacked-on-016 is the published-style follow-up the WIN recipe invites.
- μP / μTransfer (Yang 2022) — per-layer logit-temperature is a standard axis at 10M-13B; init=1, exp-parameterized is the published µP recipe.
- Pythia per-layer LR multiplier (Biderman 2023) — per-block scalar re-tuning at 70M-12B (different mechanism; same axis-shape).
- DeepNet α = 1/sqrt(2L) (Wang 2022) — fixed depth-conditional scale, validated 200-1000L; 188 is the *learned, per-block* analog, stacked on the WIN.
- **Transfer-risk: low** — per-block learned scalar is a recognized depth-conditional lever with multiple ≥100M validations; stacking it on a normalization layer is a published µP recipe (init=1, exp-parameterized).

## Why it's worth a slot
The bet, in one sharp sentence: **the closed per-block temperature on the plain baseline (161) DRIFTed because the lever fought the un-normalized Q/K magnitude prior; stacked on the WIN 016-qk-norm the lever operates on a *post-norm* signal where |QK^T| is bounded and centered, so a per-block scalar has a clean axis to re-tune depth-conditional sharpness without the magnitude-noise swamping that 161 hit** — a null at 0.94M would close the "norm + per-block temp" axis (distinct from 161's "plain + per-block temp"), and a win would amplify the WIN 016 by an additional Δ≤−0.005.

## Pass/fail bar at tiny1m3m (seed 42)
- **Baseline (this run's ctrl)**: `Tiny1M3MQKNormConfig` (the existing 016 WIN), val ≈ 6.39.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01` (i.e., 016 + per-block temp ≈ 016 alone).
- **DRIFT**: `trt_val > ctrl_val + 0.01` (per-block temp re-sharpens past the 016 optimum).

## Distinct from closed axes (defensive)
- **161-dyt-temp (DRIFT) — same parameterization, different stack target.** 161 = plain baseline + per-block τ_l init 1/sqrt(d_k); 188 = 016-qk-norm (WIN) + per-block s_l init 1.0. The lever is the same family, but the gradient landscape is different (post-norm vs un-normalized). r1 missed this; r2 corrects it.
- 016-qk-norm (WIN) — 188 *stacks on* 016, does not replace it. ctrl here is the 016-alone baseline, not the plain baseline.
- 155-per-head-temp (NULL) — per-head axis (48 params), closed null at 0.94M. 188 is per-block (12 params), different axis.
- 152-attn-logit-bias (NULL) — per-head *additive* bias, different mechanism (additive vs multiplicative).
- 184-logit-scale — global scalar on LM-head *output* logits, not pre-softmax QK^T.
- 169-qk-norm-depth (NULL) — placement of QK norm (which block gets the norm), not a scalar multiplier on scores.
- The r1 "Distinct from closed axes" section listed five of these but conspicuously omitted 161; r2 corrects.

## Plan
- **Files**:
  - `configs/llm_config.py` — add `use_qk_rms_scaling: bool = False` on `LLMConfig` (next to `use_per_layer_temp`). Add `Tiny1M3MQKRMSConfig(Tiny1M3MQKNormConfig)` with `use_qk_norm: bool = True, use_qk_rms_scaling: bool = True`.
  - `models/layers.py` — `MultiHeadAttention.__init__`: add `use_qk_rms_scaling: bool = False` kwarg and store `self.qk_rms_param = nn.Parameter(torch.zeros(1))` when flag is on. `MultiHeadAttention.forward`: after `scores = QK^T * scale` (post-016-norm step), multiply `scores = scores * self.qk_rms_param.exp()` before the causal mask and softmax. Add `or self.use_qk_rms_scaling` to the manual-path forcing list.
- **Config flag**: `use_qk_rms_scaling: bool = False` (off by default; baseline path bit-identical).
- **Param count**: 1 scalar per MHA × 12 blocks = 12 scalars total (+0.0013% of 0.94M).
- **Step-0 byte-identity**: `nn.Parameter(torch.zeros(1))` → `s = exp(0) = 1.0` exactly in IEEE 754 → `scores * 1.0 = scores` exactly.
- **Run command**: `autoresearch/bin/run-idea.sh 188-qk-rms-scaling tiny1m3m 42` (or equivalent pipeline slot — see `PIPELINE.md`). Val read: from `autoresearch/remote-results/<run-dir>/trt_*.log` — search for the final `val` line emitted by `train.py`. Champion reference `Tiny1M3MQKNormConfig` (016 WIN) val 6.3906.
- **Pass/fail**: WIN if `trt_val ≤ ctrl_val − 0.005` AND clears two-ctrl rule. NULL `|Δ| < 0.01`. DRIFT `trt_val > ctrl_val + 0.01`.
