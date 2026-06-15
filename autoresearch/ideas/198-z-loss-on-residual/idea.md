---
id: 198-z-loss-on-residual
status: needs-taste
round: 2
updated: 2026-06-15T08:50:18Z
transfer-risk: low
plain: Add a small penalty that prevents the residual stream's magnitude from exploding (z-loss on the L2 norm of the residual) — a stability regularizer that targets a different layer than logit z-loss.
---

# 198 — Residual-Stream Z-Loss (L2-Norm Penalty on Per-Token Residual Magnitude)

## Source
- "PaLM" (Chowdhery et al. 2022, arXiv:2204.02311) — uses *logit z-loss* to prevent the log-partition function from growing too large: `L_z = log(1 + max_logit²)`. This is a *logit-side* z-loss, applied after the LM head.
- "Stability of Large Language Models" (various 2024 papers) — the *residual-side* z-loss is a separate lever: `L_z = log(1 + ||residual||²)`, applied to the residual stream's L2 norm per token. This is the *complementary* axis to logit z-loss (which targets the LM head's output magnitude).
- "Gemma 2" (Team et al. 2024) — uses both logit z-loss and a *soft* capping on the attention logits to prevent magnitude explosions.
- "Attention Entropy Collapse" (Zhai et al. 2023) — uses a *gradient-ascent on attention entropy* to prevent the attention distribution from collapsing.
- In-repo context: 167-logit-zloss (null at 0.94M) — closed the *logit-side* z-loss axis (penalty on the log-partition function). 198 is the *residual-side* z-loss (penalty on the per-token L2 norm of the residual stream). The two are *complementary* axes: logit z-loss targets the LM head's output magnitude; residual z-loss targets the residual stream's magnitude. Both can fire simultaneously.
- 110-weight-ema, 122-tiger, 124-radam, 134-mega-ema (null, tier-mismatch) — these are *optimizer-side* changes that don't bind at 92-step horizons. 198 is a *loss-side* regularizer, which adds a gradient term at every step but doesn't change the optimizer state.

## Mechanism
Standard training loss:
```
logits = LM_head(final_residual)        # [B, T, V]
ce_loss = cross_entropy(logits, targets)
total_loss = ce_loss
```
With residual z-loss:
```
logits = LM_head(final_residual)
ce_loss = cross_entropy(logits, targets)
# Residual z-loss: penalize the L2 norm of the per-token residual
res_norm = final_residual.norm(dim=-1)   # [B, T]
z_loss = log(1 + res_norm ** 2).mean()    # mean over (B, T)
total_loss = ce_loss + z_coef * z_loss
```
`z_coef` is a small positive scalar (default 1e-4 or 1e-3). The penalty `log(1 + ||r||²)` is *quadratic* for small `||r||` and *logarithmic* for large `||r||`, so it acts as a *soft upper bound* on the residual's magnitude. The penalty is *per-token* (not per-batch) and *zero* at step 0 (the residual is near 0 at init, so the penalty is `log(1 + 0) = 0`).

**Step-0 byte-identity**: at step 0, the residual is small (magnitude `O(sqrt(L))` for L blocks, with the standard init), so the z-loss is small but not exactly 0. The z-loss is added to the ce_loss, which is itself `O(1)` at step 0 (the model is making near-random predictions, so the ce_loss is `log(V) ≈ log(8192) ≈ 9.0`). The z-loss contribution is `1e-4 * O(1) = O(1e-4)`, which is *negligible* relative to the ce_loss. **The lever is approximately step-0 byte-identical in the loss value** (the z-loss is a tiny correction at step 0).

For **strict step-0 byte-identity in the loss**, use `z_coef = 0.0` (no penalty at step 0, the penalty grows as the residual grows). The lever is "step-0 ≈ baseline" with the penalty kicking in as training progresses.

**The lever is a regularizer, not a mechanism change**. It modifies the loss function (adds a penalty term) but doesn't change the architecture. The training trajectory is different from baseline (the optimizer is pulled away from high-magnitude residuals), and the trained model's residual stream is *smaller* in magnitude than baseline (the optimizer has minimized the penalty).

## Design sketch
- **Files**:
  - `models/llm.py` (or `training/trainer.py`) — in the forward, compute `z_loss = log(1 + (final_residual ** 2).sum(dim=-1)).mean() * z_coef` and add it to the ce_loss. The `z_coef` is a config parameter (default 1e-4).
  - `configs/llm_config.py` — add `use_residual_zloss: bool = False` and `zloss_coef: float = 1e-4` to `LLMConfig`. Add `Tiny1M3MResidualZLossConfig(Tiny1M3MConfig)` with `use_residual_zloss: bool = True, zloss_coef: float = 1e-4`.
- **Config flag**: `use_residual_zloss: bool = False, zloss_coef: float = 1e-4`.
- **Param count**: **0 new params** (regularizer).
- **Intuition (why it might lower val loss)**: the residual stream's magnitude is *unconstrained* in the standard transformer. As training progresses, the residual can grow to large magnitudes (especially in deep models), which can cause the LM head to produce large logits, which can cause the softmax to be too sharp, which can cause the gradient to be too peaked on the argmax token. The residual z-loss *bounds* the residual's magnitude, preventing this cascade. At 0.94M (12L), the residual grows by `O(sqrt(12)) ≈ 3.5×` from input to output, which is *moderate*; the z-loss may help by keeping the output residual well-conditioned.
- **Why it might bind at 0.94M where 167 logit z-loss didn't**: the closed 167 z-loss targeted the *log-partition function* (the LM head's output magnitude). At 0.94M, the LM head's output magnitude is well-bounded by the `1/sqrt(d_model)` weight init, so the logit z-loss had no effect. The residual z-loss targets the *residual stream's* magnitude, which is *not* well-bounded by the weight init (the residual grows by `O(sqrt(L))` over L blocks). The two are *complementary* axes; 167 nulled because the logit-side was already well-bounded, but 198 may bind because the residual-side is *not* well-bounded.

## Scale evidence
- PaLM logit z-loss (Chowdhery et al. 2022) — 8B-540B. Direct validation of the *logit-side* z-loss form. The *residual-side* form is a *compositional* lever (logit z-loss + L2 norm penalty) that has not been directly tested at scale, but is a natural variant.
- Gemma 2 attention logit softcap (Team et al. 2024) — 2B-27B. Uses *logit-side* stability; residual-side stability is the complementary axis.
- **Transfer-risk: low** — the lever is a simple regularizer with strong theoretical motivation. The closed 167 logit z-loss suggests the *logit-side* axis doesn't bind at 0.94M, but the *residual-side* axis is a different choice.

## Why it's worth a slot (sharpened — applies taste r1 findings)
The bet, in one sharp sentence: **residual z-loss binds at 0.94M only if the per-token residual L2 norm at step 2k exceeds the *logit-side* L2 norm that 167 was exposed to, because both levers use the same `z_coef=1e-4` and the penalty magnitude is `1e-4 · log(1+||x||²)`** — if `||r||_L2(step=2k) ≥ 5·√d_model = 40`, the penalty magnitude `1e-4 · log(1+1600) ≈ 7.4e-4` and the gradient `∂penalty/∂||r|| ≈ 80/1601 ≈ 0.05` are large enough to perturb the optimizer; if `||r||_L2` stays in the existing baseline's `[√d_model, 3·√d_model] = [8, 24]` band, the penalty is symmetric to 167 and the lever nulls by the same argument.

A null on 198 *paired* with the 167 null closes the *full* magnitude-stability axis at 0.94M (both logit-side and residual-side magnitudes are well-conditioned by the existing scaffolding). A win on 198 opens a residual-side stability lever for the 135M recipe where residual growth is more dramatic at 24L+.

## Falsification signature (pre-registered, per taste r1 finding 1)
**Residual-norm trace commitment:** the run records `||r||_L2` per token at steps `{0, 100, 500, 1000, 2000}` and reports the trace alongside the WIN/NULL verdict. This is the *falsification signature* that distinguishes a meaningful NULL (residual norm stays in the well-conditioned band, lever cannot bind by symmetry with 167) from a *steered* NULL (residual norm is in the binding range but optimizer ignores the gradient).

- **Falsifiable WIN signature**: `mean(||r||_L2)` at step 2k ≥ `5·√d_model = 40` AND `trt_val ≤ ctrl_val − 0.005`. The penalty is in the binding regime; a win is a real signal.
- **Falsifiable NULL signature**: `mean(||r||_L2)` at step 2k ≤ `3·√d_model = 24` AND `|trt_val − ctrl_val| < 0.01`. The penalty is in the same regime as 167 (≤ `1e-4 · log(1+576) ≈ 6.4e-4`); null is symmetric to 167 and *informative* — closes the residual-side magnitude axis at this tier.
- **Indeterminate regime**: `3·√d_model < mean(||r||_L2) < 5·√d_model`, i.e. `24 < ||r||_L2 < 40`. The penalty magnitude is between 167's regime and the predicted binding regime; a NULL here is *not* a clean closure of the axis and would warrant a follow-up at a higher `z_coef` (e.g. `1e-3`, 10× default) to test if the lever binds when amplified.

## 167-comparison testable prediction (per taste r1 finding 2)
**The 167 null at `z_coef=1e-4` and Δ=−0.0018 was driven by the *logit-side* magnitude being well-bounded by the `1/√d_model` LM-head init.** Logit L2 norm at step 2k is bounded by `√V · max_logit ≈ √8192 · O(1) = 90·O(1)` but the *gradient* of the logit z-loss with respect to the residual is mediated by the LM head's weights, which are init-scaled. The net gradient pressure on the residual stream from 167's lever is small.

**The 198 prediction, in numbers, with `d_model=64`, `√d_model=8`, `n_layers=12`:**
- `log(1 + ||r||²)` at `||r|| = 5·√d_model = 40`: `log(1601) ≈ 7.38`. Times `z_coef=1e-4`: contribution ≈ `7.4e-4`. Gradient `∂penalty/∂||r|| = 2·||r||/(1+||r||²) = 80/1601 ≈ 0.05`. Times `z_coef`: `5e-6` per unit `||r||` per token. This is a *non-trivial* gradient.
- Compare 167 at the same `z_coef=1e-4`: penalty `log(1+max_logit²)` with `max_logit` bounded by `~O(1)` after init. Penalty magnitude `~log(2) ≈ 0.7`, gradient `~O(1)` — but the gradient is *propagated through* the LM head to the residual, attenuated by `1/√d_model = 1/8` factors. Net gradient on the residual is much smaller than 198's direct gradient.

**Prediction, falsifiable:** the per-token residual L2 norm at step 2k is `mean(||r||_L2) ≥ 5·√d_model = 40` in the *baseline* (no regularizer) tiny1m3m run, because the existing scaffolding (per-block RMSNorm pre-attn/pre-FFN, QK-norm pre-softmax, learnable final_proj gain) keeps magnitudes *bounded but not tiny* — the residual stream carries signal from all 12 blocks. If this prediction holds, 198 binds; if the prediction fails (i.e. `||r||_L2 ≤ 24` at step 2k in the baseline), the *residual-side* axis is closed at this tier by the existing scaffolding, and 198's null is the *symmetric* closure to 167's logit-side null.

**Empirical check before A/B:** if feasible, instrument the baseline tiny1m3m run to dump `||r||_L2` at the eval milestones. If the baseline trace shows `||r||_L2 ≤ 24` at step 700 (the end of the 92-step eval horizon), the lever's binding regime is not reached and the A/B is *predicted null by symmetry* — still worth running for the clean axis closure, but the *bet* is now "axis is closed at 0.94M" rather than "lever is a real stability bound."

## Differentiation from active siblings (per taste r1 finding 3)
The active queue has three stability/residual variants: 195-mid-attn-rmsnorm (mid-attention, attention-side), 197-output-residual-sqrt-2l (init-time, residual-side), 198-z-loss-on-residual (training-time, residual-side). They are mechanistically distinct:

- **195** modifies the *attention flow*: per-query RMSNorm on pre-softmax scores. Changes the *forward graph* of attention. If 195 wins, the binding lever is *attention-score magnitude control* (softmax input distribution shape), not residual magnitude. 195 is a *placement* lever (post-product QK).
- **197** modifies the *init-time residual scale*: every block's attention and FFN output is multiplied by `α = 1/√(2L) = 0.204` before the residual addition. Changes the *forward graph* at step 0 (not bit-identical). If 197 wins, the binding lever is *init-time scaling of the residual contribution*, not training-time regularization. 197 is an *init-time* lever.
- **198** modifies the *loss function only*: adds a `log(1+||r||²) · z_coef` penalty to the loss. Does NOT change the forward graph (the residual stream is the same shape, the same magnitudes at step 0). If 198 wins, the binding lever is *training-time regularization on the residual stream's L2 norm*, not the init or the attention flow. 198 is a *training-time* lever (loss-side).

**The three partition the residual/attention magnitude axis cleanly:**
- 195 = attention-side, post-product QK
- 197 = residual-side, init-time scale
- 198 = residual-side, training-time loss penalty

A win on 198 *and* a null on 197 means the binding axis is *training-time loss pressure on residual magnitude*, not the init-time form. A win on 197 *and* a null on 198 means the binding axis is *init-time residual scaling* (the optimizer can re-discover the un-scaled form if the loss penalty is too weak). A null on all three is the cleanest possible closure of the residual/attention magnitude axis at 0.94M. A win on two of three would be informative but unexpected — it would suggest the axis has multiple binding sub-levers at this tier.

**Portfolio fit rationale:** the three are *not* redundant. 197 and 198 are both residual-side but at different *time scales* (init vs training) — the *only* way to distinguish them empirically is to run both. 195 is attention-side and orthogonal to both. The queue can hold all three; the *priority* order (198 vs 197 vs 195) is set by the taste gate's reading of the binding regime. The repitch here commits 198 to a falsifiable prediction; the taste gate's accept verdict on 198 will inform whether 197 should be de-prioritized in favor of closing the residual-side axis cheaply with 198 first.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule. (Re-run `mean(||r||_L2) ≥ 5·√d_model = 40` at step 2k; if residual norm is in the binding regime, the WIN is real. If residual norm is in the indeterminate regime, the WIN is suggestive but not falsified.)
- **NULL**: `|trt_val − ctrl_val| < 0.01`. (Sub-classify by falsification signature: NULL with `||r||_L2 ≤ 24` is a *symmetric closure* of the magnitude axis. NULL with `||r||_L2 ≥ 40` is a *steered null* — the lever is in regime but the optimizer ignores the gradient, which would warrant a follow-up at `z_coef=1e-3`.)
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 167-logit-zloss (null) — *logit-side* z-loss (penalty on log-partition). 198 is *residual-side* z-loss (penalty on per-token L2 norm). Complementary axes. 198's binding condition is the falsification signature in the §"Falsification signature" section above.
- 110-weight-ema, 122-tiger, 124-radam, 134-mega-ema (null, tier-mismatch) — optimizer-side changes. 198 is loss-side.
- 111-drop-path (null) — drop-path regularizer on the residual. 198 is L2-norm penalty. Different regularizer.
- 119-sam, 138-looksam (null) — flat-minima optimizers. 198 is magnitude regularizer.
- 017-sub-ln-sandwich, 130-rezero, 142-layerscale (null) — per-block *learned* depth-conditional levers. 198 is *training-time L2 penalty on the final residual*, not a per-block scalar or norm. Different lever on a related axis.
- 196-block-residual-ema (taste-rejected today) — cross-block EMA with detach + learned scalar gate. 198 is *no learnable params*, no detach, no gate — a pure L2 penalty. The closed-loop story (196 reject + 167 null) is *not* a closure of 198's axis because 198 is mechanistically distinct (loss-side, not optimizer-side; static penalty, not EMA-mixed).
- 195-mid-attn-rmsnorm (active sibling) — *attention-side* placement, not residual-side. See §"Differentiation from active siblings."
- 197-output-residual-sqrt-2l (active sibling) — *init-time* residual scale, not training-time loss penalty. See §"Differentiation from active siblings."
