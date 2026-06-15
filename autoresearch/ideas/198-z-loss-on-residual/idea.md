---
id: 198-z-loss-on-residual
status: tasting
round: 1
updated: 2026-06-15T08:33:30Z
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

## Why it's worth a slot
The bet, in one sharp sentence: **residual z-loss is a stability regularizer that targets the *residual stream's* magnitude (unbounded by the weight init), and the closed 167 logit z-loss targeted the *logit-side* magnitude (well-bounded by the `1/sqrt(d_model)` init)** — the two are *complementary* axes; a null at 0.94M would tell us that *both* magnitude axes are well-conditioned at this tier, and a win would give a stability lever that bounds the residual's growth as depth increases.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 167-logit-zloss (null) — *logit-side* z-loss (penalty on log-partition). 198 is *residual-side* z-loss (penalty on per-token L2 norm). Complementary axes.
- 110-weight-ema, 122-tiger, 124-radam, 134-mega-ema (null, tier-mismatch) — optimizer-side changes. 198 is loss-side.
- 111-drop-path (null) — drop-path regularizer on the residual. 198 is L2-norm penalty. Different regularizer.
- 119-sam, 138-looksam (null) — flat-minima optimizers. 198 is magnitude regularizer.
