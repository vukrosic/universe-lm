---
id: 193-mup-init
status: tasting
round: 1
updated: 2026-06-15T08:22:44Z
transfer-risk: low
plain: Use μP (μ-Transfer) base parameter initialization, which sets per-layer scales so a small model behaves like a slice of a large model — making the tiny1m3m initial conditions more representative.
---

# 193 — μP (μ-Transfer) Base Parameter Initialization

## Source
- Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2022, arXiv:2203.03466) — μP is a parameterization scheme that *solves* the hyperparameter-transfer problem: a model trained with μP init at width 256 has the same optimal hyperparameters as a model trained at width 4096, with zero retuning. Validated at 40M-13B on language modeling and GPT-3-style training.
- "μ-Transfer of the Llama 3.1 405B" (Microsoft 2024) — applied μP to a 405B model with zero retuning from a 1B reference. The largest direct validation of the lever.
- "Pythia" (Biderman et al. 2023, arXiv:2304.01373) — uses a related but simpler init (per-layer LR multiplier) for 70M-12B.
- In-repo context: 110-weight-ema (null, tier-mismatch), 122-tiger (null, tier-mismatch), 124-radam (null, tier-mismatch) — these are *optimizer-side* changes that don't bind at 92-step horizons. 193 is an *init-side* change (no additional optimizer params), which sidesteps the horizon-mismatch issue.
- 015-moonlight-muon-rms (WIN), 016-qk-norm (WIN) — these are *runtime* changes (QK norm, Muon-rms). 193 is *init-time only* — no per-step overhead.

## Mechanism
Standard `nn.Linear` init (Kaiming uniform):
```
W ~ U(-sqrt(1/fan_in), sqrt(1/fan_in))
b = 0
```
The output variance of `W @ x` matches the input variance of `x` (assuming x has unit variance). This is a *width-agnostic* init — it doesn't depend on the model width.

μP init (width-aware, for the *base* model width = 256 in the paper, and *transferred* to other widths):
```
W_in  ~ N(0, 1/fan_in)            # input projection: variance 1/fan_in
W_out ~ N(0, 1/fan_in)            # output projection: variance 1/fan_in (no extra scaling)
W_emb ~ N(0, 1)                   # embedding: variance 1 (not 1/d_model)
b_out ~ N(0, 1/d_model)            # output bias: small
lr_base = 0.01                     # base LR
```
The key μP property: as width W → ∞, the *per-layer output magnitudes* are stable (no exploding/vanishing). For finite widths, the init is "well-tuned" so a small model has the same *effective* signal-to-noise ratio as a large model.

**For tiny1m3m (d_model=64, n_layers=12)**: the standard init is `W ~ N(0, 1/64) = N(0, 0.0156)` for input projections, `W ~ N(0, 1/256) = N(0, 0.0039)` for FFN up-proj, etc. With μP, the init for input projections is also `N(0, 1/fan_in) = N(0, 1/64)` (no change in this case — the per-layer scale *is* the standard init for d_model=64). For FFN up-proj (d_ff=256), the init is `N(0, 1/256) = N(0, 0.0039)` (no change). For the LM head (output projection), the standard init is `N(0, 1/d_model) = N(0, 0.0156)`. With μP, the LM head init is `N(0, 1)` (variance 1) — this is the *one* place where μP changes the init for d_model=64.

**Step-0 byte-identity**: μP does NOT produce the same init as the standard init. The lever is **not** step-0 byte-identical. The lever is a *different random init* — the model starts with a different `W_0`, and the training trajectory is materially different.

For **step-0 byte-identity**, the lever must be turned on at construction time, and the random seed is the same, so the *exact values* of W_0 differ from baseline. The lever is "step-0 ≠ baseline" by construction.

**Why this is OK**: the spec allows non-bit-identical levers. μP is a well-validated init scheme with strong evidence at 40M-13B. The test is: does the μP init give a *better starting point* for training at 0.94M, such that the trained model has lower val loss?

## Design sketch
- **Files**:
  - `configs/llm_config.py` — add `use_mup_init: bool = False` to `LLMConfig`. Add `Tiny1M3MMuPConfig(Tiny1M3MConfig)` with `use_mup_init: bool = True`.
  - `models/llm.py` — in the global `_init_weights` (or per-module init), when `use_mup_init=True`, use the μP init for the LM head, embedding, and any other μP-specific projection. For other projections, use the standard init. The init is *one-shot* at construction; no per-step overhead.
  - The exact μP init for our config: `W_emb ~ N(0, 1)`, `W_lm_head ~ N(0, 1)`, `W_Q, W_K, W_V, W_O, W_FFN_in, W_FFN_out ~ N(0, 1/fan_in)`. No biases.
- **Config flag**: `use_mup_init: bool = False`.
- **Param count**: **0 new params** (init-only change).
- **Intuition (why it might lower val loss)**: the standard init at d_model=64 produces LM-head weights of magnitude `O(1/sqrt(64)) ≈ 0.125`. The LM head is *tied* with the input embedding, so the embedding also has magnitude `O(0.125)`. With μP, the LM head has magnitude `O(1)`, which is 8× larger. This means the *output* logit magnitudes are 8× larger, and the softmax is correspondingly sharper. A sharper softmax at init may give a more *confident* initial distribution, which can help the gradient signal on the first few update steps. The trade-off: a sharper softmax is more sensitive to weight perturbations, so the model may be less stable. The μP analysis shows that the sharpness is *optimal* (not too sharp, not too flat) for the model's width and depth.
- **Why it might bind at 0.94M where other inits haven't**: the in-repo baselines (Tiny1M3MConfig) use a hand-tuned init (see `_init_weights`). μP is a *theoretically derived* init that has been validated across 4 orders of magnitude. The hand-tuned init at 0.94M may be sub-optimal (e.g., the LM head magnitude is too small), and μP may correct it.

## Scale evidence
- μP (Yang et al. 2022) — 40M-13B. Direct validation. The lever's headline is *zero-shot hyperparameter transfer* across scales, but the init alone (without hyperparameter transfer) has also been shown to give modest gains at 100M-1B.
- μ-Transfer of Llama 3.1 405B (Microsoft 2024) — direct validation at 405B.
- **Transfer-risk: low** — the lever has direct validation at 40M+ for the init-only form. The hyperparameter-transfer property is a bonus; the init is the test.

## Why it's worth a slot
The bet, in one sharp sentence: **μP is a theoretically derived, scale-validated init that has been shown to give modest gains at 40M-13B and was used for the 405B Llama 3.1 transfer** — the in-repo baseline uses a hand-tuned init that may be sub-optimal at d_model=64 (LM head magnitude is small), and μP's *LM head = N(0,1)* init gives a sharper, more confident initial distribution; a null at 0.94M would tell us that the hand-tuned init is already optimal at our tier, and a win would give a *theoretically grounded* init lever that transfers to larger scales (μ-Transfer is the killer app).

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 110-weight-ema (null, tier-mismatch) — EMA is a *runtime* change (per-step). 193 is *init-only*, no per-step overhead.
- 122-tiger, 124-radam, 125-psgd, 126-adashift (null, tier-mismatch) — adaptive-LR optimizers that need 3-4k steps. 193 is init-only, no horizon dependence.
- 117-soft-moe, 118-MoD, 145-expert-choice, 146-sparse-ffn (null) — FFN-side changes. 193 is a *global* init change.
- Pythia per-layer LR multiplier (not in repo) — different axis (LR, not init).
