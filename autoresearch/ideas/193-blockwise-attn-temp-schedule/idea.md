---
id: 193-blockwise-attn-temp-schedule
status: needs-run
round: 2
updated: 2026-06-15T16:47:19Z
transfer-risk: low
plain: Make the attention "sharpness" vary smoothly with depth using a fixed cosine schedule (no learned parameters), starting with a flat schedule so step-0 matches the baseline — a depth-aware attention prior without any trainable knobs.
---

# 193 — Blockwise Attention Temperature Schedule (Cosine-Depth Soft Schedule, No Learnable Params)

## Source
- 155-per-head-temp (closed null Δ=−0.0063 inside band) — per-head learnable temperature scalar; learned but per-head axis closed at 0.94M.
- 175-alibi-slopes (in-repo WIN Δ=−0.1585) — fixed per-head ALiBi slopes; depth-uniform bias that decayed with distance. Different shape (additive, not scale).
- 188-qk-rms-scaling (in-repo, planning) — per-block learnable scalar on post-norm QK^T (init 1, exp-parameterized). 188 is the *learnable* counterpart on the same axis. 193 is conditional on 188 (see "Why it's worth a slot").
- Press et al., "ALiBi" (arXiv:2108.12409, 2022) — fixed-position bias, depth-uniform; 193 is a depth-varying scale (analogous to ALiBi but on the multiplicative side).
- Su et al., "RoPE" (arXiv:2104.09864) — fixed-position rotation; 193 is fixed-depth scale.
- Roy et al., "Efficient Content-Based Sparse Attention" (2019/2020) — blockwise-local attention patterns; 193 is a continuous blockwise scale, not a hard mask.
- Possible related lever: cascaded / curriculum attention — early layers attend softly (broad context), late layers attend sharply (specific context). Cited in some "annealed attention" ablations (no canonical paper found).

## Mechanism
Standard attention: `scores = QK^T / √d_k` (constant temperature 1/√d_k across all blocks).

Blockwise temperature schedule: each block b has a multiplicative temperature `τ_b` on its scores:
```
scores_b = Q_b K_b^T / (τ_b · √d_k)
attn_b = softmax(scores_b)
```

**Sign convention (resolved in r2).** Standard usage: `τ < 1 ⇒ scores are amplified ⇒ softmax is **sharper**`, and `τ > 1 ⇒ scores are shrunk ⇒ softmax is **softer**`. (r1 flipped this — fixed below.)

**Schedule (committed, single sign):** `τ_b = 1 + α · cos(π · b / L)` with `b ∈ [0, L-1]`, `L = n_layers` (12 at tiny1m3m). At α=0, all `τ_b = 1` (bit-identical baseline). We commit to **α = −0.3** (a single scalar value for the A/B — r1's "explore both signs" is closed):

- **b=0 (early)**: `τ_0 = 1 + (−0.3) · cos(0) = 0.7` → scores amplified 1.43× → **sharper softmax** (early layers attend locally).
- **b=L-1=11 (late)**: `τ_11 = 1 + (−0.3) · cos(11π/12) ≈ 1 + 0.29 = 1.29` → scores shrunk 0.78× → **softer softmax** (late layers integrate context).

So α = −0.3 = "sharpen early, soften late" — the published cascaded-attention / curriculum-attention prior: early layers pick out local token-identity patterns, late layers mix broad context.

**Mechanistic argument for α < 0** (not α > 0): the 175-alibi WIN (Δ=−0.1585) shows that *locality bias helps at tiny1m3m*. ALiBi is additive and per-head; 193's scale-side depth-schedule is the multiplicative analog, and the sharpen-early sign is the one consistent with "early layers do the local pattern-matching that ALiBi rewarded."

**Bit-identity at step 0**: α=0 ⇒ `τ_b = 1` for all b ⇒ scores / (1 · √d_k) = standard path exactly. The A/B is α=0 (ctrl) vs α=−0.3 (trt), and the 16 (or 24, pending) ctrl bracket handles the noise floor.

## Design sketch
- **File**: `models/layers.py` — modify the manual attention path to apply `scores / (τ_b · √d_k)`. Add `or self.use_block_temp_schedule` to the manual-path forcing list (so SDPA flash doesn't perturb step-0 numerics at α=0).
- **Config flags** (added to `LLMConfig` and a new `Tiny1M3MBlockTempConfig`):
  - `use_block_temp_schedule: bool = False` (off by default; baseline path bit-identical).
  - `block_temp_alpha: float = 0.0` (default = flat schedule; trt = −0.3).
- **No params**: the schedule is hard-coded; no per-block scalar learned (this is the deliberate contrast to 188).
- **Schedule shape**: `τ_b = 1 + α · cos(π · b / L)`, computed once at forward time from `b` and `L` (or cached as a `Buffer` of shape `[L]`).
- **Bit-identical at step 0**: α=0 ⇒ `τ_b = 1` for all b ⇒ `scores / (1 · √d_k) = scores / √d_k` exactly.

## Scale evidence
ALiBi validated at 0.4B–6.7B (Press et al. 2022); cascaded-attention literature exists but is mostly empirical. Transfer-risk: low (fixed-function lever; the schedule shape is a single HP, init at 0 is the identity, and the multi-paper literature on depth-varying temperature supports a low-risk tag).

## Why it's worth a slot
**Conditional follow-up to 188-qk-rms-scaling.** 188 is the *learnable* per-block scalar on the same axis; 193 is the *fixed-shape* per-block schedule. They sit at the same depth-conditional multiplicative axis and the information content of 193 is conditioned on 188's result:

- **If 188 NULLs** (per-block learnable scalar doesn't bind in 92 steps): 193 tests whether the *fixed cosine shape* is the right prior — i.e. whether the issue with 188 is "the optimizer can't find the schedule" (193 should win) or "the depth-conditional scale axis is hostile at 0.94M" (193 should also null, closing the axis decisively).
- **If 188 WINs** (the optimizer finds a useful per-block schedule in 92 steps): 193 is **subsumed** — 188 already gives the depth-conditional scale. The 193 GPU slot can be redirected to a different axis; the conditional run is still informative as a "fixed shape vs learned shape" control.
- **In all cases**, 193 is informative: null closes the axis, win opens a new fixed-prior lever, conditional-on-188 reframes the result cleanly.

**Attribution insight (preserved from r1)**: 175-ALiBi WIN (Δ=−0.1585, the largest in-repo WIN at tiny1m3m) is the anchor — depth-uniform additive bias binds. 188 tests whether depth-varying learnable multiplicative binds. 193 tests whether depth-varying fixed multiplicative binds. Together they tile the {additive × multiplicative} × {uniform × varying} × {fixed × learned} plane, and 193 specifically isolates the *fixed shape* axis.

**Predicted Δval at tiny1m3m, seed 42, α=−0.3**: the 175 WIN gives us a 0.1585 reference for a depth-uniform *additive* lever. 193 is depth-varying *multiplicative* on a fixed shape, with a smaller amplitude (τ ∈ [0.7, 1.29] vs ALiBi's m-head slopes). Expected band: Δval ∈ [−0.01, −0.05] if the multiplicative axis binds as the additive one did, |Δval| < 0.01 (NULL) if 188 is already capturing the axis, Δval > +0.01 (DRIFT) if sharpening the early layer past the optimum breaks the 175-style locality reward.

## Pass/fail bar at tiny1m3m (seed 42)
- **Baseline (this run's ctrl)**: plain `Tiny1M3MConfig` (val 6.4216, or 6.4044/6.4091 for the bracket). Bit-identical to trt at α=0.
- **WIN**: `trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl rule. The 175 reference is 6.4216 − 0.1585 = 6.2631, so a 0.01 bar is conservative.
- **NULL**: `|trt_val − ctrl_val| < 0.01` — closes the fixed-shape axis (and, conditional on 188 NULL, the whole per-block scale axis).
- **DRIFT**: `trt_val > ctrl_val + 0.01` — sharpening the early layer past the optimum breaks locality.
- **Conditional stop**: if 188-qk-rms-scaling reports a WIN ≥−0.005 before 193 runs, redirect 193 to a different axis; the run is informative as a fixed-shape control only if 188 has not already captured the depth-conditional scale.
