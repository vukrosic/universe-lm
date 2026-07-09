---
id: 038-swan
status: needs-run
round: 1
updated: 2026-06-11T01:51:33Z
transfer-risk: low
---

# 038 - SWAN

## Source
SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training (arXiv:2412.13148, 2024). Ma, Gong, Scetbon, Meeds — Dec 2024 / Feb 2025 v3. Reports LLaMA 350M / 1.3B pretraining and ~2x token-to-perplexity vs Adam. Citation and authors verified.

## Mechanism
Pre-process gradients with normalization and whitening before the SGD step, so the optimizer needs no first- or second-moment state. In this repo the practical test is to apply the transform to matrix gradients and keep scalar/norm parameters on the normal path.

## Scale evidence
The paper reports LLaMA pretraining at 350M and 1.3B parameters and about a 2x speedup to the same perplexity with half as many tokens as Adam. transfer-risk: low - the source is already at transformer scale and the mechanism is directly about LLM training.

## Why it's worth a slot
If SWAN works here, it says the model benefits more from gradient geometry normalization than from classic adaptive optimizer memory, which is a strong signal for the larger ladder.

## Control
**The meaningful A/B is SWAN-on-the-Muon-slot vs Muon-on-the-Muon-slot** (1-D / norm / embedding stay on AdamW in both arms). The current baseline in `training/trainer.py:109-176` routes `param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name` to Muon and everything else to AdamW. The SWAN swap touches only that 2-D non-embedding, non-norm slot; the control is today's Muon-AdamW routing with all current HPs unchanged. The delta then isolates "whitening replaces Newton-Schulz", not "whitening replaces Adam-style adaptive memory". This mirrors the wording used in 011-cautious-lion (Lion replaces Muon on the same 2-D slot).

## Non-matrix routing (pinned, no improvisation)
The "normal path" the SWAN swap leaves untouched is exactly the existing Muon-AdamW split in `training/trainer.py:109-176`:
- **1-D scalars** (biases, gains, etc.) → AdamW.
- **`*.norm.weight`** (LayerNorm/RMSNorm gains) → AdamW.
- **`token_embedding` / `emb_proj`** → AdamW.
- All other **2-D matrix params** (attention Q/K/V, `out_proj`, FFN W1/W2) → SWAN in the treatment arm, Muon in the ctrl arm.

No `muon_for_1d_norm` / `muon_for_embed` / `muon_for_output` flags flipped (those would change the ctrl).

## Hyperparameters
- **LR / momentum:** inherit Muon's `muon_lr=0.024, muon_momentum=0.95` from `configs/llm_config.py:465-466`. SWAN's update is taken on the *whitened* gradient with the same scalar LR — no separate LR slot is introduced. This pins the A/B to "same step budget, different preprocessing"; a paper-default port would conflate preprocessing and LR.
- **Gradient clipping:** unchanged from baseline (`grad_clip=1.0` by default in `configs/llm_config.py`).
- **No first/second-moment buffers** in the treatment arm — that is the point of the lever. Step-0 is therefore *not* bit-identical to ctrl even at HPs-match, because ctrl carries Muon's running momentum buffer; see Identity case.

## Pass bar
Box noise at tiny1m3m is ~±0.01 val loss (ctrl gap from `closed.md` WIN rows is 0.0047–0.0175). `Δ := trt_val − ctrl_val`. **Pass iff `Δ ≤ −0.01` vs Muon-on-matrix ctrl at fixed step count (seed 42, tiny1m3m).** Sub-noise (|Δ| < 0.005) → log null and close. The paper's "2x speedup" is iso-perplexity at variable tokens; this pipeline runs fixed steps, so the bar is Δ val at fixed compute.

## Identity case
Setting the whitening transform to the identity matrix reduces SWAN to ordinary SGD-with-momentum on matrix params (it does **not** reduce to the current Muon baseline, because Muon carries a Newton-Schulz orthogonalization and a separate momentum buffer). Therefore a "step-0 bit-identical to ctrl" check does not apply here — compare end-of-training only. A useful diagnostic on the runner side: log the spectral norm of `G_whitened` vs `G_raw` over the first 100 steps; if the ratio is ~1 the whitening is doing nothing and the run is effectively bare SGD (and should null per the pass bar, not win).

## Hypothesis
Δ in [−0.01, −0.03] val loss on tiny1m3m / seed 42 vs the Muon-on-matrix ctrl. Mechanism: whitening the per-step gradient removes the need for Muon's running momentum orthogonalization while preserving update direction quality — a stateless recipe-shape win. Null is also informative: tells us Muon's specific Newton-Schulz route matters and SWAN's batched whitening loses to it at this scale.

## Wiring (what the plan must carry)
- `optimizers/swan.py` — new `SWAN` class: per-parameter whitening on matrix grads, no state. ~60–80 LoC.
- `LLMConfig.use_swan: bool = False` (default off → bit-identical to today's Muon-AdamW baseline).
- `training/trainer.py:_setup_optimizers` — add a branch mirroring the Lion path at `trainer.py:170-173`: when `config.use_swan` is True, route 2-D non-embedding, non-norm params to `SWAN` and 1-D / norm / embedding / head to `AdamW`. The routing condition is the same as Muon's — no flag changes on the AdamW side.
- No new HP fields (`muon_lr=0.024`, `muon_momentum=0.95` are reused for SWAN's step).
- ctrl = `use_swan=False`; trt = `use_swan=True`. Same seed 42, same step count, same data order.

## Reviser note (r2)
Reviewer's findings applied in full:
- Pinned the control to SWAN-on-Muon-slot vs Muon-on-Muon-slot (`## Control`).
- Added a numeric `## Pass bar` (`Δ ≤ −0.01`; sub-noise < 0.005 → null).
- Specified LR/momentum inheritance from `configs/llm_config.py:465-466` and grad-clip unchanged.
- Spelled out the non-matrix routing in `## Non-matrix routing (pinned, no improvisation)`.
- Added `## Identity case` and the spectral-norm diagnostic.

No disagreements with the reviewer.
