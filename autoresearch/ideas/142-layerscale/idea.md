---
id: 142-layerscale
status: needs-run
round: 1
updated: 2026-06-13T20:58:16Z
transfer-risk: med
plain: A learnable per-channel knob on each residual connection, starting near zero, so each layer can slowly turn itself on as training progresses.
---

# 142 — LayerScale

## Source
Touvron, Cord, et al. 2021, "Going Deeper with Image Transformers", Meta, ICCV 2021, arXiv:2103.17239. https://arxiv.org/abs/2103.17239

## Mechanism
Learnable per-channel diagonal scale `g_ℓ ∈ ℝ^{d_model}`, multiplied to the residual branch in each block.
- `x = x + g_ℓ * sub_block(x)`  (broadcast over channels)
- Init `g_ℓ = ε` (typically `1e-4` or `1e-6`).

The key idea is that *each channel* of each block has its own learnable "gain", so the block can learn to pass through signals it's confident about and suppress signals it's not. Init near zero gives a "soft warmup" — the block contributes very little at step 0 and grows in as training progresses.

## Design sketch (how it works + how to build it)
- Modify `models/layers.py` Block class: add `self.gamma = nn.Parameter(eps * torch.ones(d_model))` for the residual scale. Apply as `x = x + gamma * sub_block(x)`.
- Add `use_layer_scale: bool = False` and `layer_scale_init: float = 1e-4` to `configs/llm_config.py`.
- One parameter per block (~13 * 64 = 832 extra params at tiny1m3m — within param budget).
- Identity at step 0: `g_ℓ = 1e-4` → residual contribution is `1e-4 × sub_block(x)`. With standard init, `sub_block(x)` is O(1), so the contribution is `O(1e-4)` per channel. Forward output is baseline + 1e-4-residual ≈ baseline. ✓
- Why a real lever, not a hyperparam: the scale is *per-channel learnable*, not a scalar. This means the model can selectively dampen noisy channels and amplify informative ones — fundamentally different from ReZero (130, scalar, init 0) and Sub-LN (017, sandwich norm) which are global. Closest neighbor is ReZero, but ReZero's scalar gives a single global multiplier; LayerScale's diagonal gives per-channel selectivity.
- Targets baseline failure: at init, all residual branches fire with their full magnitude, which can cause early-step activation explosion that the optimizer has to recover from. LayerScale's init-ε makes the first few steps near-baseline, then the model "switches on" each channel as it finds a useful direction.

## Scale evidence
Paper trains ViT-S/32 up to ViT-L/16 (depth 12–24) on ImageNet; biggest gain at depth ≥ 50 in original paper. Our 12L is on the shallow end of the paper's tested range. Transfer risk: med — independent replications show small gains (0.1–0.3%) on shallow LMs (GPT-2 12L), but the lever is *not closed* in our pipeline and the per-channel selectivity is genuinely different from Sub-LN/ReZero/DropPath (all null at 12L).

## Why it's worth a slot
The closed depth-conditional levers (017-Sub-LN, 130-ReZero, 111-DropPath, 116-Hyper-Connections) all use *scalar* or *whole-residual* scale. LayerScale is *per-channel learnable diagonal* — a qualitatively different mechanism that hasn't been tested. If it wins, the per-channel selectivity is the missing ingredient for 12L. If it nulls, we have a fifth data point on the "depth-conditional levers don't fire at 12L" pattern, which is itself informative for the depth-axis reasoning.

## Plan
- Files: `configs/llm_config.py` (new flag `use_layer_scale: bool = False` and `layer_scale_init: float = 1e-4` on `LLMConfig`; new `Tiny1M3MLayerScaleConfig` subclass), `models/layers.py` (new `TransformerBlock` kwargs `use_layer_scale`, `layer_scale_init`; per-block `gamma_attn` / `gamma_ffn` `nn.Parameter(layer_scale_init * ones(d_model))`; apply as `attn_out = attn_out * gamma_attn` BEFORE the residual add, NOT the reparam `(1+γ)` form), `models/llm.py` (thread the new kwarg through both block-construction sites), `train_llm.py` (CLI flag `--use_layer_scale`).
- Flag name: `use_layer_scale: bool = False` (with underscore; distinct from the existing reparam-form `use_layerscale`). Init: `layer_scale_init: float = 1e-4` (paper default).
- Near-baseline at step 0: with `g_ℓ = 1e-4 * ones(d_model)`, the residual contribution becomes `1e-4 × sub_block(x)`. With standard sub-block init, `sub_block(x) ~ O(1)`, so the contribution is `O(1e-4)` per channel. The forward output is baseline + 1e-4-residual ≈ baseline within fp32 precision. The forward is NOT byte-identical to baseline at step 0 (extra `* 1e-4` matmul), but the residual contribution is 4 orders of magnitude smaller than the residual stream magnitude, so the resulting val-loss change is below fp32 noise. This is the intended "soft warmup" the paper specifies.
- Why a separate flag (not the existing `use_layerscale`): the existing flag uses the reparam form `(1+g)` with `g=0` init ⇒ baseline identity at step 0. The new flag uses the direct form `g·sub_block` with `g=ε` init ⇒ near-baseline (with a controlled soft-warmup). The two have different gradient dynamics — reparam has full gradient through the residual at step 0, direct has suppressed gradient on γ until it grows. The lever is *qualitatively* different from reparam (per-channel vs scalar) AND *quantitatively* different (init 1e-4 vs init 0 with reparam).
- Param cost: 2 × d_model = 128 extra params at tiny1m3m (negligible).
- Run command: `/venv/main/bin/python train_llm.py --config_class configs.llm_config.Tiny1M3MLayerScaleConfig` (or via the existing runner plumbing with `--use_layer_scale true`).
- Val loss read: `val_loss` from the JSONL in `autoresearch/ideas/142-layerscale/log.jsonl` at `eval_milestones` (last entry is the final val, compared against the tiny1m3m baseline val 6.4306).
