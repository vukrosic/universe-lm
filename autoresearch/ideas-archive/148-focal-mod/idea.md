---
id: 148-focal-mod
status: done
round: 1
updated: 2026-06-13T20:32:25Z
transfer-risk: high
plain: Replace the attention block with a focal modulation block, which aggregates context hierarchically (local to global) without softmax attention.
---

# 148 — Focal Modulation Networks

## Source
Yang, Li, et al. 2022, "Focal Modulation Networks", Microsoft Research, NeurIPS 2022, arXiv:2203.11926. https://arxiv.org/abs/2203.11926

## Mechanism
Replaces the attention block with a three-stage focal modulation:
1. **Hierarchical Context Aggregation**: stack of depthwise convs at multiple scales (e.g. kernel sizes 3, 5, 7) producing a multi-scale context feature.
2. **Gather**: linear projection of the context to the modulation space.
3. **Modulate**: gated linear projection — `output = x + γ(h) * (W_q x ⊙ context_aggregated)`, where `γ(h)` is a sigmoid gate computed from the input.

The key difference from attention: the interaction between query and context is *additive-modulated* (after a linear projection), not *softmax-weighted*. No QKᵀ dot product, no softmax, no O(T²) memory.

## Design sketch (how it works + how to build it)
- Add a `FocalModulationBlock` to `models/layers.py`:
  - Hierarchical context: `nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model)` (and 5, 7) — init as identity per kernel. ~50 LoC.
  - Gather: `nn.Linear(d_model, d_model)` — small init. ~10 LoC.
  - Modulation: `nn.Linear(d_model, d_model)` for context projection, `nn.Linear(d_model, d_model)` for query, `nn.Linear(d_model, d_model)` for gate (sigmoid). ~30 LoC.
- Modify Block class: when `use_focal_mod`, replace the attention sub-block with `FocalModulationBlock`. ~40 LoC integration.
- Add `use_focal_mod: bool = False` to `configs/llm_config.py`.
- Identity at step 0: gate init to bias such that `sigmoid(0) = 0.5`. The modulation contributes `0.5 * (W_q x ⊙ context)`. To get *strict* identity at step 0, init the gate projection's bias to `-inf` (so sigmoid → 0) or to a large negative value. Then `output = x + 0 * (anything) = x` (modulation is no-op). At training start, the gate learns to "turn on" the modulation.
- Why a real lever, not a hyperparam: focal modulation is a *fundamentally different* mechanism from softmax attention. It uses convolutions to aggregate context and a gate to modulate query — no softmax, no quadratic memory, no QKᵀ. Different inductive bias (additive modulation vs multiplicative softmax).
- Targets baseline failure: softmax attention has known issues — entropy collapse, attention sinks, quadratic memory. Focal modulation sidesteps all three. At 0.94M, the memory issue is invisible, but the entropy/sink issues may bite.

## Scale evidence
Paper trains FocalNets on ImageNet classification, COCO detection, ADE20K segmentation — not language modeling. Independent replications on LMs are limited. Transfer risk: high — focal modulation is unproven for LMs, and the mechanism's benefits (long-context efficiency) are invisible at 0.94M (T=512 or 1024 is well within attention's O(T²) budget).

## Why it's worth a slot
High-risk, high-reward. The closed "logit softcap / SSMax" axis (020–025) shows we can win by changing the *softmax* structure. Focal modulation goes one step further and removes softmax entirely. A win would be a paradigm shift — softmax attention is replaceable at 0.94M; a null would close the "non-softmax attention" axis and confirm softmax attention is the right inductive bias for tiny LMs. Filing this is cheap (one A/B); not filing it would leave a wide class of mechanisms untested.

## Plan

### Files changed
- `models/layers.py` — add `FocalModulationBlock` (~85 LoC) and wire `use_focal_mod` into `TransformerBlock.__init__`/`forward` (3 call-site branches + a constructor flag pass-through).
- `configs/llm_config.py` — add `use_focal_mod: bool = False` and `focal_mod_kernels: tuple = (3, 5, 7)` to `LLMConfig` (off by default → baseline bit-identical).
- `models/llm.py` — pass `use_focal_mod=self.use_focal_mod, focal_mod_kernels=self.focal_mod_kernels` into both `TransformerBlock(...)` instantiation sites (lines 569 and elsewhere for the YOCO branch).

### Mechanism (the FocalModulationBlock)
- **Hierarchical context aggregation**: stack of depthwise Conv1d at kernel sizes `focal_mod_kernels` (default `(3, 5, 7)`), each `groups=d_model, bias=False`, with left-padding `(k-1, 0)` along the time axis for causality. Identity-init: center tap=1, rest=0 (so at step 0 each conv is a pass-through and `context = x` plus the Kaiming-uniform-default linear `gather`).
- **Gather**: `nn.Linear(d_model, d_model, bias=True)`, **zero-init** (`W=0, b=0`) so the modulation signal is exactly `0` at step 0 regardless of context. This is the *single* parameter that controls step-0 identity — the conv init doesn't matter for parity because `gather` absorbs it.
- **Modulate**: `q_proj = nn.Linear(d_model, d_model)` (xavier init) for `W_q x`; `h_proj = nn.Linear(d_model, d_model)` (zero-init) for `W_h · context`; `gate_proj = nn.Linear(d_model, d_model)` (zero-init, bias `-10`) for `γ(h) = σ(W_g x + b_g)`. Output: `x + γ(h) * (W_q x ⊙ W_h · context)`. At step 0 with the zero-init lines above, `W_h · context = 0` exactly, so the residual update is 0 → step-0 ≡ x (bit-identical to baseline).

### Wiring
- In `TransformerBlock.__init__`: when `use_focal_mod=True`, build `self.focal_mod = FocalModulationBlock(d_model, kernels=focal_mod_kernels, dropout=dropout)`. The MHA is still built (cheap, `n_heads × d_head²`) but never called.
- In `TransformerBlock.forward`: in the 3 attention call sites (parallel-block line 2953, post-norm line 2975, pre-norm line 3004), branch on `self.use_focal_mod`. Focal branch: `attn_out = self.focal_mod(<normed_x>)`. MHA branch: existing call unchanged. This keeps the rest of the residual/FFN/sub-LN/LayerScale machinery byte-identical.

### How it stays zero-init at step 0
- `gather` weight+bias are `nn.init.zeros_` ⇒ `z = 0`.
- `h_proj` weight+bias are `nn.init.zeros_` ⇒ `h_mod = 0`.
- Therefore `g * (q * h_mod) = g * 0 = 0` regardless of `g`, `q`, or the conv outputs.
- `output = x + 0 = x` exactly at step 0.
- With `use_focal_mod=False` (default), the focal module is never built → baseline forward graph is bit-identical.

### Run command (tiny1m3m, seed 42)
```bash
/venv/main/bin/python -m autoresearch.bin.run_idea --idea 148-focal-mod
```
(runner will set `config_class=Tiny1M3MConfig`, set the seed to 42, and turn on `use_focal_mod=True`.)

### Reading the result
- `autoresearch/ideas/148-focal-mod/evidence.md` — final val_loss vs. the tiny1m3m ctrl in `autoresearch/closed.md` (baseline 6.4216). PASS ≤ −0.01 vs. ctrl, NULL band |Δ| < 0.01, DRIFT > +0.01.
- Final val loss is in the `final_val_loss` line of `log.jsonl` at the run's tail.

### Budget
~85 LoC module + ~25 LoC wiring + ~3 LoC config = ~115 LoC. Well under the 200 LoC ceiling.
