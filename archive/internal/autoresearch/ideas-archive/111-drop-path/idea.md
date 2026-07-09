---
id: 111-drop-path
status: done
round: 1
updated: 2026-06-13T10:26:26Z
transfer-risk: low
plain: It tries to randomly skip whole transformer blocks during training so the model can't lean on any single layer and has to spread the work.
---

# 111 — DropPath / Stochastic Depth

## Source
Huang et al. 2016, "Deep Networks with Stochastic Depth",
arXiv:1603.09382 https://arxiv.org/abs/1603.09382.
Validated at 110L-1202L ResNets (CIFAR/ImageNet), and at 12-24L
ViT/ConvNeXt in modern vision transformers. Open question for 0.94M
/ 12L LM pretraining.

## Mechanism
For each transformer block during training, with survival probability
`p_l` (linearly scheduled from 1.0 at the first block to `1 - p_max` at
the last, with `p_max` a small knob like 0.1), drop the block's
contribution entirely: replace the residual update `x ← x + Block(x)`
with `x ← x` (block skipped), and at inference scale all blocks by
`1 / p_l` — except the canonical implementation rescales the *kept*
blocks' outputs by `1 / p_l` (keep-probability) so the expected
magnitude is preserved without an inference-time factor.

Concrete shape: the `TransformerBlock` forward, gated by a `drop_path`
flag, samples `keep = Bernoulli(p_l)`. When `keep=1`, output is
`x + Block(x) / p_l`; when `keep=0`, output is `x`. The Bernoulli
sample is shared across the batch dim (one coin flip per block per
step) — this matches the original paper and avoids the per-token
noisiness that hurts causal LM. `p_l` is a function of layer index
`l` (1-indexed) and `n_layers`: `p_l = 1 - p_max · (l - 1) / (n_layers - 1)`.

## Design sketch
- `configs/llm_config.py`: add `use_drop_path: bool = False`,
  `drop_path_max: float = 0.1` (a 10% max drop prob is the original
  paper's default; 0.05 is the ViT/ConvNeXt modern default; 0.1
  matches the original).
- `models/layers.py` (or wherever `TransformerBlock` lives): wrap the
  block's residual add. In training (`model.training=True`):
  `keep = torch.rand(()) < p_l; out = x + (Block(x) / p_l) if keep else x`.
  In eval: `out = x + Block(x)` (no stochastic).
- LoC: ~15.
- Identity at step 0: at the first layer `p_1 = 1.0`, so `keep=1`
  always. As long as `p_max=0` (the `use_drop_path=False` default),
  the gate is short-circuited and step-0 is bit-identical. Even with
  `use_drop_path=True` and `p_max=0.1`, at step 0 the only effect is
  that the first block is never dropped (because `p_1=1.0`); deeper
  blocks may be dropped on the first forward pass — but the expected
  value of the residual magnitude is unchanged, and the random coin
  flips average to `1/p_l` rescale.
- The intuition: at 12L the model is on the boundary of "deep enough
  for stochastic depth to help" (ViT-B/12 and ViT-L/24 both used it
  with `p_max` between 0.1 and 0.4). The mechanism is a regularizer:
  it forces every block to be "droppable" and prevents the model from
  leaning on a few key blocks for the work. At 0.94M with the entire
  model fitting comfortably in a 12-block stack, the regularizer might
  or might not bite — a null would tell us the network is too
  shallow / underparameterized for skip-dropping to help, which is
  itself a useful sign for future depth-scaling ideas.

## Scale evidence
- Original paper: 110L-1202L ResNets on CIFAR/ImageNet (≥100M-equivalent
  in compute, though not parameter count).
- ViT-B/16, ViT-L/16: 12L and 24L respectively — exactly our depth
  range — with `drop_path=0.1`. That's a published, code-default
  setting in timm (the canonical vision transformer library).
- ConvNeXt (Liu et al. 2022, arXiv:2201.03545) at 18L-36L:
  `drop_path=0.1-0.4` is the default.
All in the "12-24L" depth range that brackets our 12L; transfer risk
is **low** (though the source domain is vision, not language).

## Why it's worth a slot
Tests whether a regularizer that was designed for 100L+ networks and
is a default in modern 12-24L vision transformers does anything for
12L causal LMs. The mechanism is genuinely orthogonal to every
attention/FFN/position lever in the closed list — it changes the
*training-time compute path*, not the architecture. A null at
`p_max=0.1` would say "skip-dropping is too aggressive for this
parameter count" and inform how much room there is for stochastic
regularization in future experiments.

## Plan

**Files**
- `configs/llm_config.py`: add `use_drop_path: bool = False`, `drop_path_max: float = 0.1`
- `models/layers.py`: extend `TransformerBlock.__init__` to accept `use_drop_path` / `drop_path_max` (and reuse the existing `n_layers` kwarg), store as instance attrs. In `forward`, take an optional `layer_index` kwarg. When `use_drop_path=True` and `self.training=True` and `layer_index is not None`: sample one Bernoulli coin per step `keep = torch.rand(()) < p_l` where `p_l = 1 - drop_path_max * layer_index / (n_layers - 1)` (1.0 if `n_layers==1`); if `keep=0` return `x` (skip the whole block); if `keep=1` multiply the block's residual contribution by `1/p_l` via `out = x_orig + (out - x_orig) / p_l`. Identity at step 0 because (a) the flag is off by default so the entire branch is skipped, and (b) even with flag on, `drop_path_max * 0 = 0` for `layer_index=0` gives `p_l=1.0` so the first block is never dropped and the rescale is `1.0`.
- `models/llm.py`: forward `use_drop_path` / `drop_path_max` from `LLMConfig` into `TransformerBlock(...)` for every block; pass `layer_index=i` to `block(...)` in the forward loop (the actual position, not the unique-block index, so layer tying still gets the right survival probability per depth).

**Run command**
```
/venv/main/bin/python -m training.train --config-name Tiny1M3MDropPath
```
(after we add the `Tiny1M3MDropPathConfig` subclass — for now use `LLMConfig(use_drop_path=True, drop_path_max=0.1)` passed at runtime; we'll add the config class in the same PR).

**Reading val loss**
Tail the last eval-milestone line in `runs/tiny1m3m_droppath_<ts>/metrics.json` (or whichever the runner writes) — the same place `runner.md` points the tiny1m3m screen reads from. PASS criterion: ≤ `Tiny1M3MConfig` ctrl − 0.005 (matches the "small/null" bet on a 12-block stack). NULL band |Δ| < 0.005. DRIFT > +0.005.
