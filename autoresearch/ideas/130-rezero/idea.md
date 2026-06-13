---
id: 130-rezero
status: done
round: 1
updated: 2026-06-13T15:01:39Z
transfer-risk: high
plain: It puts a small learnable multiplier on each residual branch, starting at zero — so the network starts as the identity function and learns how much each layer should add as training proceeds.
---

# 130 — ReZero: Residual Scaling for Better Initialization

## Source
Bachlechner, Majumder, Mao, Cottrell, McAuley, "ReZero is All You
Need: Fast Convergence at Large Depth" (arXiv:2003.04887, ICML
2020 workshop / UAI 2021). https://arxiv.org/abs/2003.04887

Validated on a 100-layer Transformer (T2T-style), 12-layer GPT
from scratch, and language modeling at small scale (WikiText-103,
GPT-2 small 125M). The lever is the simplest residual-connection
modification — a learnable scalar per residual branch, initialized
to zero.

## Mechanism
Standard residual: `x_{l+1} = x_l + Block(x_l)`. ReZero adds a
learnable scaling `α_l` per layer:
  `x_{l+1} = x_l + α_l · Block(x_l)`

Where `α_l` is initialized to `0` (or `0.01` in some variants).
At init, `x_{l+1} = x_l` for all layers, so the entire network
is the **identity function** at step 0. As training proceeds,
`α_l` grows from 0 toward its optimal value.

The intuition: in a deep network with L layers, the residual
contribution of each layer accumulates. At init, the random
Block outputs are uncorrelated noise; stacking L layers of
random noise corrupts the input. ReZero starts at the identity
(no noise), then *learns* how much each layer should contribute.
This avoids the "deep network starts as noise" pathology.

**Identity at step 0**: with `α_l = 0`, the entire network is
`x_L = x_0` (the input embedding). The model is bit-identical to
"no model" at step 0 — **bit-identical to baseline only if the
baseline also starts at identity, which it doesn't**. The lever
is *not* bit-identical to baseline at step 0; the deviation is
`O(α)` per layer, and after `α_l` grows to `~1`, the model is
approximately equivalent to a standard residual network (but
with possibly better init dynamics).

## Design sketch
- `models/layers.py` (modified): in the standard block's forward
  pass, replace `x = x + sublayer(x)` with `x = x + α · sublayer(x)`
  where `α` is a learnable scalar per block. ~15 LoC.
- `models/llm.py`: when `use_rezero=True`, wrap each block with a
  learnable α. The `nn.ParameterList` of α's is initialized to 0
  and trained jointly. ~10 LoC.
- `configs/llm_config.py`: add `use_rezero: bool = False`,
  `rezero_init: float = 0.0` (initial α), `rezero_lr_mul: float = 1.0`
  (LR multiplier for α, optional). ~10 LoC.
- LoC: ~35 total (under 200 ceiling).
- Identity at step 0: with `α_l = 0`, the model is the identity
  function at step 0. This is **not** bit-identical to baseline
  (which has standard residual scaling `1.0`). The lever has
  inherent step-0 cost: the model has to *grow* α from 0 to its
  optimal value, which takes a few steps.
- The intuition: at 0.94M with 12L, the deep-network initialization
  pathology is mild (12 layers is not "very deep"), but ReZero's
  residual scaling can still help by giving each layer a
  layer-specific magnitude. A null would say "at 12L the standard
  residual is fine and ReZero's slow ramp-up costs more than it
  saves"; a win would say "the layer-specific α gives a better
  per-layer residual contribution".

## Scale evidence
- arXiv:2003.04887 (Bachlechner et al. 2020): validated on a
  100-layer Transformer (T2T) for image classification (CIFAR-10),
  and on a 12-layer GPT-2 small (125M) for language modeling on
  WikiText-103. Reports 50%+ training-time speedup at 100-layer
  scale, modest gains at 12-layer scale.
- Transfer risk: **high**. The paper's headline gains are at
  100-layer depth, not 12-layer. At 12L the residual noise is
  not catastrophic and the α ramp-up is wasted. The 0.94M
  context is *least* favorable to ReZero (shallow depth is
  exactly where the lever doesn't fire).

## Why it's worth a slot
ReZero is the simplest residual-scaling lever and is distinct
from SubLN-sandwich (017 closed null) which normalizes the
post-residual. ReZero scales the *pre-residual contribution*,
not the post-residual. The lever is also distinct from
multi-residual (closed axes) and DropPath (111 closed drift).
A win would say "layer-specific residual scaling helps even
at 12L"; a null would confirm the paper's "ReZero wins only
at extreme depth" claim. Either outcome is informative — the
high transfer-risk is honest about the depth mismatch.

## Plan

### Files to change

- `configs/llm_config.py` (modified): add `use_re_zero: bool = False`
  field on `LLMConfig` (inherited by `Tiny1M3MConfig`). ~12 LoC of
  docstring + 1 field.
- `models/layers.py` (already wired): `TransformerBlock.__init__`
  already accepts `use_re_zero` and builds
  `self.re_zero_alpha_attn = nn.Parameter(torch.zeros(1))` +
  `self.re_zero_alpha_ffn = nn.Parameter(torch.zeros(1))`. The
  pre-norm forward already branches on `self.use_re_zero` and
  computes `x = x + self.re_zero_alpha_attn * self.dropout(attn_out)`
  (and the FFN equivalent). Zero-init ⇒ step-0 == baseline (the
  gate contributes 0·f = 0). Off by default → the
  `elif self.resid_mode: ... else: x = x + self.dropout(...)` else-
  branch keeps the baseline path bit-identical.
- `models/llm.py` (already wired): `MinimalLLM.__init__` already
  passes `use_re_zero=getattr(config, "use_re_zero", False)` into
  every `TransformerBlock` constructor. No new code required —
  just the new config flag activates the existing branch.

### Config flag

`use_re_zero: bool = False`. Inherits to `Tiny1M3MConfig` for free.
The plan-sketch proposed `rezero_init: float = 0.0` and
`rezero_lr_mul: float = 1.0` knobs but those aren't needed: the
two scalars are zero-init by `nn.Parameter(torch.zeros(1))` and the
optimizer routes 1-D parameters to AdamW (existing convention),
so the lever fires with one bool. Saves LoC.

### Zero-init at step 0

With `use_re_zero=False` (default), the new branch is not taken
and the model is bit-identical to baseline at step 0. With
`use_re_zero=True`, the new branch is taken but `α=0` ⇒
`α·f(x) = 0` exactly (within fp32), so the residual add becomes
`x = x + 0 = x` and the block is the identity. The model becomes
the *identity function at step 0* — bit-identical to the baseline
forward graph (one extra multiply by 0 + add, but the result is
the baseline value). Note: this is *not* the same as
"byte-identical to baseline" because the baseline has
`x = x + f(x)` (no scaling), while ReZero has `x = x + 0·f(x)`.
The numerical difference is `f(x)` (a vanishing small perturbation
of the residual stream relative to `x`), so the step-0 output
drifts from baseline by ~O(‖f(x)‖·‖x‖^-1).

### Run command

The A/B launcher is `_arq_130-rezero.py` (mirrors `_arq_111-drop-path.py`):

```
cd /root/universe-lm && /venv/main/bin/python _arq_130-rezero.py
```

It subclasses `Tiny1M3MReZeroConfig` (which sets `use_re_zero=True`)
and shells `train_llm.main()` with `--config_class __main__.C
--seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
Baseline ctrl is `_arq_111-drop-path.py` pattern but for the
un-flagged config: use `Tiny1M3MConfig` (no `use_re_zero`).

Per `prompts/runner.md` the runner uses `config_class` (a config
subclass) rather than CLI flags — so we follow the existing
pattern with a dedicated A/B launcher script that lives at the
repo root and a corresponding `Tiny1M3MReZeroConfig` subclass
in `configs/llm_config.py`.

### Reading the final val loss

The script writes a JSON metrics blob to
`autoresearch/runs/<run_id>/metrics.json` with the final val loss
as `val_loss` (and the wallclock `step` it occurred at). Read the
last `val_loss` line — that's the A/B target. The two-ctrl
bracket (per `feedback-one-seed-only`) gives the variance band;
two `Tiny1M3MConfig` runs without `--use_re_zero` should land
within ±0.01 of each other.
