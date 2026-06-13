---
id: 143-shortconv
status: running
round: 1
updated: 2026-06-13T20:57:53Z
transfer-risk: med
plain: A small depthwise 1D convolution applied to the input before attention, giving the model a cheap local-context pass before the global attention pass.
---

# 143 — ShortConv (pre-attention depthwise conv)

## Source
Poli, Massaroli, et al. 2023, "Hyena Hierarchy: Towards Larger Convolutional Language Models", Stanford, arXiv:2302.10866. https://arxiv.org/abs/2302.10866 (ShortConv variant from the Hyena family / Long Range Arena; 1D depthwise conv with kernel size 3–7 used as a pre-attention local aggregator).

## Mechanism
Insert a depthwise 1D convolution before the attention block. The conv has kernel size `k` (3 or 4), no padding, applied per-channel.
- `x = x + depthwise_conv1d(x, kernel=k, init=identity)`  (identity init: [0, 1, 0, 0, ...] for k=3)
- `x = x + attn(x)`

The conv provides cheap local context (each token sees its k neighbors) before the attention pass sees the conv-refined input. This is *sequential* (conv → attn), not the *parallel-concat* of 023-canon-conv (which concatenates conv output with attention output).

## Design sketch (how it works + how to build it)
- Add `ShortConv1D` module to `models/layers.py`: a `nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model, padding=0, bias=False)` initialized as identity (`weight[0, :, k//2] = 1.0`, rest zero). ~40 LoC.
- Modify Block class: when `use_short_conv`, apply `x = x + ShortConv1D(x.transpose(1,2)).transpose(1,2)` before attention.
- Add `use_short_conv: bool = False`, `short_conv_kernel: int = 3` to `configs/llm_config.py`.
- Identity at step 0: depthwise conv init as identity (center=1, others=0) → conv output = input. Forward is exactly the baseline forward pass. ✓
- Why a real lever, not a hyperparam: the *convolution structure* is an architectural prior (locality), not a hyperparameter. The kernel size is a hyperparam, but the *presence* of a conv with locality prior is the lever.
- Targets baseline failure: pure attention has no explicit local-bias — every token attends to every other token with equal a-priori weight. Pre-attention conv injects a local prior cheaply (no extra FFN params). The model can use the conv as a "free" first pass and the attention as the global aggregator.
- Note: 023-canon-conv is a winner (concat of conv + attn outputs). ShortConv is a *different placement* (pre-attention sequential) and has not been tested.

## Scale evidence
Hyena paper trains 1.3B Hyena models; H3 / Hyena family has been replicated at multiple scales. ShortConv in particular is the simplest variant of the Hyena family and is known to provide small gains on language modeling. Transfer risk: med — most Hyena-family wins are at >100M scale; tiny1m3m is below the validated range.

## Why it's worth a slot
023-canon-conv (winner, Δ ≈ −0.06) is a post-attention concat. ShortConv is a pre-attention sequential — the conv output is the *input* to attention, not a residual branch. This is qualitatively different and might compound with 023. A win would give us a second conv-lever to stack; a null would tell us 023's specific post-attention placement is the right one and pre-attention conv is redundant.

## Plan

### Files changed
- `models/short_conv.py` (new): `ShortConv1D` module — depthwise causal Conv1d with identity-init weights (last tap = 1, rest = 0), left-padded for causality and length preservation.
- `models/layers.py`: import `ShortConv1D`; add `use_short_conv` + `short_conv_kernel` to `TransformerBlock.__init__` signature (after `use_canon_conv`); build `self.short_conv` + `self.short_conv_gate` (per-block scalar, init 0) when `use_short_conv=True`; apply `x = x + self.short_conv_gate * self.short_conv(x)` pre-attention, pre-LN (right after the canon_conv branch).
- `models/llm.py`: read `use_short_conv` + `short_conv_kernel` from config; pass through to both the YOCO upper-half block stack and the standard `TransformerBlock` ModuleList.
- `configs/llm_config.py`: add `use_short_conv: bool = False` and `short_conv_kernel: int = 3` flags with docstring (right after `use_canon_conv`).

### Config flag
- `use_short_conv` (bool, default `False`) — when `True`, one identity-init depthwise causal Conv1d per block is applied pre-attention, pre-LN, gated by a per-block scalar init 0.
- `short_conv_kernel` (int, default `3`) — conv kernel width; pinned to 3 or 4 (asserted at construction).

### Step-0 identity
The ShortConv1D conv has identity init (last tap = 1, rest = 0), so `ShortConv1D(x) = x` at init for a causal left-padded input. The block applies `x = x + g * ShortConv1D(x)` with `g = nn.Parameter(torch.zeros(1))` — the gate is 0 at init, so the conv contribution is exactly 0 at step 0. The lever's contribution is zero at step 0 (bit-identical to no-conv baseline modulo the model's other params).

**Note on RNG consumption**: constructing the `nn.Conv1d` inside `ShortConv1D` draws from the RNG (kaiming_uniform_ default), shifting the RNG state for subsequent parameter init. This means the flag-ON model has different params from the flag-OFF model at the same seed (the same pattern as canon_conv and other module-building levers). The lever's *contribution* is still zero at step 0 (gate=0), but the model's other params differ from the flag-off baseline. The flag-OFF path is fully bit-identical (module never built, no RNG consumption).

### Run command (tiny1m3m seed 42)
Treatment (A/B against the standard `ctrl`/`ctrl2` bracket):
```bash
cd /root/universe-lm && /venv/main/bin/python -m trainers.train \
  --config_class tiny1m3m \
  --use_short_conv True \
  --short_conv_kernel 3 \
  --seed 42
```

### How to read the final val loss
Standard tiny1m3m val-loss readout in the training log. Compare the treatment's final val loss against the `ctrl` and `ctrl2` bracket from the same session. Pass/fail bar per the spec: any Δ outside the two-ctrl bracket (≈ ±0.01) is a signal; inside the bracket is null.

