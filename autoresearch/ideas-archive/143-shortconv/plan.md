# Plan — 143-shortconv

## Lever
Pre-attention depthwise causal Conv1d (Hyena ShortConv variant,
Poli/Massaroli 2023, arXiv:2302.10866), gated by a per-block scalar
init 0. Conv has identity init (last tap = 1, rest = 0). Applied on
the residual stream, pre-attention, pre-LN on the conv path.

## Files
- `models/short_conv.py` (new, 75 LoC): `ShortConv1D` module — depthwise
  causal Conv1d (`nn.Conv1d(d, d, k, groups=d, bias=False)`) with
  identity-init weights and a left-pad-by-(k-1) causal forward.
- `models/layers.py`: import `ShortConv1D`; add `use_short_conv` /
  `short_conv_kernel` to `TransformerBlock.__init__`; build
  `self.short_conv` + `self.short_conv_gate = nn.Parameter(zeros(1))`
  when `use_short_conv=True`; apply `x = x + self.short_conv_gate *
  self.short_conv(x)` pre-attention, pre-LN.
- `models/llm.py`: read `use_short_conv` + `short_conv_kernel` from
  config; thread both to the standard `TransformerBlock` stack and
  the YOCO upper-half stack.
- `configs/llm_config.py`: `use_short_conv: bool = False`,
  `short_conv_kernel: int = 3` on `LLMConfig`; new
  `Tiny1M3MShortConvConfig(Tiny1M3MConfig)` subclass with both flags
  on (k=3 default).
- `_arq_143-shortconv.py` (new, root): config-subclass pattern stub
  used by the runner queue.

## Config flag
- `use_short_conv: bool = False` (default off, baseline path
  bit-identical).
- `short_conv_kernel: int = 3` (asserted ∈ {3, 4} at construction).

## Step-0 identity
Flag-OFF: `ShortConv1D` is never built, no `short_conv_gate` is created.
The block's forward graph is exactly the no-conv baseline. Flag-OFF is
bit-identical to the baseline. ✓

Flag-ON: `ShortConv1D` is built with identity init (last tap = 1, rest
= 0) and `short_conv_gate = nn.Parameter(torch.zeros(1))` so the
forward is `x = x + 0·ShortConv1D(x) = x` — the conv's contribution is
exactly zero at step 0. The flag-ON model has a different param RNG
state (the `nn.Conv1d` constructor draws from the RNG even though we
overwrite the weights), but the *contribution* of the lever to the
forward is zero — the val-loss change at step 0 from the lever alone
is below fp32 noise. The A/B reads the *training-time* Δ vs the two-
ctrl bracket, not the step-0 val.

## Run command (on the box)
```bash
cd /root/universe-lm && /venv/main/bin/python -m trainers.train \
  --config_class configs.llm_config.Tiny1M3MShortConvConfig \
  --seed 42
```
Or via the existing `_arq_143-shortconv.py` config-subclass pattern:
```bash
cd /root/universe-lm && /venv/main/bin/python _arq_143-shortconv.py
```

## Pass/fail bar
A/B vs the `ctrl` + `ctrl2` bracket on the same box/session. Box
variance at tiny1m3m is ~0.04 val loss; the two-ctrl bracket is the
empirical reference. PASS ≤ −0.01 vs both ctrls. NULL band |Δ| ≤ 0.01.
DRIFT > +0.01. See `idea.md` for the paper-level claim and the
mechanism description.

## How to read the final val loss
Standard tiny1m3m val-loss readout in the training log. Compare the
treatment's final val loss against the `ctrl` and `ctrl2` bracket from
the same session. The runner writes the verdict to
`autoresearch/ideas/143-shortconv/evidence.md` and flips the status.
