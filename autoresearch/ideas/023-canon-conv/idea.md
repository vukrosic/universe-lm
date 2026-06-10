---
id: 023-canon-conv
status: running
round: 2
updated: 2026-06-10T07:59:38Z
---

# 023 — Canon layers (gated depthwise causal Conv1d on the residual stream)

## Source
Allen-Zhu et al., "Physics of Language Models" Canon-layer line (2024-2025);
the same short causal depthwise convolution is used as the local token-mixing
block in Griffin (De/Smith/Fernando, "Griffin: Mixing Gated Linear
Recurrences with Local Attention for Efficient Language Models",
arXiv:2402.19427, 29 Feb 2024) and in Mamba blocks. Griffin alone validates
the mechanism; Allen-Zhu's Canon is the named ancestor. Filed here as a
standalone drop-in mixer.

## Mechanism
Insert a single causal depthwise Conv1d over the sequence on the residual
stream **once per block, immediately before the attention sublayer's pre-LN**
(so the per-block forward becomes `x ← x + g·DWConv(x); h ← attn(LN(x)); x ←
x + h; …`). The conv is depthwise, kernel size 3, single scalar learnable
output gate `g` per block init to 0 so step-0 ≡ baseline: `x ← x + g·DWConv(x)`,
`g = 0` at init. ~45 LoC: one `CanonConv` module + flag wiring; no
sublayer-input variant, no FFN-side conv, no second conv per block.

### Shape (pinned)
- **Conv**: `nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
  groups=d_model, bias=False)`. Depthwise = one filter per channel; bias
  off because the gate absorbs any constant offset.
- **Causality**: enforced by **left-padding** the time axis with
  `kernel_size − 1 = 2` zeros *before* the conv (not by `padding=2` which
  pads both sides). Implementation: `F.pad(x.transpose(1, 2), (2, 0))`
  along the time axis after the channel transpose.
- **Gate**: a single scalar `nn.Parameter(torch.zeros(1))` per `CanonConv`
  module — one per block, not per-channel, not per-token. Step-0 identity
  is exact because `g·… = 0` for every input.
- **LN interaction**: the conv reads the **pre-LN** residual stream
  (cheaper; matches Griffin's placement). No extra LN on the conv path.

## Why it's worth a slot
We expect a val drop because explicit *local* token mixing offloads
short-range n-gram modeling from attention, freeing the few attention heads
of a 0.94M-param model to model longer-range structure — local conv + global
attention is the hybrid that powers Griffin/Mamba. Distinct from the closed
`SWA` and `dilated-attn` levers (those reshape the attention *window*) and
from `attn-sink` / NSA / diff-attn (inside-attention). This is a separate
parameter-cheap convolutional mixer *on the residual stream*, orthogonal
to attention. Zero-init gate guarantees a clean identity start and fires
every step. A null tells us local mixing is redundant with RoPE+attention
at 6L; a win is a sub-50-LoC, transferable structural lever.

## Definition (gate 2)

### Ctrl vs trt
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (`configs/llm_config.py:773`; FIRE flag passed at run time per the
  `Tiny1M3MCoPEOnFireConfig` docstring at `:693-694`). The 009 WIN
  signature is `trt=6.3234 vs ctrls 6.3875/6.4050` (`closed.md:44`). Pinned
  to the FIRE-equipped baseline so the A/B partitions the orthogonal-axis
  question (does a separate local-mixing conv add anything *on top of* the
  best attention-side win?), not the "does FIRE win?" question.
- **Trt**: same config + `use_canon_conv=True`.
- **Config class**: new `Tiny1M3MCanonOnFireConfig(Tiny1M3MConfig)` with
  `use_fire_pe: bool = True` + `use_canon_conv: bool = True` (mirrors
  `Tiny1M3MCoPEOnFireConfig` / `Tiny1M3MFOXOnFireConfig`).

### Pass bar (tiny1m3m box noise)
Run-to-run val-loss variance at this tier is ≈ ±0.01
(`closed.md` ctrls span 6.3875–6.4050 = 0.018 spread). With a single seed
the bar must clear the box and not just sit inside it. Three non-overlapping
bands tile the real line without overlap (mirrors 020-forgetting-attn's
`idea.md:92-96`):
- **WIN**: `trt_val < ctrl_val − 0.01` (clears the cited noise floor —
  trt must be strictly better than the ctrl by more than the box spread).
- **NULL**: `|trt_val − ctrl_val| ≤ 0.01` (sub-noise; the lever does not
  fire on top of FIRE at this scale; the inclusive bound closes the gap
  with WIN so no result can satisfy both).
- **FAIL**: `trt_val > ctrl_val + 0.01` (the conv is actively interfering
  with attention, not just neutral).

### Seed
**Seed 42 only.** Single fixed seed, no multi-seed sweep, no per-seed
mean. A sub-noise delta is *inconclusive, not real*; never add "run more
seeds to confirm" — log null and move on.

### Placement (pinned — kills a hidden A/B axis)
- **One conv per block, on the residual stream, immediately before the
  attention sublayer's pre-LN.** Not before the FFN sublayer, not at the
  sublayer input, not twice per block.
- Matches Griffin's canonical pre-attention Conv1d block placement and
  isolates the lever to a single variable.
- Per-block forward (pseudo): `x = x + g · DWConv(left_pad(x, 2))` →
  `h = attn(LN(x))` → `x = x + h` → … → `h = ffn(LN(x))` → `x = x + h`.

### Padding & causality (pinned)
- Use `F.pad(x.transpose(1, 2), (2, 0))` (left-pad 2 along time) before
  `Conv1d(..., padding=0, ...)`. **Do not** set `padding=2` on the
  `Conv1d` — that pads both sides and leaks future tokens.
- Add a causality test: a `+1` perturbation added to position `t` of the
  input must change only positions `≥ t` of the output (assert per-element
  equality at `s < t`).

### Gate shape (pinned)
- Single scalar `nn.Parameter(torch.zeros(1))` per `CanonConv` module.
  Not a per-channel vector, not a per-token gate. The model has `n_layers`
  such gates total, one per block.
- At init `g = 0` exactly, so `g · DWConv(x) = 0` and the block forward is
  bit-equivalent to the no-conv path (test (e) below).

### LoC budget (≤ 50 LoC, well under the 200 ceiling)
- (a) `CanonConv` module class (`DWConv` + scalar gate, init zeros) ≈ 12 LoC
- (b) integration in `TransformerBlock.forward` (left-pad, conv, gate-mul,
  residual-add — one branch) ≈ 8 LoC
- (c) flag wiring (`use_canon_conv: bool = False` in `LLMConfig` +
  `TransformerBlock`) + new config class `Tiny1M3MCanonOnFireConfig` ≈ 10 LoC
- (d) causality test (`+1` at `t` only affects `≥ t`) ≈ 8 LoC
- (e) step-0 identity test (`use_canon_conv=True, g=0` ≡ baseline within
  `1e-6`) ≈ 8 LoC

Total ≈ 46 LoC.
