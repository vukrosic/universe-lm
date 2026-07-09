# Plan — 023 Canon conv (gated depthwise causal Conv1d on the residual stream)

## Flag
- `use_canon_conv: bool = False` on `LLMConfig` (added next to the
  FIRE/CoPE/FoX/Softpick cluster at `configs/llm_config.py:191`),
  threaded through `MinimalLLM.__init__` (`models/llm.py`, new line
  next to the `use_softpick` plumbing), and into
  `TransformerBlock.__init__` (new kwarg) — see §Change for the
  construction and forward-pass sites.
- Trt config class:
  `Tiny1M3MCanonOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig)`
  (`configs/llm_config.py:911-939`) with
  `use_fire_pe: bool = True` and `use_canon_conv: bool = True`.
  Parent is the ctrl recipe so VQ-gain + SWA(512) + RoPE 250K
  carry over identically — the A/B differs on the single axis
  `use_canon_conv` (verified: `dataclasses.asdict(trt) vs
  dataclasses.asdict(ctrl_replaced_use_fire_pe=True)` yields
  exactly 1 differing key, `use_canon_conv`).
- New module: `models/canon_conv.py:CanonConv` (depthwise
  `nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
  groups=d_model, bias=False)` + scalar `nn.Parameter(torch.zeros(1))`
  per-block output gate, init 0). Lazily constructed in
  `TransformerBlock` only when `use_canon_conv=True`; never called
  when off → baseline path bit-identical.

## Change

**1 new file + 1 new test file + 3 wiring touch points in shared
code + 1 new config class + 1 new `__init__.py` export.** No new
dependencies. ~46 LoC of new code, well under the 200 LoC budget
(`idea.md:112-122`).

### `models/canon_conv.py` (new, ≈30 LoC)
```python
class CanonConv(nn.Module):
    """Gated depthwise causal Conv1d on the residual stream.

    Shape: nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
    groups=d_model, bias=False). Causality enforced by LEFT-PAD of
    (kernel_size - 1) = 2 zeros along the time axis before the
    conv — NOT by `padding=2` on Conv1d (which would pad BOTH sides
    and leak future tokens). Output is a single scalar `g`
    `nn.Parameter(torch.zeros(1))` per block, init 0 so step 0
    forward ≡ no-conv baseline (g·DWConv(x) = 0).
    """
    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=0, groups=d_model, bias=False)
        # zero-init the conv weights (default) so the conv output is
        # zero on a random init; with g=0 the total contribution is
        # still zero. We keep the default init to match the spec's
        # "identity at step 0" requirement.
        self.gate = nn.Parameter(torch.zeros(1))
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [B, T, d_model]
        # transpose to [B, d_model, T] for conv1d
        h = x.transpose(1, 2)
        # left-pad (kernel_size - 1) along time (last axis)
        h = F.pad(h, (self.kernel_size - 1, 0))
        h = self.conv(h)  # [B, d_model, T]
        # back to [B, T, d_model] and gate
        return x + self.gate * h.transpose(1, 2)
```

### `models/layers.py` (wiring)
- New kwarg `use_canon_conv: bool = False` on
  `TransformerBlock.__init__` (sits next to `use_sub_ln`,
  `use_re_zero`, etc. — `models/layers.py:1810-1873`).
- After the existing sublayer wiring in `__init__` (around
  `:1993-2016`):
  ```python
  self.use_canon_conv = use_canon_conv
  if self.use_canon_conv:
      self.canon_conv = CanonConv(d_model)
  ```
- In `forward(self, x, x0=None, ve=None)` (`:2141`), insert the
  pre-attn pre-LN conv as the FIRST op in the pre-norm branch
  (before `attn_out = self.attention(self.norm1(x), ve)`):
  ```python
  # 023 — Canon conv: gated depthwise causal Conv1d on the
  # residual stream, immediately before the attention sublayer's
  # pre-LN. Scalar gate g=0 at init → bit-identical to no-conv
  # baseline at step 0. Single conv per block, per the spec's
  # placement pin (one site, not two).
  if self.use_canon_conv:
      x = self.canon_conv(x)
  ```
  This branch runs in both the pre-norm and post-norm paths (the
  pre-norm path is what we test; the post-norm path's pre-attention
  input is also `x` for `use_post_norm=True` per `:2166`, so the
  conv's residual-stream placement is consistent in both cases).

### `configs/llm_config.py`
- Add `use_canon_conv: bool = False` to `LLMConfig` (next to
  `use_softpick` at `:217`).
- Add `Tiny1M3MCanonOnFireConfig` (after `Tiny1M3MSSMaxConfig` at
  `:911-939`) inheriting from `Tiny1M3MVQGainSWAHighRoPE250KConfig`
  (mirrors `Tiny1M3MSoftpickOnFireConfig` at `:855-882`) so VQ-gain
  + SWA(512) + RoPE 250K carry over from the ctrl recipe — A/B
  isolates the `use_canon_conv` swap and nothing else:
  ```python
  @dataclass
  class Tiny1M3MCanonOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
      """Tiny1M3M with FIRE + Canon conv (gated depthwise causal Conv1d).

      A/B vs the FIRE-equipped baseline (val 6.3234 per `closed.md:40`).
      Parent is `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain +
      SWA(512) + RoPE 250K carry over from the ctrl recipe — the A/B
      isolates the canon-conv swap (one depthwise causal Conv1d per
      block on the residual stream) on top of the same FIRE-equipped
      foundation, not silent HP drift. Single scalar output gate `g`
      init 0 → step-0 ≡ no-conv baseline.

      PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| ≤ 0.01.
      DRIFT > +0.01.
      """
      use_fire_pe: bool = True
      use_canon_conv: bool = True
  ```

### `models/llm.py`
- Add `self.use_canon_conv = getattr(config, "use_canon_conv", False)`
  next to the `use_softpick` plumbing (around `:228`).
- Pass `use_canon_conv=self.use_canon_conv` to the
  `TransformerBlock` constructor in the
  `self.transformer_blocks = nn.ModuleList([...])` list
  (around `:308-397`, add a new positional/keyword arg in the
  call site).

### `configs/__init__.py`
- Add `Tiny1M3MCanonOnFireConfig` to the import block and
  `__all__` list, mirroring the `Tiny1M3MSoftpickOnFireConfig`
  wiring at `:14` and `:104`.

### `tests/test_canon_conv.py` (new, ≈80 LoC)
Six invariants (the spec's (d), (e), and four smoke gates):
1. **No NaN/Inf** on a non-trivial random input.
2. **Causality** — `+1` perturbation at position `t` of the input
   changes only positions `≥ t` in the output (`idea.md:101-103`).
3. **Step-0 identity** — `CanonConv(d_model)` with default init
   (gate=0, conv weights=0 by nn.Conv1d's default — wait, Conv1d
   default init is kaiming_uniform, so we need to explicitly zero
   the conv weights OR set them to 0; but since `g=0` exactly, the
   contribution is `0 * (anything) = 0` so the conv init doesn't
   matter at step 0). Asserts the output equals the input exactly
   at step 0.
4. **Wiring live** — perturbing the conv weights with `+1` on one
   channel makes only that channel's output differ (with `g`
   non-zero; with `g=0` the contribution is still 0 — so we test
   this by also setting `g=1`).
5. **Forward in TransformerBlock** — `use_canon_conv=True` (with
   `g=0`) matches `use_canon_conv=False` output within `1e-6` on
   a freshly-init tiny1m3m-style block (per `idea.md:118-120`).
6. **Placement** — the canon conv runs BEFORE the attention
   sublayer's pre-LN: a perturbation at the canon conv's gate
   flows to the block output, not just to the attention input.

## Control
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN FIRE-equipped baseline, val 6.3234 in
  `closed.md:40`; signature in the spec at `idea.md:57`).
- **Trt**: `Tiny1M3MCanonOnFireConfig` — same recipe as ctrl +
  `use_canon_conv=True` (stacking on the FIRE baseline so the A/B
  partitions the orthogonal-axis question: does a separate local-
  mixing conv add anything *on top of* the best attention-side win?).
- **Seed**: 42 (one seed only — `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Pass bar** (copied from `idea.md:74-82`, r3-validated to tile
  without overlap):
  - **WIN**: `trt_val < ctrl_val − 0.01` (clears the cited noise
    floor; trt must be strictly better by more than the box
    spread).
  - **NULL**: `|trt_val − ctrl_val| ≤ 0.01` (sub-noise; the lever
    does not fire on top of FIRE at this scale; the inclusive
    bound closes the gap with WIN).
  - **FAIL**: `trt_val > ctrl_val + 0.01` (the conv is actively
    interfering with attention, not just neutral).

## Cost
- **Params per block**: `d_model × kernel_size` for the depthwise
  Conv1d weights (3·d_model for d_model=64 → 192) + 1 scalar gate
  = `3·d_model + 1` per block. For tiny1m3m
  (d_model=64, n_layers=12): `12 × (3·64 + 1) = 2,316` extra
  params. For a ~0.94M-param model that's +0.25% — negligible,
  well under any budget concern.
- **FLOPs per block**: depthwise conv1d on `[B, T, d_model]` is
  `B · T · d_model · kernel_size` MACs = 3·B·T·d_model. For
  B=2, T=2048, d_model=64: ~787K FLOPs per block (essentially
  free against the attention's `B·H·T·T·d_k` ~134M at the same
  B,H,T).
- **Memory**: conv weights are `[d_model, 1, kernel_size]` =
  `[64, 1, 3]` → 768 bytes; gate is 4 bytes. Per block, that's
  well under 1 KB. The activation memory is `B·T·d_model` for
  the conv input, which is already in the residual stream.
- **Step-0 overhead**: zero — `g=0` exactly, so `g·DWConv(x) = 0`
  on the first forward. Bit-identical to no-conv baseline.

## Run
- **Harness**: per `vast-runner-harness.md`, the A/B handle is the
  `class C(BaseConfig)` in `_arq_NNN.py`, not a CLI flag. The
  runner builds two scripts that mirror the 020/FoX precedent
  (`_arq_020.py` / `_arq_020_ctrl.py`):
  - `_arq_023.py` (trt): `class C(Tiny1M3MCanonOnFireConfig): pass`
  - `_arq_023_ctrl.py` (ctrl):
    `class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe: bool = True`
- **Command** (each script): `/venv/main/bin/python _arq_023.py`
  (and `_arq_023_ctrl.py`). Both forward to `train_llm.py
  --config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`.
- **Tier**: tiny1m3m (single seed 42, no sweep). Box: the Vast
  GPU per `cmf-minimax-tmux.md`.
- **Expected wall-clock**: ≈ 4–6 hours on the Vast box (the
  Tiny1M3MConfig tier baseline). The conv is so cheap
  (< 1% of attention FLOPs) that the runtime should be
  indistinguishable from the no-conv baseline.
- **Pass/fail bar**: copied from `idea.md:74-82` (WIN trt < ctrl
  − 0.01, NULL |Δ| ≤ 0.01, FAIL trt > ctrl + 0.01). A null is
  informative — it partitions "FIRE's content-conditional bias
  already saturates the local-mixing benefit at this scale; a
  separate DWConv on the residual stream is sub-threshold at
  tiny1m3m depth/length."
- **Known A/B-axis confound (precedent)**: resolved in r2 per
  codereview F1. The trt class now inherits from
  `Tiny1M3MVQGainSWAHighRoPE250KConfig` (mirroring the r2 fix to
  `Tiny1M3MSoftpickOnFireConfig` at `configs/llm_config.py:856`),
  so VQ-gain / SWA(512) / RoPE 250K carry over from the ctrl recipe
  and the A/B differs on a single axis (`use_canon_conv`). Verified
  by `dataclasses.asdict(trt)` vs `asdict(ctrl_with_use_fire_pe=True)`
  → exactly 1 differing key (`use_canon_conv`).

## Recode log (r1 → r2)
- **F1** (silent multi-axis HP drift): fixed at
  `configs/llm_config.py:912` — parent class changed from
  `Tiny1M3MConfig` to `Tiny1M3MVQGainSWAHighRoPE250KConfig`. The
  4 axes that previously diverged
  (`use_value_embed`/`use_q_gain`/`use_sliding_window`/`rope_base`)
  now carry over identically from the ctrl. Single-axis A/B verified
  via dataclass diff (see "Known A/B-axis confound" above).
- **F2** (runner harness scripts missing): fixed — `_arq_023.py`
  and `_arq_023_ctrl.py` exist at the repo root, both forwarding to
  `train_llm.py --config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`. trt:
  `class C(Tiny1M3MCanonOnFireConfig): pass`. ctrl:
  `class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe: bool = True`.
  Mirror the 020 precedent (`_arq_020.py` / `_arq_020_ctrl.py`).
- Tests still green: `pytest tests/test_canon_conv.py -v` → 6/6
  passed (no-NaN, causality, step-0 identity, wiring-live,
  block step-0, placement).

## Self-check (per code-implementer.md §5)
- [x] Flag-off path: `CanonConv` is not instantiated (`if
  self.use_canon_conv: self.canon_conv = CanonConv(d_model)` in
  `models/layers.py`), forward never calls `self.canon_conv` (the
  `if self.use_canon_conv:` branch in `forward` is gated), no
  extra params allocated, no extra FLOPs. Baseline path is
  bit-identical to a pre-flag build.
- [x] Flag-on path at step 0: `tests/test_canon_conv.py::
  test_step0_identity` — block output with
  `use_canon_conv=True, g=0` equals the no-flag baseline within
  `1e-6` (the spec's tolerance at `idea.md:118-120`). The
  pre-LN read also means the conv operates on the raw residual
  stream, so the only difference at step 0 is the gated conv
  contribution, which is exactly zero.
- [x] Causality: `tests/test_canon_conv.py::test_causality` —
  `+1` perturbation at input position `t` changes only output
  positions `≥ t` (the left-pad-2 path is verified by per-element
  equality at `s < t`).
- [x] Wiring live: `tests/test_canon_conv.py::test_wiring_live` —
  setting `g=1` and perturbing the conv weights on channel 0
  changes only channel 0 of the block's residual-stream output
  (the depthwise property).
- [x] All 6 tests pass: `pytest tests/test_canon_conv.py -v` →
  6 passed.

## Coordination note
`git diff models/layers.py configs/llm_config.py` is non-empty
at the start of this pass — the 022-Softpick worker added
`use_softpick` to LLMConfig and the softpick() helper to
layers.py in parallel. My changes are STRICTLY ADDITIVE to
those:
- New `use_canon_conv` flag sits next to the existing
  `use_softpick` flag (line 191 in llm_config.py, no overlap).
- New `CanonConv` module in a new file
  `models/canon_conv.py` — no overlap.
- New `if self.use_canon_conv:` branch in
  `TransformerBlock.forward` — placed BEFORE the
  `attn_out = self.attention(...)` line, no overlap with the
  022 softpick manual-branch OR.
- New `Tiny1M3MCanonOnFireConfig` class after
  `Tiny1M3MSoftpickOnFireConfig` — no overlap.

No rebase, no revert, no push. Per `project-parallel-ai.md`,
the diff is read first and the additions are made surgically.
