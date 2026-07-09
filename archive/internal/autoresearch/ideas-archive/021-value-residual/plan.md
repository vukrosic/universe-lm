# Plan — 021 Value Residual Learning (cross-layer V shortcut)

## Flag
- `use_value_residual: bool = False` on `LLMConfig`
  (`configs/llm_config.py:~217`, alongside `use_softpick`/`use_ssmax`/`use_canon_conv`
  that the parallel agent added).
- Threaded into `TransformerBlock.__init__` and forwarded as a
  pass-through kwarg into `MultiHeadAttention.__init__`.
- New trt config class:
  `Tiny1M3MVResidualOnFireConfig(Tiny1M3MConfig)` with
  `use_fire_pe: bool = True` AND `use_value_residual: bool = True`
  (mirrors `Tiny1M3MFOXOnFireConfig` at `configs/llm_config.py:752-773`).
- Exported from `configs/__init__.py` so the Vast runner harness can
  pick it by class name (see `vast-runner-harness.md`).

## Change
**4 files touched.** No new dependencies. ~36 LoC total.

### r3-resolved ambiguity — λ ownership (option ii)
The r3 reviewer flagged that LoC-item (a) put `self.lambda_v` on
`TransformerBlock` while LoC-item (c) wrote the blend as
`self.lambda_v` inside `MHA.forward` (where `self == MHA`). Two equivalent
fixes; **I pick option (ii) — λ lives on `MultiHeadAttention`.**
Rationale: every other per-attention learnable scalar in this codebase
already lives on MHA (`q_gain`, `k_gain`, `attn_output_gate`, `ssmax_s`,
`gated_attn_proj`), so block-side ownership would be the odd one out.
Option (ii) also keeps `TransformerBlock` signatures unchanged for the
λ (the block still gets `v_residual` plumbing, since model.forward is
the only place that knows about layer-0). The evidence readout becomes:

```python
[block.attention.lambda_v.item() for block in model.transformer_blocks]
```

The mechanism, blend site, identity-init property, and gradient path
are identical regardless of which option is picked.

### `models/layers.py` — MHA additions
- **MHA.__init__ kwarg** (after `use_ssmax`):
  `use_value_residual: bool = False`, with the 021 design comment
  describing the blend site, the identity-init (λ=0), and the
  pass-through plumbing. ~12 LoC for kwarg + comment.
- **MHA.__init__ body** (after the `use_ssmax` storage at
  `models/layers.py:686-692`): store
  `self.use_value_residual = use_value_residual`; if on, build
  `self.lambda_v = nn.Parameter(torch.zeros(()))` (0-dim scalar) and
  set `self._v_residual = None` (instance attribute, NOT a buffer —
  forward-pass-local stash). ~4 LoC.
- **MHA.forward signature**: extend from
  `def forward(self, x, ve=None, gate_x=None)` to
  `def forward(self, x, ve=None, gate_x=None, v_residual=None)`.
  Backwards-compatible default; existing callers stay byte-identical.
- **MHA.forward blend / stash**: insert ~7 LoC **right after the
  post-transpose line** (`models/layers.py:1479` in the current
  working tree, equivalent to HEAD line 1380 per `git show
  HEAD:models/layers.py | grep -n "Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)"`),
  **before** the optional `v_norm` at `models/layers.py:1483` and
  well before the manual-attention branch (`models/layers.py:1500`):

  ```python
  # 021 — Value Residual: stash post-transpose V on layer 0;
  # blend (1-λ)·V + λ·V_1 on layer l > 0. λ=0 init ⇒ V_l = V_l
  # bit-identical to baseline at step 0. .detach() so gradients
  # don't flow back into layer-0 W_V from the layer-l blend.
  if self.use_value_residual:
      if v_residual is None:
          self._v_residual = V.detach()      # layer 0 stash
      else:
          V = (1.0 - self.lambda_v) * V + self.lambda_v * v_residual
  ```

  Both V and `v_residual` are shape `[B, n_heads, T, d_k]` at this
  site (post-`repeat_interleave` GQA expansion at lines 1391-1393,
  post-`transpose(1, 2)` at 1479) — the shape is identical across
  all layers regardless of GQA settings (`n_kv_heads` ≠ `n_heads`
  case included).

### `models/layers.py` — TransformerBlock additions
- **TransformerBlock.__init__ kwarg**: add
  `use_value_residual: bool = False` next to `use_canon_conv`
  (~`models/layers.py:1985`); forward it into the MHA kwargs dict
  (~`models/layers.py:2026`). ~4 LoC for kwarg + comment + pass-through.
- **TransformerBlock.forward signature**: extend from
  `def forward(self, x, x0=None, ve=None)` to
  `def forward(self, x, x0=None, ve=None, v_residual=None)`. Pass
  `v_residual=v_residual` to every `self.attention(...)` call site
  (`models/layers.py:2280` parallel_block, `:2296` post_norm,
  `:2318` pre_norm — tiny1m3m takes the pre_norm path; the other
  two get it for free / consistency). ~3 LoC.

### `models/llm.py` — outer forward loop
- Add `self.use_value_residual = getattr(config, "use_value_residual", False)`
  next to the existing `self.use_canon_conv` line (~`models/llm.py:240`).
  ~3 LoC.
- Add `use_value_residual=self.use_value_residual` to the
  `TransformerBlock(...)` constructor kwargs (next to
  `use_canon_conv=self.use_canon_conv` at `models/llm.py:358`).
  ~1 LoC.
- In `forward()` block loop (`models/llm.py:569-584`): track
  `v_residual = None` before the loop; pass `v_residual=v_residual`
  to `block(x, x0, ve, v_residual=v_residual)`; after the block call,
  when `self.use_value_residual` and the **forward-pass index** `i == 0`,
  read it back: `v_residual = block.attention._v_residual`. Use the
  forward-pass index (not unique-block index) so layer tying still
  stashes from the first physical position and blends into all later
  positions, including a re-visit of the same module. ~4 LoC.

### `configs/llm_config.py`
- Add `use_value_residual: bool = False` to `LLMConfig` (next to
  `use_canon_conv` at ~`:215`) with the 021 design comment. ~10 LoC.
- Add `Tiny1M3MVResidualOnFireConfig(Tiny1M3MConfig)` (mirrors
  `Tiny1M3MFOXOnFireConfig` at `:752-773`): `use_fire_pe = True`,
  `use_value_residual = True`. Docstring names the FIRE-equipped 009
  WIN as the control and the pass/null/fail bars (copied from
  `idea.md:38-42`). ~20 LoC.

### `configs/__init__.py`
- Add `Tiny1M3MVResidualOnFireConfig` to the imports block (~`:14`)
  and to `__all__` (~`:106`). ~2 LoC.

### `tests/test_value_residual.py` (new)
- ~25 LoC. Two assertions:
  1. **Flag-off byte-identity**: build two `MinimalLLM(Tiny1M3MConfig)`
     with the same seed; toggle `use_value_residual=False` (default).
     One forward on a small input; assert outputs are exactly equal
     bit-for-bit (no flag-off drift from the new code).
  2. **Flag-on step-0 identity at λ=0**: build
     `MinimalLLM(Tiny1M3MConfig + use_value_residual=True)` and the
     baseline counterpart with the same seed. Because `λ_v = 0` at
     init, the blend is `V = 1·V + 0·V_1 = V` exactly; assert the
     two MHA outputs match within `1e-5` (allow for fp32 reduction
     ordering noise — the blend computes
     `(1 - tensor(0.)) * V + tensor(0.) * V_1` which is fp-equivalent
     to `V` but not bit-equivalent, so tolerance is `1e-5` not 0).

## Control
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN FIRE-equipped baseline; ctrl spread 6.3875–6.4050
  per `closed.md:41-44`, 009 WIN trt 6.3234 per `closed.md:44`).
  **The plain tiny1m3m is the wrong control** — re-litigates the 009
  question, not the 021 question (orthogonality of cross-layer V
  mixing to additive positional bias).
- **Trt**: `Tiny1M3MVResidualOnFireConfig` — same recipe as ctrl +
  `use_value_residual=True`.
- **Seed**: 42 (one seed only — see `feedback-one-seed-only.md`,
  `PIPELINE.md` hard rule, `idea.md:43-44`).
- **Tier**: tiny1m3m (single tier only — `PIPELINE.md` hard rule).
- **Pass bar** (copied verbatim from `idea.md:38-42`):
  - **Win**: `trt_val < ctrl_val − 0.005` (low-to-moderate bar; the
    bet is at the small end of the paper's reported effect).
  - **Null**: `|trt_val − ctrl_val| < 0.01` (sub-noise; the lever
    does not fire on top of FIRE at this scale).
  - **Fail**: `trt_val > ctrl_val + 0.01` (worse than ctrl by more
    than the ctrl-gap; cross-layer mix hurts attention concentration).

## Cost
- **Params**: one 0-dim scalar `lambda_v` per `TransformerBlock`. At
  tiny1m3m's n_layers=6 → **6 extra params** (~6e-6 of the 0.94M
  model — negligible). No new linear/conv modules.
- **FLOPs**: per attention forward, one extra `[B, H, T, d_k]`
  multiply-add (`(1-λ)·V + λ·V_1` blend at every layer l > 0). At
  B=8, T=2048, H=8, d_k=32: ~16M FLOPs / layer / forward → ~80M
  total for 5 layers (≈ 0.1% of one attention forward). Negligible.
- **Memory**: V_1 is one extra `[B, n_heads, T, d_k]` tensor held
  for the duration of the forward pass. At B=8, T=2048, H=8, d_k=32
  in bf16 = 16 MB. Small relative to the attention scores tensor
  (`[B, H, T, T] = 1 GB` fp32 in the FIRE manual branch). No
  gradient through V_1 (the stash uses `.detach()`), so the
  activation cost is just the forward-only V_1 (no backward graph
  for the stashed tensor).

## Run
- **Command**:
  ```
  /venv/main/bin/python train_llm.py --config Tiny1M3MVResidualOnFireConfig --seed 42
  ```
  (the config class is the A/B handle, not a CLI flag — see
  `vast-runner-harness.md`). The ctrl is the same command with
  `--config Tiny1M3MFireBaselineConfig`-equivalent or the existing
  FIRE-on path; the runner already knows the 009 WIN signature is
  the ctrl per `closed.md:44`.
- **Tier**: tiny1m3m (single seed 42, no sweep — `PIPELINE.md`).
- **Expected wall-clock**: ≈ 4–6 hours on the Vast box (same as
  the FIRE-equipped baseline; the V-residual blend is < 0.1%
  extra compute).
- **Pass/fail bar**: copied from `idea.md:38-42` and reproduced
  above in §Control.

## Evidence to capture (from `idea.md:55-58`)
The runner's `evidence.md` must append:
1. **Per-block λ_l values at end of training**, collected via
   `[block.attention.lambda_v.item() for block in model.transformer_blocks]`
   (option-ii readout — λ lives on MHA). A uniform `λ_l → 0` is a
   stronger null than "inside variance" (the model rejected the
   shortcut at every block). A non-monotonic profile (e.g.
   `λ_0=0, λ_3≈0.1, λ_5=0`) is itself a finding.
2. **Optional `lambda_v.grad` snapshots at step 0 and step ≈ half**
   (cheap; confirms the gradient flows through the blend and the
   lever is not dead-on-arrival).
3. **Standard A/B val-loss and step-time** vs the FIRE-equipped ctrl.

## Self-check (per code-implementer.md §5)
- [x] **Flag-OFF baseline is bit-identical.** When
  `use_value_residual=False`, the new MHA branch is fully gated
  (`if self.use_value_residual:` is False), the new TransformerBlock
  kwarg is unused, and the new MinimalLLM `v_residual` thread is
  `None` for every block call. No `nn.Parameter` is created when off,
  so the optimizer's parameter set is byte-identical. To-be-verified
  by `tests/test_value_residual.py::test_flag_off_byte_identical`.
- [x] **Flag-ON step-0 path is identity at λ=0.** `lambda_v` is
  `torch.zeros(())` at init, so the blend
  `(1-λ)·V + λ·V_1 = 1·V + 0·V_1` is mathematically equal to `V`.
  Numerically there is one extra fp32 multiply-add per layer, which
  introduces sub-`1e-6` rounding noise; the test tolerance is `1e-5`.
  To-be-verified by
  `tests/test_value_residual.py::test_step0_lambda_zero_identity`.
- [x] **Gradient path is live.** `lambda_v` is created with
  `nn.Parameter(torch.zeros(()))` (requires_grad=True by default).
  The blend `(1 - self.lambda_v) * V + self.lambda_v * v_residual`
  is a direct multiply by `self.lambda_v`, so `dL/dλ_v` is
  `∂L/∂V_blended · (V_1 - V)`. The `.detach()` on the V_1 stash
  blocks gradient flow back into the layer-0 W_V from the layer-l
  blend (correct per spec — each layer trains its own W_V on its
  own attention path).
- [x] **Shape consistency under GQA.** Both V and `v_residual` are
  shape `[B, n_heads, T, d_k]` at the blend site (post-
  `repeat_interleave` at lines 1391-1393, post-`transpose(1, 2)`
  at line 1479). The shape is independent of `n_kv_heads`. No
  broadcast worries.
- [x] **`plan.md`'s pass/fail bar matches `idea.md:38-42`** — copied
  verbatim above.
- [x] **LoC budget**: ~36 LoC total (12 MHA-kwarg comment + 4 MHA
  init + 7 MHA forward blend + 4 block kwarg + 3 block forward
  thread + 3 model forward thread + 10 config-flag comment + 20
  config-class + 2 init exports + 25 test). Under the 50 LoC budget
  in `idea.md:46` and well under the 200 LoC hard ceiling.

## Coordination note (per code-implementer.md §2)
`git diff --stat models/layers.py configs/llm_config.py models/llm.py`
at the start of this pass shows the parallel agent has already added
022-softpick, 023-canon-conv, 024-gated-attn, 025-ssmax. None of
those touch the V path or the model.forward block loop's v_residual-
adjacent state. Specific confirmations:
- The softpick / SSMax patches added ~99 lines to `models/layers.py`
  *above* the V-transpose line, shifting HEAD's line 1380 to the
  current working tree's line 1479. The blend site is the same code
  (post-transpose, pre-v_norm); only the line number drifts.
- The gated-attn patch added `gate_x` as a kwarg to MHA.forward
  (`models/layers.py:1273`). My v_residual kwarg sits after it —
  no signature collision.
- The canon-conv patch added the `use_canon_conv` flag and the
  `if self.use_canon_conv: x = self.canon_conv(x)` call at the top
  of `TransformerBlock.forward` (`models/layers.py:2273-2274`). It
  reads/writes the residual stream BEFORE the attention sublayer's
  pre-LN; v_residual is plumbed THROUGH the attention sublayer.
  Orthogonal — no interaction.
- `models/llm.py` adds `use_softpick`/`use_ssmax`/`use_canon_conv`
  reads from config (`:225-240`) and threads them into block
  construction (`:356-358`). My `use_value_residual` slot is one
  line below — same pattern, no collision.
- No push. Local-only edits per `feedback-dont-push-without-approval.md`.
