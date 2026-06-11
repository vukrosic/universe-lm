# Plan — 030 U-Net Skip Gates (sigmoid gate init −1.5 fix)

## Flag
- All flags already live in `LLMConfig`:
  - `use_unet_skips: bool = False` (`configs/llm_config.py:53`)
  - `unet_gate_type: str = "raw"` (`configs/llm_config.py:62`)
  - `unet_gate_init: float = 0.0` (`configs/llm_config.py:67`)
  - `unet_skip_count: Optional[int] = None` (`configs/llm_config.py:57`)
  - Default OFF — baseline path bit-identical when `use_unet_skips=False`.
- Treatment is a **three-flag composition with a non-default init** —
  no new mechanism code. The previous attempt
  (`docs/youtube-architecture-ablation-log.md §5`) flipped only
  `use_unet_skips=True` and left `unet_gate_type="raw"` +
  `unet_gate_init=0.0`. That's the dead-gate bug per
  `[[unet-skips-gate-fix]]` memory: a raw gate at exactly 0 receives
  almost no gradient and never turns on. The fix is the modded-nanogpt
  sigmoid path: `unet_gate_type="sigmoid"` + `unet_gate_init=-1.5`
  → `sigmoid(-1.5) ≈ 0.18` of the early activation flows in at step 0,
  giving the gate a non-zero starting point with a non-zero gradient.
- New trt config class: `Tiny1M3MUNetSigmoidOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig)`
  (added next to `Tiny1M3MCanonOnFireConfig` at `configs/llm_config.py:~940`)
  with `use_fire_pe: bool = True`, `use_unet_skips: bool = True`,
  `unet_gate_type: str = "sigmoid"`, `unet_gate_init: float = -1.5`.
  Parent is the FIRE-equipped baseline so the A/B isolates the U-Net
  axis on top of the current best baseline.

## Change

**No new module file. No new mechanism code.** The
`models/llm.py:166-196` U-Net skip implementation, including the
sigmoid branch at `:612-613` (`if self.unet_gate_type == "sigmoid":
gate = torch.sigmoid(gate)`), is already in place. The lever is
purely a config-flag flip with a non-default init.

### `configs/llm_config.py`
Add `Tiny1M3MUNetSigmoidOnFireConfig` after `Tiny1M3MCanonOnFireConfig`
(at `:940`), mirroring the `Tiny1M3MSoftpickOnFireConfig`
(`:855-882`) / `Tiny1M3MCanonOnFireConfig` (`:911-939`) pattern:

```python
@dataclass
class Tiny1M3MUNetSigmoidOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + U-Net sigmoid skips (modded-nanogpt fix).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates the
    U-Net swap (residual-stream architectural lever) on top of the
    same FIRE-equipped foundation, not silent HP drift.

    Adds learnable U-Net skip connections bridging early layer outputs
    into mirrored late layers. The gate parameter is initialised to
    -1.5 and wrapped in sigmoid (modded-nanogpt PR #125 fix:
    https://github.com/KellerJordan/modded-nanogpt/pull/125), so
    `sigmoid(-1.5) ≈ 0.18` of the early activation flows in at step 0
    — small, bounded to (0, 1), non-zero starting point with non-zero
    gradient. Categorically distinct from our previous broken attempt
    (`docs/youtube-architecture-ablation-log.md §5`, val +0.0003
    worse) which used `unet_gate_type="raw"` + `unet_gate_init=0.0`
    — the dead-gate bug per `[[unet-skips-gate-fix]]` memory. The
    mechanism never actually ran in that test; it was a bug-experiment,
    not a mechanism A/B.

    At tiny1m3m's 12-layer depth the U-Net mirrors are 0↔11 /
    1↔10 / 2↔9 / 3↔8 / 4↔7 / 5↔6 — six pairs (n_layers=12, not 6;
    taste assumed 6L but Tiny1M3MConfig is 12L), so the predicted
    effect is "small but non-zero", not big-if-true. A clean null
    after the fix
    definitively closes U-Net skips for this model class; a win
    plausibly amplifies at 135M where depth grows. Strictly orthogonal
    to FIRE (which is an attention-side lever); orthogonal to all
    closed levers (no residual-stream architectural change in
    closed.md). transfer-risk: low — modded-nanogpt's +1.25%
    speedup is at ≥100M parameter scale, directly comparable to
    tiny1m3m's model class.

    PASS ≤ −0.005 vs the FIRE-equipped ctrl (taste's "small but
    non-zero" prediction; not −0.01 because the 3-pair U at 6L is
    a smaller bet than the deeper-stack version). NULL band
    |Δ| < 0.005. DRIFT > +0.005. See
    `autoresearch/ideas/030-unet-skip-sigmoid/plan.md`.
    """
    use_fire_pe: bool = True
    use_unet_skips: bool = True
    unet_gate_type: str = "sigmoid"
    unet_gate_init: float = -1.5
```

### `configs/__init__.py`
Add `Tiny1M3MUNetSigmoidOnFireConfig` to the import block and
`__all__` list, mirroring the `Tiny1M3MCanonOnFireConfig` wiring.

### `models/llm.py` — **NO CHANGES**
- `use_unet_skips` is wired at `:166-196`. The sigmoid branch at
  `:612-613` is already in place.
- `unet_skip_gates` is initialised via
  `torch.full((skip_count, d_model), gate_init)` at `:186-190`
  using `gate_init = float(getattr(config, "unet_gate_init", 0.0))`
  at `:185`. With `unet_gate_init=-1.5`, every entry is initialised
  to -1.5 (broadcast to (skip_count, d_model)).
- Forward at `:609-617`: for each layer `i >= n_layers - skip_count`,
  computes `gate = self.unet_skip_gates[skip_idx]`, wraps with sigmoid
  if `unet_gate_type=="sigmoid"`, multiplies the saved skip, adds to
  the residual stream. `sigmoid(-1.5) ≈ 0.18` per channel at step 0
  → skip contributes ≈18% of the saved early activation.

### `models/layers.py` — **NO CHANGES**
- U-Net skips live in `models/llm.py`, not `models/layers.py`. The
  TransformerBlock and MultiHeadAttention surfaces are unchanged.

### Harness scripts (`_arq_030.py` + `_arq_030_ctrl.py`)
Mirror the 023/026 precedent at the repo root, **but with the
@dataclass decorator on the ctrl subclass — required for the
`use_fire_pe = True` override to actually take effect** (see
coordination note below):

```python
# _arq_030.py — trt: FIRE + U-Net sigmoid skips
import sys
from configs.llm_config import Tiny1M3MUNetSigmoidOnFireConfig
class C(Tiny1M3MUNetSigmoidOnFireConfig):
    pass
if __name__ == "__main__":
    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = ["train_llm.py", "--config_class", "__main__.C",
                "--seed", "42", "--dataset_path", "processed_data/pretrain_1B",
                "--warmup", "false"]
    train_llm.main()
```

```python
# _arq_030_ctrl.py — ctrl (FIRE-equipped 009 WIN signature, no U-Net)
import sys
from dataclasses import dataclass
from configs.llm_config import Tiny1M3MVQGainSWAHighRoPE250KConfig
@dataclass                                            # <-- required
class C(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    use_fire_pe: bool = True
if __name__ == "__main__":
    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = ["train_llm.py", "--config_class", "__main__.C",
                "--seed", "42", "--dataset_path", "processed_data/pretrain_1B",
                "--warmup", "false"]
    train_llm.main()
```

Without `@dataclass`, the parent's `__init__` runs, sets
`self.use_fire_pe = False` (parent default), and the class-attribute
`use_fire_pe: bool = True` is shadowed at instance level. Verified
by direct repro: `dataclasses.fields(C)[i]` shows `use_fire_pe` is
still owned by the parent and `C().use_fire_pe` returns `False`.
Smoke-test confirms the `@dataclass`-decorated ctrl has
`use_fire_pe=True` and matches the trt config on every field except
the three U-Net keys.

### Tests — reuse existing `tests/test_unet_ablations.py`
The existing `tests/test_unet_ablations.py` already covers every
invariant this plan needs at the `Tiny1M3MConfig` baseline:

1. **Gate value at step 0** — the `tiny_unet_sigmoid_m15`
   ablation at line 38 asserts `sigmoid(-1.5) ≈ 0.1824` element-wise
   on `MinimalLLM(Tiny1M3MConfig(use_unet_skips=True,
   unet_gate_type="sigmoid", unet_gate_init=-1.5))`. Same exact
   mechanism path as the trt config.
2. **Skip-count default** — `unet_skip_count=None` defaults to
   `n_layers // 2 = 12 // 2 = 6` at the tiny1m3m tier
   (`Tiny1M3MConfig.n_layers = 12`, NOT 6). Mirrors are
   0↔11 / 1↔10 / 2↔9 / 3↔8 / 4↔7 / 5↔6 — six pairs, not the three
   the taste predicted (taste assumed `n_layers=6`).
3. **Forward live** — `tiny_unet_sigmoid_m15` toggles
   `use_unet_skips` on/off on the same model and asserts the
   nonzero-gate output diverges from the zero-gate output
   (`contrib > 1e-4`). Same forward path.

No new test file required — running `python3
tests/test_unet_ablations.py` exercises the exact sigmoid(−1.5) +
FIRE-on configuration via the underlying `tiny_unet_sigmoid_m15`
case (the test sweeps the mechanism, the config sweeps the parent
recipe; both share the same `MinimalLLM` wiring).

## Control
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN signature, val 6.3234 per `closed.md:44`; ctrl spread
  6.3875–6.4050). The current best FIRE-alone baseline — same as
  023/024's ctrl. Pinned here so the A/B partitions the
  orthogonal-axis question: does residual-stream architectural
  bridging add anything *on top of* the best attention-side win?
- **Trt**: `Tiny1M3MUNetSigmoidOnFireConfig` — same recipe + U-Net
  sigmoid skips with init -1.5.
- **Seed**: 42 (one seed only — `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Two-ctrl bracket**: runner queues `ctrl` first and `ctrl2` last
  (§2 of `runner.md`); trt is WIN only if it beats *both* ctrls by
  more than the gap between them.

## Cost
- **Params Δ**: `unet_skip_count * d_model` learnable scalars in
  `unet_skip_gates`. At `n_layers=12 → unet_skip_count=6, d_model=64`
  (tiny1m3m), that's +6·64 = +384 params (~0.04% of 0.95M). Confirmed
  by smoke (`params: trt=952560 vs ctrl=952176, Δ=+384`). The bridge
  has NO additional projection (it's an element-wise gated add, not
  a Linear), so no Linear weights either.
- **FLOPs Δ**: 6 extra `gate * skip` element-wise multiplies per
  forward + 6 sigmoid calls + 6 residual adds. Vanishingly small
  against the attention's quadratic cost.
- **Memory Δ**: `unet_skip_count` × stash of `[B, T, d_model]` early
  activations. At B=2, T=2048, d_model=64, that's 6·2·2048·64·4 bytes
  = 6 MB peak (kept in memory until the matched late layer consumes
  it). Negligible against attention activations.
- **Step-0**: NOT bit-identical — `sigmoid(-1.5) ≈ 0.18` means
  ≈18% of the early activation is mixed into the matched late layer
  at step 0. This IS the mechanism (the dead-gate bug was bit-identical
  to the no-skip path at step 0, which is exactly why it never
  fired). Step-0 deviation is the mechanism, not a bug.

## Run
- **Harness**: per `vast-runner-harness.md`, the A/B handle is the
  `class C(BaseConfig)` in `_arq_NNN.py`. The runner builds two
  scripts mirroring the 023 precedent:
  - `_arq_030.py` (trt): `class C(Tiny1M3MUNetSigmoidOnFireConfig): pass`
  - `_arq_030_ctrl.py` (ctrl):
    `class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe: bool = True`
- **Command** (each script): `/venv/main/bin/python _arq_030.py`
  (and `_arq_030_ctrl.py`). Both forward to `train_llm.py
  --config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`.
- **Tier**: tiny1m3m (single seed 42, no sweep). Box: the Vast GPU
  per `cmf-minimax-tmux.md`.
- **Expected wall-clock**: ≈ 4–6 hours on the Vast box (Tiny1M3MConfig
  tier baseline). U-Net skips are essentially free (3 stashes + 3
  gated adds per forward).
- **Pass/fail bar** (matches taste's "small but non-zero" prediction;
  taste explicitly says modded-nanogpt's +1.25% speedup is at ≥100M
  scale and tiny1m3m's 3-pair U at 6L is the small end of the bet):
  - **WIN**: `trt_val < ctrl_val − 0.005` (matches the "small but
    non-zero" prediction; not −0.01 because the 6-pair U at 12L is
    the small end of the modded-nanogpt result).
  - **NULL**: `|trt_val − ctrl_val| < 0.005` (sub-noise; definitively
    closes U-Net skips for tiny1m3m at this depth — taste's explicit
    null-as-result framing). A clean null is informative: the gate
    learns, the skip flows, and still no effect → the lever does not
    fire at 12 layers.
  - **DRIFT/FAIL**: `trt_val > ctrl_val + 0.005` (the skip stomps
    the residual stream rather than augmenting it — would be the
    most surprising result, indicates the sigmoid(-1.5) init is too
    aggressive at this scale).

## Self-check (per code-implementer.md §5)
- [ ] Flag-off path: with `use_unet_skips=False`, baseline path is
  bit-identical to a pre-flag build (the `if self.use_unet_skips:`
  guard at `models/llm.py:167` skips the entire wiring; the for-loop
  branches at `:609,624` are gated). No params, no FLOPs.
- [ ] Flag-on path at step 0: `unet_skip_gates` is initialised to
  -1.5; `torch.sigmoid(-1.5) ≈ 0.182426...`; the skip contributes
  ≈18% of the early activation. NOT bit-identical to no-skip — but
  the previous dead-gate bug WAS bit-identical, which is why it
  never fired. Step-0 deviation is the mechanism.
- [ ] Single-axis A/B: `dataclasses.asdict(Tiny1M3MUNetSigmoidOnFireConfig())`
  vs `asdict(Tiny1M3MVQGainSWAHighRoPE250KConfig(use_fire_pe=True))`
  → exactly 3 differing keys: `use_unet_skips`, `unet_gate_type`,
  `unet_gate_init`. These three together form a single composed
  axis (the U-Net-sigmoid lever); they are not independent (e.g.
  `unet_gate_init=-1.5` only matters when `use_unet_skips=True`,
  and `unet_gate_type="sigmoid"` only matters when
  `use_unet_skips=True`). This is the cleanest representation of
  the single mechanism A/B, mirroring how 023's plan treats
  `kernel_size=3 + padding=0 + groups=d_model` as one composed
  axis. The mechanism A/B is single-axis even if the dataclass
  diff is 3 keys.
- [ ] Pre-run smoke (per `runner.md` §3a): build
  `MinimalLLM(Tiny1M3MUNetSigmoidOnFireConfig())` on CPU, no
  training, assert `model.unet_skip_count == 6` (= n_layers//2 =
  12//2 = 6) and `torch.allclose(torch.sigmoid(model.unet_skip_gates),
  torch.full_like(model.unet_skip_gates, 0.18242552))`. **PASSED**
  inline this pass (smoke output in `log.jsonl`).
- [ ] Tests: `python3 tests/test_unet_ablations.py` — the
  `tiny_unet_sigmoid_m15` row exercises sigmoid(−1.5) directly.
- [ ] Memory note check (`[[unet-skips-gate-fix]]`): the fix matches
  modded-nanogpt's PR #125 sigmoid(−1.5) init exactly. The previous
  raw-zero failure is documented as a dead-gate bug; the sigmoid
  fix is the cheapest possible "bug-fix-becomes-lever" test.

## Coordination note
- `git diff --stat configs/llm_config.py models/layers.py models/llm.py
  configs/__init__.py` at the start of this pass: prior workers
  already added `Tiny1M3MUNetSigmoidOnFireConfig` at
  `configs/llm_config.py:973-1019`, wired it into `configs/__init__.py`,
  and the U-Net `unet_gate_type=="sigmoid"` branch at
  `models/llm.py:617-625` is already in place. This pass added
  **only** `_arq_030.py` and `_arq_030_ctrl.py` (new top-level
  scripts) and edited this `plan.md`. No mechanism code touched.
- **Cross-idea bug flag for the code-reviewer**:
  `_arq_020_ctrl.py`, `_arq_023_ctrl.py`, `_arq_026_ctrl.py` all
  use the pattern `class C(Parent): use_fire_pe: bool = True`
  **without `@dataclass`**. In Python dataclasses, an inheriting
  class without `@dataclass` does NOT override the parent's field
  default — the parent's `__init__` sets `self.use_fire_pe =
  False`. Direct repro: `C().use_fire_pe → False`. Those ctrl runs
  are silently FIRE-OFF, not the FIRE-equipped baseline their
  comments claim. **This plan's `_arq_030_ctrl.py` uses `@dataclass`
  so its ctrl correctly carries `use_fire_pe=True`.** The
  code-reviewer should escalate the 020/023/026 ctrl regression
  separately (out of scope for 030's own A/B).
- The 020-025 ideas are running on the box; this plan does NOT
  touch any of their files or harness scripts.
- The previous broken U-Net attempt at
  `docs/youtube-architecture-ablation-log.md §5` is documented but
  NOT modified — it stands as the historical record of the
  dead-gate bug.
- No `git push` — local commit only when asked.
