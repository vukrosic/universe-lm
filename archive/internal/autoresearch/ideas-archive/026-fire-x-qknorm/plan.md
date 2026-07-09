# Plan — 026 FIRE × QK-Norm (stack two attention-domain wins)

## Flag
- Both flags already live in `LLMConfig`:
  - `use_fire_pe: bool = False` (`configs/llm_config.py:144`)
  - `use_qk_layernorm: bool = False` (`configs/llm_config.py:559`)
  - Default OFF — baseline path bit-identical when both flags are off.
- Treatment is **two-flag composition**, no new mechanism code. The
  treatment config flips both to `True`; the ctrl flips only `use_fire_pe`.
- New trt config class: `Tiny1M3MQKNormOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig)`
  with `use_fire_pe: bool = True` and `use_qk_layernorm: bool = True`
  (added next to `Tiny1M3MCanonOnFireConfig` at `configs/llm_config.py:911-939`).
  Parent class is `Tiny1M3MVQGainSWAHighRoPE250KConfig` so the
  V-embed + q-gain + SWA(512) + RoPE 250K recipe carries over from the
  ctrl identically — the A/B differs on the single axis `use_qk_layernorm`
  (verified by `dataclasses.asdict(trt) vs asdict(ctrl_with_use_fire_pe=True)`
  → exactly 1 differing key).

## Change

**No new module file. Pure config composition.** All mechanism code
already exists from 009 (FIRE) and 016 (QK-Norm).

### `configs/llm_config.py`
Add `Tiny1M3MQKNormOnFireConfig` after `Tiny1M3MCanonOnFireConfig`
(at `:911-939`), mirroring the `Tiny1M3MSoftpickOnFireConfig`
(`:855-882`) / `Tiny1M3MCanonOnFireConfig` pattern:

```python
@dataclass
class Tiny1M3MQKNormOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + QK-Norm (LayerNorm on Q,K head-dim).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates the
    QK-Norm swap (per-head LayerNorm on Q,K along `d_head`) on top
    of the same FIRE-equipped foundation, not silent HP drift. The
    treatment stacks `use_qk_layernorm=True` on top: bounds the
    per-head logit `Q·K/√d_head` to `|·| ≤ √d_head`. Categorically
    distinct from FIRE — FIRE is *additive* (bias added to logits
    post-dot-product); QK-Norm is *multiplicative-normalizing*
    (LayerNorm bounds the dot-product magnitude that the bias gets
    added to). The two operate at different points and on different
    mathematical axes — independence claim verified by mechanism, not
    by data (since this is the first composition test of the two).

    The 013-CoPE DRIFT (+0.143 vs FIRE-alone, `closed.md:31`) is the
    relevant prior, but 013 failed by stacking *two additive position
    bias levers* — QK-Norm does not compound additively with FIRE's
    bias; it bounds the magnitude of the Q·K product. This is the
    qualitative difference that makes 026 a different bet from 013.

    Expected: additive (~−0.078 vs FIRE-alone, computed as 009's
    −0.064 + 016's −0.014). Superadditive (~−0.09+) would mean the
    per-head logit bounding makes FIRE's learned position bias more
    consistent across heads. A null or regression would mean the
    013-CoPE precedent generalises — attention-domain headroom is
    exhausted by FIRE at this scale.

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| ≤ 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/026-fire-x-qknorm/plan.md`.
    """
    use_fire_pe: bool = True
    use_qk_layernorm: bool = True
```

### `configs/__init__.py`
Add `Tiny1M3MQKNormOnFireConfig` to the import block and `__all__`
list, mirroring the `Tiny1M3MCanonOnFireConfig` wiring.

### `models/layers.py` — **NO CHANGES**
- `use_fire_pe` is already wired through `TransformerBlock.__init__` →
  `MultiHeadAttention.__init__` and into the manual attention branch
  at `:1547+` (the FIRE bias is added post-`Q·K^T/√d_k` and
  pre-softmax).
- `use_qk_layernorm` is already wired through `MultiHeadAttention.__init__`
  at `:554`. The override at `:670-672` builds `q_norm`/`k_norm` as
  `nn.LayerNorm(d_head)` when the flag is on (identity at step 0 via
  γ=1, β=0 default init). Applied in the forward at `:1407-1408`
  BEFORE the dot product (pre-RoPE for the FIRE branch, see `:1413-1414`
  for the non-FIRE branch).
- The two flags compose cleanly: in the FIRE branch (`:1547+`), Q,K
  are LayerNorm'd first (`:1407-1408`), the manual `scores = Q@K^T/√d_k`
  is computed, then the FIRE bias is added. The order is "norm
  before dot product" then "add positional bias post-dot-product" —
  matches the spec.

### `models/llm.py` — **NO CHANGES**
- `use_qk_layernorm` is plumbed through at `:212` (or similar via
  `getattr`) into the `TransformerBlock` constructor — already in
  place from 016.
- `use_fire_pe` is plumbed analogously from 009.

### Harness scripts (`_arq_026.py` + `_arq_026_ctrl.py`)
Mirror the 020/023 precedent at the repo root:

```python
# _arq_026.py — trt: FIRE + QK-Norm on the FIRE-equipped baseline
import sys
from configs.llm_config import Tiny1M3MQKNormOnFireConfig
class C(Tiny1M3MQKNormOnFireConfig):
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
# _arq_026_ctrl.py — ctrl (FIRE-equipped 009 WIN signature)
import sys
from dataclasses import dataclass
from configs.llm_config import Tiny1M3MVQGainSWAHighRoPE250KConfig

@dataclass
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

## Control
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN signature, val 6.3234 per `closed.md:44`; ctrl spread
  6.3875–6.4050 per `closed.md:41-44`). This is the *FIRE-alone*
  baseline — the bigger of the two single-lever wins being stacked.
- **Trt**: `Tiny1M3MQKNormOnFireConfig` — same recipe + `use_qk_layernorm=True`.
- **Seed**: 42 (one seed only — `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Two-ctrl bracket**: runner queues `ctrl` first and `ctrl2` last
  (§2 of `runner.md`); trt is WIN only if it beats *both* ctrls by
  more than the gap between them.

## Cost
- **Params Δ**: +2 · n_layers · d_head LayerNorm γ/β params per
  Q,K LayerNorm. At `n_layers=12, d_head` ≈ d_model/n_heads, that's
  + a few hundred params (~0.05% of 0.94M). Negligible.
- **FLOPs Δ**: LayerNorm and RMSNorm have the same asymptotic cost
  (one reduction + one mul/add per element). +1 extra add per element
  for the β bias — visible only as a fraction of a percent on the
  attention sublayer wall-clock.
- **Memory Δ**: +2·d_head per layer in LayerNorm gain+bias. Negligible.
- **Step-0**: both LayerNorms init γ=1, β=0 → identity at step 0.
  FIRE bias init = 0 (009's identity-init). Step-0 forward is
  bit-identical to a plain `Tiny1M3MVQGainSWAHighRoPE250KConfig` run
  (modulo the FIRE manual-path numerical drift documented in
  `009-fire-pe/plan.md:62`).

## Run
- **Harness**: per `vast-runner-harness.md`, the A/B handle is the
  `class C(BaseConfig)` in `_arq_NNN.py`, not a CLI flag. The runner
  builds two scripts mirroring the 020/023 precedent:
  - `_arq_026.py` (trt): `class C(Tiny1M3MQKNormOnFireConfig): pass`
  - `_arq_026_ctrl.py` (ctrl):
    `@dataclass class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe: bool = True`
    (the `@dataclass` decorator is load-bearing — without it the
    parent's generated `__init__` runs verbatim with `use_fire_pe=False`,
    shadowing the class-attr override at instance time; fix from r1
    codereview F1.)
- **Command** (each script): `/venv/main/bin/python _arq_026.py` (and
  `_arq_026_ctrl.py`). Both forward to `train_llm.py --config_class
  __main__.C --seed 42 --dataset_path processed_data/pretrain_1B
  --warmup false`.
- **Tier**: tiny1m3m (single seed 42, no sweep). Box: the Vast GPU
  per `cmf-minimax-tmux.md`.
- **Expected wall-clock**: ≈ 4–6 hours on the Vast box (Tiny1M3MConfig
  tier baseline). QK-Norm has zero extra cost; FIRE adds the
  documented ~1.6 GFLOPs/step from the manual attention path
  (`009-fire-pe/plan.md:29`).
- **Pass/fail bar** (copied from taste's hypothesis range, tile
  without overlap):
  - **WIN**: `trt_val < ctrl_val − 0.01` (clears the cited noise
    floor ~0.01; trt must be strictly better by more than the box
    spread). Expected additive Δ ≈ −0.078 (009's −0.064 + 016's
    −0.014, since 016 won by −0.014 on the *plain* tiny1m3m ctrl
    and FIRE is now the baseline) — so WIN should fire comfortably
    if the two levers are orthogonal.
  - **NULL**: `|trt_val − ctrl_val| ≤ 0.01` (sub-noise; the lever
    does not fire on top of FIRE at this scale — attention-domain
    headroom is exhausted, generalising the 013-CoPE precedent).
  - **DRIFT/FAIL**: `trt_val > ctrl_val + 0.01` (the per-head logit
    bounding interferes with FIRE's learned position bias rather
    than refining it — concrete generalisation of the 013-CoPE
    failure to a non-additive lever).

## Self-check (per code-implementer.md §5)
- [ ] Flag-off path: with `use_qk_layernorm=False` AND `use_fire_pe=False`,
  baseline path is bit-identical to a pre-flag build (both 009 and
  016 verified this individually; the composition is the same
  default-off path).
- [ ] Flag-on path at step 0: `nn.LayerNorm(d_head)` with γ=1, β=0
  is identity at step 0; FIRE bias init 0; so step-0 forward only
  differs from the FIRE-alone ctrl by the LayerNorm's reduction +
  re-scale (which is per-row centering on Q,K — already in 016's
  plan as the mechanism). Run one CPU forward on a `MinimalLLM(trt)`
  vs `MinimalLLM(ctrl_with_use_fire_pe=True)` and check the per-token
  logit drift is ≤ a few `1e-2` (the LN's centering effect, not zero
  because LN is the mechanism).
- [ ] Single-axis A/B: `dataclasses.asdict(Tiny1M3MQKNormOnFireConfig())`
  vs `asdict(Tiny1M3MVQGainSWAHighRoPE250KConfig(use_fire_pe=True))`
  → exactly 1 differing key (`use_qk_layernorm`). The parent-class
  fix from 023's r1 codereview (`configs/llm_config.py:912`) is the
  template — same single-axis discipline.
- [ ] Pre-run smoke (per `runner.md` §3a "BOX REALITY"): build
  `MinimalLLM(Tiny1M3MQKNormOnFireConfig())` on CPU, no training,
  confirm no AttributeError (catches a flag added to the dataclass
  but not threaded through — bit us on 009-fire-pe per
  `vast-runner-harness.md`).

## Coordination note
- `git diff configs/llm_config.py models/layers.py models/llm.py`
  shows no in-flight edits in those files at the start of this pass
  (only `autoresearch/ideas/023-canon-conv/plan.md` and friends are
  modified — those are documentation, not shared code).
- This plan adds **only** a new dataclass at
  `configs/llm_config.py:~940` and two new top-level scripts
  (`_arq_026.py`, `_arq_026_ctrl.py`). No mechanism code added —
  the composition rides on the existing 009 and 016 implementations.
- The 020-025 ideas are running on the box; per the user's pin,
  this plan does NOT touch any of their files or harness scripts.
- No `git push` — local commit only when asked.
