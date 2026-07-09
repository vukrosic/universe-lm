# Plan — 029 V-Norm (per-head LayerNorm on V before AV product)

## Flag
- New flag on `LLMConfig`: `use_v_layernorm: bool = False`
  (added next to `use_qk_layernorm` at `configs/llm_config.py:559`).
  Default OFF → baseline path bit-identical when the flag is off.
- Trt config class: `Tiny1M3MVNormOnQKNormConfig(Tiny1M3MQKNormConfig)`
  (added next to `Tiny1M3MQKNormConfig` at `configs/llm_config.py:721-733`)
  with `use_v_layernorm: bool = True`. The A/B differs on the single
  axis `use_v_layernorm`; QK-Norm is the inherited foundation
  (verified by `dataclasses.asdict(trt) vs asdict(ctrl)` → exactly
  1 differing key, `use_v_layernorm`).
- The 029 lever is the **symmetric partner of 016**: same per-head
  LayerNorm mechanism (γ=1, β=0 → identity at step 0), same `d_head`
  axis, separate `nn.LayerNorm(d_k)` module (no weight sharing with
  the existing q_norm/k_norm from 016). The composition test asks:
  does V-side per-head magnitude bounding fire on top of Q,K-side
  per-head logit bounding?

## Change

**~10 LoC: one flag + ~5 LoC of wiring in MHA + one new config class
+ one new harness script pair.** No new module file. The mechanism
mirrors the existing `use_qk_layernorm` override pattern at
`models/layers.py:670-672` and the existing `v_norm` site at `:1530-1531`.

### `configs/llm_config.py`

Add the flag, next to `use_qk_layernorm` (`:559`):

```python
# 029 — V-Norm (Wortsman et al. 2023, arXiv:2309.14322): per-head
# LayerNorm on V along `d_head` before the AV product, symmetric
# partner of 016's QK-Norm. Bounds per-head V vector magnitude so
# outlier V entries do not dominate the AV aggregation. The override
# is OR'd with the global `use_layernorm` so either flag flips the
# V norm to LayerNorm; v_norm_type still routes invented norms
# (pnorm1.5 etc.) for the closed #92 lever. Default off → baseline
# path bit-identical (no v_norm module built at all). See
# `autoresearch/ideas/029-v-norm/plan.md`.
use_v_layernorm: bool = False
```

Add the trt config class, next to `Tiny1M3MQKNormConfig` (`:721-733`):

```python
@dataclass
class Tiny1M3MVNormOnQKNormConfig(Tiny1M3MQKNormConfig):
    """Tiny1M3M with QK-Norm + V-Norm (per-head LayerNorm on V).

    A/B vs the QK-Norm ctrl (`Tiny1M3MQKNormConfig`, the 016 WIN
    signature val 6.3906 per `closed.md:33`). Parent is the
    QK-Norm-equipped baseline so the QK-Norm carries over identically
    — the A/B isolates the V-Norm swap (per-head LayerNorm on V along
    `d_head`) on top of the same QK-Norm foundation, not silent HP
    drift. Symmetric to 016: applies `nn.LayerNorm(d_head)` (γ=1, β=0
    init → identity at step 0) on V before the AV product. Independent
    `v_norm` module (no weight sharing with q_norm/k_norm).

    The bet: the same "bound-the-per-head-magnitude" mechanism that
    helped Q and K in 016 also acts on V — outlier V entries
    destabilise the AV output and the residual stream, and bounding
    V at the head-dim level is the symmetric fix. A null partitions
    016's win between logit-bounding (Q,K side) and output-value-
    bounding (V side) — informative regardless. The Wortsman 2023
    ViT/BERT diagnostic carries the mechanism at scale; LM-pretraining
    record is thinner than QK-Norm's Qwen3/SmolLM3 adoption (taste
    transfer-risk tag: med, mitigated by anchoring to 016's already-
    validated per-head-site).

    PASS ≤ −0.005 vs the QK-Norm ctrl (matches 016's plan bar — the
    bet is at the low end of what V-Norm *could* do per taste's
    caveat that Wortsman used it as a diagnostic, not a primary
    lever). NULL band |Δ| < 0.005. DRIFT > +0.005. See
    `autoresearch/ideas/029-v-norm/plan.md`.
    """
    use_v_layernorm: bool = True
```

### `configs/__init__.py`
Add `Tiny1M3MVNormOnQKNormConfig` to the import block and `__all__`
list, mirroring the `Tiny1M3MQKNormConfig` wiring.

### `models/layers.py` — wiring (~6 LoC)

**`MultiHeadAttention.__init__`** — add the kwarg next to
`use_qk_layernorm` (`:554`):

```python
use_v_layernorm: bool = False,
```

**`MultiHeadAttention.__init__`** — after the existing v_norm block
at `:676-678`:

```python
self.use_v_norm = v_norm_type not in ("", "none", None)
if self.use_v_norm:
    self.v_norm = make_norm(self.d_k, v_norm_type, self.use_layernorm)
# 029 — V-Norm override: if use_v_layernorm is on (and the closed-#92
# v_norm_type override is off), build a per-head LayerNorm(d_head)
# on V mirroring the use_qk_layernorm override at `:670-672`. Identity
# at step 0 via γ=1, β=0 default init. The two flags compose: when
# both v_norm_type and use_v_layernorm are set, use_v_layernorm wins
# (LayerNorm overrides the invented norm — explicit > implicit).
elif use_v_layernorm:
    self.use_v_norm = True
    self.v_norm = nn.LayerNorm(self.d_k)
```

The forward at `:1530-1531` requires no change — it already gates
on `self.use_v_norm`.

**`TransformerBlock.__init__`** — thread the new kwarg through
(mirror the `use_qk_layernorm` plumbing). Pass it from `LLMConfig`
via `getattr` in `models/llm.py`.

### `models/llm.py` — wiring (~2 LoC)

Add `self.use_v_layernorm = getattr(config, "use_v_layernorm", False)`
next to the `use_qk_layernorm` plumbing (around `:212`), and pass
`use_v_layernorm=self.use_v_layernorm` to the `TransformerBlock`
constructor in the list at `:~308-397`.

### `tests/test_v_norm.py` (new, ~60 LoC)

Five invariants mirror 016's test surface plus the spec's symmetry
test:

1. **No NaN/Inf** on a non-trivial random input through MHA.
2. **Step-0 identity** — `use_v_layernorm=True` (γ=1, β=0 default)
   produces a forward output within `1e-4` of `use_v_layernorm=False`
   on a freshly-init MHA. (LN with γ=1, β=0 is per-row centering +
   re-scaling; not bit-identical to the no-LN path, but the spec
   accepts the mechanism's centering effect as ≤ a few `1e-2` per
   token — assert `<= 5e-2` per logit.)
3. **Wiring live** — perturbing the LN γ on one channel changes only
   that channel's V output (the depthwise / per-head property).
4. **Composition with QK-Norm** — `use_qk_layernorm=True` AND
   `use_v_layernorm=True` builds three independent LayerNorms; no
   weight sharing (`id(self.q_norm) != id(self.v_norm)` and
   `id(self.k_norm) != id(self.v_norm)`).
5. **Closed-#92 precedence** — when both `v_norm_type="pnorm1.5"`
   AND `use_v_layernorm=True` are set, the existing v_norm_type
   wins (it was set first; the elif branch never fires). Document
   this in the flag comment.

### Harness scripts (`_arq_029.py` + `_arq_029_ctrl.py`)
Mirror the 023 precedent at the repo root:

```python
# _arq_029.py — trt: QK-Norm + V-Norm on QK-Norm ctrl
import sys
from configs.llm_config import Tiny1M3MVNormOnQKNormConfig
class C(Tiny1M3MVNormOnQKNormConfig):
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
# _arq_029_ctrl.py — ctrl: QK-Norm baseline (016 WIN signature)
import sys
from configs.llm_config import Tiny1M3MQKNormConfig
class C(Tiny1M3MQKNormConfig):
    pass
if __name__ == "__main__":
    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = ["train_llm.py", "--config_class", "__main__.C",
                "--seed", "42", "--dataset_path", "processed_data/pretrain_1B",
                "--warmup", "false"]
    train_llm.main()
```

## Control
- **Ctrl**: `Tiny1M3MQKNormConfig` (the 016 WIN signature; QK-Norm
  alone on plain tiny1m3m, val 6.3906 per `closed.md:33`). Pinned
  to the QK-Norm-equipped baseline so the A/B partitions the question
  "does V-Norm fire on top of QK-Norm at this scale?" cleanly. The
  taste explicitly anchors the transfer-risk argument to "composes
  with 016" rather than to "needs a new adoption story."
- **Trt**: `Tiny1M3MVNormOnQKNormConfig` — same config + V-Norm-LayerNorm.
- **Seed**: 42 (one seed only — `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Two-ctrl bracket**: runner queues `ctrl` first and `ctrl2` last
  (§2 of `runner.md`); trt is WIN only if it beats *both* ctrls by
  more than the gap between them.

## Cost
- **Params Δ**: +1 · n_layers · 2·d_head LayerNorm γ+β params per
  V LayerNorm. At `n_layers=12, d_head=16` (tiny1m3m, d_model=64
  / n_heads=4 → d_head=16), that's +12·2·16 = +384 params (~0.04%
  of 0.94M). Negligible.
- **FLOPs Δ**: one LayerNorm per token per head per layer = one
  reduction + mul/add along d_head. < 0.01% of total model FLOPs.
- **Memory Δ**: +2·d_head per layer (LN gain+bias). Negligible.
- **Wall-clock Δ**: ~0% (LayerNorm vs no-op is invisible against
  attention's quadratic cost).
- **Step-0**: `nn.LayerNorm(d_k)` with γ=1, β=0 → identity at step 0
  (modulo the LN's centering+re-scaling, which is the mechanism).

## Run
- **Harness**: per `vast-runner-harness.md`, the A/B handle is the
  `class C(BaseConfig)` in `_arq_NNN.py`. The runner builds two
  scripts mirroring the 023 precedent:
  - `_arq_029.py` (trt): `class C(Tiny1M3MVNormOnQKNormConfig): pass`
  - `_arq_029_ctrl.py` (ctrl): `class C(Tiny1M3MQKNormConfig): pass`
- **Command** (each script): `/venv/main/bin/python _arq_029.py`
  (and `_arq_029_ctrl.py`). Both forward to `train_llm.py --config_class
  __main__.C --seed 42 --dataset_path processed_data/pretrain_1B
  --warmup false`.
- **Tier**: tiny1m3m (single seed 42, no sweep). Box: the Vast GPU
  per `cmf-minimax-tmux.md`.
- **Expected wall-clock**: ≈ 4–6 hours on the Vast box (Tiny1M3MConfig
  tier baseline). V-Norm is asymptotically free (one LN per token
  per head, same as QK-Norm).
- **Pass/fail bar** (matches 016's plan bar per taste's caveat:
  taste says "the pass bar should match the 016 plan bar (~-0.005
  to -0.01)"; the bet is at the low end of what V-Norm could do):
  - **WIN**: `trt_val < ctrl_val − 0.005` (matches 016's bar; the
    bet is at the low end of the hypothesis range for the symmetric
    partner). Expected Δ ≈ −0.005 to −0.015.
  - **NULL**: `|trt_val − ctrl_val| < 0.005` (sub-noise; V-side
    bounding does not fire on top of Q,K-side bounding at this
    scale — informative: partitions 016's win to the *logit*-bounding
    half, ruling out the *value-output*-bounding half).
  - **DRIFT/FAIL**: `trt_val > ctrl_val + 0.005` (V-Norm interferes
    with the AV product rather than stabilising it — would be the
    most surprising result).

## Self-check (per code-implementer.md §5)
- [ ] Flag-off path: with `use_v_layernorm=False` AND `v_norm_type=""`,
  baseline path is bit-identical to a pre-flag build (the elif branch
  never fires, no v_norm module is built, no extra params or FLOPs).
- [ ] Flag-on path at step 0: `nn.LayerNorm(d_k)` with γ=1, β=0 is
  per-row centering + re-scaling — not bit-identical to the no-LN
  path but the deviation per logit is the mechanism (≤ a few `1e-2`
  per token from the centering). Run one CPU forward on
  `MinimalLLM(trt)` vs `MinimalLLM(ctrl)` and confirm the per-token
  logit drift sits in the expected band (LN's centering effect).
- [ ] Independent module: `id(self.q_norm) != id(self.v_norm)` and
  `id(self.k_norm) != id(self.v_norm)` — no weight sharing. Spec's
  explicit requirement at `idea.md:23`.
- [ ] Single-axis A/B: `dataclasses.asdict(Tiny1M3MVNormOnQKNormConfig())`
  vs `asdict(Tiny1M3MQKNormConfig())` → exactly 1 differing key
  (`use_v_layernorm`). Mirror the 023-canon r1→r2 single-axis
  discipline.
- [ ] Pre-run smoke (per `runner.md` §3a): build `MinimalLLM(
  Tiny1M3MVNormOnQKNormConfig())` on CPU, no training, confirm no
  AttributeError (the `use_v_layernorm` flag must thread through
  `TransformerBlock.__init__` → `MultiHeadAttention.__init__` —
  catches a flag added to the dataclass but not threaded through,
  the 009-fire-pe regression).
- [ ] Tests: `pytest tests/test_v_norm.py -v` passes 5/5.

## Coordination note
- `git diff configs/llm_config.py models/layers.py models/llm.py`
  shows no in-flight edits in those files at the start of this pass.
- This plan adds **one new flag** (`use_v_layernorm` at
  `configs/llm_config.py:~559`), **one new dataclass** at
  `configs/llm_config.py:~735`, **one new kwarg** + **~6 LoC of
  wiring** in `MultiHeadAttention.__init__` (next to `use_qk_layernorm`
  at `:670-678`), **one new test file** (`tests/test_v_norm.py`),
  and two new top-level scripts (`_arq_029.py`, `_arq_029_ctrl.py`).
- All edits are strictly additive (new flag, new branch, new module);
  no rewrite of existing mechanism code. The closed #92 v_norm_type
  lever still works (the elif branch only fires when v_norm_type is
  off and use_v_layernorm is on; precedence is documented in the
  flag comment).
- The 020-025 ideas are running on the box; per the user's pin,
  this plan does NOT touch any of their files or harness scripts.
- No `git push` — local commit only when asked.
