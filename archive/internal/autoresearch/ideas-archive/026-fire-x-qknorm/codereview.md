# Code-review log — 026 FIRE × QK-Norm

## r2 — 2026-06-10 — verdict: accept

F1 from r1 fixed at `_arq_026_ctrl.py:3-9`: `from dataclasses import dataclass` imported, `@dataclass` decorator now applied to `class C(Tiny1M3MVQGainSWAHighRoPE250KConfig)` with `use_fire_pe: bool = True`. Verified end-to-end on disk:

```bash
$ python3 -c "import sys, dataclasses; sys.path.insert(0, '.'); import _arq_026, _arq_026_ctrl; \
    da = dataclasses.asdict(_arq_026.C()); db = dataclasses.asdict(_arq_026_ctrl.C()); \
    print({k: (db[k], da[k]) for k in da if da[k] != db[k]})"
{'use_qk_layernorm': (False, True)}
$ python3 -c "import _arq_026_ctrl; c = _arq_026_ctrl.C(); \
    print('ctrl FIRE on:', c.use_fire_pe, 'QK-Norm off:', not c.use_qk_layernorm)"
ctrl FIRE on: True QK-Norm off: True
```

Single-axis A/B confirmed: **exactly 1 differing key** (`use_qk_layernorm`); ctrl instance has `use_fire_pe=True` (no longer shadowed by parent's `__init__`). The bug pattern flagged in r1 F1 (out-of-scope note for `_arq_020/023/...`) remains for those other scripts; that's the parallel-Claude's concern — out of scope here.

Re-confirmed from r1 (all still hold):
- **Mechanism faithful** — composition is "LayerNorm Q,K (`models/layers.py:1457-1458`) → RoPE (same line) → `scores = QK^T/√d_k` (`:1596`) → add FIRE bias (`:1597-1598`) → mask → softmax → @V". Matches `idea.md:17` spec. No new mechanism code; rides existing 009 + 016 implementations.
- **Identity at step 0** — `nn.LayerNorm(d_head)` init γ=1, β=0; FIRE bias init 0; step-0 logit drift = 9.5e-3 per implementer's CPU smoke (LN centering on Q,K, exactly the mechanism — not a bug).
- **No silent HP drift** — `dataclasses.asdict` diff: 1 key. VQGain+SWA+RoPE+FIRE all carry from parent untouched.
- **Flag default OFF in parent** — `LLMConfig.use_qk_layernorm: bool = False` (`configs/llm_config.py:559`); trt and ctrl flip explicitly.
- **LoC budget** — ~40 LoC across config + `__init__.py` exports + 2 harness scripts. Well under 200.
- **Plan ↔ idea consistency** — seed 42; tier tiny1m3m; ctrl = FIRE-equipped 009 WIN signature; PASS ≤ −0.01.
- **Coordination clean** — `git diff configs/llm_config.py models/layers.py models/llm.py`: only the new `Tiny1M3MQKNormOnFireConfig` block at `:1131-1167` + the `configs/__init__.py` export pair (lines 19, 116). No edits to `models/layers.py`, `models/llm.py`, or `optimizers/` from this idea. Parallel workers' in-flight wiring untouched.

Ready to run.

## r1 — 2026-06-10 — verdict: revise

Mechanism (composition): faithful — both flags already wired; trt is a pure two-flag enable on top of the existing 009 (FIRE) and 016 (QK-Norm) implementations. No new mechanism code. `models/layers.py:705-707` builds `q_norm`/`k_norm` as `nn.LayerNorm(d_head)` when `use_qk_layernorm=True` (γ=1, β=0 default → identity at step 0); `:1457-1458` applies it pre-RoPE on the standard path, `:1451-1452` on the NoPE/CoPE path. The FIRE branch at `:1591-1598` reuses the same `Q`/`K` produced after `q_norm`/`k_norm`, computes `scores = QK^T / √d_k` and adds the FIRE bias — composition is "LayerNorm Q,K → dot product → add FIRE bias → mask → softmax → @V", exactly the spec at `idea.md:17`.

Trt config (`configs/llm_config.py:1128-1163`): clean. `class Tiny1M3MQKNormOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig)` is `@dataclass`-decorated, so `use_fire_pe: bool = True` and `use_qk_layernorm: bool = True` are real field defaults. Single-axis A/B verified by `dataclasses.asdict` diff vs `Tiny1M3MVQGainSWAHighRoPE250KConfig(use_fire_pe=True)` → exactly 1 differing key (`use_qk_layernorm`); all 4 VQGain+SWA+RoPE axes carry over identically.

LoC: ~40 lines (one new config class + one `__init__.py` export pair + two harness scripts). Well under the 200 ceiling. No model/optimizer changes.

Coordination: clean. The diff only adds a new config class at `configs/llm_config.py:1128-1163` and two `__init__.py` lines; no edits to `models/layers.py` / `models/llm.py` / `optimizers/`. Parallel workers' in-flight wiring is untouched.

### Findings (must fix before run)

- **F1 — ctrl script silently drops `use_fire_pe`, invalidating the A/B** (`_arq_026_ctrl.py:6-7`). The ctrl class

  ```python
  class C(Tiny1M3MVQGainSWAHighRoPE250KConfig):
      use_fire_pe: bool = True
  ```

  is **not** decorated with `@dataclass`. Python's dataclass inheritance generates `__init__` once at parent decoration; an undecorated subclass inherits that `__init__` verbatim, which uses the parent's field defaults (`use_fire_pe=False` from `LLMConfig:150`). The class-level annotation in `C`'s body creates a class attribute, but `__init__` sets `self.use_fire_pe = False` on the instance, **shadowing** it. Verified end-to-end:

  ```python
  >>> class C(Tiny1M3MVQGainSWAHighRoPE250KConfig):
  ...     use_fire_pe: bool = True
  >>> C.use_fire_pe                 # True  (class attr)
  >>> C().use_fire_pe               # False (instance attr from __init__)
  >>> dataclasses.fields(C)[…].default  # False
  ```

  `train_llm.py:326` calls `ConfigClass()` and then reads `config.use_fire_pe` — it gets **False**. The ctrl will run **without FIRE**. The A/B as authored is therefore:

  ```
  trt : Tiny1M3MVQGainSWAHighRoPE250K + FIRE + QK-Norm  (correct)
  ctrl: Tiny1M3MVQGainSWAHighRoPE250K                   (NO FIRE, NO QK-Norm)  ← bug
  ```

  i.e. a multi-axis A/B that conflates "add FIRE" (~−0.064 by itself per closed.md:44) and "add QK-Norm" (~−0.014). The expected ~−0.078 Δ would fire spuriously from FIRE alone — the QK-Norm × FIRE composition question would be unmeasured. This is the **same wiring bug** already documented at `autoresearch/queue.md:156` ("subclass override silently dropped `use_fire_pe=False`; 4 -sh reruns all produced identical 6.3419 to 4 dp → confirms bug, not noise") for the prior day's 021/023/024/025 shared FIRE ctrls.

  **Fix** (one-line + one-import in `_arq_026_ctrl.py`):

  ```python
  """Autoresearch 026 — ctrl: FIRE-equipped 009 WIN signature."""
  import sys
  from dataclasses import dataclass
  from configs.llm_config import Tiny1M3MVQGainSWAHighRoPE250KConfig


  @dataclass
  class C(Tiny1M3MVQGainSWAHighRoPE250KConfig):
      use_fire_pe: bool = True
  …  # rest unchanged
  ```

  Re-verify after the fix:

  ```bash
  python3 -c "
  import sys; sys.path.insert(0, '.')
  import _arq_026_ctrl
  c = _arq_026_ctrl.C()
  assert c.use_fire_pe is True, c.use_fire_pe
  assert c.use_qk_layernorm is False, c.use_qk_layernorm
  print('ctrl ok: FIRE on, QK-Norm off')
  "
  ```

  Then re-check the single-axis A/B vs trt:

  ```bash
  python3 -c "
  import sys, dataclasses; sys.path.insert(0, '.')
  import _arq_026, _arq_026_ctrl
  da = dataclasses.asdict(_arq_026.C())
  db = dataclasses.asdict(_arq_026_ctrl.C())
  diff = {k: (db[k], da[k]) for k in da if da[k] != db[k]}
  assert diff == {'use_qk_layernorm': (False, True)}, diff
  print('single-axis A/B ok:', diff)
  "
  ```

  (Out of scope but worth flagging upstream: the **same bug pattern is present in `_arq_020_ctrl.py`, `_arq_023_ctrl.py`, and every other `class C(…BaseConfig): use_fire_pe: bool = True` ctrl script**. The 023 r2 codereview claimed verification of this pattern — that was wrong; the verification was a dataclass diff against the trt's *config class*, not against the ctrl script's *runtime instance*. Future ideas using a FIRE-equipped ctrl must use the `@dataclass`-decorated form.)

After F1 lands, this is `accept` — trt is correctly configured, mechanism composes cleanly, identity-init holds at step 0, single-axis A/B confirmed on the trt side, LoC tiny, no coordination conflicts.
