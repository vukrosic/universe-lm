# Plan — 028 Deep-and-Thin Config (depth/width swap at fixed ~0.94M budget)

## Flag

**No new flag added.** 028 is a *structural* config swap (architecture
field overrides on a new dataclass), not a behavioural lever. Adding a
`use_deep_thin: bool` would be theatre — every existing config field
(`d_model, n_heads, n_kv_heads, n_layers, d_ff`) is already a dataclass
knob; the A/B is simply two configs with different values for those
knobs. The "default-OFF, bit-identical baseline" guarantee is preserved
at the *config-class* level: the baseline `Tiny1M3MConfig` is untouched
(diff is purely additive — one new subclass), so any existing call that
constructs `Tiny1M3MConfig` is bit-identical pre- and post-diff.

- **Ctrl class:** `Tiny1M3MConfig` (unchanged, `configs/llm_config.py:665`).
- **Trt class:** new `Tiny1M3MDeepThinConfig(Tiny1M3MConfig)` added after
  `Tiny1M3MQKNormOnFireConfig` (around `configs/llm_config.py:1168`).

The trt overrides exactly the 5 architecture fields and inherits every
non-architectural field from `Tiny1M3MConfig`:

| field         | ctrl (`Tiny1M3MConfig`) | trt (`Tiny1M3MDeepThinConfig`) |
|---------------|-------------------------|--------------------------------|
| `d_model`     | 64                      | **48**                         |
| `n_heads`     | 4                       | **3**                          |
| `n_kv_heads`  | 2                       | **3** (MHA-tied — see confound) |
| `n_layers`    | 12                      | **20**                         |
| `d_ff`        | 256                     | **192** (= 4·d_model)          |

`d_head = d_model / n_heads = 16` preserved (was 64/4=16, now 48/3=16).
`emb_rank=8`, `ffn_variant="squared_relu"`, `vocab_size=49152` all
inherited unchanged. Non-architectural fields frozen (see §Frozen below).

## Change

**Diff surface: `configs/llm_config.py` only (one new dataclass) +
`configs/__init__.py` (export) + one new test file.** No
`models/layers.py`, no `models/llm.py`, no `models/fox.py`, no
`models/fire_pe.py`. The parallel-AI coordination memo
(`MEMORY.md`: `project-parallel-ai`) is a non-issue — confirmed by
`git diff` at the start of the planning pass (existing unstaged changes
in `models/layers.py`/`configs/llm_config.py` are 029 V-Norm
additions that do not touch the Tiny1M3M baseline fields we override).

### `configs/llm_config.py` — new dataclass (after line 1168)

```python
@dataclass
class Tiny1M3MDeepThinConfig(Tiny1M3MConfig):
    """Tiny1M3M deep-and-thin: depth/width swap at fixed ~0.94M budget.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306
    per `LEADERBOARD.md` row 14). The treatment reallocates the 0.94M
    budget across more, thinner layers: n_layers 12→20 (1.67×),
    d_model 64→48 (0.75×), d_ff 256→192 (= 4·d_model preserved),
    n_heads/n_kv_heads 4/2 → 3/3 (MHA-tied — see confound below).
    Per-head `d_head = 16` preserved (was 64/4, now 48/3); `emb_rank=8`,
    `ffn_variant="squared_relu"`, `vocab_size=49152` all inherited
    unchanged from `Tiny1M3MConfig`. Param budget arithmetic:
    per-block attn 9.2k + FFN 18.4k + norms 0.24k ≈ 27.9k; ×20 = 558k
    + embedding factorisation 393.6k ≈ 951k (+1.3% vs baseline 939k,
    inside the ±5% ceiling). MobileLLM (Ma et al., ICML 2024,
    arXiv:2402.14905) reports +2.7% / +4.3% on zero-shot benchmarks
    at 125M / 350M from this exact depth/width swap; the open question
    is whether the lever still fires at 0.94M (133× smaller than the
    paper's smallest ablation).

    **Known confound (see `idea.md:50-55`).** Baseline is GQA 2:1
    (n_heads=4, n_kv_heads=2). The depth/width swap also collapses
    kv-sharing → MHA (n_heads=n_kv_heads=3). Tied-QK / full-MHA is a
    known WIN signature at tiny1m3m (`LEADERBOARD.md` row 0 = vq-gain
    + rope250k + swa384 + tiedqk, val 6.3041) — the trt Δ partly
    reflects the kv-sharing collapse, not pure depth/width. Picked B1
    over B1' (MQA n_kv_heads=1) and B2 (d_model=32, d_ff off the
    4·d_model rule) because the `d_ff = 4·d_model` convention is more
    load-bearing for "pure depth/width swap" than the GQA ratio.
    Runner reports the confound alongside the raw val-loss Δ.

    PASS ≤ ctrl − 0.01 (clears the cited ±0.01 box-noise floor).
    NULL band |Δ| ≤ 0.01 (inclusive — sub-noise = inconclusive,
    no multi-seed rescue). DRIFT > ctrl + 0.01. ctrl_val baseline
    6.4306 (`LEADERBOARD.md` row 14) — interpreted against the
    in-session ctrl run to avoid cross-session drift. Seed 42 only.
    See `autoresearch/ideas/028-deep-thin-config/plan.md`.
    """
    d_model: int = 48
    n_heads: int = 3
    n_kv_heads: int = 3
    n_layers: int = 20
    d_ff: int = 192
```

### `configs/__init__.py` — add import + `__all__` entry

Add `Tiny1M3MDeepThinConfig` to both the `from .llm_config import (…)`
block and the `__all__` list (next to `Tiny1M3MVNormOnQKNormConfig`).

### `tests/test_deep_thin_config.py` — param-count sanity assert

New test file. Builds `MinimalLLM(Tiny1M3MDeepThinConfig())` once and
asserts total parameter count `< 990_000` (the spec's "within ±5% of
0.94M" guarantee, with 951k expected and 990k as the ceiling). Mirrors
the style of `tests/test_v_norm.py::_build`. Catches the case where
the dataclass field assignments silently drift the budget (e.g. someone
later edits `d_ff` and breaks the convention).

```python
"""Sanity check for the 028 deep-thin config — param count lands within
the ±5% of the 0.94M baseline budget. See
`autoresearch/ideas/028-deep-thin-config/plan.md`."""
import torch

from configs.llm_config import Tiny1M3MConfig, Tiny1M3MDeepThinConfig
from models.llm import MinimalLLM


def _param_count(cfg) -> int:
    torch.manual_seed(42)
    return sum(p.numel() for p in MinimalLLM(cfg).parameters())


def test_deep_thin_lands_in_budget():
    """`Tiny1M3MDeepThinConfig` builds with ≤ 990k params (≈ +5% over
    the 0.94M baseline). Expected ≈ 951k (+1.3%)."""
    n = _param_count(Tiny1M3MDeepThinConfig())
    assert n <= 990_000, f"deep-thin param count {n} exceeds 990k ceiling"


def test_baseline_unchanged_by_diff():
    """Sanity: the additive diff doesn't perturb `Tiny1M3MConfig` itself
    (no shared mutable state between the two dataclasses)."""
    n = _param_count(Tiny1M3MConfig())
    # The known baseline is ≈ 939k (see idea.md:47). Window ±2% so a
    # legitimate later embedding tweak doesn't break this test, but a
    # silent baseline drift does.
    assert 920_000 <= n <= 960_000, (
        f"baseline param count {n} drifted outside the 920-960k window"
    )
```

## Control

| | ctrl | trt |
|---|---|---|
| config class | `Tiny1M3MConfig` | `Tiny1M3MDeepThinConfig` |
| seed | 42 | 42 |
| tier | tiny1m3m | tiny1m3m |
| `train_tokens` | 3_000_000 | 3_000_000 |
| `batch_size` | 2 | 2 |
| `max_seq_len` | 2048 | 2048 |
| `muon_lr` | 0.024 | 0.024 (unchanged) |
| `adamw_lr` | 0.006 | 0.006 (unchanged) |
| `warmup_ratio` | 0.02 | 0.02 |
| `schedule_type` | `warmup_decay_to_zero` | `warmup_decay_to_zero` |

**Single axis of variation:** the 5 architecture fields in the table
above. Every other knob — optimizer settings, schedule, batch size,
token budget, eval milestones, AMP, grad-clip, weight-decay,
ffn_variant, emb_rank, vocab_size — is identical between ctrl and trt
(verified via `dataclasses.asdict(ctrl) vs asdict(trt)` → exactly 5
differing keys: `{d_model, n_heads, n_kv_heads, n_layers, d_ff}`).

**Pass bar (tiles the real line at ±0.01 — box noise ≈ ±0.01):**
- **WIN:** `trt_val < ctrl_val − 0.01` (strict)
- **NULL:** `|trt_val − ctrl_val| ≤ 0.01` (inclusive — sub-noise = inconclusive)
- **FAIL:** `trt_val > ctrl_val + 0.01` (strict)

ctrl_val baseline = 6.4306 (`LEADERBOARD.md` row 14). Interpreted
against the **in-session** ctrl run, not the leaderboard number, to
avoid cross-session drift confounds.

## Cost

**Params Δ:** baseline ≈ 939k → trt ≈ 951k (+12k, +1.3%).
- Per-block ctrl (d_model=64, n_heads=4, n_kv_heads=2, d_ff=256,
  squared_relu): attn `2·d_model² + 2·d_model·kv_size` = `2·64² + 2·64·32`
  = 8192 + 4096 = 12.3k; FFN `2·d_model·d_ff` = 32.8k; norms ≈ 0.3k →
  ~45.4k/block × 12 = 545k.
- Per-block trt (d_model=48, n_heads=n_kv_heads=3, d_ff=192,
  squared_relu): attn `4·d_model²` = 9.2k (MHA — no GQA shrink); FFN
  `2·48·192` = 18.4k; norms ≈ 0.24k → ~27.9k/block × 20 = 558k.
- Embedding factorisation unchanged: `49152·8 + 8·d_model` ≈ 393.6k both
  sides (the 16-param tail differs by emb_rank · (64−48) = 128 params,
  negligible).
- Total: 939k → 951k (verified by the test below).

**FLOPs Δ:** roughly equivalent at fixed token budget — more layers
× smaller matmuls. Per-token compute scales as `n_layers · (4·d_model²
+ 2·d_model·d_ff)`: ctrl 12·(16384 + 32768) = 590k; trt 20·(9216 +
18432) = 553k. Trt is ~6% cheaper FLOPs/token → if it WINs at equal
wall-clock, the win is real; if it NULLs, the depth lever costs the
same as the width lever at this scale.

**Memory Δ:** activation memory dominated by `n_layers · batch · seq ·
d_model`. Ctrl 12·2·2048·64 = 3.1M floats. Trt 20·2·2048·48 = 3.9M
floats (+27%). Inside the V100 budget at batch_size=2 (no OOM risk).

## Run

**Tier:** tiny1m3m (≈ 0.94M params · 3M tokens · seed 42).
**Wall-clock target:** ~5 min/job per the queue.md arq notes
(2026-06-10 batch); trt slightly slower per step (more layers, smaller
matmuls), so expect ~5-6 min for the trt arm.

**Command shape (runner will write the `_arq_028.py` and
`_arq_028_ctrl.py` shims at queue time, mirroring `_arq_024_ctrl.py` /
`_arq_025.py`):**

```bash
# ctrl
/venv/main/bin/python _arq_028_ctrl.py  # uses Tiny1M3MConfig + seed=42
# trt
/venv/main/bin/python _arq_028.py        # uses Tiny1M3MDeepThinConfig + seed=42
```

**Pass/fail bar (copied from idea.md):**
- **WIN:** `trt_val < ctrl_val − 0.01`
- **NULL:** `|trt_val − ctrl_val| ≤ 0.01`
- **FAIL:** `trt_val > ctrl_val + 0.01`

ctrl_val target ≈ 6.4306 (in-session leaderboard sanity). If the box
ctrl drifts > 0.01 from this, the box is bad and the idea stays
`needs-run`.

## Frozen (non-architectural) fields — assertion

All inherited unchanged from `Tiny1M3MConfig` via the dataclass
inheritance (no overrides):

- `max_seq_len=2048`, `batch_size=2`, `train_tokens=3_000_000`,
  `compile_model=False`
- `warmup_ratio=0.02`, `schedule_type='warmup_decay_to_zero'`
- `eval_milestones=(0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700)`
- all optimizer / Muon / AdamW settings (`muon_lr=0.024`,
  `muon_momentum=0.95`, `adamw_lr=0.006`, `weight_decay=0.2`,
  `dropout=0.0`, `grad_clip=1.0`, `use_amp=True`)
- `ffn_variant="squared_relu"`, `emb_rank=8`, `vocab_size=49152`,
  `seed=42`

**No LR bump, no batch_size change, no schedule edit to "rescue" a
deeper model.** The runner's results must reflect the structural lever
alone — any HP retune contaminates the A/B and would force the call
back to `needs-recode`.

## LoC accounting

- `configs/llm_config.py`: 1 new dataclass, 5 field overrides +
  docstring ≈ 50 lines (well under the 30 LoC code budget — most of
  the lines are docstring; the dataclass body proper is 5 lines).
- `configs/__init__.py`: 2 lines (1 import, 1 `__all__`).
- `tests/test_deep_thin_config.py`: ~30 lines (two `assert`s + a
  shared helper).
- **Net additive diff: ~85 lines, ~12 code-lines.** No existing line
  edited.

## Self-check (before flipping to `needs-codereview`)

- [x] **Flag-OFF reproduces ctrl:** trivially — there is no flag. The
  ctrl is `Tiny1M3MConfig`, which is unchanged by the diff (purely
  additive: a new subclass and a new test file).
- [x] **Trt exercises the new code:** `MinimalLLM(Tiny1M3MDeepThinConfig())`
  builds with d_model=48, n_layers=20, n_heads=n_kv_heads=3 — the
  test asserts the param count, which is the operational proof the
  config fields take effect at model-build time.
- [x] **Pass bar matches idea.md:** WIN ≤ ctrl − 0.01, NULL ≤ 0.01,
  FAIL > +0.01 — copied verbatim from `idea.md:57-61`.
- [x] **No `models/layers.py` / `models/llm.py` edit:** confirmed.
  Diff surface is config + test only; the parallel-AI coordination
  memo is a non-issue.
- [x] **Seed 42 only:** the trt inherits `seed=42` from
  `Tiny1M3MConfig`; no multi-seed protocol in this plan.
