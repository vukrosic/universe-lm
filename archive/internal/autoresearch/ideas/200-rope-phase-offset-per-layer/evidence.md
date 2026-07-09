# Evidence — 200-rope-phase-offset-per-layer

## r1 — 2026-06-15T16:34:42Z — build-smoke failed

**Daemon bounce (log.jsonl)**:
> build-smoke failed: SMOKE_FAIL: ImportError: cannot import name
> 'Tiny1M3MPerLayerKRotationConfig' from 'configs.llm_config'
> (/root/universe-lm/configs/llm_config.py)

**Root cause**: daemon-pull race. The `Tiny1M3MPerLayerKRotationConfig`
class was added in commit `13403a6` at `2026-06-15T16:34:24Z` — only
18s before the bounce at `16:34:42Z`. The daemon's `git pull` on the
box ran against the state visible at the pull moment; if the box's
auto-sync wasn't yet complete, the box's `configs/llm_config.py`
didn't have the new class and the daemon's `_box_smoke.py` import
failed. The class is verified locally (commit `13403a6` shows
`+88` to `configs/llm_config.py` for the 200 subclass).

## r1 — recode fix — build-smoke re-verified locally

- `configs/llm_config.py:7902` — `Tiny1M3MPerLayerKRotationConfig(Tiny1M3MConfig)`
  dataclass with `use_per_layer_k_rotation: bool = True`.
- `configs/llm_config.py:1438` — `use_per_layer_k_rotation: bool = False`
  on `LLMConfig` (default OFF, baseline bit-identical).
- `models/layers.py:1585` — kwarg threaded through `MultiHeadAttention.__init__`.
- `models/layers.py:2972-2981` — parameter allocation guarded by flag.
- `models/layers.py:4111-4129` — forward branch (post-RoPE / post-qk_norm /
  post-GQA-repeat, K-only).
- `models/llm.py:349` — flag read from config (`getattr(..., False)`).
- `models/llm.py:968`, `:1369` — flag passed to both MHA construction sites.

**Local CPU build-smoke**:
```
max_abs_diff: 0.0
ctrl params: 949056, trt params: 949152, delta: 96
angle param transformer_blocks.0.attention.per_layer_k_rotation_angles:
  shape=torch.Size([8]), sum=0.0
```

Bit-identical at step 0; +96 params (+0.001% of 0.94M) — matches plan.md
budget.

**Subsequent daemon auto-syncs** (commits after the bounce):
- `6a76191 daemon: auto-sync model code for box pull [2026-06-15T16:35:41Z]`
- `50d0c10 daemon: auto-sync model code for box pull [2026-06-15T16:36:41Z]`

The class is now stable on `origin/main` (HEAD ahead 5 commits vs
`origin/main`, but the local branch is exactly the daemon's box-pulled
state plus the 200/198/195 plan edits, and the 200 class is included in
all daemon commits from `13403a6` onward). The next daemon pull will
see the class and the build-smoke will pass.

## Run artifact (unchanged from r1)

- `_arq_200-rope-phase-offset-per-layer.py` (repo root) — defines top-level
  `C(Tiny1M3MPerLayerKRotationConfig)` and calls `train_llm.main()` with
  `--config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`.
- `autoresearch/ideas/200-rope-phase-offset-per-layer/run.json` —
  `{"name": "200-rope-phase-offset-per-layer", "arq_file":
  "_arq_200-rope-phase-offset-per-layer.py", "job_timeout": "12m"}`.

## Pass/fail bar (unchanged from r1)

- WIN: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule.
- NULL: `|trt_val − ctrl_val_mean| < 0.01`.
- DRIFT: `trt_val > ctrl_val_mean + 0.01` (closes the lever family until ≥135M).
- Sub-noise is inconclusive per the one-seed-only rule (seed 42 only).