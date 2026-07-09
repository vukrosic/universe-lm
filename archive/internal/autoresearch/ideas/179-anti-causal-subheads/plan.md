# Plan — 179 anti-causal-subheads

## Flag
- `use_anti_causal_subheads: bool = False` (default OFF), declared on
  `MultiHeadAttention.__init__` (`models/layers.py`) and threaded into both
  block sites in `models/llm.py`. The treatment subclass
  `Tiny1M3MAntiCausalSubHeadsConfig(Tiny1M3MConfig)` flips it on
  (`configs/llm_config.py`).

## Change
- `models/layers.py` (`MultiHeadAttention`)
  - `__init__`: add `use_anti_causal_subheads: bool = False` kwarg. Allocate
    `self.ac_subhead_gate = nn.Parameter(torch.full((n_heads,), -10.0))` when
    the flag is on (init −10 ⇒ `sigmoid(−10) ≈ 4.5e-5` ⇒ effectively causal
    for every head at step 0). When off, `self.ac_subhead_gate = None` (stub
    for attribute-lookup safety; the forward branch is never taken).
  - `forward()` (manual attention path): before the `masked_fill` at
    `models/layers.py:3516`, compute
    `gamma_h = torch.sigmoid(self.ac_subhead_gate).view(1, H, 1, 1)` and
    pass the per-head fill value
    `fill = -1e9 * (1.0 - gamma_h)` to `masked_fill` (broadcasts over
    `[B, H, T, T]`). The masked positions take
    `−1e9·(1 − γ_h)` per head; unmasked positions are unchanged.
  - Manual-path entry condition (line 3387 et al.):
    add `or self.use_anti_causal_subheads` so the manual path is forced
    (SDPA's flash kernel doesn't support a per-head additive bias on the
    mask fill).
- `configs/llm_config.py`
  - Add `Tiny1M3MAntiCausalSubHeadsConfig(Tiny1M3MConfig)` with
    `use_anti_causal_subheads: bool = True`. Inherits everything else
    from `Tiny1M3MConfig` (incl. `use_fire_pe=False`).
- `models/llm.py`
  - Read `use_anti_causal_subheads = getattr(config, "use_anti_causal_subheads", False)`
    once at construction (next to the existing `use_mqa_gated` read at
    line 313).
  - Pass it to both MHA sites: the YOCO upper-half block (next to
    `use_mqa_gated` at line 698) and the standard block (next to
    `use_mqa_gated` at line 994).

## Control
- **Control**: `Tiny1M3MConfig` (plain baseline, all lever flags OFF).
- **Treatment**: `Tiny1M3MAntiCausalSubHeadsConfig` (only
  `use_anti_causal_subheads=True`).
- Seed: **42** (one seed only — `feedback-one-seed-only`).
- Tier: **tiny1m3m** (3M tokens, 12L/4H/d_model=64).
- Cache reference: `autoresearch/baseline-cache.json` box `5b8a7fea8963`
  (RTX 3060), `ctrl_val_mean = 6.3988`, `noise_band = 0.04`, n=3.

## Cost
- Params: H=4, n_layers=12 ⇒ 4 · 12 = **48** new params (+0.005% of 0.94M).
- FLOPs/forward: a single `(1 − sigmoid)` and a per-head broadcast on the
  fill value. The `masked_fill` is the same op as baseline (one broadcast
  over the mask). Negligible.
- Memory: +48 floats (a single 4-vector broadcast 12 times). Negligible.
- Forces manual attention path: ~1% wall-clock overhead at tiny1m3m
  (per the closed 152/155/166/180 SDPA-off pattern).

## Run
- Command: the daemon's CPU build-smoke + GPU run via
  `autoresearch/bin/queue-daemon.sh` reading
  `autoresearch/ideas/179-anti-causal-subheads/run.json`. The treatment
  entry `_arq_179-anti-causal-subheads.py` defines `class C(Tiny1M3MAntiCausalSubHeadsConfig)`
  and runs `train_llm.main()` with `--seed 42`,
  `--dataset_path processed_data/pretrain_1B`, `--warmup false`.
- Tier: **tiny1m3m**, seed 42, expected wall-clock ~12 min on RTX 3060.
- Pass/fail bar (from `idea.md`, single-seed):
  - **WIN**: `trt_val ≤ 6.3888` *and* the trt beats both same-session ctrls
    by ≥ the two-ctrl gap (`PIPELINE.md` §2).
  - **NULL**: `|trt_val − 6.3988| ≤ 0.01` (modally expected per the
    closed per-head-attention-shape trio 152/155/166 + QK-norm 162/165
    nulls at 0.94M/12L/4H).
  - **DRIFT**: `trt_val > 6.4088`.
  - **Inside-band ambiguity** (`|Δ| ≤ 0.01`): logged NULL with
    `cache_authoritative: true`.
- Inference schedule: **keep γ_h as trained at both train and eval** (the
  trained γ_h IS the eval-time mask shape — measures real deployment
  behavior, not a contrived override).
