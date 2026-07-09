# Code-review log — 028 deep-thin-config

## r1 — 2026-06-10 — verdict: accept

**Faithful, identity-safe (config-level), no HP drift, no shared-file conflict.
Approve to `needs-run`.**

**What I checked against `plan.md` / `idea.md` (spec) → diff:**

- **Pinned tuple matches verbatim.** `configs/llm_config.py:1208-1212` overrides
  exactly `d_model=48, n_heads=3, n_kv_heads=3, n_layers=20, d_ff=192` — the
  B1 tuple from `idea.md:31-44`. No off-by-one, no silent fallback. ✓
- **`d_head` preserved at 16.** `d_model/n_heads = 48/3 = 16` (was `64/4 = 16`).
  Confirmed empirically: `n_heads // n_kv_heads = 3//3 = 1` → MHA path (no
  GQA `repeat_interleave`); identical attention math to the baseline, just
  thinner. ✓
- **Param count inside ±5% ceiling.** Built `MinimalLLM(Tiny1M3MDeepThinConfig())`
  locally: **961,168** params (+1.28% vs `Tiny1M3MConfig` 949,056). Within the
  ±5% / ≤990k budget. Forward pass on `[2, 16]` input is finite, shape
  `[2, 16, 49152]`. ✓
- **Confound disclosed in the docstring** (`configs/llm_config.py:1190-1199`):
  baseline GQA 2:1 → MHA tie (n_heads=n_kv_heads=3) is called out, with the
  pointer to `LEADERBOARD.md` row 0 (tied-QK + MHA is a known WIN signature).
  Runner is instructed to report the confound alongside the raw val-loss Δ.
  Honest disclosure, not silent absorption. ✓
- **Baseline path bit-identical.** `Tiny1M3MConfig` is not edited — the diff
  is purely additive (one new subclass appended after
  `Tiny1M3MQKNormOnFireConfig` at line 1170; one import + one `__all__` entry
  in `configs/__init__.py`). The "default-OFF, baseline bit-identical"
  guarantee holds at the config-class level: every existing site that
  constructs `Tiny1M3MConfig` is byte-for-byte unchanged. Test
  `test_baseline_unchanged_by_diff` enforces this with a 920–960k window. ✓
- **No silent HP drift.** Test `test_arch_fields_overridden`
  (`tests/test_deep_thin_config.py:52-84`) asserts `muon_lr, adamw_lr,
  batch_size, train_tokens, warmup_ratio, schedule_type, emb_rank,
  ffn_variant, seed` all match between ctrl and trt — exactly the "no LR
  bump to rescue a deeper model" check the spec demands
  (`idea.md:68-75`, `plan.md:240-244`). Catches the failure mode where a
  later implementer "helps" the deeper model with a learning-rate tweak. ✓
- **Flag is justified as a no-op.** `plan.md:5-13` correctly argues that a
  `use_deep_thin: bool` would be theatre because every field
  (`d_model, n_heads, n_kv_heads, n_layers, d_ff`) is already a dataclass
  knob — the A/B is two configs, not a flag. The default-OFF guarantee is
  satisfied at the config-class level. Mirrors how
  `Tiny1M3MQKNormConfig` etc. are structured. ✓
- **LoC budget respected.** The 028 contribution is a single ~45-line
  dataclass (mostly docstring; 5 real code lines for the field overrides) +
  2 lines in `configs/__init__.py` + 85 lines of test. Well under the 200
  LoC ceiling. ✓
- **Plan ↔ idea consistency.** Pass bar tiled at ±0.01 (`plan.md:163-166`
  ↔ `idea.md:57-61`). Single seed = 42 (`plan.md:271` ↔ `idea.md:65-66`).
  Ctrl = `Tiny1M3MConfig` in both. tier = tiny1m3m in both. No drift. ✓
- **Coordination.** Diff surface is config + test only; no
  `models/layers.py`, `models/llm.py`, `models/fox.py`, `models/fire_pe.py`
  edit. `git diff` confirms the 028 dataclass is appended at the end of the
  Tiny1M3M block (line 1170), not touching any existing field. The unstaged
  `models/layers.py` / `models/llm.py` work from the parallel AI (V-Norm,
  QK-Norm wiring, etc.) is left intact — no revert, no stomp. ✓

**Tests run locally (`pytest tests/test_deep_thin_config.py -v`):**
- `test_baseline_unchanged_by_diff` PASS
- `test_deep_thin_lands_in_budget` PASS  (961,168 ≤ 990,000)
- `test_arch_fields_overridden` PASS

**Forward smoke (built `MinimalLLM(Tiny1M3MDeepThinConfig())`, ran forward on
random `[2, 16]` input):** finite output, shape `[2, 16, 49152]`, no crash on
the n_heads=3 / d_head=16 MHA path.

**Disposition:** ready to run. Runner shim follows the standard pattern
(`_arq_028_ctrl.py` uses `Tiny1M3MConfig` + seed 42; `_arq_028.py` uses
`Tiny1M3MDeepThinConfig` + seed 42). The confound (MHA tie-collapse) must
be reported alongside the raw Δ in the run note — already specced in
`idea.md:50-55` and the docstring.
