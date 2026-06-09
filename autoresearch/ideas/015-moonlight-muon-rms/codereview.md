# Code-review log — 015 Moonlight Muon RMS

## r1 — 2026-06-09 — verdict: accept
- **Mechanism faithful.** `optimizers/muon.py:156-157` computes
  `scale = c * (max(p.size(-2), p.size(-1)) ** 0.5)`, applied as
  `p.add_(g, alpha=-lr*scale)` at line 175. Matches idea.md's
  `update = c * sqrt(max(d_in, d_out)) * O` — `c` is the paper's single
  global knob (0.2), `O` is the orthogonalized update `g`. Math is right.
- **Identity / step-0 holds.** When `use_moonlight_muon=False` (the
  default, `configs/llm_config.py:445`), `training/trainer.py:216-220`
  falls through to `getattr(config, "muon_scale_mode", "shape_aspect")`
  — the existing baseline. The new `moonlight_c` constructor arg
  (`muon.py:75`) is wired into `defaults` (`muon.py:76`) but only
  *read* on the `"moonlight"` branch (`muon.py:156`), so the OFF path
  is bit-identical. Step-0: NS5 of a zero gradient = zero update, and
  `0 * scale = 0` regardless of formula.
- **No silent HP drift.** All three touched files (`muon.py`, `trainer.py`,
  `llm_config.py`) have only additive edits for this idea. `muon.py`
  adds one `elif` branch and one constructor kwarg; `trainer.py` adds
  the scale_mode ternary + the `moonlight_c=...` kwarg; `llm_config.py`
  adds the two new fields and the new config class. No LR/schedule/init
  constants, no seed, no other knobs were smuggled in.
- **Single boolean, default OFF.** `use_moonlight_muon: bool = False`
  at `configs/llm_config.py:445`; `moonlight_muon_c: float = 0.2` at
  line 446. The treatment path actually exercises the new code: when
  True, `scale_mode` becomes `"moonlight"` and the new branch fires on
  every Muon param per step.
- **LoC budget.** ~12 LoC in muon.py (branch + comments), ~10 LoC in
  trainer.py (comment + ternary + kwarg), ~10 LoC in llm_config.py
  (field block + dataclass). Well under the 200 LoC cap.
- **Plan ↔ idea consistency.** Control = `Tiny1M3MConfig` (plain Muon
  `shape_aspect`); treatment = `Tiny1M3MMoonlightMuonConfig` (flips
  `use_moonlight_muon=True` only). Seed pinned 42, tier tiny1m3m, PASS
  bar `trt ≤ ctrl − 0.01` / NULL `|Δ| < 0.01` / DRIFT `trt > ctrl + 0.01`
  per `plan.md`. Single global knob `c=0.2` matches the paper.
- **Coordination.** The 015 footprint is `muon.py` + `trainer.py` +
  `llm_config.py` only; `llm_config.py` is purely additive (110
  insertions, 0 deletions in this commit's slice), placed after the
  existing `Tiny1M3MCautiousMuonConfig` so it doesn't reorder the
  pre-existing config classes. No `models/layers.py` / `models/llm.py`
  stomp, no rebase, no push.
- **Minor noted, not blocking.** `muon.py:156` does
  `c = float(group.get("moonlight_c", 0.2))` even though `moonlight_c`
  is in `defaults` (line 76) and the trainer always passes it
  explicitly (`trainer.py:225`). The `.get(..., 0.2)` fallback is
  defensive, not a bug — keeps a safe default if someone instantiates
  `Muon` directly with the old constructor signature.
