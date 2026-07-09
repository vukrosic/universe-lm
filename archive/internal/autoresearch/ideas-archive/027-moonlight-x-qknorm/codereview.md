# 027 — code-review log

## r1 — 2026-06-10 — verdict: accept

- **Recipe defined as specced.** `configs/llm_config.py:767-796` adds
  `Tiny1M3MMoonlightMuonQKNormConfig(Tiny1M3MConfig)` with exactly the two
  flags flipped: `use_moonlight_muon: bool = True` and
  `use_qk_layernorm: bool = True`. No other field overrides — inherits
  `muon_lr=0.024`, depth, width, data, schedule from `Tiny1M3MConfig`
  unchanged. No silent HP drift.
- **Faithful to mechanism.** Moonlight wires through
  `training/trainer.py:209-225` → `scale_mode="moonlight"` →
  `optimizers/muon.py:146-157` where `c = float(group.get("moonlight_c", 0.2))`
  and `scale = c · sqrt(max(d_in, d_out))`. This is the closed-#015 path
  (verified by closed.md:32 evidence Δ −0.0138). QK-Norm wires through
  `models/llm.py:274,396` → `models/layers.py:579 → 705-707` where
  `_qk_use_ln = bool(use_layernorm) or bool(use_qk_layernorm)` flips both
  `q_norm` and `k_norm` to `nn.LayerNorm(d_k)` via `make_norm`. This is
  the closed-#016 path (Δ −0.0138 evidence). The two paths share no state.
- **Pure additive insert.** `git diff configs/llm_config.py` shows the
  027 contribution is the new dataclass block at 767-796 only — no edits
  to existing lines (the surrounding 029/026/030 inserts in the same file
  are from the parallel-Claude unstaged edits, not stomped). No edits to
  `optimizers/muon.py`, `training/trainer.py`, `models/llm.py`,
  `models/layers.py` for 027 — plan's "no edits to those files" claim
  holds.
- **OFF→OFF identity.** Both flags default `False` on `LLMConfig`
  (lines 546, 559). With both off: Moonlight reads
  `getattr(config, "use_moonlight_muon", False)` and falls through to
  `scale_mode="shape_aspect"` (trainer.py:218-220) — bit-identical to
  the plain-Muon path. QK-Norm reads
  `getattr(config, "use_qk_layernorm", False)` and `_qk_use_ln=False`
  → `make_norm` returns the default `RMSNorm(d_k)` — bit-identical to
  the closed baseline. Step-0 identity: Moonlight is identity on the
  first step because the orthogonalized update `g` is independent of the
  scale factor when `momentum=0` and the gradient buffer is empty
  (NS5(0)=0; any finite scale ×0 = 0). QK-Norm is identity at step 0
  because `nn.LayerNorm` inits γ=1, β=0 → preserves Q,K up to the
  centering+rescaling that doesn't change the angle (the dot-product
  direction is preserved at init under unit-variance Q,K). Both step-0
  invariances are independent.
- **Build-smoke matches.** `log.jsonl` reports
  "build-smoke params Δ=+384 matches QK-Norm-alone; differ-from-ctrl:
  exactly the two flags". The +384 = `n_layers·2·d_head = 12·2·16` —
  RMSNorm→LayerNorm adds only the β bias term (γ already existed under
  RMSNorm), per norm, two norms per layer (q_norm + k_norm), 12 layers.
  The plan's earlier `+768 = 2·n_layers·2·d_head` estimate
  double-counted γ; the actual build-smoke `+384` is the correct
  signature. Optimizer-side has zero parameter Δ (Moonlight only changes
  the scalar in the Muon step, no new state) — consistent with the
  expected one-sided params delta.
- **Plan ↔ idea consistency.** Plan's PASS bar (`trt ≤ ctrl − 0.01`,
  NULL `|Δ| < 0.01`, DRIFT `> +0.01`) matches the dataclass docstring
  (lines 791-792) and is the same bar the review approved. Plan uses
  the correct flag name `use_qk_layernorm` throughout (not the loose
  `use_qk_norm` shorthand from `idea.md`). Tier = tiny1m3m, seed = 42
  (single seed, as required by the project rule). Control =
  `Tiny1M3MConfig`. Treatment = `Tiny1M3MMoonlightMuonQKNormConfig`.
- **LoC budget.** The 027 contribution is ~30 LoC including the
  docstring — well under the 200-LoC mined-idea budget.
- **Coordination.** No edits to shared files for this idea; the
  llm_config.py insert is positioned after the parallel-Claude's 029
  `Tiny1M3MVNormOnQKNormConfig` insert (line 747), avoiding collision
  exactly as the plan claimed.
- **Non-blocking note.** `configs/__init__.py` does **not** export
  `Tiny1M3MMoonlightMuonQKNormConfig`. The runner harness imports configs
  via `from configs.llm_config import <Name>` directly (see
  `_arq_026.py`, `_arq_029.py`), not via `from configs import …`, so
  the missing `__init__.py` re-export does not block the run. The
  pre-existing `Tiny1M3MMoonlightMuonConfig` (015 closed) is also not
  in `__init__.py` and has run successfully in the past. Flag this as a
  cleanup item, not a blocker.
- **Verdict: accept.** Composition of two closed WINs at non-overlapping
  code paths, both pre-wired and validated, two-line treatment dataclass,
  baseline path bit-identical, pass/fail bar above the in-session ctrl-
  to-ctrl noise floor. Ready to run. Flip → `needs-run`.
