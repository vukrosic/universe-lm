# Code-review log — 030 U-Net Skip Gates (sigmoid init −1.5)

## r1 — 2026-06-10 — verdict: accept

**Mechanism faithful to idea.md / plan.md.** `models/llm.py:185-191` already
fills `unet_skip_gates` with `unet_gate_init` (here −1.5) on a
`(unet_skip_count, d_model)` parameter; the forward at `:617-625` reads
`gate = self.unet_skip_gates[skip_idx]`, wraps with `torch.sigmoid` iff
`self.unet_gate_type == "sigmoid"`, and applies `x = x + gate * skip`.
`sigmoid(−1.5) ≈ 0.1824` → ≈18% of the saved early activation mixes into the
mirrored late layer at step 0 — exactly the modded-nanogpt PR #125 fix the
idea calls for. Mechanism code was already in tree; this idea is a config-only
flag flip plus two harness scripts (well under the 200-LoC budget).

**Single-axis A/B confirmed.**
`Tiny1M3MUNetSigmoidOnFireConfig` differs from
`Tiny1M3MVQGainSWAHighRoPE250KConfig(use_fire_pe=True)` on exactly the three
keys that together encode the one U-Net-sigmoid lever
(`use_unet_skips=True`, `unet_gate_type="sigmoid"`,
`unet_gate_init=-1.5`); the three are not independent (the latter two are
inert when `use_unet_skips=False`), so this is the cleanest single-mechanism
representation. Implementer-side smoke (logged in log.jsonl) confirms
`Δparams=+384 = unet_skip_count·d_model = 6·64` and
`sigmoid(unet_skip_gates) ≈ 0.182426` element-wise.

**Flag-off path bit-identical.** `if self.use_unet_skips:` guards
construction at `models/llm.py:167` and both forward branches at `:617,632` —
the baseline path adds no params, no FLOPs, no ops. The 029 / 026 / 022 etc.
diff entries riding along in `models/layers.py` / `models/llm.py` are
unrelated parallel work and do not interact with the U-Net path.

**Flag-on step-0 deviation is the mechanism, not a bug.** Plan explicitly
justifies non-bit-identical step-0 (idea 015 NULL band — the previous
zero-init was bit-identical, which is exactly why the gate never fired). The
dead-gate bug is documented in `[[unet-skips-gate-fix]]` and the present
sigmoid(−1.5) fix is the modded-nanogpt PR #125 prescription verbatim.

**Harness scripts correct.**
- `_arq_030.py` (trt): `class C(Tiny1M3MUNetSigmoidOnFireConfig): pass` —
  parent is `@dataclass`-decorated and `C` adds no new typed fields, so the
  parent's auto-generated `__init__` runs and all four flag defaults
  (`use_fire_pe=True`, `use_unet_skips=True`, `unet_gate_type="sigmoid"`,
  `unet_gate_init=-1.5`) are honored.
- `_arq_030_ctrl.py` (ctrl): `@dataclass class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe: bool = True`
  — `@dataclass` IS required here. Without it, `use_fire_pe: bool = True` is
  a bare annotation and the parent dataclass's `__init__` sets
  `self.use_fire_pe = False` (parent default), silently flipping ctrl
  FIRE-OFF. The implementer's docstring comment in `_arq_030_ctrl.py:2-10`
  documents this explicitly. Correct.
- Both forward to `train_llm.main()` with `--seed 42 --warmup false
  --dataset_path processed_data/pretrain_1B` — matches runner.md §2 and the
  one-seed rule.

**Plan ↔ idea consistency.** Tier=tiny1m3m (one tier). Seed=42 (one seed).
Ctrl = `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` (the 009 WIN
signature, val 6.3234) — the FIRE-equipped baseline both 023 and 026 use, so
the A/B isolates a residual-stream architectural lever on top of the
current attention-side best. PASS bar (≤ −0.005), NULL band (|Δ| < 0.005),
DRIFT (> +0.005) match between idea.md, plan.md, and the in-tree docstring.

**Cosmetic finding (non-blocking).** The `Tiny1M3MUNetSigmoidOnFireConfig`
docstring at `configs/llm_config.py:1031-1033, 1043-1044` says
"6-layer depth", "3 short pairs", "3-pair U at 6L". `Tiny1M3MConfig.n_layers
= 12` (confirmed `configs/llm_config.py:674`), so the actual mirror at
default `unet_skip_count = n_layers // 2 = 6` is
**0↔11 / 1↔10 / 2↔9 / 3↔8 / 4↔7 / 5↔6 — six pairs at 12L**, not three at
6L. Plan.md §Flag and §Tests both correctly call this out
("n_layers=12, not 6; taste assumed 6L but Tiny1M3MConfig is 12L") and the
smoke output asserts `model.unet_skip_count == 6`, so the actual experiment
runs the correct six-pair U. The numerical PASS bar (−0.005, "small but
non-zero") is conservative enough to be appropriate for either reading and
doesn't change. **Doc nit only — not blocking. Fix on a future pass if
touching this config.**

**Cross-idea escalation (out of scope for 030's verdict).** The implementer
flagged in plan.md:295-306 that `_arq_020_ctrl.py`, `_arq_023_ctrl.py`,
`_arq_026_ctrl.py` use the pattern `class C(Parent): use_fire_pe: bool =
True` **without** `@dataclass`. Confirmed by inspection: in Python
dataclasses, an inheriting class without `@dataclass` does NOT override the
parent's field default — the parent's auto-generated `__init__` runs first
and sets `self.use_fire_pe = False`. Those three ctrl runs are silently
FIRE-OFF, not the FIRE-equipped baseline their comments claim. This is a
real regression that contaminates the ctrl side of 020 / 023 / 026's A/Bs,
but is **out of scope for 030's review** (030's own ctrl uses `@dataclass`
correctly). Logging here so it isn't lost; the runner / 020/023/026 reviewers
should treat their currently-running ctrls as suspect and re-issue with
`@dataclass`-decorated harness scripts.

→ `accept`: ship to runner. Single-axis lever, bit-identical flag-off path,
mechanism faithful, harness scripts correct, seed 42, one tier.
