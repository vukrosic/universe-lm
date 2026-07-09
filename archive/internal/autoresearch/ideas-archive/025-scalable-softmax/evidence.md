# Evidence — 025 Scalable-Softmax (SSMax, length-aware attention temperature)

## Verdict: WIN (within-session, with TWO wiring bugs)
- tier: tiny1m3m, seed 42, box: vast-81.45.65.189 (V100-PCIE-32GB)
- control val (plan-025 ctrl = Tiny1M3MConfig + use_fire_pe=True, **buggy**): **6.4269**
- treatment val: **6.3359** · Δ: **−0.0910**
- pass/fail bar (plan.md): WIN trt < ctrl − 0.01 / NULL −0.01 < Δ ≤ 0 / REGRESS Δ > 0
- bpb: n/a (pending harness)
- raw: remote-results/2026-06-10-vast-tiny1m3m/{025-ssmax.log, 025-ctrl.log, 025-ssmax-sh.log}
- date: 2026-06-10

## Two wiring bugs (CRITICAL)
1. **Plan-025 ctrl** (`_arq_025_ctrl.py`) used `class C(Tiny1M3MConfig): use_fire_pe: bool = True` — override dropped at runtime (config dump shows `use_fire_pe: False`). Ran as plain Tiny1M3MConfig → val 6.4269.
2. **SSMax trt config itself is missing FIRE**: `Tiny1M3MSSMaxConfig` is pre-baked with `use_ssmax=True` but **NOT** `use_fire_pe=True`. The other 020-023 trt configs (`*OnFireConfig`) are pre-baked with both; the SSMax one is an outlier. Model-config dump in `025-ssmax.log` (8:45Z) confirms `use_fire_pe: False`. **The trt was supposed to be SSMax+FIRE but ran as SSMax-alone.**
3. **Shared -sh rerun** (`_arq_025_shared.py`) tried `class C(...): use_fire_pe=True, use_ssmax=True` — both dropped → val 6.3419. Wasted rerun.

The 025-ssmax val 6.3359 is therefore the SSMax-alone effect (no fire), not SSMax+FIRE. Compared to a hypothetical SSMax+FIRE run, the no-fire penalty would push the val up by ~0.07 (FIRE effect), so SSMax+FIRE would land near 6.27 — still a clear WIN, but the within-session delta understates the joint.

## Cross-day sanity
- 025-ctrl on this box: 6.4269 (plain Tiny1M3MConfig at seed 42)
- Leaderboard ctrl (2026-06-08): 6.4287 — matches the buggy 025-ctrl
- 025-ssmax: 6.3359. Plan-025 §Control says the trt should be SSMax+FIRE on
  top of the FIRE-equipped baseline, mirroring the 020-023 shared-baseline
  pattern. As implemented, the trt is SSMax-alone.

## Pass/fail adjudication
Plan bar: WIN < −0.01. Within-session Δ = −0.0910 ≪ −0.01. **WIN in the within-session sense** — but the comparison is SSMax-alone (6.3359) vs no-fire-no-ssmax (6.4269), which is a stronger contrast than the spec'd SSMax+FIRE-vs-FIRE comparison. The within-session WIN overstates the isolated SSMax effect; the spec'd A/B (SSMax+FIRE vs FIRE) is unmeasured. Phase-2: re-run with `use_fire_pe=True` baked into the SSMax config + a proper fire-equipped baseline.

## Transfer note
SSMax (arXiv:2501.19399) is a per-head learnable `s·log(n)` temperature on
attention logits. Numerically trivial (one log + one multiply per head, no
accumulation or renormalisation). Should transfer cleanly. The mechanism is
additive on top of FIRE per the plan; the within-session WIN is real, but
needs a re-run with the correct `*OnFireConfig` base. **Phase-2: add
`use_fire_pe=True` to `Tiny1M3MSSMaxConfig` in llm_config.py, re-test.**
