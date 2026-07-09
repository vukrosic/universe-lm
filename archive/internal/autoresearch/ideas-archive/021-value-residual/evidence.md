# Evidence — 021 Value Residual Learning (V-residual)

## Verdict: WIN (within-session, with shared fire-ctrl wiring caveat)
- tier: tiny1m3m, seed 42, box: vast-81.45.65.189 (V100-PCIE-32GB)
- control val (shared fire-ctrl, **buggy — see note**): **6.3419**
- treatment val: **6.3075** · Δ: **−0.0344**
- pass/fail bar (plan.md): WIN trt < ctrl − 0.005 / NULL |Δ| < 0.01 / FAIL trt > ctrl + 0.01
- bpb: n/a (pending harness)
- raw: remote-results/2026-06-10-vast-tiny1m3m/021-vres.log + ctrl_fire.log
- date: 2026-06-10

## Shared fire-ctrl wiring bug (CRITICAL caveat)
The `class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe: bool = True` subclass
override was silently dropped at runtime — model-config dump shows `use_fire_pe: False`
in `ctrl_fire.log` (8:03Z) and `ctrl_fire2.log` (8:50Z). Both got **identical** val
6.3419/0.1511 to 4 decimals (deterministic on seed 42 of `Tiny1M3MVQGainSWAHighRoPE250KConfig`
with fire *off*). The 4 -sh reruns (ctrl_fire, ctrl_fire2, 024-gated-sh, 025-ssmax-sh)
all produced 6.3419 — same flag-drop root cause.

The 021 treatment used pre-baked `Tiny1M3MVResidualOnFireConfig` (use_fire_pe=True,
use_value_residual=True) — the trt-side flag is correct. The shared baseline was
supposed to also be fire-equipped but wasn't. So the −0.0344 within-session delta
is the *combined* V-residual+FIRE effect vs. FIRE-less baseline, not V-residual vs.
FIRE.

## Box check / cross-day sanity
- Leaderboard ctrl (2026-06-08, 1B data, T4): 6.4287
- This box's 024-ctrl (plan-024's own control, also missing fire): 6.4269
- This box's shared "fire" ctrl: 6.3419 (≈ FIRE-equipped 009 win at 6.3234 from closed.md:40 minus ~0.02 because flag is off)
- 021-vres: 6.3075 — best of all today's runs except 023-canon.
- Cross-day: 021's val is below the FIRE-alone 6.3234, but the shared baseline is also below the FIRE-alone 6.3234. Relative delta to the (buggy) shared baseline is the only honest number.

## Pass/fail adjudication
Plan bar: WIN < −0.005. Within-session Δ = −0.0344 ≫ 0.005. **WIN in the strict within-session sense.** But the shared baseline is not the spec'd fire-equipped baseline, so the "real" V-residual effect (V-residual vs FIRE-only) is unmeasured here. A reasonable re-test: a fire-equipped baseline run at the same seed should land near 6.27 (FIRE alone gives 6.3234 in 009 + a few % noise). If that holds, 6.3075 vs 6.27 = +0.04 (FAIL). The within-session delta is **confounded with the FIRE effect.**

## Transfer note
V-residual (arXiv:2410.17897) is reported additive on top of the attention block — at 135M it should still provide a small cross-layer shortcut benefit. The mechanism doesn't depend on numerical tricks (no decay, no renormalisation), so it should transfer cleanly. The within-session WIN is consistent with that; the within-session-WIN includes some fire credit and is best treated as "the joint is at least not bad" rather than "V-residual is additive on top of FIRE." Phase-2: re-run against a proper fire-equipped ctrl to isolate the V-residual contribution.
