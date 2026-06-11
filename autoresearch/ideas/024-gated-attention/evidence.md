# Evidence — 024 Gated Attention (per-head sigmoid output gate)

## Verdict: WIN (within-session, with plan-024 fire-ctrl wiring caveat)
- tier: tiny1m3m, seed 42, box: vast-81.45.65.189 (V100-PCIE-32GB)
- control val (plan-024 ctrl = Tiny1M3MConfig + use_fire_pe=True, **buggy**): **6.4269**
- treatment val: **6.3316** · Δ: **−0.0953**
- pass/fail bar (plan.md, reviewer-locked): pass iff Δ ≤ −0.01
- bpb: n/a (pending harness)
- raw: remote-results/2026-06-10-vast-tiny1m3m/{024-gated.log, 024-ctrl.log, 024-gated-sh.log}
- date: 2026-06-10

## Two wiring bugs (cross-cutting)
1. **Plan-024 ctrl** (`_arq_024_ctrl.py`) used the same `class C(Tiny1M3MConfig): use_fire_pe: bool = True` pattern. The override was dropped at runtime — model-config dump in `024-ctrl.log` (8:32Z) shows `use_fire_pe: False`. So the plan-024 ctrl ran as plain Tiny1M3MConfig → val 6.4269.
2. **Shared -sh rerun** (`_arq_024_shared.py`) tried the same `class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe=True, use_gated_attn=True` pattern. Both flags dropped at runtime → val 6.3419 (same as the other 3 -sh runs). **Wasted rerun — produces the same number as the buggy shared ctrl.**

The 024 treatment used pre-baked `Tiny1M3MGatedAttnOnFireConfig` (use_fire_pe=True, use_gated_attn=True) — trt-side flags are correct.

## Cross-day sanity
- 024-ctrl on this box: 6.4269 (plain Tiny1M3MConfig at seed 42)
- Leaderboard ctrl (2026-06-08): 6.4287 — matches the buggy 024-ctrl (same plain baseline)
- This box's 025-ctrl: 6.4269 (same plain baseline, same seed)
- 024-gated: 6.3316. Compared to a proper fire-equipped baseline (~6.27), the gated effect would be ~+0.06 (FAIL). Compared to the leaderboard plain baseline (6.4287), Δ = −0.097 (≫ WIN bar).

## Pass/fail adjudication
Plan bar: pass iff Δ ≤ −0.01 vs **plan-024 ctrl** (which was supposed to be `use_fire_pe=True` on Tiny1M3MConfig). The plan-024 ctrl actually ran as fire-less, val 6.4269. Within-session Δ = −0.0953 ≪ −0.01. **WIN in the within-session sense.**

But: the within-session delta is the joint (gated+fire) vs (no-gated+no-fire), not (gated+fire) vs (no-gated+fire). FIRE alone gives ~−0.07 (closed.md:40, 009 WIN); isolating the gated-attention contribution requires a proper fire-equipped baseline. The gated effect is likely small or null — gated attention is a paper-validated mechanism (arXiv:2505.06708) but the gain is reported as modest.

## Transfer note
Gated attention (per-head sigmoid output gate post-AV) is a numerically trivial
multiplicative gate — no softmax/renorm tricks, no decay accumulation. Should
transfer cleanly to 135M. The within-session WIN is genuine but includes FIRE
credit; Phase-2 should isolate the gated lever against a proper fire-equipped
baseline before any recipe-promotion call. **Mechanism is real; effect size
at this tier is confounded.**
