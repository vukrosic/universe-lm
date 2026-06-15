# Evidence — 158 gau

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03)
- baseline: cached mean=6.4346 ±0.0458 (box key 5b8a7fea8963, measured 2026-06-14T06:58:42Z, n=3 ctrls on commit 42ed363)
- treatment val: 6.5441   Δ vs baseline: +0.1095
- bpb: n/a (pending harness)
- pass/fail bar (from plan.md): PASS Δ≤-0.01; NULL |Δ|≤0.01; DRIFT Δ>+0.01
  → Δ=+0.1095 > +0.01 → plan DRIFT (loss); cache NULL (trt sits above mean+band → "wrong sign" per §5)
- box check: ctrl mean 6.4346 vs box-class leaderboard 6.4394±0.04 — within 0.005, no drift
- raw: remote-results/2026-06-14-vast-tiny1m3m/{158-gau_52674.log,run_06-54.json}
- date: 2026-06-14

## Transfer note
GAU fuses Attention+FFN into a single gated unit (−32.5% params at this tier).
Plan explicitly framed PASS as a "−37% param cut recovered by architectural
win" — a strong claim that requires the attention/FFN split to be a binding
bottleneck. The Δ=+0.1095 loss falsifies that: the freed parameter budget is
not re-absorbed into the GAU gating pair at 0.94M/12L. This joins the closed
FFN-side nulls (146-sparse-ffn, 156-moa, 157-conv-ffn) as evidence that
*capacity-and-mixing levers on the FFN axis do not bind at 0.94M*. The
−32.5% param cut simply under-trains relative to the same-token-budget control.
The GAU design point is parameter efficiency at large scale (where the FFN
intermediate dominates), invisible at this tier. Transfer-risk: low;
re-evaluate at Phase-2 ≥135M where the FFN is the binding bottleneck.