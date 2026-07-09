# Evidence — 145 Expert-Choice MoE

## Verdict: NULL (DRIFT)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (5b8a7fea8963, RTX 3060, sm_86, driver 580.159.03, commit 42ed363)
- baseline: cached mean=6.4376 ±0.04 (box 5b8a7fea8963, 3 ctrls measured 2026-06-14T04:56:33Z); CACHED path (commit unchanged since 113-galore MEASURE)
- treatment val: **6.5547**   train=6.5265   acc=0.1297   Δ vs baseline: **+0.1171**
- bpb: n/a (pending harness — never omit)
- pass/fail bar (plan.md): PASS Δ≤-0.01; NULL band |Δ| < 0.01; DRIFT > +0.01  → **Δ=+0.1171 ≫ +0.01 → DRIFT (degenerate)**
- box check: cached mean 6.4376 vs leaderboard 6.4306 — Δ=+0.0070 (within 0.04 noise, **NO DRIFT**)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log 145-expert-choice_52674.log alongside)
- date: 2026-06-14

## Run trace
- step 0: loss=10.8123 (bit-identity preserved at init ✓)
- mid-training: ~6.7–6.5 (in ctrl cluster)
- final (step 725): val_loss=**6.5547**, train=6.5265, val_acc=0.1297 (vs ctrl val_acc 0.1443)

Train loss is *better* than ctrl (6.5265 < 6.4242 by ~0.10) — the +1.18M routed-FFN capacity *is* absorbing loss on the train set. But val_acc collapses (0.1297 vs 0.1443) and val_loss regresses (+0.1171) — classic over-fit signature from the extra parameters at 0.94M/12L. The expert-choice load-balance lever can't pay for the capacity overhead at this tier.

## Transfer note
Expert-Choice MoE's headline claim is **load balance by construction** (no auxiliary balance loss) plus parity val at large scale. At 0.94M the 4-expert/1.18M-param overhead swamps the lever — the 12L × d_model=64 stack has no headroom to absorb it. The 117-soft-moe and 118-MoD runs at the same tier already closed the MoE-vs-MoD axis (both NULL). Expert-Choice is the natural third data point and confirms the same conclusion: routing-lever × parameter-golf is not the binding constraint — these mechanisms are sized for ≥135M. Phase-2 tier decisions should **not** re-evaluate expert-choice here; the question for any future mid-scale tier is whether the load-balance property beats 117/118 (token-choice + balance-loss) on a *parameter-matched* FFN budget, not the absolute number.
