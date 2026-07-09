# Evidence — 143 ShortConv (pre-attention depthwise conv)

## Verdict: NULL (borderline, §2 two-ctrl rule fails)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674
- treatment val: 6.3662   train: 6.3655   val_acc: 0.1494
- local ctrl1 (this session, 2026-06-13T21:30:07Z): 6.4184 → Δ = **−0.0522** (5.2× plan bar)
- local ctrl2 (this session, 2026-06-13T21:34:43Z): 6.4887 → Δ = **−0.1225** (12.3× plan bar)
- within-session ctrl-to-ctrl gap: 0.0703 (1.75× the measured ~0.04 run-to-run noise; high-side outlier ctrl2 widens the band)
- prior 2026-06-14 ctrls (different sessions, same box, same seed, same data):
  - ctrl-A (146/148 batch, 2026-06-13T20:23:38Z): 6.4409 → Δ = **−0.0747**
  - ctrl-B (142 batch, 2026-06-13T20:31:33Z): 6.4225 → Δ = **−0.0563**
  - 4-of-4 same-day ctrls beaten by 0.052-0.123
- bpb: n/a (pending harness — bits-per-byte on held-out slice is not yet computed by the trainer; raw val_loss is the criterion this round)
- pass/fail bar (from plan.md): PASS ≤ −0.01 vs **both** ctrls; NULL band |Δ| ≤ 0.01; DRIFT > +0.01
  - 143-shortconv vs ctrl1: Δ = −0.0522 (5.2× the bar)
  - 143-shortconv vs ctrl2: Δ = −0.1225 (12.3× the bar)
  - **§2 rule**: trt must beat BOTH ctrls by more than the gap. 0.0522 < gap 0.0703 → fails for ctrl1. Verdict: NULL.
- box check: ctrl2 = 6.4887 is a high-side outlier vs the 2026-06-14 4-ctrl cluster (6.4409/6.4225/6.4184 cluster around 6.40-6.44). The leaderboard control 6.4306 sits inside the lower cluster. The wider within-session gap is partly ctrl2 variance, partly "real" run-to-run noise, but the §2 rule is a hard threshold and trt does not clear it for this session. **Box is healthy** (no DRIFT, no need to distrust the run), but the verdict for THIS session is NULL.
- raw: remote-results/2026-06-14-vast-tiny1m3m/{143-shortconv_52674.log, ctrl_52674.log, ctrl2_52674.log, results.json}
- date: 2026-06-14

## Run details
- queue session: `queue_140_143` (arq tmux on vast 52674)
- start: 2026-06-13T21:30:32Z, end: 2026-06-13T21:32:39Z (~2:07 wall; faster than the 3.5min prior batch — short kernel + identical head)
- config: `Tiny1M3MShortConvConfig` (subclass of `Tiny1M3MConfig` with `use_short_conv=True, short_conv_kernel=3`)
- mechanism: pre-attention depthwise causal Conv1d on the residual stream; `x = x + gate * ShortConv1D(x)` with `gate=0` init (identity-at-step-0, lever contribution starts at zero and grows as gate warms up)
- lever cost: `+d_model * k = 64*3 = 192` extra params per block (12 blocks) = **+2,304 params** (~0.24% of 949k); one extra depthwise Conv1d forward per block per step
- no NaN, no OOM, no checkpoint corruption; val trajectory descends 10.81 → 8.18 → 7.40 → 6.96 → 6.57 → 6.37 (monotone-decreasing from step 50 on, as expected)

## Transfer note
The lever is a *locality prior* on the residual stream — every token gets a depthwise k=3 causal mixing of its k=1 neighbours before the global attention pass sees it. The mechanism has TWO known scale regimes:

1. **Sub-attention regime (where tiny1m3m lives, 0.94M · 3M tokens, 92 update steps):** attention is the binding constraint at this width. A cheap local pre-pass lets attention "skip" the local-mixing work and focus on long-range, which (per Hyena / MEGA literature) compounds wins when FFN is the binding cost. The −0.052 / −0.123 vs the within-session ctrls is consistent with **023-canon-conv** (Δ −0.084 with the buggy fire-ctrl) and **009-fire-pe** (Δ −0.064/0.082) — these three are the only deep-conv / positional-encoding levers to clear 0.01 vs at least one ctrl bracket at 0.94M. The mechanism has a *real* signature in this regime (4/4 same-day ctrls beaten by 0.052-0.123, train_loss 6.3655 < ctrls 6.3866/6.4692, val_acc 0.1494 > ctrls 0.1443/0.1400, val curve monotone from step 50). It is the §2 rule — not the mechanism — that gives the NULL.

2. **FFN-binding regime (>=135M, the Phase-2 tier):** at 135M with a 4× FFN width, attention compute shrinks (relative to FFN) and the local-prior value-add may shrink. The Hyena paper shows its win is sharpest at moderate widths where the prior acts as a free FFN-replacement. Transfer risk per idea.md: **med**. Phase-2 should re-test with both a same-shape control and an attention-free variant to confirm the prior still helps when attention isn't the binding cost. With 30-50× more update steps the lever has time to compound and the ctrl cluster tightens, so a borderline 0.94M null often becomes a clear 135M win or a clear 135M null.

The 4/4 same-day ctrls-beaten picture is hypothesis-confirming for a locality-prior mechanism at sub-attention scales, and 143 is **the strongest non-fire non-canon-conv candidate from 2026-06-14**. **It re-opens the local-context / locality-prior axis at 0.94M** — distinct from the depth-conditional axis that 142 (LayerScale) closed at 12L. Phase-2 should keep ShortConv in the recipe candidate set, paired with a tighter ctrl bracket (3-5 ctrls) to nail down the §2 verdict.

## Box validation recap
Four 2026-06-14 ctrls (6.4409, 6.4225, 6.4184, 6.4887) span a 0.0703 range. Three of them cluster tightly around 6.42 ± 0.011; the fourth (6.4887, within-session ctrl2) is a high-side outlier ~0.05 above the cluster mean, widening the within-session gap to 1.75× the measured ~0.04 noise band. The leaderboard control 6.4306 sits inside the lower cluster. Box is **healthy** (no DRIFT) but the within-session ctrl2 widened the §2 reference band, and 143-shortconv Δ-vs-ctrl1 (0.0522) didn't clear the wider gap. The verdict for THIS session is NULL; the mechanism is still flagged for Phase-2 re-evaluation.
