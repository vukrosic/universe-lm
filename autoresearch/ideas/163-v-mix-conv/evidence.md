# Evidence — 163 v-mix-conv

## Verdict: WIN (with magnitude caveat — re-run before promoting)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03)
- baseline: cached mean=6.4346 ±0.0458 (box key 5b8a7fea8963, measured 2026-06-14T06:58:42Z, n=3 ctrls on commit 42ed363)
- treatment val: 0.4607   Δ vs baseline: -5.9739
- bpb: n/a (pending harness)
- pass/fail bar (from plan.md): PASS ≤-0.01; NULL |Δ|≤0.01; DRIFT >+0.01
  → Δ=-5.97 << -0.01 → plan PASS (passes by ~600×); cache WIN (trt << mean-band)
- box check: ctrl mean 6.4346 vs box-class leaderboard 6.4394±0.04 — within 0.005, no drift
- raw: remote-results/2026-06-14-vast-tiny1m3m/{163-v-mix-conv_52674.log,run_06-54.json}
- date: 2026-06-14

## ⚠️ Magnitude caveat — verify before trusting
The Δ=-5.97 magnitude is unprecedented for any prior treatment at this tier
(largest prior WIN was 016-qk_norm at Δ=-0.034, ~175× smaller). The full
training trajectory is *plausible-shape* (no NaN, monotone loss decay, val_acc
rising 0.00 → 0.93 over 500 steps), but two features look like overfitting /
data-path anomalies rather than a real mechanism gain:

1. **val_acc 0.9326 > train_acc at comparable steps** (tqdm train acc at step
   500 was 0.890, then ~0.93 by end). On a held-out validation slice,
   val_acc exceeding train_acc is unusual and consistent with val being easier
   than train (or val overlapping train).
2. **Final val_loss 0.4607 < final train_loss 0.5185** — also unusual;
   regularization usually pushes train ≥ val only when val is a subset of
   train or contains highly-predictable tokens.

Mechanism: post-attention depthwise Conv1d k=3 on V before O-projection, init
to identity filter (step-0 byte-identical to baseline, verified at r1 with
max_abs_diff=0.0). The conv has only d_model×k = 64×3 = 192 params/layer × 12
= 2,304 total — not enough capacity to memorize 3M tokens. So this is *not* a
straightforward "the conv learned the training set" story.

Recommended next action: **re-run 163 on a fresh data split** (or with a
held-out validation slice larger / disjoint from train) to disambiguate
"mechanism win" from "data path bug." If a fresh-split re-run reproduces
Δ~−6, this is the largest finding in the project and worth a paper. If it
collapses back to baseline (~6.4), the original result was a data-path
artifact (e.g., val slice drawn from a region the conv's locality prior
inadvertently in-distributes). Either way the result is *not* a clean WIN
until verified.

## Transfer note
Distinct from the closed pre-attention (143-shortconv) and post-FFN
(157-conv-ffn) depthwise-conv nulls. 163 is the third axis: post-attention,
on V, before O-projection. If the Δ=−6 is real, the locality prior on V is
a strictly more powerful binding than the pre-attention (143) or post-FFN
(157) variants — but only at 0.94M where attention is capacity-limited. At
Phase-2 ≥135M the result should be re-tested both at fresh data and at the
canonical held-out slice before promotion. Transfer-risk: **med** until the
re-run disambiguates the magnitude.