# Evidence — 147 DropKey

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (5b8a7fea8963, RTX 3060, sm_86, driver 580.159.03, commit 42ed363)
- baseline: cached mean=6.4376 ±0.04 (box 5b8a7fea8963, 3 ctrls measured 2026-06-14T04:56:33Z); CACHED path
- treatment val: **6.4612**   train=6.4301   acc=0.1388   Δ vs baseline: **+0.0236**
- bpb: n/a (pending harness — never omit)
- pass/fail bar (idea.md sketch): NULL |Δ| < 0.01; DRIFT > +0.01  → **Δ=+0.0236 sits OUTSIDE |Δ|<0.01 null band and on the wrong side → NULL (DRIFT, marginal)**
- box check: cached mean 6.4376 vs leaderboard 6.4306 — Δ=+0.0070 (within 0.04 noise, **NO DRIFT**)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log 147-dropkey_52674.log alongside)
- date: 2026-06-14

## Run trace
- step 0: loss=10.8123 (bit-identity preserved at init ✓)
- mid-training: ~6.7–6.5 (in ctrl cluster)
- final (step 725): val_loss=**6.4612**, train=6.4301, val_acc=0.1388 (vs ctrl val_acc 0.1443)

Train loss is essentially the same as ctrl (6.4301 vs 6.4242) — the key-drop Bernoulli regularizer is doing its job (no train-side degradation). Val_acc and val_loss both slip marginally (Δ=+0.0236, +0.0055 acc). The regularization lever is alive but at this tier it neither helps nor hurts enough to clear the null band.

## Transfer note
DropKey is a **regularizer for the attention pattern** — keys are randomly masked during QKᵀ so each surviving key must "earn its keep". The original paper and the follow-on ViT applications show a *parity-or-better* val curve on classification (ImageNet, DeiT) at scale, not a quality gain. At 0.94M/12L the per-head d_k=16 means the surviving-key softmax operates on 4–5 keys after the 0.3 default drop — a much harsher regime than the paper's d_k=64. The transfer hypothesis is intact: the lever should compose with any 135M tier without breakage, but it is *not* expected to win at this tier. This closes the key-drop axis at tiny1m3m; do not re-mine DropKey-style variants until the mid-scale tier exists.
