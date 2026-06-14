# Evidence — 150 Cross-Layer Feedback Attention (Holtzman et al. 2020)

## Round 1 — needs-recode (failed 2026-06-13)

## Verdict: FAILED → needs-recode
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674
- baseline: cached mean=6.4302 ±0.04 (box 5b8a7fea8963, measured 2026-06-14)
- treatment val: 11.3931   Δ vs baseline: +4.9629 (catastrophic — 125× DRIFT threshold)
- step-0 val: 10.8125 (acc=0.0000) — collapsed from the first forward
- bpb: n/a (pending harness)
- pass/fail bar: PASS trt < cached_mean − 0.04; DRIFT > +0.01  → **met (DRIFT, catastrophic)**
- box check: cached mean 6.4302 vs leaderboard ctrl 6.4306 (within noise)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 150-xlayer-feedback_52674.log)
- date: 2026-06-14

## Failure mode (round 1)
The lever's spec claimed step-0 bit-identity: `xlayer_gate = 0` ⇒
`xlayer_gate * y_xa = 0` ⇒ forward graph identical to baseline. The
empirical reality disagrees — step-0 val_loss=10.81 with acc=0.0000
is far from baseline ~6.43. The model is degenerate from step 0;
the training loss never decreases (final 11.46 ≈ step-0 10.81 +
log(3/0.94)/2 noise). The rc=0 (queue moved on) but the lever does
not train.

## Bounce note (round 1)
Implementer must find why `xlayer_mem` plumbed into the block
changes the forward graph even with `xlayer_gate=0`. Likely
suspects: (1) `XLayerCrossAttn` returns non-zero output even when
Q==0 (catastrophic path); (2) `xlayer_mem.append(x_pre_ffn)` mutates
the residual stream via autograd (inplace op on a node that the
next block reads from); (3) the cross-attn's softmax over a
`(B, K·T, qk_dim)` tensor with K=2, T=2048 saturates and the gate
multiplication is not bit-exact in fp32 when K·T > 4096. The block
smoke (`MinimalLLM(cfg).forward()` bit-identical to off-path) is
not enough — it must include the full forward pass through ALL
12 blocks to catch cross-block mutation.

## Transfer note (round 1)
Cross-layer feedback attention is a real mechanism at 1B+ scale
(Holtzman et al. 2020) but the bit-identity guarantee failed here
at the implementation level, not the mechanism level. The mechanism
bet (selective cross-layer transport beats linear mixing) cannot be
evaluated until step-0 is actually equivalent to baseline. Re-test
only after a verified step-0 smoke (block-level AND 12-block
forward, fp32 max-abs-diff < 1e-6).

---

## Round 3 — needs-recode (failed 2026-06-14)

## Verdict: needs-recode (round 3, cap reached)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03, commit 7f7fe90)
- baseline: cached mean=6.4394 ±0.04 (box 5b8a7fea8963, measured 2026-06-14 from 3 ctrls this queue)
- treatment val: **11.8856**   Δ vs baseline: **+5.4462**   (~136× DRIFT band)
- pass/fail bar: Δ≤-0.01 vs baseline (real WIN only) → **not met** (Δ is +5.45, wrong sign and 545× the bar)
- bpb: n/a (pending harness — never omit)
- box check: cached mean 6.4394 vs leaderboard 6.4306 Δ=+0.0088 (within noise, **NO DRIFT**)
- raw: `autoresearch/remote-results/2026-06-14-vast-tiny1m3m-2/{results.json, 150-xlayer-feedback.log}`
- date: 2026-06-14

## Run trace (round 3)
- step 0: val_loss=10.8125, val_acc=0.0000 → **bit-identity preserved at step 0** ✓ (matches ctrl step-0)
- step 25: val_loss=8.2137, val_acc=0.0228 (learning)
- step 50: val_loss=7.7781, val_acc=0.0562
- step 75: val_loss=7.5528, val_acc=0.0770
- step 100: val_loss=7.3600, val_acc=0.0842 (best)
- step 150: val_loss=7.3041, val_acc=0.0861 (still best)
- step 200: val_loss=7.3384 (turning)
- step 300: val_loss=8.2506 (diverging)
- step 400: val_loss=10.1050 (collapsing)
- final (step 732): val_loss=**11.8856**, train_loss=12.0845, val_acc=0.0191

## Failure mode (round 3)
The tanh-gate fix delayed divergence past step 200 (round 2 diverged at step ~100) but the positive-feedback loop in `xlayer_mem` reasserts: gate opens → xlayer output grows → next block's pre-FFN input grows → that block's xlayer_mem update is larger → gate output grows further. tanh(·) bounds the gate in [-1,1] but does not bound the rate of change in the upstream residual stream. Three failed recodes in a row (round-1 val=11.39, round-2 val=9.77, round-3 val=11.89) — the pattern is consistent and the mechanism is structurally destabilizing at depth-12 × d_model=64.

## Bounce note (round 3, cap reached)
Round=3 hit; per the 3-round cap the implementer must call this — either:
- `done` (null with verdict "mechanism diverges at 0.94M — close axis") and append a line to closed.md, OR
- `rejected` and move folder to `_closed/`.
A 4th recode attempt is **not** in scope per the cap.

## Transfer note (round 3)
The mechanism (cross-block attention with zero-init gate) is real at 1B+ (Feedback Transformer, persistent memory papers) but the d_model=64 / 12L regime is hostile — small width means the cross-attn output vector dominates the residual stream at any nonzero gate magnitude. At 135M (d_model=1024, 24L), the same cross-attn output is proportionally smaller, so the instability would be milder. Recommend: if a future 135M tier exists, retry 150 there as a fresh idea; do not recode 4× at tiny1m3m.

---

## Round 2 — needs-recode (failed 2026-06-14)

## Verdict: needs-recode (round 2)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03, commit 7f7fe90)
- baseline: fresh 3 ctrls same-seed-42 this queue (ctrl=6.4366, ctrl2=6.4287, ctrl3=6.4528) → **mean=6.4394 ±0.04** (cache rewritten — commit-changed trigger)
- treatment val: **9.7731**   Δ vs baseline: **+3.3337**   verdict: NULL by band rule, **but DIVERGED — train loss 9.81, val acc 0.0071** ⇒ not a clean null
- pass/fail bar (plan.md): Δ≤-0.01 vs baseline (real WIN only) → **not met** (Δ is +3.33, wrong sign and 833× the bar)
- bpb: n/a (pending harness — never omit)
- box check: ctrl range 0.0241 within cached noise band 0.04; baseline mean 6.4394 vs leaderboard 6.4306 Δ=+0.0088 (within noise, **NO DRIFT**); cache was rewritten for this box_key — runs_since_measure reset to 0
- raw: `autoresearch/remote-results/2026-06-14-vast-tiny1m3m-2/{results.json, ctrl.log, ctrl2.log, ctrl3.log, 150-xlayer-feedback.log}`
- date: 2026-06-14

## Run trace (round 2)
- step 0: val_loss=10.8125, val_acc=0.0000 → **bit-identity preserved at step 0** (matches ctrl step-0 of 10.81) ✓
- step 100: val_loss=7.3575, val_acc=0.0843 (descending, looks like learning)
- final (step 732): val_loss=9.7731, train_loss=9.8096, val_acc=0.0071 (back up — diverged)

The trajectory (10.81 → 7.36 → 9.77) shows the model trained for ~100 steps then **exploded** in the second half. The mem.detach() fix (round-2 recode) addressed the gradient cascade through earlier blocks' pre-FFN states, but the training is still unstable past ~step 100. Step-0 bit-identity is no longer the failure mode; the failure is now mid-training instability, likely either (a) xlayer_gate opens faster than q/k/v gradients can stabilize, or (b) the `xlayer_mem` truncation + detach leaves a residual coupling that compounds across the 12-block stack.

## Comparison to round 1 (val=11.3931, Δ=+4.96)
The mem.detach() fix helped: Δ went from +4.96 to +3.33 — about 33% reduction in divergence. But Δ is still ~83× the 0.04 noise band. The lever is **not yet runnable at tiny1m3m**. Two recodes have not yielded a stable run; this is a structurally hard problem for cross-layer attention at this depth/width (12L × d_model=64) and warrants either (a) a smaller-window / harder-gated variant (xlayer_k=1, or xlayer_gate=zero with a smaller learnable lr multiplier), or (b) closure of the axis — file it as `closed` if the implementer can't produce a third stable variant.

## Transfer note (round 2)
The mechanism (cross-block attention with zero-init gate) is qualitatively different from value-residual (021, WIN). At 135M (the only "scale" we ever consider, per pipeline), the depth-width ratio is much more favorable to cross-layer attention — 24L × d_model=1024 with the same `K=2` window covers proportionally less depth. If a third recode succeeds at tiny1m3m, it should transfer cleanly to 135M (low transfer-risk despite this miss). If the third recode also diverges, recommend `closed` — the depth-12 × d_model=64 regime appears structurally hostile to cross-layer attention at this scale, but the upstream papers (Feedback Transformer, persistent memory) report wins at 1B+, so the mechanism itself is sound; the cost is the small-tier overhead.