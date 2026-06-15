# Evidence — 144 Mixture of Softmaxes (Yang et al. 2017)

## Verdict: NULL (round 2, K=2)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674
- baseline: cached mean=6.4302 ±0.04 (box 5b8a7fea8963, measured 2026-06-14)
- treatment val: 6.4584   Δ vs baseline: +0.0282 (inside noise band)
- bpb: n/a (pending harness)
- pass/fail bar: PASS trt < cached_mean − 0.04; NULL inside band  → **met (NULL band)**
- box check: cached mean 6.4302 vs leaderboard ctrl 6.4306 (within noise)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 144-mos_52674.log)
- date: 2026-06-14

## Transfer note
Round-2 K=2 chunked mixture of softmaxes. The round-1 K=4 path OOM'd
on the RTX 3060 12GB; round-2 cut K to 2 and chunked along B*T at
chunk=128 to fit. Param count is still 4× the baseline (4.1M vs
0.94M) — the 3.15M fresh vocab-sized heads dominate. Train_loss trt
6.4301 sits in the ctrl cluster (6.42–6.43), confirming the
forward-pass is working but the rank expansion at the output head
does not deliver at this tier. The lever's bet — that the body is
*not* the binding constraint, the output rank is — is wrong at 0.94M
in the same-band regime: even with 4× the params, the model can't
extract the extra rank into val_loss improvement. Closes the
MoS / output-rank axis at this tier. Re-test at 135M where the body
is wider and the rank lever has more room is a future play; the
3× param inflation is the lever's headline confound so same-band is
uninformative between "rank lever null" and "param injection ate
rank benefit."
