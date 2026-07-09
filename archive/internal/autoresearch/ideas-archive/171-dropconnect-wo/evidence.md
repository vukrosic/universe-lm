# Evidence — 171 dropconnect-wo

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- baseline: cached mean=6.3975 ±0.04 (box 5b8a7fea8963, measured 2026-06-15T05:15:46Z, n=4)
- treatment val: 6.4453   Δ vs baseline: +0.0478 (wrong sign)
- bpb: n/a (pending harness)
- pass/fail bar: Δ ≤ -0.020 for signal, otherwise null per idea.md:148-160 → not met (Δ +0.048 wrong-sign)
- box check: baseline mean 6.3975 vs cached prior 6.4447 = DRIFT (cache re-baselined this queue)
- raw: remote-results/2026-06-15-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-15

## Transfer note
DropConnect without dropout (drop weights, keep activations) is a weight-level regularization axis that hurt rather than helped at tiny1m3m — the +0.048 wrong-sign Δ exceeds the band, meaning the lever is unambiguously bad at 0.94M. At vision scale (DropConnect was originally proposed on CNNs by Wan et al. 2013) the lever is well-validated at ≥100M, but for tiny decoder-only LMs the regularization tax is paid without compensating generalization gain — likely because the model is already small enough that further weight perturbation degrades the loss landscape more than it regularizes. Closes the **weight-level** axis (no-dropout variant) at tiny1m3m; the **activation-level** axis (Dropout, DropPath, DropKey, DropConnect-with-dropout) is independent and may be re-evaluated at ≥135M where FFN/attention capacity becomes the binding bottleneck.