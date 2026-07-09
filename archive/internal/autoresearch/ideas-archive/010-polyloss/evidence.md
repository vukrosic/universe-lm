# Evidence — 010 polyloss

## Verdict: NULL (inside variance)
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060)
- control val: 6.5991   treatment val: 6.5938   Δ: **−0.0053**
- ctrl2: 6.6050 (two-ctrl bracket; ctrl-to-ctrl gap **0.0059**)
- pass/fail bar: PASS ≤ −0.005 vs ctrl. Treatment beats ctrl1 by 0.0053 and ctrl2
  by 0.0112, BUT the margin over the nearest ctrl (0.0053) is **smaller than the
  ctrl-to-ctrl spread (0.0059)** → fails the two-ctrl WIN rule → **NULL**.
  PolyLoss (ε₁=1.0) nudges loss down but the effect is inside session variance.
- ⚠️ box check: same +0.19 baseline drift as the 006 batch (session ctrl ~6.60 vs
  prior-day ~6.39) — within-session A/B valid, cross-day not. See 006 evidence.
- raw: remote-results/2026-06-09-vast-tiny1m3m/logs/010-polyloss.log
- date: 2026-06-09
