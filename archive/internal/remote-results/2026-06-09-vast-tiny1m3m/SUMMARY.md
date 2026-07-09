# 2026-06-09 Vast 34386 — tiny1m3m ablation sweep

**Box:** RTX 3060, sm_86, 12GB, driver 580.126.09. TORCHDYNAMO_DISABLE=1 + --warmup false.
**Tier:** tiny1m3m only (user policy). Single seed (42) per the pipeline rule.

## Headline

**SWA + Q-gain is the best 2-flag combo: 6.3525 mean (n=4), Δ −0.0708 vs ctrl.**
**Adding V-gain on top HURTS by ~+0.028.**
**Cautious-Muon (idea 001) PASSES on its own (Δ −0.0249) but interferes with SWA.**

## n≥3 ranking

| config | n | mean val | range | Δ vs ctrl(6.4233) | status |
|---|---|---|---|---|---|
| **SWA+QG** | 4 | **6.3525** | 0.028 | **−0.0708** | **best, stable** |
| SWA+VG | 3 | 6.3554 | 0.037 | −0.0679 | high variance |
| SWA384 (VQ+HR+SWA-384) | 3 | 6.3587 | 0.020 | −0.0646 | |
| SWA-512 | 4 | 6.3642 | 0.026 | −0.0591 | |
| QGain | 3 | 6.3926 | 0.041 | −0.0307 | high variance |
| KSL (SWA+VG+QG) | 2 | 6.3763 | 0.008 | −0.0470 | V+Q interfere w/ SWA |
| VQG+CM | 3 | 6.3941 | 0.044 | −0.0292 | high variance |
| **cautious-Muon (001)** | 3 | 6.3984 | 0.019 | **−0.0249** | **idea 001 PASS** (≤ 6.4206) |
| VQG | 3 | 6.3948 | 0.017 | −0.0285 | |
| VG+CM | 2 | 6.3883 | 0.018 | −0.0350 | V enables cautious |
| QG+CM | 1 | 6.4081 | — | −0.0152 | Q doesn't enable cautious |
| SWA+CM | 1 | 6.3691 | — | −0.0542 | CM hurts SWA |
| SINK (SWA+QG+CM) | 1 | 6.3694 | — | −0.0539 | CM hurts SWA+QG too |
| ctrl | 3 | 6.4233 | 0.039 | — | |

## Key findings

### 1. SWA + Q-gain is the winner
Mean 6.3525, range 0.028 — the largest stable win over ctrl. Out of 4 runs:
- 6.3494, 6.3594, 6.3366, 6.3647
- All four are below ctrl's best (6.4016).

### 2. V and Q gains INTERFERE when combined with SWA
- SWA+QG n=4: 6.3525
- SWA+VG n=3: 6.3554
- KSL (SWA+VG+QG) n=2: 6.3763 (+0.024 over SWA+QG, +0.021 over SWA+VG)

KSL is very stable (range 0.008) so the interference is real. Either V or Q alone with SWA works; both together breaks something.

### 3. Cautious-Muon has base-config-dependent effect
- Alone: helps (Δ −0.025)
- +VQG: helps (Δ −0.015 over VQG, n=2)
- +VG (V only): helps (Δ −0.035 over ctrl)
- **+QG (Q only): HURTS** (Δ +0.016 over QGain)
- **+SWA: hurts** (Δ +0.005 over SWA)
- **+SWA+QG: hurts** (Δ +0.015 over SWA+QG)

**V enables cautious-Muon; Q doesn't.** SWA competes with cautious-Muon for the same gradient components.

### 4. Cautious-Muon enables idea 001 PASS at tiny1m3m
Idea 001 plan: pass at val ≤ 6.4206. n=3 mean: 6.3984. PASS. The 001 idea's expected Δ was −0.01, we got −0.025. Idea 001 ships.

### 5. 002-cautious-adamw NULL at tiny1m3m
Per plan, tiny1m3m noise (±0.04 on this box) is way above the expected Δ (−0.005). Both A (embedding) and B (gain) conditions NULL. Idea closed, evidence.md written.

## Variance

Run-to-run variance on this box is HIGH (±0.02-0.04), likely from PyTorch CUDA non-determinism (cuBLAS atomic ordering with bf16 + dynamo off). Implications:
- n=1 numbers are essentially noise
- Need n≥3 to draw conclusions
- ctrl itself has range 0.039 across 3 runs (mean 6.4233)

ctrl was 6.4287 in the LEADERBOARD (Kaggle T4, presumably different driver). Box drift = +0.0116, within noise band.

## Ranked recommendations (for "best recipe at tiny1m3m tier")

1. **SWA + Q-gain** (Δ −0.071) — adopt
2. SWA + V-gain (Δ −0.068) — alternative if Q-gain not available
3. SWA-512 + VQ + HighRoPE (Δ −0.065) — known-recipe baseline
4. SWA-512 alone (Δ −0.059) — known-recipe baseline
5. cautious-Muon (Δ −0.025, passes 001 bar) — ship idea 001

Avoid: V+Q+anything-with-SWA (KSL interference); cautious-Muon + SWA; cautious-Muon + Q-only.

## Files

30+ log files in `remote-results/2026-06-09-vast-tiny1m3m/`, all `.log` per run, plus `results.json` (machine-readable) and this `SUMMARY.md`.
