# QK-Gain Leaderboard

![Gain vs Val Loss](runs/qk_gain_vs_loss.png)

## Lock10M200MConfig — 10M params · 200M tokens · ~48828 steps (~33min/run)

| gain | final_vl | vs_baseline | time |
|------|----------|-------------|------|
| 0.0 (baseline) | 5.015 | — | 33m 05s |

![Lock10M200MConfig training curves](runs/lock10m200m/lock10m200m_baseline.png)

## Mini10M20MConfig — 10M params · 20M tokens · ~4880 steps (~4min/run)

| gain | final_vl | vs_baseline | time |
|------|----------|-------------|------|
| 4.0 | 4.9816 | -0.222 | 3m 51s |
| 0.0 (baseline) | 5.2041 | — | 3m 44s |

![Mini10M20MConfig training curves](runs/mini10m20m/qk_gain_mini.png)

## Micro7M1MConfig — 6.7M params · 1M tokens · ~244 steps (avg ~23s/run)

| gain | final_vl | vs_baseline | time |
|------|----------|-------------|------|
| 3.0 | 6.4175 | -0.050 | 22.9s |
| 4.0 | 6.4238 | -0.044 | 23.4s |
| 2.0 | 6.4322 | -0.035 | 23.0s |
| 1.0 | 6.4444 | -0.023 | 22.7s |
| 0.0 (baseline) | 6.4675 | — | 26.9s |

![Micro7M1MConfig training curves](runs/toy_small/qk_gain_fine.png)

## Nano3M32KConfig — 3.2M params · 32k tokens · 8 steps (avg ~9s/run)

| gain | final_vl | vs_baseline | time |
|------|----------|-------------|------|
| 2.2 | 9.2894 | -0.087 | 9.2s |
| 1.6 | 9.3012 | -0.076 | 9.2s |
| 2.3 | 9.3044 | -0.072 | 9.1s |
| 2.1 | 9.3081 | -0.069 | 9.2s |
| 1.8 | 9.3119 | -0.065 | 9.2s |
| 1.9 | 9.3138 | -0.063 | 9.2s |
| 2.0 | 9.3025 | -0.074 | 8.9s |
| 2.5 | 9.3231 | -0.054 | 9.1s |
| 3.0 | 9.3119 | -0.065 | 13.3s |
| 3.5 | 9.3325 | -0.044 | 9.1s |
| 1.7 | 9.3194 | -0.057 | 9.1s |
| 1.5 | 9.3163 | -0.061 | 9.0s |
| 1.0 | 9.3775 | +0.001 | 9.0s |
| 0.0 (baseline) | 9.3769 | — | 13.2s |

![Nano3M32KConfig training curves](runs/toy/qk_gain_fine.png)

## Scaling Trend
Optimal gain: 2.2 (nano/32k) → 3.0 (micro/1M) → 4.0 (mini/20M)