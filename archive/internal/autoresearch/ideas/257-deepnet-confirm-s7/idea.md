---
id: 257-deepnet-confirm-s7
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T03:30:37Z
transfer-risk: low
plain: 3-seed confirm of 253-deepnet-alpha (the −0.0230 candidate, best signal of the session) at seed 7. Tests whether deepnet-alpha's right-direction gain is real or a lucky seed, vs the champion 3-seed mean 6.2539. No new model code.
---

# 257 — deepnet-alpha confirm (seed 7)

Re-run of `Tiny1M3MAlibiConfig + use_deepnet_alpha` at seed 7. See 253-deepnet-alpha-alibi.
Part of the deepnet-alpha 3-seed confirm: 253@42 (Δ−0.0230), 256@123, 257@7.
Champion 3-seed mean 6.2539 (42/123/7 = 6.2650/6.2556/6.2412). If deepnet's 3-seed
mean is clearly below 6.2539 with low scatter, it is the first real challenger to
alibi this session (promotion/stack candidate); if it straddles, residual-scaling is
noise-bound at this tier like the curvature axis.
