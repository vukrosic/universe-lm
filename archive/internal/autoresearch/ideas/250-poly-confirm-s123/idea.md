---
id: 250-poly-confirm-s123
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T02:21:35Z
transfer-risk: low
plain: 3-seed confirm of 230-poly-alibi (the −0.0111 candidate) at seed 123. Tests whether poly-alibi's right-direction nudge is real or single-seed noise, by building a poly 3-seed mean (seeds 42/123/7) to compare against the champion's known 3-seed mean (6.2539). No new model code.
---

# 250 — poly-alibi confirm (seed 123)

Re-run of `Tiny1M3MPolyAlibiConfig` at seed 123. See 230-poly-alibi for the mechanism.
Part of the poly-alibi 3-seed confirm: 230@42 (Δ−0.0111), 250@123, 251@7.
Champion 3-seed mean 6.2539 (42/123/7 = 6.2650/6.2556/6.2412). If poly's 3-seed mean
is clearly below 6.2539 with low scatter, poly-alibi is a real (small) improvement and
a champion-promotion candidate; if it straddles the champion mean, the curvature axis
is noise-bound at this tier.
