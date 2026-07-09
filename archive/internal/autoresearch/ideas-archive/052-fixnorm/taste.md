# Taste log — 052 FixNorm

## r1 — 2026-06-11 — verdict: accept
- Crisp bet: one sentence ("if embedding norm is the bottleneck, FixNorm moves loss; if not, embedding magnitude isn't the culprit"). Sharp, falsifiable, no vibes.
- Niche-fit is strong at tiny1m3m: the embedding table dominates the 0.94M param budget (vocab × d_model ≫ block params), so an embedding-side norm lever is a first-order effect at this tier, not a rounding error. Mechanism (clamp L2 radius) is identity-able and well under any reasonable LoC budget.
- Information value on either outcome. WIN → confirms embedding magnitude is the lever and feeds 135M. NULL → kills embedding magnitude as a hypothesis and *decongests* the broader normalization family in queue (051-scalenorm, 055-deepnorm, 056-branchnorm, 057-normformer, 060-ngpt) — a useful pruning signal even when it loses.
- Family overlap is acknowledged but the bet is *the right place to start* the family. FixNorm is the cheapest isolation: embedding-side only, leaves attention/FFN untouched. 051 (ScaleNorm, same paper) targets the hidden-state norm sites — a different axis. 060 (nGPT) is the strict superset (unit-norm everything); running FixNorm first tells us whether the hypersphere geometry buys anything *beyond* clamping embedding magnitude.
- Non-obvious in this repo: the closed `Norm zoo` (pnorm/manhattan/center/squash/clip/channelscale, closed.md:24) was at hidden-state norm sites — embedding-radius is uncovered. 016-qk-norm already WON (Δ -0.0138, closed.md), so magnitude-control levers do fire at tiny1m3m; embedding-side is the logical next probe.
- Transfer-risk: med is honest — primary evidence is NMT, not LM pretraining — but the paper reports FixNorm remains competitive at high-resource WMT14 EN-DE, and the mechanism is portable (no tiny-vocab exploit). Carries toward 135M.
- Accept: enter definition gate.
