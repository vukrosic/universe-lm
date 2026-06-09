# Taste log — 008 gated-deltanet

## r1 — 2026-06-09 — verdict: reject
- **Off-niche on tier.** The proposal explicitly asks for `screen20m+` ("need enough tokens to see a val-loss signal on a 135M target"). The pipeline runs **one tier only — `tiny1m3m`** (0.94M params · 3M tokens, seed 42). Per the taste gate: "An idea whose payoff only appears at larger scale, or needs data/infra we don't have, has no taste here regardless of how good the paper is."
- **Leverage (in our tier)**: actually arguably real — linear vs softmax attention is a structural lever and a null at tiny1m3m would be informative. But that's a *re-pitch* (drop the screen20m+ framing; the bet becomes "linear attention closes the val-loss gap at tiny1m3m"). A re-pitch is a fresh mine, not a revision of this one.
- **Why reject instead of revise**: the gap isn't "bet is dull" — it's "you proposed a tier we don't run." That's an off-niche reject, not a bet-sharpen revise. The miner is welcome to mine a tiny1m3m-framed linear-attention variant from scratch (e.g. as a swap-in for 004-retnet-retention's result).
- **Re-pitch path**: if a tiny1m3m A/B is desired, file as a new idea with that tier explicit and a crisp null/positive bet. Don't reuse this slug.
