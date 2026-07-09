# Taste log — 010 polyloss

## r2 — 2026-06-09 — verdict: accept
- **Crisp bet — now lands**: the re-pitch took path (1) and frames the distinct mechanism cleanly: CE is the zeroth-order polynomial expansion of `-log p_t` in `(1-p_t)`; the j=1 coefficient fills the *next* Taylor term, the principled correction to CE's truncation error. One sharp "we expect X because Y" sentence — no longer "another over-confidence fix."
- **Portfolio crowding is now moot**: the r1 concern was "2nd loss-shape in a row after 007." 007-sigmoid-loss was **rejected** (arXiv ID unverifiable, closed.md 2026-06-09). 010 is now the *only* loss-shape idea in the active queue — no back-to-back redundancy, no overlapping-mechanism dilution.
- **Info value**: clean standalone A/B. Win or null, it answers "does filling CE's next Taylor term move val loss at tiny scale?" A null is loggable (CE truncation error is negligible at this scale); a win is a cheap, transferable, loss-only lever. Either outcome teaches something.
- **Leverage**: loss-shape edits (content-dependent label-smoothing flavor) are exactly the kind of thing that moves tiny1m3m val loss. Big-if-true, ~3-5 LoC, zero compute cost.
- **Niche fit ✓**: loss-only, identity/zero-init safe (ε=0 ≡ CE), tiny1m3m / seed 42, no model-shape change, no infra/data needs.
- **Dedup ✓**: not in closed.md or LEADERBOARD; poly-expansion of CE not previously run.
- Accept → definition gate (round reset to 1). Spec gate should pin the single ε value (no sweep) and a sharp pass/fail bar so the null stays informative.

## r1 — 2026-06-09 — verdict: revise
- **Leverage**: real — polynomial expansion of CE is a known, cheap lever; ~3-5 LoC, loss-only, identity-safe. The bet ("higher-order term prevents over-confidence and over-smoothing") is *almost* crisp.
- **Info value**: solo, this would be a clean ablation. In context (post-007 sigmoid-loss accept), the marginal info value drops — both are loss-shape, both target the same pathology (CE gradient vanishing on confident predictions / logit-magnitude drift). Whichever wins, the other may be redundant.
- **Portfolio fit**: ⚠️ 2nd loss-shape in a row. 007 (sigmoid BCE + z-loss) and 010 (poly-expansion of CE) probe overlapping territory. Per the taste gate: "diversify even if each is individually fine."
- **Niche fit**: ✓ loss-only, identity-safe, tiny1m3m ✓.
- **Crisp bet**: weak as written. The miner frames it as "another over-confidence fix", which makes it sound derivative of 007. The *distinct* bet is the polynomial-expansion hypothesis: "CE is a low-order Taylor approximation; the j=1 correction is the principled next term." That's a real bet — frame it that way and it stands apart from 007.
- **Revise direction**: re-pitch with a mechanism-distinct framing. Two acceptable paths:
  1. "We expect PolyLoss's j=1 term to win because it corrects a specific Taylor-truncation defect in CE — orthogonal to the bounded-gradient story that drives sigmoid loss." This is the same idea, sharper claim, distinct from 007's mechanism.
  2. Wait: park this one and mine a non-loss-shape idea (architecture / init / PE) for the next round, so we don't run two loss-shape ablations back-to-back.
- The miner picks (1) or (2). Round 2 will accept if (1) lands a distinct-mechanism framing; otherwise a re-pitch to (2) is fine and round 3 forces the call.
