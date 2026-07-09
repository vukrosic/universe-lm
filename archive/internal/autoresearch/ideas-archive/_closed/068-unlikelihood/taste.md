## r1 — 2026-06-11 — verdict: reject

- **Info-free on our screen.** Welleck et al. explicitly claim "maintaining
  perplexity" — the lever's wins are on generation-quality metrics (repetition
  rate, self-BLEU), not next-token loss. Our screen is val loss (CE on a held-out
  set). A null on val loss is *guaranteed by the paper's own framing*, so a
  tiny1m3m run cannot distinguish a working lever from a broken one. The A/B
  teaches us nothing whether it wins or loses.
- **Unsharpened bet for the niche.** "Tiny models often fall into repetition
  before they lose perplexity; unlikelihood attacks that failure mode directly"
  is a generation-quality argument with no "we expect val loss to drop by X
  because Y" sentence attached. The whole pitch lives on a metric we don't
  measure. If the screen grew a generation-quality axis (rep-rate, distinct-n,
  self-BLEU on greedy) this could revive — but as filed, the bet doesn't face
  the right judge.
- **Crowded portfolio (5th loss-shaper in a row).** 066 label-smoothing, 067
  confidence-penalty, 068 unlikelihood, 069 focal-loss, 070 mtp-head — all five
  adjacent slots are loss-side tweaks. Even setting aside the screen-mismatch,
  this is a `revise` on portfolio fit alone: diversify before adding a 6th.
- **Mechanism is a family of HPs, not a lever.** "A tiny set of sampled
  negatives per sequence" punts the actual mechanism: which tokens count as
  negatives (previous-context n-grams? low-MI tokens? high-frequency tokens?
  model-sampled proposals?) and the count/coef of the auxiliary loss are all
  unspecified. The hand-wave is HP-shaped, not mechanism-shaped; the spec gate
  would have to invent the lever.
- **Transfer doesn't rescue it.** Even at 135M the lever's expected win is
  generation quality, so a tiny1m3m val-loss null doesn't bound the 135M claim
  either way.

**Why reject, not revise:** to make this worth a slot the miner would have to
(a) argue against the paper's own "perplexity maintained" claim — i.e. posit a
*mechanism* by which the unlikelihood auxiliary loss *does* move CE on
tiny1m3m (not "attacks repetition," but "regularizes the gradient in a way CE
benefits from") — and (b) name the negative-sampling rule as a single
mechanism, not a knob grid. That's a fresh idea, not a re-pitch; the miner is
better off filing it as a new proposal with the sharpened bet than burning
rounds 2-3 on this card.
