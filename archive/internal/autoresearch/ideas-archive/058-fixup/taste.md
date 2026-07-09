## r2 — 2026-06-11 — verdict: accept

- **All three r1 findings closed.** The miner re-pitched to test only
  the *cheapest* Fixup ingredient (zero-init of the last linear in each
  branch) instead of the full no-norm recipe, and the step-0 identity is
  depth-invariant by construction (every block is exact identity at init
  whether 6L or 10000L), so the **mechanism fires at tiny1m3m**. The
  previous depth-mismatch gap is gone.
- **Both outcomes now carry distinct, pre-specified signal at 6L.** WIN
  = ship the implemented `zero_init_resid` flag (zero LoC, no param cost,
  no architectural cost, no schedule change). NULL = the existing RMSNorm
  is doing work *beyond* init-rescaling at 6L — a sharper finding than
  the closed norm-zoo "norm helps at 6L" because it eliminates the
  "norm papers over bad init" sub-hypothesis. The bet as rewritten
  *can* win whether it wins or loses, which is the r1 gap explicitly
  fixed.
- **Mechanistically distinct from the closed norm-zoo and 017.** 017
  adds sub-norms; 019 swaps the norm op (DyT); norm-zoo (pnorm/manhattan/
  center/squash/clip/channelscale) transforms the normalizer. 058 is
  *init-only with the normalizer kept* — same RMSNorm, different
  starting state for the branch outputs. The r1 "3rd norm-axis pitch"
  crowding concern is real but mitigated: this is a different family
  (init-stability, not normalization-operation).
- **Dead-code retirement is a legitimate slot use.** The flag is
  implemented at `models/llm.py:531-538` and committed (4be65bb "residual-
  stream levers") but never ablated. If we never test it, it's misleading
  surface area for future authors. The A/B retires the flag either way
  (ship on WIN, delete on NULL). Marginal cost: ~5 min on a ctrl-bracketed
  10M/30M prep slot; effectively free.
- **Transfer-risk (`med`) is acceptable.** The miner correctly notes
  that the specific "keep norm + zero-init branches only" ablation has
  no published precedent at any scale. The step-0 identity mechanism is
  depth-invariant in *property* (init-time identity holds at any L), but
  the *value* at 135M could be larger (deeper stacks need init
  stabilization more) or smaller (other mechanisms dominate). `med` is
  the right tag; the A/B at tiny1m3m screens the property, and the 135M
  recipe filters the value later.
- **Verdict: accept.** Sharp bet, fits the niche, free to test, retires
  dead code. Resetting `round` to 1 for the definition gate's budget.

---

## r1 — 2026-06-11 — verdict: revise

- **Depth mismatch — the lever doesn't fire at tiny1m3m.** Fixup's headline
  evidence is *10,000-layer* ResNets and deep MT encoder-decoders trained
  without norm. We screen at **6 layers**. The pattern is the same one
  [[017-sub-ln-sandwich]] hit (null at 6L; per DeepNet the lever fires at
  100+ layers) and the one [[018-ademamix]] was taste-rejected for ("bet
  can't fire at tiny1m3m"). At 6L the residual stream's scale drift is
  bounded by depth itself, so *any* sane init (Fixup, GPT-2 0.02, std
  Kaiming) trains fine without norm — that's not new information.
- **Both outcomes are pre-predicted, so info value is ~0 as framed.**
  Null = "norm helps at 6L," already known and consistent with the norm-zoo
  closures (closed.md:25). A small win wouldn't license the headline
  conclusion either: 6L no-norm training succeeds for trivial reasons
  unrelated to Fixup's branch-scaling argument, so it would not actually
  prove "norms are compensating for bad init." The bet as written can't
  win whether it wins or loses.
- **Portfolio: 3rd norm-axis pitch in a row.** [[017-sub-ln-sandwich]] →
  null. [[019-dyt]] → duplicate-reject of closed `squash`. Norm-zoo
  (pnorm/manhattan/center/squash/clip/channelscale) is closed at
  closed.md:25. Fixup is not a *mathematical* duplicate — it removes the
  normalizer entirely rather than swapping it — so it earns a re-pitch,
  but the family is crowded and the bar to enter is higher than r1
  default.
- **How to close the gap (concrete re-pitch).** Reframe so the **mechanism
  fires at 6L** without depending on no-norm training to succeed:
  - Test the Fixup *init recipe alone* (residual-branch scale-down by
    `L^(-1/(2m-2))`, zero-init the last conv/linear of each residual
    branch, keep biases-as-bias) **with RMSNorm still in place**. The
    sharp bet becomes: "if the init recipe alone reduces loss at 6L, it
    proves the existing norm is partly papering over branch-scale
    miscalibration, even at our depth." A null then *is* informative
    (norm is doing real work, not just rescaling) — both outcomes carry
    signal at tiny1m3m.
  - Or — narrower but cleaner — ablate just the **zero-init of the last
    linear in each branch** (the cheapest Fixup ingredient, ~5 LoC) and
    state the bet as "branch-residual identity at step 0 helps the
    AdamW/Muon warmup at 6L." This is identity-able, niche-fitting, and
    doesn't depend on the no-norm regime to win.
  - Either reframe needs one sentence of "we expect X because Y at 6L,
    independent of the 10,000-layer regime."
- **Transfer-risk tag (`med`) is honest but irrelevant if the bet
  doesn't fire here first.** The miner correctly flagged that the
  original evidence is ResNets + MT encoder-decoder, not decoder-only LM.
  Fix the scale-fire problem first; transfer-risk becomes the next gate's
  problem.
