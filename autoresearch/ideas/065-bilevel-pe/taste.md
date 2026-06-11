# Taste log — 065 bilevel-pe

## r2 — 2026-06-11 — verdict: accept
- **All 5 r1 findings closed materially.** (1) Length-extrap framing dropped;
  bet is now a concrete val-loss target (Δ −0.005 to −0.015 vs FIRE-stacked
  ctrl 6.3234). The single-sentence lever — "two-scale frequency partition
  lets the model resolve intra-segment order more sharply than single-band
  RoPE" — is sharp and tier-specific. (2) Segmenter committed to fixed
  S=64 (deterministic, no text metadata, no learned boundary). A null with
  S=64 is interpretable as "the gates didn't move", not "segmentation
  failed" — info-value is preserved. (3) Differentiator named in one
  sentence: "only 065-BiPE tests a two-scale structural partition of the
  position index itself", cleanly disjoint from 061-ALiBi, 062-PosInterp,
  064-XPOS, 072-T5-RPE, 073-DeBERTa. Portfolio-fit concern answered by
  mechanism uniqueness, not queue size. (4) FIRE-stacking story is
  defensible: BiPE is a *rotation* lever on Q/K, FIRE is an *additive
  logit bias* — orthogonal axes, no shared scalar. The CoPE contrast
  (closed.md: trt 6.4659, +0.069) is the right reference: CoPE made
  *position itself* input-dependent and collided with FIRE's content-aware
  bias on the same "content × position" axis; BiPE keeps positions
  deterministic, so there's no double-content-dependence. (5)
  Identity/zero-init pathway is now explicit: pre-scaled inter-band freqs
  + gates g_intra=g_inter=1.0 → composed per-channel rotation equals
  R(p·θ_k) exactly at init, i.e. step 0 is **bitwise identical to plain
  RoPE**, and stacked on FIRE is bitwise identical to 009-fire-pe's step
  0. A null is therefore "the gates stayed near 1.0", not "init pulled
  the model off the baseline manifold".
- **Niche fit is clean.** Mechanism, not HP. Identity/zero-init-able.
  tiny1m3m-testable. Single seed. ~80 LoC. Transfer-risk `med` is honest:
  no published sub-200M evidence either way, but the construction is
  structurally simple and the bitwise-equivalence to baseline at init
  caps the worst case. Even a clean null is worth logging — it cheaply
  retires the two-scale-rotation family of PE designs at our tier.
- **Verdict: accept.** Resetting round to 1 for the definition gate.

## r1 — 2026-06-11 — verdict: revise
- **Headline benefit is OOS at tiny1m3m.** BiPE's paper evidence is *length
  extrapolation*; tiny1m3m is a fixed-length 3M-token run — we cannot observe
  extrapolation here (same reason 009-fire-pe's "## Transfer argument" explicitly
  parks length-extrap for a future tier). The current `Why it's worth a slot`
  ("cheap notion of within/across chunks … may be more useful than token-only
  distance") is a vibe, not a sharp tiny1m3m bet. Re-pitch with a concrete
  val-loss mechanism specific to our tier: e.g. "intra-segment index gives the
  model precise local order while inter-segment index reduces destructive
  interference on attention to the *current* sentence/paragraph, predicting a
  Δ ≈ X on the train distribution".
- **Segmenter rule is unspecified — that's the whole crux.** At tiny1m3m our
  context length is short and there's no document/paragraph metadata in
  pretraining tokens. What defines a segment? Newline? Fixed stride? Learned
  boundary? The mechanism's behavior at our tier is dominated by this choice and
  the idea must commit to one. Without it, a null result is uninformative ("did
  BiPE fail, or did the segmentation fail?") and the info-value bar fails.
- **PE family is over-crowded right now.** The active needs-taste queue has
  061-ALiBi, 062-PosInterp, 063-YaRN, 064-XPOS, 072-T5-RPE, 073-DeBERTa-disent,
  plus 065. Seven PE ideas in a row is portfolio-fit failure even if each is
  individually OK. To survive triage 065 must clearly differentiate from the
  others — *what does BiPE test that no other PE-queue idea tests?* Name it in
  one sentence.
- **Stacking story vs FIRE (closed winner) is missing.** 009-fire-pe is the
  current PE winner (Δ -0.064 to -0.082 ≫ bar). 013-CoPE stacked on top of FIRE
  was destructively closed (closed.md: trt 6.4659 vs ctrls ≈6.39, +0.069). The
  re-pitch must answer: does BiPE *replace* FIRE (then the bet is "BiPE-only
  beats FIRE-only", a high bar) or *stack* on FIRE (then justify why it won't
  reproduce the CoPE+FIRE destructive interaction)? Either choice is fine — but
  it has to be chosen and defended.
- **Identity/zero-init pathway not stated.** Per the niche-fit bar an idea must
  be identity/zero-init-able. The current mechanism note is "small segmenter
  plus two position lookups" — describe the init that recovers RoPE/FIRE at
  step 0 so a null is "the lever didn't fire" not "the init was bad".
