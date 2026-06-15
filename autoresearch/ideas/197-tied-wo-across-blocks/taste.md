# Taste log — 197 Tied W_O Across Blocks

## r1 — 2026-06-15 — verdict: revise
- **Engage the closed priors or lose me.** Two W_O-adjacent nulls are on the
  record. (a) `closed.md:23` — "layer tying" closed (full Universal-Transformer
  parameter sharing, null). (b) `171-dropconnect-wo` — closed 2026-06-15,
  Δ=+0.0478 wrong-sign on a *different* W_O intervention. The current pitch
  names the layer-tying axis but treats it as background, and never mentions
  171 at all. 197 sits on a third W_O-adjacent slot. Re-pitch must explicitly
  defend non-redundancy: "171 is a *regularizer* on W_O weights (noise at
  training); 197 is *parameter sharing* of W_O (structural collapse at
  inference). Even if both null, they kill different mechanistic stories —
  171 says 'W_O has no slack to absorb dropconnect noise'; 197 says 'W_O has
  no slack to absorb block-collapse.'" If the re-pitch can't draw that line,
  this is just the 3rd W_O null in a row and the slot is crowded.
- **Pick ONE of {hard, soft} and commit. The design sketch contradicts itself.**
  "Hard version (proposed for 197)" says block b's W_O slot is removed, but
  then the param accounting in the next paragraph adds 12 α scalars
  (`(1-α_b)·W_O_b + α_b·W_O_shared` at α_b=0 init) — that's the *soft*
  blend. Two different levers, two different param counts, two different
  bit-identity claims. A re-pitch must state: "the proposed A/B is the soft
  blend, α_b init 0, step-0 bit-identical, treatment gains 12 scalars
  (4,108 params), control unchanged. The hard version is a secondary ablation
  reserved for the definition gate, not this pitch." Right now the slot is
  ambiguous and the implementer will pick wrong.
- **Param-reduction fairness if you go hard.** Hard version drops 12 × 4096 =
  49,152 params (-5.2%). Treatment has strictly fewer parameters than
  control — that's a model-size lever, not a parameter-shape lever. If the
  re-pitch insists on the hard version, propose a re-allocation: freed budget
  moves to a wider FFN or an extra head so the A/B is "tied W_O + re-alloc"
  vs "per-block W_O + same compute", not "tied W_O is just smaller." A
  smaller model losing to a bigger model is the wrong null to log.
- **Sharpen the leverage claim with a mechanism, not a NULL-prediction.** The
  crisp-bet sentence inverts its own logic: "a 197 NULL would confirm the
  layer-tying null generalizes to W_O-only" — but the pitch also claims
  W_O-only is *narrower* (less aggressive) than full tying. A *narrower*
  lever's null does not "confirm" a *stronger* lever's null — full tying's
  null is *weaker* (more regularized), so 197's null would *bound the
  failure mode from below*, not from above. Rewrite the bet as: "if W_O-only
  tying wins at 0.94M, the binding constraint of full tying was FFN-tie
  (depth-specific learning died on FFN), not W_O-tie. If W_O-only tying
  nulls, FFN-tie is not the binding constraint and the failure mode is
  elsewhere (e.g. shared QK, shared LayerNorm)." Pick the U-T-paper failure
  mode you are actually testing and say so.
- **Leverage read.** At 0.94M, 5% param reduction is ~49k params — small.
  Combined with the closed-tying-axis prior, the win case is "small
  param-reduction win" and the null case is "third W_O null in a row." The
  taste bar is "big-if-true beats safe-but-tiny" — this leans safe-tiny
  until the re-pitch demonstrates the discriminator is worth a slot.
- **Niche fit / mechanism / transfer-risk: med** all check out. The lever is
  identity/zero-init-able (α_b=0 init = baseline step 0), mechanism-shaped
  (not HP), no data/infra needed, transferable to 135M (ALBERT/U-T partial
  tying is well-validated at 110M-235M). The 3-round cap allows one more
  pitch — use it to (a) commit to the soft version, (b) defend against
  171 + closed-tying, (c) sharpen the bet to name the *mechanism* you're
  isolating.
