# taste.md — review notes per round (newest on top)

## r1 — 2026-06-15 — verdict: accept

- **Sharp two-sided bet.** Tests an unexplored slice of the orthogonal-rebase
  axis that 154 established as the strongest 0.94M lever (WIN Δ=-3.48). The bet
  is positioned correctly: "if the rebase axis is binding, pre-RoPE also binds;
  if the axis is specifically post-position, pre-RoPE nulls" — both outcomes
  are informative about the rebased-attn family.
- **Fresh slice, not a duplicate.** 154 = fixed random rebase on K and V
  (pre-softmax). 185 = learned K-only post-RoPE (closed via recode-cap, never
  cleanly nulled on data). 192 = learned Q+K pre-RoPE. Three different
  placements on the same family is the right portfolio density — neither
  crowded nor thin.
- **Mechanism, not HP.** 4 heads × 8 pairs × 12 blocks = 384 φ scalars
  (+0.041%) — structural lever, not a sweep parameter.
- **Identity init, byte-identical at step 0.** φ=0 ⇒ cos(0)=1, sin(0)=0 exact
  in IEEE 754 fp32, so the static rotation is `R=I` exactly ⇒ forward graph
  matches baseline exactly. The definition gate can verify this cheaply with
  `max_abs_diff(step0_logits) == 0.0`.
- **Niche fit.** Tiny1m3m capable (params +0.041%); transfer-risk med is
  correctly tagged — RoPE family wins at LLaMA-1/2/3 / Mistral / Qwen ≥7B,
  and the lever's mechanism (learned basis change before position) is
  structurally independent of sequence length, so transfer is plausible.
- **Lower implementation risk than 185.** 185's churn was post-RoPE K-only
  with placement after position mix; 192 is pre-RoPE on Q+K — the static
  rotation layers cleanly BEFORE RoPE with no interaction with the
  position-dependent angle path. The "where to insert" question has one
  obvious answer (right before the existing RoPE block). Implementation is
  simpler than 185's, not equivalent.
- **Why not revise.** The bet is already crisp; the niche-fit checklist is
  clean; the closed-axis background check passes (154 WIN, 185 closed-recode
  not closed-data). The only thing a revise could ask for is a tighter
  pre-registration of the expected Δ, but at this stage one clean A/B teaches
  more than another spec round.
- **Pass to definition gate.** Round reset to 1 (definition's own budget).