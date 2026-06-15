# Taste log — 207 W_O Low-Rank Bottleneck

## r1 — 2026-06-15 — verdict: accept

**Why this is worth a slot:** the 194 (FFN low-rank) taste-reject explicitly invited "drop the FFN-side mechanism and propose the same low-rank correction on a different sub-block (attention Q/K/V/O projection, the residual stream itself, or the embedding-to-hidden lift)." 207 is the exact re-pitch: the rank-r residual correction moves from FFN to **W_O** — a placement where the axis is not exhausted (FFN has 6+ closed nulls; W_O has none).

**Findings:**

- **Crisp bet.** "W_O at 0.94M has intrinsic rank ≤ 16, so a learned rank-r correction exploits this redundancy and improves val loss by Δ ≤ −0.01." Pre-registered, falsifiable on one seed. The mechanism is a structural prior (rank-r subspace for the residual write) — not a vibe, not a HP.
- **Identity/zero-init works.** `α_raw = -10`, `sigmoid(α) ≈ 0` ⇒ `W_O_eff = W_O` exactly at step 0, bit-identical baseline. The +2.6% param inflation only kicks in once the optimizer pulls α away from 0. The reviewer (definition gate) can verify step-0 fp32 max-abs-diff < 1e-6 across all 12 blocks cheaply.
- **Information value is high either way.** WIN = W_O has exploitable low-rank structure at 0.94M (a real, transferable finding — would carry to 135M). NULL = W_O is full-rank at this tier and the rank-r correction adds noise (clean axis-closure at 0.94M, still informative for "is W_O the binding constraint?"). Both outcomes log to closed.md.
- **Transfer risk: med is the right tag.** W_O intrinsic rank scales sub-linearly with model size (effective rank of projection matrices tends to grow ~ √d_model for trained transformers, so 0.94M→135M ≈ 1.7× effective rank growth). A rank-16 correction at 0.94M is a more aggressive prior at 135M but still well within the LoRA literature's r=8-256 sweet spot. Not a tier-mismatched lever like 110-ema/122-tiger/134-mega-ema that physically cannot fire in 92 steps.
- **Portfolio fit: not redundant with the W_O family.** 197 (tied W_O — sharing axis), 199 (spectral W_O — Lipschitz axis), 203 (pre-W_O SE — pre-projection axis) are *different* axes on the same matrix. 207 is the **rank** axis. Tying answers "do blocks need distinct W_O?"; low-rank answers "what's the effective dimensionality of the W_O subspace?"; spectral answers "is W_O Lipschitz-bounded?" — orthogonal hypotheses. Running all four is well-spaced, not crowded. W_O is **not closed** at this tier (no closed W_O-structural-prior nulls; 160-rms-gain-per-head was post-AV magnitude, not W_O structure; 142-layerscale was per-channel diagonal gain, not W_O structure).
- **The 194 prior taste-reject did the right refutation work for us.** FFN low-rank was killed because (a) FFN axis was exhausted and (b) the author predicted null. W_O has neither failure mode. The mechanism itself is clean (residual additive rank-r path); the only question is whether the W_O subspace is low-rank at 0.94M/12L, which is a *new* testable claim.
- **One flag for the runner (not the taste gate).** 197 is in `needs-repitch` and not yet running. If 197 runs first and ties W_O successfully, 207's W_O-share-style information is partly captured — but 207's rank axis is still orthogonal. The runner can sequence, but the taste verdict stands.

**Verdict routing:** `accept` → `needs-review` (round reset to 1 for the definition gate).
