# Review log — 203 pre-W_O SE channel attention

## r1 — 2026-06-15 — verdict: revise

**Why it doesn't yet clear the definition gate:**

The mechanism has an internal inconsistency. The plain frontmatter, the
title parenthetical ("Per-Token Channel Reweighting"), and the design-sketch
comment (`se_weight: [B, T, d_model] → [B, T, d_model]`) all describe
**per-token** channel attention. The formula pools over T
(`x_pooled = mean over T axis`), which collapses T and yields
**per-sample** channel weights (the original SE-Net pattern for CNNs).
The lever the taste review accepted is the per-token content-dependent
cell of the post-AV axis family — the formula must match.

Two more tightenings needed: missing numeric pass/fail bar, and the
"exactly bit-identical at step 0" wording is literally wrong (the SE
block's internal sigmoid at init is ~0.5, not 1.0; γ-silence still
gives a clean A/B but the "exactly" claim will fail a strict
baseline-parity check).

**Source real and current.** Hu et al. SE-Net (arXiv:1709.01507, TPAMI
2019) — real, foundational. Woo et al. CBAM (arXiv:1807.06521, ECCV
2018) — real, channel branch is exactly the SE block. In-repo pointers
to 142 / 160 / 181 / 191 all resolve.

**Mechanism is structural, not a hyperparameter.** A small per-token
channel-attention MLP is an architectural lever (per-channel
resolution + content-dependence). γ_raw=-10 keeps the SE branch silent
at step 0 — this survives the per-token fix (the silence is via the
gate, not via the SE block returning 1). Pass.

**tiny1m3m only.** Plan and idea both reference 0.94M / 3M tok /
seed-42 exclusively. No tier-mismatch.

**Not already closed.** 142 (LayerScale — per-channel diagonal,
content-INdependent, residual-stream placement — null +0.0172
wrong-sign), 160 (per-head gain on attention output, content-INdependent
— null -0.0023 inside band), 181 (cross-head RMSNorm, normalization
not attention — null +0.1722 above band), 191 (per-token *scalar* on
attention output, content-INdependent — pending in needs-taste). 203
is the missing "content-dependent + per-channel resolution" cell of
the post-AV axis family — orthogonal to all four.

**LoC well under 200.** SE block (W_1 + W_2 + GELU + sigmoid + multiply
+ γ-gate blend) ≈ 30-50 LoC in `models/layers.py` at the
`out_proj` apply site. Easy.

**Transfer-risk: med, defensible.** SE-Net validated at ImageNet across
all scales (ResNet/EfficientNet). LM-attention-output placement is
novel but the conditioning primitive (small MLP → sigmoid gate → multiply)
is well-known (SE/CBAM/FiLM). Med is the right tag — same logic as
per-token gain at LLaMA/Gemma scale.

**Falsifiable.** Both WIN and NULL are informative: WIN = content-
dependent per-channel reweighting binds; NULL = channel reweighting is
redundant with W_O at 0.94M. The closed neighbors (142 / 160 / 181 /
191) all nulled at 0.94M, so a NULL here closes the post-AV axis
family cleanly.

### Findings (for the reviser; name the section, name the fix)

- **A. Resolve per-token vs per-sample in `## Mechanism`.** This is
  the load-bearing fix. The formula pools over T → se_weight is
  `[B, d_model]` → broadcast to `[B, T, d_model]` → **per-sample**
  channel attention (the original SE-Net for CNNs pattern). The plain
  frontmatter ("each token softly up- or down-weight its own
  channels"), the title parenthetical ("Per-Token Channel
  Reweighting"), the design-sketch comment (`se_weight: [B, T,
  d_model] → [B, T, d_model]`), and the closing intuition in "Why
  it's worth a slot" ("the *content* of the token matters for which
  channels are informative") all describe **per-token**. The lever
  the taste review accepted is per-token content-dependent channel
  reweighting — the formula must match. **Fix:** drop the T-axis
  pooling step. The MLP is applied per-token to the channel vector:
  `se_weight(x_t) = sigmoid(W_2 · gelu(W_1 · x_t))`. Param count is
  unchanged (W_1, W_2 are still `d_model × d_model/r` matrices shared
  across all tokens/positions; no T-axis pooling means no `[B,
  d_model/r]` intermediate). Bit-identity behavior at init is
  unchanged (γ-gate silences the branch, internal sigmoid value doesn't
  matter). Update the design-sketch `## Compute` line and the
  `## Params` count note accordingly.

- **B. Add a `## Pass bar` section with a concrete Δval number against
  a real control.** The idea names a directional bet ("WIN binds /
  NULL redundant") but no numeric Δ. The cache band at tiny1m3m is
  ±0.04 (4-ctrl cluster mean = 6.4394 ± 0.04) and the box-noise band
  is ±0.01. **Suggested bar:** `Δval ≤ -0.01 vs the 4-ctrl cluster
  mean (6.4394) — beats the noise band, must also beat both individual
  ctrls in the cluster per §2 two-ctrl rule`. Both WIN (binds at
  0.94M, transfer to 135M) and NULL (closes post-AV axis family at
  0.94M) are informative — but the bar must be explicit.

- **C. Tighten the "bit-identical at step 0" claim.** `sigmoid(-10) ≈
  4.54e-5`, not 0. With `nn.Linear` defaults (kaiming-uniform init for
  weights, zero init for biases), at step 0:
  - `gelu(W_1 · x_t)` has small magnitude
  - `W_2 · gelu(...)` is near zero
  - `sigmoid(near-zero)` ≈ 0.5 (NOT 1)
  So `se_weight ≈ 0.5` per token, and the blend becomes
  `attn_out_post ≈ attn_out · (1 − 4.54e-5 + 4.54e-5·0.5) ≈ attn_out ·
  (1 − 2.27e-5)`. This is **at the fp32 noise floor** (clean A/B),
  but it is not "exactly bit-identical." **Fix:** replace "exactly"
  with `step-0 max-abs-diff < 1e-5 vs the same-seed baseline run, fp32`
  — a one-sentence precision fix in the `## Design sketch` block.
  The implementer's self-check then has a concrete number to hit, not
  a claim it can't deliver.

- **D. Document the param group for γ_raw in `## Design sketch`.**
  Following 021's value-residual pattern (the only-learned-scalar
  precedent in this repo) and the spec in 207-wo-lowrank-bottleneck's
  review (finding D), `γ_raw` should ride in the **Muon** param group
  (1-D gain parameters benefit from Muon's LR scale; AdamW at peak LR
  0.024 is ~10× too hot for a scalar). One sentence in the design
  sketch, no debate.

### Routing

revise → `needs-revision`, round bumped to 2. Three-revise budget is
available before the definition-gate cap; this lever is sound once
finding A is fixed, so the reviser should be able to land it on round
2 or 3.