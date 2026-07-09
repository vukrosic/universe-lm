# Review log — 203 pre-W_O SE channel attention

## r3 — 2026-06-15 — verdict: approve

All four r1 findings land cleanly. Round 3 hits the definition-gate cap
(3-round budget exhausted), so the verdict is forced — the lever is
sound, fully specified, and ready to plan.

**Finding A (per-token vs per-sample) — RESOLVED.** The r3
`## Mechanism` block now applies the MLP per-token to the channel
vector, no T-axis pooling:
`se_weight_t(x_t) = sigmoid(W_2 · gelu(W_1 · x_t))`, with `W_1 ∈
R^{d_model × d_model/r}` and `W_2 ∈ R^{d_model/r × d_model}` shared
across all tokens/positions. Matches the plain frontmatter, the
title parenthetical ("Per-Token Channel Reweighting"), the design-
sketch comment, and the closing intuition in "Why it's worth a slot."

**Finding B (concrete pass bar) — RESOLVED.** `## Pass bar` names
the 4-ctrl cluster mean 6.4394, sets WIN at `Δval ≤ -0.01` (clears the
±0.01 box-noise band; must also beat both individual ctrls per the
§2 two-ctrl rule), NULL at `|Δ| ≤ 0.04` (cache band; closes the
post-AV axis family at 0.94M when paired with 142/160/181/191), and
Above-band at `Δ ≥ +0.04` (consider harmful, abandon). The bit-
identity check is now the **gate for the WIN/NULL read** — the
implementer must report `max-abs-diff < 1e-5` at step 0 vs the
same-seed baseline, with the diagnostic that a larger diff means a
wiring bug (γ not on the residual, W_1/W_2 init non-default), not
real signal. Good.

**Finding C (bit-identity wording) — RESOLVED.** "Exactly bit-
identical" replaced with `step-0 max-abs-diff < 1e-5 vs the same-seed
baseline run at fp32`; the text now explicitly notes the internal
`se_weight_t` is ~0.5 (not 1.0) at init and that the γ-gate silences
the branch *anyway* — implementer's self-check is the concrete
number, not an unverifiable claim.

**Finding D (γ_raw → Muon param group) — RESOLVED.** The design
sketch now says: route `γ_raw` to the **Muon** group (1-D gain
scalars benefit from Muon's LR scale; AdamW at peak LR 0.024 is ~10×
too hot for a scalar). Two implementer options are spelled out: (a)
name the param with a `norm`-suffixed key so `muon_for_1d_norm=True`
catches it, or (b) add a small explicit `if 'se_gamma' in name`
branch in the Muon routing. Cites 021/207 reviewer precedent. Fine.

**Other gates (re-confirmed).** Source: Hu et al. SE-Net
(arXiv:1709.01507, TPAMI 2019) + Woo et al. CBAM (arXiv:1807.06521,
ECCV 2018) — both real, well-known. Mechanism is structural (a
small per-token channel-attention MLP with γ-gated residual blend),
not a hyperparameter lever. tiny1m3m only, seed 42. LoC ~30–50 in
`models/layers.py` at the out_proj apply site (well under 200).
Not a duplicate of any closed lever: 142/160/181/191/176 are all
content-INdependent (per-channel *static* gain, per-head *static*
gain, cross-head *static* RMSNorm, per-token *static* scalar, V-side
*static* norm) and 203 is the **content-dependent + per-channel
resolution** cell of the post-AV axis family. Transfer-risk: med
defensible — SE/CBAM validated at ImageNet across all scales; the
*placement* in attention output is novel but the conditioning
primitive (small MLP → sigmoid → multiply) is well-known. Falsifiable
on one seed.

**Routing.** Approve → `needs-plan`, round reset to 1 (per the §3
reset rule for `approve` on round 3 — the code gate gets a fresh
budget).

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