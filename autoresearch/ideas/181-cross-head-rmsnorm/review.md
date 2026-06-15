# Review log — 181 cross-head-rmsnorm

## r1 — 2026-06-15 — verdict: revise

**Source check — passes.**
- NormFormer (Shleifer et al. 2021, arXiv:2110.09423) — real, ViT-class
  + small LM validation at 100M-300M.
- RMSNorm (Zhang & Sennrich 2019, arXiv:1910.07467) — canonical
  reference, well-validated at 7B-70B (LLaMA-3, Qwen-2, Gemma-2).
- The **cross-head axis** (normalize across H heads within each d_k slice,
  as opposed to standard post-AV RMSNorm which normalizes over the
  concatenated d_model axis) is novel at this tier. Citations hold.

**Mechanism check — passes (with cleanup finding).**
- Structural change: per-(b,t,k) position, RMS-normalize across H heads
  of `out = attn_w @ V ∈ R^{B×H×T×d_k}`, then apply a per-(h,k) gain
  γ_h[k] with init that yields identity at step 0.
- Step-0 byte-identity: the sketch first states `γ_h[k] = 1` ⇒
  byte-identical, then correctly catches the error (the bare RMSNorm
  *is* a rescaling even with γ=1) and proposes `γ_h[k] = 1 + tanh(γ_raw_h[k])`
  with γ_raw init 0 ⇒ γ=1 ⇒ RMSNorm normalized value × 1 = normalized
  value, but that is **still not byte-identical to baseline** — the
  bare RMSNorm divides by RMS even at γ=1.
- **Cleanup finding #1 (reviser-actionable, blocking):** the sketch
  has two contradictory paragraphs in `## Mechanism` and `## Design
  sketch`. The correct byte-identity formulation needs an additional
  gating scalar, **or** the lever is acknowledged to be NOT
  byte-identical at step 0 but only approximately-identity (RMSNorm
  is a rescaling; tanh gate on the gain does not turn off the
  RMSNorm itself). Two valid paths:
  1. **Single-gate approach** (recommended): add a learnable
     scalar `α ∈ [0,1]` init 0 (e.g. `α = sigmoid(α_raw)` init α_raw=-5
     so sigmoid(-5)≈0.0067, OR `α = relu(α_raw)` clamped), with
     `out_norm = (1-α)·out + α · (out / rms) · γ`. At α=0, the
     output is exactly `out` (byte-identical). α is a single scalar
     per block, +12 params total. γ_h[k] gain stays as proposed
     (init 1 via tanh).
  2. **Acknowledge non-byte-identity** and require the runner to
     verify step-0 max-abs-diff < 1e-4 (fp32 noise floor) instead
     of byte-identical. Less clean; the working tree's norm zoo
     closed nulls assume exact byte-identity for cheap A/B, so the
     gate-α approach is preferred.
  Either path: lock the parameterization in one paragraph and remove
  the contradiction. The sketch's `α-gate` mention at line 42 is
  the right direction — promote it to the canonical formulation.

- **Cleanup finding #2 (reviser-actionable, blocking):** the design
  sketch has an arithmetic error at the H=4, d_k=16, n_layers=12
  param count. H×d_k = 64 per block × 12 blocks = 768 params total
  (the sketch states 768 in `## Design sketch` line 43 but says
  "Per block: 4·16 = 64 params" then "Total: 768 params (+0.081% of
  0.94M)" — this is consistent). Add the per-block param count
  explicitly to the plan, and if the gate-α approach is used, add
  +12 for the 12 per-block α scalars.

**Tier check — passes.** Plan runs at tiny1m3m (0.94M · 3M tok,
seed 42). No reference to screen20m, full ladder, or multi-tier.

**Closed-axis dedup — passes.**
- 160-rms-gain-per-head (NULL) — per-head scalar gain on
  `o_h = (A·V)_h`, independent per head, post-AV. 181 normalizes
  *across* heads (couples them) before applying a per-(h,k) gain.
  Different axis: 160 doesn't change relative head magnitudes, 181
  removes them. The cross-head axis is **not** what 160 closed.
- 176-v-pre-av-norm (in queue, approved r2) — V-pre-AV
  RMSNorm. Different position (V pre-AV vs attention output
  post-AV), different normalization target (V ∈ R^{B×H×T×d_k}
  same shape as 181's `out`, but 176 normalizes each head's V
  independently along d_k via per-head RMSNorm, while 181
  normalizes ACROSS heads along H).
- 162-q-only-norm (NULL), 165-k-only-norm (NULL) — pre-softmax QK
  RMSNorm. Different tensor, different position.
- 152-attn-logit-bias (NULL), 155-per-head-temp (NULL), 166-t5-rpe
  (NULL) — per-head attention-shape levers, smooth perturbations.
- 154-rebased-attn (WIN) — rebases K and V pre-softmax. Different
  operation.
- 173-entmax-15 (in queue) — softmax replacement. Different
  operation.
- 107-exclusive-self-attn (NULL), 109-kda-channel-gate (NULL),
  024-gated-attn (WIN w/ caveat), 045-attn-output-gate — output-side
  gates (per-head scalar / channel / input-conditional sigmoid).
  None of them change the *relative magnitudes between heads*. 181
  is the first closed-or-active lever that couples H axes.
- closed axes line "NSA / diff-attn / hybrid heads" — diff-attn
  is a smooth post-QK operator replacement. 181 is a normalization,
  not an operator replacement.
- Not in `_closed/`. Not in `closed.md`. ✓

**LoC budget — passes.** Cross-head RMSNorm is ~10 LoC (compute
RMS over H dim, normalize, multiply gain), MHA plumbing ~15 LoC
(field + parameter + apply), config subclass ~10 LoC, llm.py
threading ~6 LoC per site. Total ~40-50 LoC, well under 200.

**Falsifiable bar — MISSING (reviser-actionable, blocking).** No
`## Pass / fail bar` section with numerical thresholds. Lock:
- **control** = unmodded `Tiny1M3MConfig` (no cross-head RMSNorm)
- **WIN** = Δval ≤ -0.005 vs cached baseline (≈6.4394 ± 0.04 per
  baseline-cache.json); mirrors 016-qk-norm's bar; clears the
  ±0.04 box noise at tiny1m3m by ≥8×
- **NULL** = |Δ| < 0.005
- **DRIFT** = Δval ≥ +0.005 (cross-head coupling breaks W_O's
  pre-trained magnitude assumptions ⇒ expected to be small but not
  catastrophic; +0.005 catches any wrong-sign axis collapse)
- Sub-noise is **inconclusive** per one-seed-only — no re-run with
  extra seeds. Use the cached baseline as the reference (per the
  154-rebased-attn precedent).

**Transfer-risk — passes.** Tag `transfer-risk: med` is justified:
- Primitive (RMSNorm on attention output) is scale-validated at
  1B+ (LLaMA-3, Qwen-2, Gemma-2). NormFormer at 100M-300M.
- The cross-head axis is novel at 100M+. Direct LM validation
  absent.
- `med` is the right tag (not `low` because the cross-head axis
  is novel at scale, not `high` because the primitive is
  scale-tested in adjacent forms).

**Plan section missing — MUST ADD (reviser-actionable, blocking).**
`idea.md` has no `## Plan` section. The 169/165/176/173 reviewers
all flagged the same thing. Add a `## Plan` that locks:
- **Field name**: `use_cross_head_rmsnorm: bool = False` on
  `LLMConfig` (sibling of `use_head_gain` at `configs/llm_config.py`
  — the 160 flag already exists).
- **Config subclass**: `Tiny1M3MCrossHeadRMSNormConfig(
  Tiny1M3MConfig)` with `use_cross_head_rmsnorm: bool = True`,
  `@dataclass` decorated (per the 162/165/155/161/176 precedent
  that bare-class annotation breaks dataclass field inheritance).
- **MHA kwarg plumbing**: add `use_cross_head_rmsnorm: bool = False`
  kwarg to `MultiHeadAttention.__init__` (sibling of
  `use_head_gain` at line 809 in current `models/layers.py`).
  Register parameters when flag is on:
  - `self.cross_head_rmsnorm_alpha_raw = nn.Parameter(torch.zeros(n_heads))`
    (per-head gate-α, init 0 ⇒ α=0 ⇒ byte-identical at step 0)
  - `self.cross_head_rmsnorm_gain_raw = nn.Parameter(torch.zeros(n_heads, d_k))`
    (per-(h,k) gain, init 0 ⇒ γ = 1 + tanh(0) = 1)
- **Apply site**: after computing `out = attn_w @ V` (shape
  `[B, H, T, d_k]`) at the AV-product step in MHA.forward. Compute
  `rms = sqrt(mean(out.pow(2), dim=1, keepdim=True) + eps)` (mean
  over H axis). Apply:
  `out = (1 - alpha_h) * out + alpha_h * (out / rms) * (1 + tanh(gain_raw))`
  where `alpha_h` is the per-head gate broadcast over d_k. Apply
  BEFORE the existing `use_head_gain` site at line 3616 (so the
  160 post-AV per-head scalar gain composes on top, not
  underneath; if both are off, no branch taken, baseline graph
  bit-identical).
- **Mutual exclusion asserts** (top of MHA.forward, mirror the
  `use_cope ∧ use_qk_norm_post_rope` assertion pattern):
  - `assert not (self.use_cross_head_rmsnorm and self.use_head_gain)`
    — the two compose mathematically, but isolating the cross-head
    axis requires turning 160 OFF; assert prevents the implementer
    from accidentally turning both on.
  - `assert not (self.use_cross_head_rmsnorm and self.use_attn_output_gate)`
    — same reasoning.
  - `assert not (self.use_cross_head_rmsnorm and self.use_gated_attn)`
    — same reasoning.
- **TransformerBlock pass-through**: add `use_cross_head_rmsnorm:
  bool = False` to `TransformerBlock.__init__` (sibling of
  `use_head_gain`), pass into the MHA constructor.
- **llm.py read+thread**: add `self.use_cross_head_rmsnorm =
  getattr(config, "use_cross_head_rmsnorm", False)` in
  `MinimalLLM.__init__` (sibling of `self.use_head_gain`), thread
  into both `TransformerBlock(...)` constructor sites.
- **Param count**: per block = H × 1 (alpha_raw) + H × d_k
  (gain_raw) = 4 + 64 = 68 params × 12 blocks = 816 params
  (+0.087% of 0.94M). Mirrors 176-v-pre-av-norm's exact param
  count (which is also H × (1 α + d_k γ) per block = 68 per
  block × 12 = 816).
- **Runner stub**: `_arq_181-cross-head-rmsnorm.py` mirroring the
  162/165/169/170/176 pattern (`build
  Tiny1M3MCrossHeadRMSNormConfig`, `config_class`,
  `/venv/main/bin/python _arq_181-cross-head-rmsnorm.py`).
- **Identity-init tolerance**: with the gate-α approach, step-0
  is byte-identical (max-abs-diff = 0.0). Plan should call for a
  fp32 max-abs-diff test in the runner (`trt_step0_logits ==
  ctrl_step0_logits` byte-exact). If the second path
  (non-byte-identity) is chosen, require max-abs-diff < 1e-4.

**Coordination note (non-blocking).** Working tree has 162/165/
166/167/168/169/170/171/172/176 hunks landed (per `git log --oneline`).
181 hunks are not yet present. No conflict. Running `git diff` to
verify before editing is the reviser's job, not the reviewer's
(the review gate is plan-not-code).

**Verdict**: revise → `needs-revision`. Mechanism is sound and
the cross-head axis is genuinely novel (not closed by 160's
per-head scalar gain, 176's V-pre-AV RMSNorm, or any of the
pre-softmax QK levers). Findings are:
1. Promote the gate-α to the canonical parameterization (cleanup
   finding #1) — or commit to non-byte-identity.
2. Re-derive param count cleanly with gate-α included (cleanup
   finding #2).
3. Add `## Plan` section locking field name, config subclass, MHA
   plumbing, apply site, mutual-exclusion asserts, param count,
   runner stub.
4. Add `## Pass / fail bar` with numerical thresholds (WIN ≤ -0.005,
   NULL |Δ| < 0.005, DRIFT ≥ +0.005).

All four are reviser-actionable without further review. Round
stays at 1 (no round reset on revise).