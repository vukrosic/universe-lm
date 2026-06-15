---
id: 202-grouped-value-projection
status: needs-review
round: 1
updated: 2026-06-15T08:33:28Z
transfer-risk: med
plain: Probe that isolates the V-axis from the K-axis in MQA/GQA sharing: V-only soft-blend with per-head sigmoid(α_h) gate; reads out α_h trajectory to test the bet that V-redundancy is the binding axis and K-redundancy is not.
---

# 202 — V-Only Soft-Blend Probe (Isolate V-Sharing From K-Sharing)

## Source
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245, 2023). GQA interpolates MQA↔MHA via group size. Validated at Llama-2 7B/13B/70B.
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA, arXiv:1911.02150, 2019). MQA single-shared K, V at PaLM-540B (quality-preserving).
- 178-mqa-gated — null: Δ=+0.1647 at tiny1m3m (inside variance); **evidence.md records val loss only, NOT the per-head β_k/β_v gate trajectory**. Without the trajectory read-out, 178's null is mechanistically ambiguous: we cannot tell whether the optimizer found NO per-head K/V-sharing gradient (family dead) or found one but the val landscape didn't reward it (axis closed at the loss level only).
- closed.md line "MHA vs GQA, MLA, Tied QK" — closed fixed-group-size GQA arch sweep. The closed sweep and 178 both leave the **K vs V attribution** unanswered.

## Mechanism
Per head h, soft-blend per-head V with a group-shared V via per-head sigmoid(α_h):
```
V_h_eff = (1 − σ(α_h)) · V_h_local + σ(α_h) · V_group
```
where `V_group ∈ R^{d_model × d_k}` is one projection shared by the heads in group g (G groups of size h_per_group = H/G). K is **unchanged** — every head keeps its own W_K_h.

At init `α_h = -10` ⇒ σ(α) ≈ 4.5e-5 ⇒ `V_h_eff ≈ V_h_local` (bit-identical to MHA at step 0). K is never touched, so the K-axis is fully held-out and acts as the implicit control.

G=2, h_per_group=2, H=4: 2 V_group matrices + 4 α scalars per block × 12 blocks = 24 V_group + 48 α = 24 × d_model·d_k + 48 = 24,624 params (replaces 48 × d_model·d_k = 49,152; net −24,528, ~−2.6% of 0.94M).

**Why V, not K — the mechanism, not the vibe.** V is mixed post-softmax: `attn @ V = softmax(QKᵀ/√d) · V`, which is a linear combination of V rows weighted by attention scores. There is no nonlinearity between V and the output. K, by contrast, enters through a softmax *argument*; the gradient w.r.t. K is mediated by the softmax Jacobian `diag(p) − p·pᵀ`, which both (a) saturates in the high-score regime and (b) penalizes sharing because the softmax already de-duplicates "what to attend to" across heads. So the V-axis has a direct, linear, near-ungated gradient for sharing; the K-axis has a saturating, head-specific gradient that resists sharing. That is the bet, derived from the softmax path, not asserted as a vibe.

## Design sketch
- **File**: `models/layers.py` — add to `MultiHeadAttention.__init__`: `use_grouped_v: bool = False`, `v_group_size: int = 2`. Allocate `self.W_V_group = nn.ParameterList` of G `nn.Linear(d_model, d_k, bias=False)` matrices. Allocate `self.v_group_alpha = nn.Parameter(torch.full((H,), -10.0))`. K projection is **unchanged**.
- **Forward**: compute per-head V as today (`V_h_local = x @ W_V_h`). For each head h, look up its group g, compute `α_h = sigmoid(self.v_group_alpha[h])`, then `V_h_eff = (1 − α_h) · V_h_local + α_h · V_group_g(x)`. Reshape and continue as today.
- **Init**: `W_V_group_g` initialized to the *elementwise mean* of the in-group per-head W_V_h weights at construction time (so V_group_g(x) is the average of the in-group local V projections at step 0); `v_group_alpha = -10` so `σ(α) ≈ 0` and `V_h_eff = V_h_local` exactly. Bit-identical at step 0.
- **Config**: `use_grouped_v: bool = False`, `v_group_size: int = 2` in `configs/llm_config.py`. New `Tiny1M3MGroupedVConfig(use_grouped_v: bool = True)`. Thread through both `TransformerBlock` sites in `models/llm.py`.

## Scale evidence
GQA at Llama-2 7B/13B/70B (the family is well-validated). The asymmetric V-only form is **novel** — no published paper isolates V-sharing from K-sharing in the soft-blend framing. The closed.md arch-sweep's fixed-group GQA and 178's symmetric gated-blend both touched K and V together. Re-tag `transfer-risk: med` because (a) the mechanism (V-mixing is post-softmax and linear) is a mechanistic argument, not scale evidence, and (b) no published work validates the V-only form at any scale. If the probe produces a clean trajectory read-out at 0.94M, the lever form could be re-pitched with `low` risk.

## Why it's worth a slot — the probe framing (r2)
**This is a probe, not a lever.** The bet, in one sharp sentence: **we expect at least one α_h to move measurably off −10 (i.e. σ(α_h) > 0.05) over the 92-step run, because V enters the loss surface through a near-linear post-softmax path; we do not expect the K-axis to be the binding constraint because the K-axis gradient is softmax-saturated and head-specific.**

Primary metric (success criterion): per-block, per-head final `α_h` values (H=4 per block × 12 blocks = 48 scalars). Recorded at end of training, single tensor dump, no second run needed.
Secondary metric: val loss Δ vs baseline (informative but not deciding).
Falsifiable outcomes:
- (a) **All α_h stay near 0** (σ(α) < 0.05 throughout): the optimizer has no per-head V-sharing gradient. The V-axis is closed mechanistically at 0.94M; 178's null was the K-axis and the V-axis together. Family is dead — append to closed.md.
- (b) **At least one α_h moves off 0 (σ(α) > 0.05) but val loss is null** (inside band): the V-axis has a real per-head gradient that the val landscape didn't reward. Distinct from 178 — that one couldn't separate the axes. This is the high-info-value outcome: it isolates V as the axis that almost-but-doesn't pay off at 0.94M, and tells us the lever form (V-only GQA at the right group size) might win at scale where the val landscape has room to absorb it.
- (c) **At least one α_h moves AND val loss Δ < band** (real win): V-only GQA binds at 0.94M where symmetric MQA-gated doesn't. Promote to a lever idea with sharper scope; the closed.md GQA entry is too broad and should be split.
- (d) **Mixed** (some blocks move, some don't): layer-dependent V-redundancy map. Read out and log; revisit on a 12L+ tier where the pattern matters.

The probe is what 178 was supposed to be but didn't record. This one records.

## Why not just close the family
178 is a val-Δ null; the closed arch-sweep is a val-Δ null. Neither produced a per-axis gradient signal. The closed line is "GQA is closed at tiny1m3m by val loss" — that is a flat statement, not a mechanistic explanation. 202 produces the explanation: it splits the axis and reads the gates. If outcome (a), 202 *strengthens* the closure (mechanistic close). If outcome (b) or (c), 202 *re-opens* the V-axis with attribution. Either is information; neither is wasted compute.

## What the runner must record
Beyond val loss, dump at end of run:
- Per-block, per-head final `α_h` (48 scalars) — the primary signal.
- Per-block, per-head `α_h` trajectory summary (min, max, mean over the run) — secondary.

Single tensor dump. Cost negligible.

## Plan
**Files**
- `models/layers.py` — add `use_grouped_v: bool = False`, `v_group_size: int = 2` to `MultiHeadAttention.__init__`. When on, allocate `self.W_V_group` (`nn.ParameterList` of G `nn.Linear(d_model, d_k, bias=False)`) and `self.v_group_alpha` (`nn.Parameter(torch.full((H,), -10.0))`). At construction, initialize each W_V_group_g to the elementwise mean of the in-group per-head W_V_h weights. In `forward`, compute per-head V as today, compute group-V from `W_V_group_g(x)`, then `V_h_eff = (1 − sigmoid(α_h)) · V_h_local + sigmoid(α_h) · V_group_g(x)`.
- `configs/llm_config.py` — add `use_grouped_v: bool = False`, `v_group_size: int = 2`. New `Tiny1M3MGroupedVConfig(use_grouped_v: bool = True)`.
- `models/llm.py` — thread `use_grouped_v` and `v_group_size` into both `TransformerBlock` sites.

**Config flag**: `use_grouped_v: bool` (default off).

**Step-0 byte-identical**: α_h = −10 ⇒ sigmoid(α) ≈ 0 ⇒ V_h_eff = V_h_local. K is untouched, so the K-axis is the held-out control.
