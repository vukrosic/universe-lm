# Review log — 173 entmax-15

## r1 — 2026-06-15 — verdict: approve

**Note on gate flow:** Idea arrived with `status: needs-run` (the implement-button flipped past `needs-review` without a reviewer pass; prior workers exited before writing review.md). Reclaimed to `reviewing` and re-running the definition gate from this review. Code already on disk is consistent with the plan and passable on its own terms — this verdict stands on the *idea spec*, not on the code.

**Source check — passes.**
- Peters/Niculae/Martins, "Sparse Sequence-to-Sequence Models", ACL 2019, arXiv:1905.09018. Real, current (still cited by 2024-2025 LLM-attention surveys as the canonical sparse-attention reference).
- Lorena et al. arXiv:2307.13011 — confirms α=1.5 as the practical default. Cross-references the original.
- arXiv IDs resolve; authors and venue are plausible. No fabrication.

**Mechanism check — passes.**
- Structural change: replace `torch.softmax(scores, dim=-1)` with `entmax_15(scores, mask, alpha_h, dim=-1)` — a Tsallis α-entmax projection via bisection on λ.
- Step-0 ≡ baseline: `α_raw_h = 0` ⇒ `α_h = 1 + 0.5·(1 + tanh(0)) = 1` exactly. The helper short-circuits to `torch.softmax` when `amp1.abs().max() == 0` (models/layers.py:156) — perfect step-0 bit-identity, not approximate.
- Per-head learnable scalar (12L × 4H = 48 params, +0.005% overhead) — not an LR/init-constant lever.
- α ≥ 1 only (parmeterization `α_h = 1 + 0.5·(1 + tanh(α_raw_h)) ∈ [1, 2]`); softmin / "negative-sparsity" modes impossible by construction. Safe.

**Tier check — passes.** Plan runs at tiny1m3m (0.94M · 3M tok, seed 42). No reference to screen20m, full ladder, or multi-tier promotion.

**Closed-axis dedup — passes.**
- Closed levers in `closed.md` that touch the softmax-replacement axis:
  - 148-focal-mod (NULL) — *replaces attention block* with gated-additive context. Different architecture.
  - 156-moa (NULL) — parallel-attention-experts + router. Different mechanism.
  - 152/155/160/162/165/166 — per-head attention-shape levers (smooth perturbations of softmax). Different family.
- Closed axes line "NSA / diff-attn / hybrid heads" — diff-attn is smooth post-QK (`out = φ(Q)φ(K)^T · V` with smooth φ), entmax-1.5 is a non-smooth simplex projection. Different family. The taste-reviewer's r2 verdict explicitly separated entmax-1.5 from this axis on this point and that's correct.
- Entmax-1.5 is not a duplicate of any closed lever.

**LoC check — passes.** `entmax_15` helper ≈107 LoC (lines 97-224 of models/layers.py), MHA integration ≈15 LoC (lines 1356-1370), config field 1 LoC, llm.py pass-through 2 LoC per site. Total ≈130 LoC, under the 200 budget.

**Falsifiable bar — passes.** Plan commits to "Δ ≤ -0.015 WIN OR Δ ≥ +0.05 DRIFT; anything inside the |Δ|<0.01 null band is a clean close." Tight, falsifiable, with the noise band explicitly anchored to the cache (6.4394 ± 0.04). Box noise ~±0.01 val loss acknowledged and the bar is above it.

**Transfer-risk tag — passes.** `transfer-risk: med` is justified:
- Mechanism is scale-free (Tsallis entropy has no depth/width prior).
- Closest validation: Peters et al. 2019 on WMT'14 De-En Transformer (~140M) and BERT-base GLUE (~110M). ≥100M direct validation.
- Domain mismatch: those are encoder-decoder translation + classification, not decoder-only causal LM. The path is one domain removed from our setting.
- Lorena 2023 analyzes gradient flow on α=1.5 — supports the choice without being a LM-specific validation.
- `med` is the right tag (not `low` because of the domain gap, not `high` because the mechanism is general).

**Plan correctness — passes (already on disk).**
- `models/layers.py:97-224` — `entmax_15` helper with bisection, fully-masked-row handling, fp32 internal compute, α=1 short-circuit.
- `models/layers.py:1356-1370` — `entmax_alpha_raw` registered when `use_entmax=True`, gated out when off (no Parameter registered, baseline graph untouched).
- `models/layers.py:3096-3108` — `use_entmax` correctly added to the elif chain that forces the manual attention path (entmax-1.5 can't go through SDPA's flash kernel — correct).
- `models/layers.py:3251-3265` — swap site. Replaces `torch.softmax` with `entmax_15(scores, window.view(...), alpha_h, dim=-1)`. Per-head `α_h = 1 + 0.5·(1 + tanh(α_raw_h))` derived on the fly.
- `configs/llm_config.py:397` — `use_entmax: bool = False` default off.
- `models/llm.py:323-324, 697-698, 976-977` — kwargs plumbed through both YOCO upper-half and standard transformer block paths.

**Distinguishing r2 claims — verified.**
- Family 1 (operator perturbation: 152/155/160/162/165/166) — smooth, small-Lipschitz, absorbed by Q/K gradients. Closed nulls at this tier.
- Family 2 (operator replacement non-attention: 148) — focal modulation replaces the block. Different architecture.
- Family 3 (capacity injection: 117/118/146/156) — MoE/router/expert levers add parameters.
- Entmax-1.5: bit-identical at step 0 AND non-perturbative as `α_h` moves (a single bit of α_h crossing 1.0 introduces a hard-zero mass regime — ∂L/∂V_i = 0 exactly for zeroed-out rows). It is *isolated* to one axis (`α_h`) — no other parameter to absorb the change. The "non-smooth operator change" framing is real and meaningfully different from the 8 smooth siblings.

**Honest Δ prior.** 70% in-band null, 20% mild WIN, 10% DRIFT/strong WIN. The reviewer-tighter bar `Δ ≤ -0.015` committed at r2 is appropriate; this is not a vibes-claim.

**Implementation risk — low.** Step-0 byte-identity is *exact* (the helper short-circuits to `torch.softmax`), not approximate. The bisection converges to <1e-7 in ≤25 iterations for n_keys ≤ 4096 (T=2048 in tiny1m3m). Masked positions use `-inf` and the helper handles fully-masked rows defensively (sets them to zero output). fp32 internal compute, cast back to input dtype at the end. The only concern would be if `amp1.abs().max().item()` causes a CUDA sync per forward call — this is a known `.item()`-induced sync, but for tiny1m3m at 4H × T² = 16M ops per layer the sync is negligible. Worth noting in evidence.md if it ever shows up in profiling, not blocking here.

**What a null teaches (per the r2 framing).** Even if entmax-1.5 closes null, it closes the *operator-replacement* family with a stronger test than the 8 smooth siblings — if a non-perturbative operator change can't bind at this tier, the soft-perturbation nulls are independently confirmed. DRIFT (Δ ≥ +0.05) saves a Phase-2 slot. All three outcomes informative. This is a clean r2 → run.

**Resetting round to 1** for the code gate's fresh budget.

### Verdict: approve → needs-plan (round 1)