## r1 — 2026-06-15 — verdict: approve

- **Source real and current**: arXiv:2407.06641 (Shi et al. 2024, "Rebased: Linear Attention with Efficient Rebasing") verified — the 154 in-repo WIN is the closest analog and the source paper is correctly cited. RoPE (Su et al. 2021) and 154/172/176/180/181 family are all real, distinct in-repo or external references. No fabricated citation.
- **Mechanism, not HP**: per-head learned orthogonal rotation matrix on K (static, position-independent). Init at θ=0 ⇒ R_h = I ⇒ K = R_h·K exactly ⇒ step-0 byte-identical to baseline (the implementer should verify `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0` as the idea proposes). The orthogonal constraint is preserved because each R_h is a product of 2D rotations on disjoint planes — preserves norms and dot products, no softmax temperature shift. Architectural change, not an LR/schedule/init-constant lever.
- **tiny1m3m only**: every reference is tiny1m3m, seed 42, single-cache reference `5b8a7fea8963` (val 6.3988 ±0.04). No screen20m or larger-tier leakage. ✅
- **Not already closed**: distinct from
  - 154-rebased-attn (WIN, fixed *random shared* rebase of K and V) — 185 is *learned, per-head, K-only*; tests whether optimizer can beat random per-head basis choice.
  - 172-per-head-rope-base (closed null, per-head *position-dependent* RoPE base) — 185 is *position-independent*, a different lever shape (static basis vs angular-frequency modulation).
  - 176-v-pre-av-norm (closed null, V-norm pre-AV) — different tensor and different op.
  - 180-qk-logit-conv (rejected, pre-softmax smoothing) — different op.
  - 181-cross-head-rmsnorm (rejected, cross-head post-AV norm) — different placement and op.
  - 152/155/160/166 per-head scalar family (closed) — 185 is per-head *matrices* on K, not scalars on scores.
  Genuinely fresh lever at this tier.
- **Implementable in < 200 LoC**: small change to `MultiHeadAttention.__init__` (one nn.Parameter, shape `[n_heads, d_k//2]`, init zeros) and `forward` (build block-diagonal R_h from angles, einsum `"hij,bhtj->bhti"`). ~30–50 LoC across `models/layers.py`, `configs/llm_config.py`, `models/llm.py`. Param budget: +384 params (+0.041% of 0.94M) — negligible. No file outside the 154-style footprint.
- **Falsifiable pass/fail bar**: clear WIN/NULL/DRIFT bins tied to the cached control `5b8a7fea8963` (mean 6.3988, band 0.04). WIN = trt ≤ ctrl − 0.005 with two-ctrl clearance; NULL = |trt − ctrl| < 0.01; DRIFT = trt > ctrl + 0.01. Δ bar is tight-but-resolvable at tiny1m3m and matches the project's standard bar convention. Sub-noise effects correctly flagged as inconclusive per one-seed-only rule.
- **transfer-risk justified**: `med` is correct — the lever *form* is novel at ≥100M (no published paper tests exactly this), but the underlying orthogonal-rebase mechanism is well-validated by 154's WIN at 0.94M. The Scale evidence section cites the largest source-paper scale (1B+) and RoPE's validation at 1B–405B for the underlying mechanism. Tag matches the citation.
- **Information value**: high in all three outcomes. WIN → learned per-head rebase binds, unlocks for ≥135M Phase-2 where per-head gradient signal is richer. NULL → 154's binding was the *random* rebase itself (noise-injection aspect), not learnability — closes the learned axis cleanly. DRIFT → learned basis is *worse* than none, strong negative. No "meh" branch.
- **Concrete handoff to plan gate**:
  - Config flag: `use_static_k_rotation: bool = False` on `MultiHeadAttention.__init__` and `LLMConfig`.
  - New subclass: `Tiny1M3MStaticKRotationConfig(Tiny1M3MConfig)` with `use_static_k_rotation: bool = True`.
  - Param tensor: `self.k_rotation_angles = nn.Parameter(torch.zeros(n_heads, d_k // 2))` (init 0 ⇒ identity).
  - Forward: build block-diagonal R_h from angles (no full d_k×d_k materialization needed — apply each 2D rotation to its (2i, 2i+1) plane pair directly via reshape + small matmul on pairs); einsum applies per-head.
  - Verify step-0 byte-identity in smoke before launching the full run.
  - Two-ctrl bracket required for WIN clearance.
- **Verdict: approve** — sound, falsifiable, novel, implementable, information-dense. Proceeding to plan gate; round reset to 1.