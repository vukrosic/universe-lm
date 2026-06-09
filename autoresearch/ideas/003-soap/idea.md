---
id: 003-soap
status: planning
round: 2
updated: 2026-06-09T00:39:45Z
---

# 003 — Soap (Shampoo + Adam)

## Source
Vyas, Morwani, Zhao, Kwun, Shapira, Brandfonbrener, Janson, Kakade — "SOAP: Improving and Stabilizing Shampoo using Adam" (arXiv:2409.11321, Sep 2024; v2 Jan 2025). Code: https://github.com/nikhilvyas/SOAP.

## Mechanism
Showed Shampoo (1/2-power) is mathematically equivalent to Adafactor in the preconditioner's eigenbasis. SOAP runs Adam inside that rotated basis and refreshes the eigenbasis only every K steps (one new hyperparameter: preconditioning frequency). Inherits Adam's simplicity + Shampoo's curvature benefits. Paper: 40%+ fewer iterations, 35%+ wall-clock savings vs AdamW on 360M/660M LM pre-training. Implementation is < 200 LoC. Propose: SOAP replaces AdamW (1D + embedding), Muon stays for 2D hidden.

## Pass / fail bar
- pass: screen20m val ≤ 4.5887 (ctrl 4.6364, target Δ = −0.0477). V+q+SWA+HighRoPE baseline still applies.
- fail: screen20m val > 4.6364
- noise: |Δ| ≤ 0.05 — within the screen20m single-seed noise band; treat as inconclusive
- expected Δ ≈ −0.03 to −0.05; lower values are below the single-seed noise floor

## Routing (committed)
SOAP replaces the AdamW path only. Concretely:
- **SOAP optimizer**: `token_embedding.weight` (vocab × d_model, ~91% of grads), `emb_proj.weight` (r × d_model, when `emb_rank < vocab`), `out_proj.weight` (when `muon_for_output=False`).
- **Stays on plain AdamW**: every 1D param — `*.norm.weight` (RMSNorm γ), and any 1D learnable scalar (`q_gain`, `k_gain`, `smear_gate`, `output_temp` τ, `vocab_bias` b_v, etc.). Eigendecomposition is meaningless on 1D params.
- **Stays on Muon**: all 2D hidden weights (attn Q/K/V/O projections, FFN up/down/gate). Muon's orthogonalization on hidden is the load-bearing mechanism; SOAP on hidden would discard it.
- If `use_cautious_adamw` (idea 002) ships first, the AdamW path that survives 1D-param routing can be swapped to `CautiousAdamW` (sign-mask on the update). SOAP and the cautious mask are independent levers — both can ship in the same config.

## bf16 stability — pre-flight (≤5 min, must pass before full run)
Train 100 steps on screen20m with bf16 enabled. After every step, log the eigenvalue spectrum of the preconditioner on `token_embedding.weight` (largest 2D SOAP param):
- any NaN / Inf in eigvals → abort, re-promote as `use_soap_fp32_only` or close
- imaginary part of any eigval > 1e-3 → abort, same as above
- condition number (λ_max / λ_min) > 1e6 → abort, same as above
- All three clear after 100 steps → proceed to the full 19m run.

Memory cost: eigenbasis is (d_out, d_out) per 2D param. For our scale (d_model ≤ 576, emb ≤ 49152) the basis for `token_embedding` is the dominant term at ~288 MB bf16. The other 2D params are negligible by comparison. Acceptable.

## Seed protocol
3 seeds (42/43/44) when |Δ| ≤ 0.03 (i.e. the lower half of the expected range, which single-seed can't resolve). If single-seed pass and |Δ| > 0.03, ship the single seed. If single-seed pass and |Δ| ≤ 0.03, run the other two seeds before promoting to plan.md.

## Transfer argument
The eigenbasis converges in O(1) steps at any scale — it's a preconditioner, not a learned feature — so the curvature benefit is scale-invariant. The unknown is whether the eigenbasis is *well-conditioned* at small scale; that's exactly what the pre-flight measures. If the basis is well-conditioned at screen20m, the same mechanism applies at 25M/135M. If it's not, the bf16 pre-flight aborts before the full run, and we close or re-promote as fp32-only.

## Wiring
Add `use_soap: bool = False` to `LLMConfig` on the line after `use_cautious_muon: bool = False` (line 360). New `optimizers/soap.py` (~200 LoC, paper's own impl is <200) — copy the `Adam._single_tensor_adam` body, prepend the periodic eigenbasis update (every K steps via a small `state["step"]` counter), and apply the rotated-Adam update just before `param.add_(update, alpha=-lr)`. In `training/trainer.py:142`, gate the swap: `SOAP(adamw_params, ...) if config.use_soap else torch.optim.AdamW(adamw_params, ...)`. ~200 LoC total (new optimizer ~190 + 4-line gate in trainer.py + 1 line in config), bit-identical to baseline when `use_soap=False`.
