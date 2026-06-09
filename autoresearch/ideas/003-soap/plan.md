# Plan — 003 Soap (Shampoo + Adam)

## Flag
- `configs/llm_config.py:371` (line after `use_cautious_adamw: str = "none"`)
  - `use_soap: bool = False` — default OFF → baseline path bit-identical.

## Change

| File | Edit |
|---|---|
| `optimizers/soap.py` | NEW (~155 LoC). `SOAP(Optimizer)`. Per-tensor: 1D params → plain AdamW; 2D params → Adam in Shampoo's eigenbasis. Maintains `L = GG^T` (d_out×d_out) and `R = G^T G` (d_in×d_in) as running preconditioner estimates; refreshes eigenbasis every `precondition_frequency` steps (default 10) via `torch.linalg.eigh` on the symmetric, regularized preconditioner. Sub-steps: (1) `p.mul_(1 - lr*wd)`; (2) update `L`, `R`; (3) every K steps, `Q_L, Λ_L = eigh(L + εI)`, `Q_R, Λ_R = eigh(R + εI)`; (4) `G' = Q_L^T G Q_R`; (5) `m = β1 m + (1-β1) G'`, `v = β2 v + (1-β2) G'²`; (6) `u' = m̂/(√v̂ + ε)`; (7) `u = Q_L u' Q_R^T`; (8) `p -= lr * u`. The eigenbasis is initialized to identity on step 0, so the first step is bit-identical to AdamW (Q_L = Q_R = I → G' = G → standard Adam direction). |
| `optimizers/__init__.py` | Add `from .soap import SOAP` and `SOAP` to `__all__`. |
| `training/trainer.py:75-126` | Routing. Insert SOAP branch in the param-classification loop: if `use_soap and ndim==2 and not is_muon_candidate and requires_grad` → `soap_params.append(p); continue`. Otherwise the existing Muon / AdamW split. Skip when `use_soap=False` (bit-identical). |
| `training/trainer.py:128-180` | Optimizer construction. Add a new `soap_params` list alongside `muon_params` / `adamw_params`. Gate the SOAP optimizer: `if use_soap: soap_optimizer = SOAP(soap_params, lr=adamw_lr, weight_decay=...)`; else pass-through. `adamw_optimizer` is built from `adamw_params` only (which now excludes SOAP-eligible 2D params). Return `[muon_optimizer, soap_or_adamw_optimizer]`. |

Step-0 (flag OFF) — `use_soap=False`, `soap_params=[]`, no SOAP optimizer instantiated, all 2D non-Muon params flow to the existing `AdamW`. The chain is bit-identical to the current path.

## Control
- **Control**: V+q+SWA+HighRoPE + AdamW on `(token_embedding, emb_proj, out_proj, *.norm.weight, 1D scalars)` + Muon on 2D hidden. Seed 42. Tier `screen20m` (19M).
- **Treatment**: control + `use_soap=True` → those same 2D non-Muon params are routed to `SOAP` (1D params stay on plain AdamW). Seed 42. Tier `screen20m`.

## Cost
- **Params**: 0 (re-routes existing params).
- **FLOPs/step**: per 2D SOAP param, ~1× GG^T + 1× G^T G + 1× eigh(L) + 1× eigh(R) every K=10 steps. At step-0, eigenbasis is identity so cost ≈ AdamW. Per full step (after first), projected grad = Q_L^T G Q_R ≈ 2 × params FLOPs. Dominant overhead: the periodic eigh (cubic in min(d_out, d_in), so negligible for hidden layers; visible only for `token_embedding` where d_out=vocab=49152, d_in=d_model=576). For `token_embedding`, eigh on (576×576) is cheap; on (49152×49152) it would be expensive, but the routing uses L=G G^T (d_out×d_out=49152×49152). ⚠️ check: eigh on 49152×49152 fp32 is ~3min/step and 18GB. **Will fall back to fp32-only for token_embedding if this becomes a bottleneck.** Pre-flight catches this.
- **Memory**: per SOAP param, 4 new state tensors: `L` (d_out×d_out), `R` (d_in×d_in), `Q_L` (d_out×d_out), `Q_R` (d_in×d_in), plus the existing `exp_avg`, `exp_avg_sq`. For `token_embedding` (vocab=49152, d_model=576): `L` is 49152×49152 fp32 ≈ 9 GB, `Q_L` ≈ 9 GB. This is the dominant term. Other 2D params (emb_proj, out_proj) have d_out×d_out ≈ d_model×d_model ≈ 576² = negligible.
- **Mitigation for token_embedding memory**: store `L` / `Q_L` in fp32 only when `use_soap=True` (the bf16 path may be lossy). The eigenbasis itself is fp32 by default; the projection step casts to match param dtype. **Hard precondition: run the bf16 pre-flight (see Run) before committing to the full run.**

## Run

### Step 0 — bf16 pre-flight (hard gate, ≤5 min wall-clock)
Train 100 steps on `screen20m` with `use_soap=True` and `bf16` enabled. After every step, on the largest 2D SOAP param (`token_embedding.weight`):
- log eigvals of `L` and `R` (largest / smallest / any NaN / any Inf)
- log `||imag(eigvals)||_max` (must be 0; >1e-3 = abort)
- log `λ_max / λ_min` (must be < 1e6; else abort)

Any one of {NaN/Inf, imaginary > 1e-3, condition > 1e6} → **abort**: do NOT promote to a full run. Re-file as `use_soap_fp32_only` (raises same flag but forces eigenbasis to fp32 with no bf16 path) or close.

### Step 1 — full A/B on `screen20m`
```bash
# Control (baseline AdamW path, use_soap=False)
python train_llm.py --config screen20m --seed 42 --out runs/screen20m-soap-ctrl

# Treatment
python train_llm.py --config screen20m --seed 42 --use_soap True --out runs/screen20m-soap-trt
```
Wall-clock: ~30-45 min each on a single A100 (the screen20m tier). Pass/fail bar from `idea.md`:
- pass: treatment val ≤ 4.5887 (ctrl 4.6364, target Δ = −0.0477)
- fail: treatment val > 4.6364
- noise: |Δ| ≤ 0.05

### Step 2 — seed escalation
If treatment passes AND |Δ| ≤ 0.03, run the other two seeds (43, 44) before promoting past `screen20m`. If |Δ| > 0.03, single seed is enough.

## Self-check (before release to code-reviewer)
- `use_soap=False` reproduces the control (no numeric drift) — confirm by inspecting the routing code path: with the flag off, `soap_params=[]`, the SOAP optimizer is never instantiated, and the AdamW path is unchanged.
- The treatment path actually exercises SOAP — confirm by a 10-step dry run on a tiny config with `use_soap=True`: `token_embedding.weight` should have non-zero `state["Q_L"]` after step 10, and the param update should be visibly different from a control dry-run.
- `plan.md` pass/fail bar matches `idea.md` — both: pass ≤ 4.5887, fail > 4.6364, noise |Δ| ≤ 0.05.
