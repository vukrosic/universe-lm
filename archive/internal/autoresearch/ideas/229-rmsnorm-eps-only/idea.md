---
id: 229-rmsnorm-eps-only
status: needs-taste
round: 1
updated: 2026-06-16T01:05:00Z
transfer-risk: low
plain: Make the small numerical-stability epsilon inside RMSNorm a learnable scalar (init at 1e-6, the standard value). At init this is bit-identical to baseline RMSNorm; the model can tune the regularization strength. Smaller variant of 219 (which adds bias + eps).
---

# 229 — Learnable Epsilon in RMSNorm (eps-only variant)

## Source
219-rms-eps-affine filed two changes: (1) learnable eps, (2) bias added to RMSNorm. 229 files *only* (1) — a simpler, smaller lever. Direct sources: RMSNorm (Zhang & Sennrich 2019, arXiv:1910.07467) uses fixed eps; PowerNorm (Shen et al. 2020) also uses fixed eps in its power-mean. The closest novel lever is "RMSNorm with learnable eps" which has been informally explored but is not a published standard.

## Mechanism
```
# baseline RMSNorm:
denom = sqrt(mean(x^2, dim=-1, keepdim=True) + 1e-6)
y     = (x / denom) * gain

# 229:
eps_l = self.rms_eps.abs() + 1e-9                  # ensure positive, init 1e-6
denom = sqrt(mean(x^2, dim=-1, keepdim=True) + eps_l)
y     = (x / denom) * gain                          # gain still per-feature, init 1
```

Init: `eps_l = 1e-6`, `gain = 1.0`. Step-0 bit-identical to baseline RMSNorm. The optimizer can move eps_l up (more regularization on the denom, smaller effective y) or down (less regularization, larger y).

## Design sketch
- **Files**: `models/layers.py` — modify `RMSNorm` class. Add `self.rms_eps = nn.Parameter(torch.tensor(1e-6))` initialized to 1e-6. Use `self.rms_eps.abs() + 1e-9` in the denominator to keep positive. **NO bias added** (the bias piece is filed separately as 219-rms-eps-affine; 229 is just the eps half).
- **Config flag**: `use_learnable_rms_eps: bool = False`, `learnable_rms_eps_init: float = 1e-6`.
- **Cost**: 1 scalar per norm × 24 norms (2 per block × 12 blocks) = +24 params, +0.0026% of 0.94M. Free.
- **Why it should help at tiny1m3m**: the eps controls the *denominator's bias term* — at d_model=64 the variance `mean(x^2)` has high variance itself, and the eps term smooths this. At init eps=1e-6 the smoothing is negligible; the model may want a larger eps (e.g., 1e-3) to dampen the variance outliers, or smaller (e.g., 1e-9) for sharper normalization. The 24 eps values (one per norm) might learn different schedules per norm-instance.
- **Why it might be null**: the existing gain parameter in RMSNorm (per-feature learnable) can already absorb the eps's effect — increasing eps is equivalent to shrinking gain. So the lever might be perfectly redundant with gain at 0.94M.

## Scale evidence
RMSNorm is well-validated at scale (LLaMA-1/2/3, Mistral, etc.). Learnable eps is novel but the relationship to gain means it's plausibly redundant. Transfer-risk **low** (architecturally trivial, scale-agnostic).

## Why it's worth a slot
229 is a *smaller, more focused* variant of 219 (which adds bias AND eps). A win on 229 specifically isolates the eps axis from the bias axis — useful attribution info if 219 wins. A null would say eps is redundant with gain, suggesting 219's WIN (if any) is from the bias axis not eps. The lever is cheaper than 219 and cleanly separable as a single-axis test.
