---
id: 199-attn-output-lowrank
status: repitching
round: 1
updated: 2026-06-15T16:21:23Z
transfer-risk: med
plain: Replace the W_O output projection with a low-rank factorization (W_O = A·B with rank r < d_model) — fewer parameters, forces a structured output, and acts as a soft bottleneck.
---

# 199 — Low-Rank Attention Output Projection (LoRA-Style W_O Factorization, Bit-Identity at r=d_model)

## Source
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022, arXiv:2106.09685) — LoRA factorizes a frozen pre-trained weight matrix as `W + A·B` where `A, B` are low-rank. Validated at 7B-65B. The lever here is the *non-LoRA* version: replace `W_O` with `A·B` (a single low-rank factorization, no pre-trained `W`).
- "LPLR" and "LyCORIS" (various 2023) — extensions of LoRA. The non-LoRA *low-rank from-scratch* form is the "low-rank projection" axis, which has been explored in "Low-rank Transformer" (2020) and "Linformer" (Wang et al. 2020, arXiv:2006.04768) for the *attention* matrix (not the output projection).
- "Intrinsic Dimensionality" (Aghajanyan et al. 2020, arXiv:2012.13255) — shows that the *intrinsic dimension* of fine-tuning is much smaller than the full parameter count, motivating low-rank factorizations.
- "Compacter" (Mahabadi et al. 2021, arXiv:2106.04647) — uses *Kronecker-factored* low-rank adapters. Validated at 100M-300M.
- In-repo context: 178-mqa-gated (null) — closed the *multi-query attention* (MQA) axis, which shares the V projection across heads. 199 is *intra-head* low-rank on W_O (factorizes W_O into two low-rank matrices). Different axis.
- 156-moa (null) — parallel attention experts + per-token router. 199 is a *single* W_O with low-rank factorization, no router.
- 146-sparse-ffn (null) — sparse FFN. 199 is *low-rank* attention output, not sparse FFN.

## Mechanism
Standard attention output projection:
```
V = W_V @ x            # W_V: [d_model, d_k * H]
out = softmax(QK^T / sqrt(d_k)) @ V   # [B, T, d_k * H]
head_out = out @ W_O   # W_O: [d_k * H, d_model]
```
With low-rank W_O factorization:
```
V = W_V @ x
out = softmax(QK^T / sqrt(d_k)) @ V
# W_O = A @ B, where A: [d_k * H, r], B: [r, d_model], r < min(d_k * H, d_model)
head_out = out @ A @ B
```
For tiny1m3m (H=4, d_k=16, d_model=64), the standard W_O is `[64, 64]` (4096 params). With low-rank factorization, `W_O = A @ B` where `A: [64, r]`, `B: [r, 64]`. For `r = 32` (half-rank), the total params are `64 * 32 + 32 * 64 = 4096` (same as full-rank). For `r = 16` (quarter-rank), the total is `64 * 16 + 16 * 64 = 2048` (half). For `r = 8` (eighth-rank), the total is `64 * 8 + 8 * 64 = 1024` (quarter).

**Step-0 byte-identity**: with `A` and `B` initialized by the standard Kaiming init, the product `A @ B` has expected magnitude `O(1/sqrt(d_k * H))` per component (from the Kaiming init on each factor). The full-rank `W_O` (also Kaiming-init) has expected magnitude `O(1/sqrt(d_k * H))` per component. **The two have the same magnitude distribution at step 0**, but the *actual values* are different (random draws are different). The lever is **not** step-0 byte-identical; it's a *different random init*.

For **step-0 byte-identity**, the factorization must match the full-rank init at step 0. This is possible via the "SVD init" technique: initialize `A, B` such that `A @ B = W_O_full` at step 0. With SVD of `W_O_full = U · Σ · V^T`, set `A = U · sqrt(Σ)`, `B = sqrt(Σ) · V^T`. This makes `A @ B = W_O_full` at step 0 exactly. The optimizer then *destroys* the factorization over training (the rank grows beyond `r` as `A` and `B` move independently). The SVD init gives step-0 byte-identity; the standard Kaiming init doesn't.

**Alternative form (recommended)**: use the SVD init for strict step-0 byte-identity. Add `use_lowrank_wo: bool = False` and `lowrank_wo_r: int = 32` config flags.

## Design sketch
- **Files**:
  - `models/llm.py` (or `models/layers.py`) — in the attention forward, replace `self.W_O` (a single `nn.Linear(d_k * H, d_model)`) with two `nn.Linear` modules: `self.W_O_A = nn.Linear(d_k * H, r, bias=False)` and `self.W_O_B = nn.Linear(r, d_model, bias=False)`. The forward becomes `head_out = self.W_O_B(self.W_O_A(out))`. With SVD init, decompose a "ghost" full-rank W_O and set `W_O_A.weight = U · sqrt(Σ)`, `W_O_B.weight = sqrt(Σ) · V^T`. Without SVD init, use Kaiming on both.
  - `configs/llm_config.py` — add `use_lowrank_wo: bool = False` and `lowrank_wo_r: int = 32` to `LLMConfig`. Add `Tiny1M3MLowRankWOConfig(Tiny1M3MConfig)` with `use_lowrank_wo: bool = True, lowrank_wo_r: int = 32`.
- **Config flag**: `use_lowrank_wo: bool = False, lowrank_wo_r: int = 32`.
- **Param count**: for `r = 32` (half-rank), same as full-rank (4096). For `r = 16` (quarter-rank), **−2048 params (−0.22% of 0.94M)**.
- **Intuition (why it might lower val loss)**: W_O is a `[64, 64]` matrix that projects the concatenated per-head values back to the residual stream. A low-rank factorization forces the projection to be a *structured* linear map (rank ≤ r), which is a *regularizer* on the attention output. At 0.94M, the data is limited; a structured projection may help by reducing the *effective* number of free parameters in the attention path. The LoRA paper shows that low-rank *adaptations* (not from-scratch) work well at 7B-65B; the from-scratch form (this lever) is a more aggressive regularization.
- **Why it might bind at 0.94M where other capacity-injection levers didn't**: the closed 117-soft-moe, 118-MoD, 146-sparse-ffn, 156-moa levers all *add* capacity (more experts, more routes, more projections). 199 *reduces* capacity (low-rank factorization). The closed capacity-injection levers nulled because the *added* capacity was not absorbed by the optimizer at 0.94M. The *reduced*-capacity lever is a different story: the *less* capacity may be easier to optimize at this tier.

## Scale evidence
- LoRA (Hu et al. 2022) — 7B-65B. The *adaptation* form (not from-scratch). Direct validation that low-rank works at scale.
- "Low-rank Transformer" (2020) — 100M-1B. The from-scratch low-rank form.
- **Transfer-risk: med** — the lever has direct validation at 100M+ for the *adaptation* form (LoRA), but the *from-scratch* form is a more aggressive regularization. The closed 146-sparse-ffn suggests that *capacity-reduction* levers don't always bind at 0.94M.

## Why it's worth a slot
The bet, in one sharp sentence: **low-rank W_O is a *capacity-reduction* lever (LoRA's structural analog, from-scratch) with direct validation at 100M+ for the *adaptation* form, and the closed capacity-*injection* levers (117, 118, 146, 156) suggest that *added* capacity doesn't amortize at 0.94M, but *reduced* capacity is a different choice** — a low-rank W_O forces a structured attention output, which may be a useful regularizer at 0.94M where the data is limited; a null at 0.94M would close the *capacity-reduction* axis (the full-rank W_O is already optimal at this tier), and a win would give a 0.22-2% param-reduction lever that also improves val loss.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 178-mqa-gated (null) — *cross-head* V sharing. 199 is *intra-head* W_O factorization. Different axis.
- 156-moa, 158-gau, 157-conv-ffn (null) — capacity-*injection* FFN/attention levers. 199 is *capacity-reduction* on W_O.
- 146-sparse-ffn (null) — sparse FFN. 199 is low-rank attention output, not FFN.
- 190-w0-wv-tied — tied W_O and W_V. 199 is *low-rank* W_O (parameter sharing via factorization), not tying.
