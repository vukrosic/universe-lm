---
id: 166-t5-rpe
status: needs-review
round: 1
updated: 2026-06-14T06:23:27Z
transfer-risk: med
plain: Replace RoPE with T5-style bucketed relative-position bias on the attention logits — each pair of token positions gets a learnable additive bias from a small set of logarithmic distance buckets, initialized to zero so step-0 is bit-identical to the baseline.
---

# 166 — T5-Style Bucketed Relative Position Bias (RPE)

## Source
- Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, JMLR 2020, arXiv:1910.10683) — introduced the bucketed relative position bias (32 logarithmic distance buckets per direction, learned bias per bucket per head). Validated at T5-11B (T5-XXL).
- RoPE (Su et al. 2021) is the current autoregressive-LM standard; closed in our sweep as 500k-base winner. FIRE (Li et al. 2024, 009 WIN) is the depth-aware successor.
- ALiBi (Press et al. 2022) — linear distance bias with head-specific slopes. *Not* closed in our axis sweeps, but largely redundant with RoPE/FIRE for autoregressive LMs.
- T5-RPE has been re-used in BigBird (Zaheer et al. 2020), REALM (Guu et al. 2020), and LongT5 (Guo et al. 2021) — all validated at 100M+.
- Closed PE axes: NoPE, post-norm tying, RoPE base sweep, FIRE, CoPE.

Distinct from RoPE (rotary Q/K), ALiBi (linear slope), FIRE (depth-aware frequency). T5-RPE is *additive logit bias* on QK^T, bucketed by relative distance.

## Mechanism
Replace the rotary position embedding applied to Q and K with an *additive bias* on the QK^T logits, indexed by relative distance `|i − j|` via a logarithmic bucket function. The bias is a learned per-head per-bucket scalar:
```
buckets(i, j) = floor(log2(|i - j| + 1))   # 0..B-1, B=32
bias_h[buckets] = self.rpe_bias[h, buckets]  # (H, B)
logits = Q @ K^T / sqrt(d_head) + bias_h[buckets][None, None, :, :]
```
At init, `self.rpe_bias = nn.Parameter(torch.zeros(H, B))` ⇒ all biases are 0 ⇒ logits are bit-identical to the un-PEd baseline (QK^T/sqrt(d) with no rotation, no bias). The lever *adds* the RPE to whatever positional scheme is currently active (or replaces it — both are valid; starting as "added on top of un-rotated Q/K" gives the cleanest step-0 identity). ~30 LoC; +`H × B = 4 × 32 = 128` params per block (negligible).

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_t5_rpe: bool = False` and `t5_rpe_buckets: int = 32` kwargs. When on, registers `self.rpe_bias = nn.Parameter(torch.zeros(H, t5_rpe_buckets))`. In `forward`, *after* the QK matmul and *after* the causal mask, add `bias = self.rpe_bias[bucket_index_matrix]` to the logits before softmax. The bucket index matrix is a precomputed `(T, T)` tensor built once per call (cheap, T ≤ 2048). Optionally the lever can be implemented to *replace* the RoPE path (config flag selects one or the other).
- **Config flag**: `use_t5_rpe: bool = False`, `t5_rpe_buckets: int = 32` (default).
- **Step-0 identity**: `self.rpe_bias = nn.Parameter(torch.zeros(H, B))` init ⇒ all biases are 0 ⇒ added logit bias is 0 ⇒ logits are bit-identical to no-RPE baseline. The RoPE path can remain unchanged (RPE is added on top) or replaced via a separate config.
- **Intuition**: T5-RPE is a *logit-bias* positional encoding, structurally different from the *Q/K rotation* family (RoPE, FIRE, ALiBi). The hypothesis: an *additive* logit bias may capture distance-dependent attention priors that a *rotational* transform cannot. Specifically, T5-RPE can express "attend less to distant tokens" as a direct prior without needing a rotation to implicitly encode it. If T5-RPE wins alongside FIRE (009), the positional information is *over-determined* at this tier and the two encodings are partially redundant (info-value: how much extra positional info can be encoded). If T5-RPE wins *instead of* FIRE, the additive bias is the better mechanism at 0.94M.
- **Why now**: the closed PE axes (RoPE / NoPE / FIRE / CoPE) are all *Q/K-rotation* family. The additive-logit-bias family (T5-RPE, ALiBi) is largely unclosed at our tier. T5-RPE is the canonical, well-validated additive variant.

## Scale evidence
T5-XXL (11B parameters) used 32-bucket RPE on attention logits, achieving SOTA on SuperGLUE/GLUE/TriviaQA at release. BigBird, REALM, and LongT5 all re-used T5-RPE at 100M+. Transfer risk is **med**: T5-RPE is encoder-decoder-native and the autoregressive-LM case has less direct validation, but the mechanism is structurally simple and the bucket parameterization is known to work.

## Why it's worth a slot
A win would tell us the *additive logit bias* family of positional encodings works at autoregressive LMs at 0.94M, orthogonal to the *rotational Q/K* family (RoPE/FIRE) that dominates the closed axes. A null would close the additive-bias PE family at our tier and confirm the rotational family is the binding axis for autoregressive LMs. Either result is informative for future PE-level work.
