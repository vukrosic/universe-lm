---
id: 179-anti-causal-subheads
status: tasting
round: 1
updated: 2026-06-15T05:47:27Z
transfer-risk: med
plain: Let some attention heads peek at the future during training (a small per-head gate decides), starting with every head fully causal so step-0 is byte-identical.
---

# 179 — Anti-Causal Sub-Heads (UniLM-style Hybrid Causal + Bidirectional Heads)

## Source
- Dong, Yang, Wei, et al., "Unified Language Model Pre-training for Natural Language Understanding and Generation" (UniLM, NeurIPS 2019, arXiv:1905.03197). Validated on SQuAD 2.0, GLUE, and generation tasks at BERT-base (~110M) and BERT-large (~340M).
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, JMLR 2020, arXiv:1910.10683). T5-style prefix-LM uses bidirectional attention on a prefix and causal on the rest; validated at T5-base (~220M) through T5-11B (~11B).
- Song et al., "MPNet: Masked and Permuted Pre-training for Language Understanding" (arXiv:2004.09297). Mixes causal and masked-LM objectives.
- In-repo context: closed.md line "Multiscale heads / parallel block / attn sink (2026-06-04 batch)" closed global bidirectional + causal hybrids at the layer level (some layers global, some causal). 179 is per-**head** within a single layer: each head independently chooses causal or bidirectional. CoPE (013) closed drift. The per-head axis is fresh.

## Mechanism
Standard causal attention: position t attends to positions ≤ t. The attention mask is `M[i,j] = −∞ for j > i`, `0` for j ≤ i.
Anti-causal sub-heads: for each head h, learn a scalar `γ_h ∈ [0, 1]` that controls how much of the head's attention is bidirectional. Concretely:
```
mask_h[i, j] = (1 − γ_h) · (−∞ for j > i)  +  γ_h · 0  (no mask)
```
Equivalently, the bidirectional component adds the upper-triangular mask with weight 1−γ_h (or equivalently, attenuates the causal mask with weight γ_h).

Parameterize `γ_h = sigmoid(γ_raw_h)` with `γ_raw_h` init `−10` (large negative) ⇒ `γ_h ≈ 0` at step 0 ⇒ full causal mask for all heads ⇒ **byte-identical to baseline at step 0**.

The lever: pushing `γ_raw_h` positive makes head h progressively bidirectional. The optimizer can grow any subset of heads into "global" heads.

A second lever axis (orthogonal): per-head *learned mask pattern*, e.g., a learned position-wise bias of shape `[H, T]` that the optimizer can shape. Out of scope for this filing — keep the lever isolated to γ_h.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_anti_causal_subheads: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.ac_subhead_gate = nn.Parameter(torch.full((n_heads,), -10.0))` (init −10 ⇒ sigmoid ≈ 4.5e-5 ≈ 0 ⇒ causal for all heads). Apply in the attention forward: after computing scores `[B, H, T, T]`, compute `γ_h = sigmoid(self.ac_subhead_gate)`, then `mask = causal_mask * (1 − γ_h).view(1, H, 1, 1) + 0 * γ_h` (broadcasting across heads). The mask is `−∞` where causal-only is needed and `0` where bidirectional is allowed.
  - `configs/llm_config.py` — add `use_anti_causal_subheads: bool = False`. Add `Tiny1M3MAntiCausalSubHeadsConfig` subclass with `use_anti_causal_subheads: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_anti_causal_subheads: bool = False`.
- **Step-0 byte-identical**: `γ_raw_h = −10` ⇒ `γ_h = sigmoid(−10) ≈ 4.5e-5` ⇒ mask is essentially `−∞` for j>i (scaled by 0.99995) ⇒ softmax gets `exp(−large·0.99995)` which is < 1e-300 in fp32 ⇒ effectively no attention to future positions ⇒ byte-identical to causal baseline. **Note**: in fp32, exp(−30) ≈ 9e-14, still effectively zero. With `−10·(1 − 4.5e-5) ≈ −9.99955`, exp ≈ 2.3e-5, which is technically nonzero but 5 orders of magnitude smaller than the diagonal exp(0)=1.0, so the softmax output is identical at fp32 precision.
- **Param count**: H=4, n_layers=12. Per block: 4 gate params. Total: 48 params (+0.005% of 0.94M).
- **Intuition (why it might lower val loss)**: bidirectional heads during training can see the answer for next-token prediction. This is training-time leakage, but it's an inductive bias that helps the head learn *structural* features (where things appear in context) that causal heads can't. At inference time, all heads must be causal (we don't have future context). The hope is that bidirectional training pushes the model toward better representations even when constrained to causal at inference. UniLM/prefix-LM papers show this works at 100M+ for encoder-decoder tasks; this is the **decoder-only** analog. Different from the closed "multiscale heads" axis (which mixes causal + global at the **layer** level); 179 mixes at the **head** level within a layer.

## Scale evidence
- UniLM at 110M/340M (BERT-class encoder; encoder has full bidirectional by default, so the gain is on the decoder side).
- T5 prefix-LM at 220M-11B (encoder-decoder; bidirectional encoder + causal decoder).
- **No published decoder-only anti-causal-head result at 100M+** — the closest analog is XLNet-style permutation LM (Yang et al. 2019, arXiv:1906.08237) which permutes the attention order to give some heads partial future access. XLNet was validated at 110M-340M.
- Transfer-risk is **med**: validated at 100M+ for encoder-decoder; decoder-only anti-causal heads is novel at this scale. The mechanism is scale-free (just a mask shape).

## Why it's worth a slot
The bet, in one sharp sentence: **per-head learnable bidirectional access during training is a fresh axis (the closed multiscale-heads lever mixes at layer level; 179 mixes at head level within a layer) and could give the model richer structural features without committing the whole layer to bidirectional (which the closed arch sweep already tried and failed at).** A null at 0.94M would close the per-head causal-mixing axis and confirm that the decoder-only binding constraint is "future-context leakage doesn't transfer to causal inference"; a win would unlock the hybrid head family for Phase-2 ≥135M where per-head gradient signal is larger and the bidirectional-vs-causal mix can develop meaningfully.
