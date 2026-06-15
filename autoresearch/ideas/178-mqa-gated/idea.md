---
id: 178-mqa-gated
status: implementing
round: 1
updated: 2026-06-15T05:46:20Z
transfer-risk: low
plain: Make every attention head share one set of key/value projections, but let each head learn how much of its own key/value to keep vs. borrow from the shared pool, starting with full per-head keys/values so step-0 is byte-identical.
---

# 178 — Gated Multi-Query Attention (G-MQA: Per-Head Mix Between Head-Local and Shared K/V)

## Source
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (arXiv:1911.02150, 2019). Validated at T5/PaLM-class scale (~100M-540B) for inference speed; the training-time quality effect is positive at all scales (no perplexity regression vs MHA).
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245, 2023). GQA interpolates between MQA (group=1) and MHA (group=H). Validated at Llama-2-7B/13B/70B.
- In-repo: closed.md line "MHA vs GQA, MLA, Tied QK" closed as null in the arch sweep — but that closed specific group-size choices (group=H, group=2, etc.), not a **gated learnable per-head blend**. The lever here is a smooth, per-head gate that can grow toward any group_size at training time.

## Mechanism
Standard MHA: `K_h = x @ W_K_h`, `V_h = x @ W_V_h` for each head h.
MQA: `K_shared = x @ W_K_shared`, `V_shared = x @ W_V_shared` — one K/V used by all heads.
G-MQA (this lever): per head h,
```
K_h = α_h · (x @ W_K_h) + (1 − α_h) · K_shared
V_h = α_h · (x @ W_V_h) + (1 − α_h) · V_shared
```
with `α_h ∈ ℝ^{H}` learnable, init α_h=1 for all h ⇒ K_h, V_h are full per-head ⇒ **byte-identical to baseline at step 0**.

The gate is `α_h = sigmoid(α_raw_h)` with `α_raw_h` init `+10` ⇒ `sigmoid(10) ≈ 0.99995` (essentially 1, but not bit-identical in fp32). To get *exact* bit-identity, parameterize as `α_h = 1 - relu(1 - α_raw_h)` with `α_raw_h` init 1 ⇒ `α_h = 1` exactly. Or even simpler: `K_h = x @ W_K_h + β_h · (K_shared − x @ W_K_h)`, init β_h=0 ⇒ `K_h = x @ W_K_h` exactly (bit-identical). Use the **β-form** for clean step-0 identity.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_mqa_gated: bool = False` to `MultiHeadAttention.__init__`. When on, allocate `self.mqa_gate_k = nn.Parameter(torch.zeros(n_heads))` and `self.mqa_gate_v = nn.Parameter(torch.zeros(n_heads))` (init 0 ⇒ head-local K/V used). Compute `K_shared = x @ self.W_K_shared`, `V_shared = x @ self.W_V_shared` as `[B, T, d_model]` (single projection), then per-head mix: `K_h = K_h_local + β_k_h * (K_shared_h − K_h_local)` (broadcasting across heads), same for V. The shared K/V projection is one `nn.Linear(d_model, d_model)` per block (vs H per-head projections).
  - `configs/llm_config.py` — add `use_mqa_gated: bool = False`. Add `Tiny1M3MMQAGatedConfig` subclass with `use_mqa_gated: bool = True`.
  - `models/llm.py` — thread `use_mqa_gated` into both `TransformerBlock` sites (~lines 685, 941).
- **Config flag**: `use_mqa_gated: bool = False` (off by default; baseline path bit-identical because no K/V_shared projections are allocated when off).
- **Step-0 byte-identical**: `β_k_h = β_v_h = 0` for all heads ⇒ per-head K_h_local is used; the K_shared/V_shared projections are computed but multiplied by 0 ⇒ output unchanged. Max-abs-diff vs baseline = 0.0.
- **Param count**: H=4, d_model=64, n_layers=12. Per block, gate adds 4 + 4 = 8 params (negligible). The shared K/V projections add 2 × d_model × d_model = 8192 params per block, **BUT** the per-head W_K, W_V projections can be removed when the gate goes to 0 ⇒ at full MQA, K+V params drop from 4·2·4096 = 32,768 to 2·4096 = 8192 per block. **However**, at step 0 the per-head W_K, W_V still exist (we need them for the head-local branch), so the gate adds ~25% param overhead during the head-local phase, then drops as the optimizer grows the gate to use the shared path. The "right" implementation is: at init, both branches exist; as the gate β_h → 1, the head-local W_K_h, W_V_h become dead-weight. We accept the overhead since the gate is the lever.
- **Intuition (why it might lower val loss)**: MQA at 0.94M with H=4 reduces K/V parameters from 4·d_model² to 1·d_model² (75% savings on K/V). At our scale where capacity is already tight, this trade-off is risky — but the *gated* version lets the optimizer keep per-head K/V when it helps and share when it doesn't. The lever is a smooth interpolation between MHA (β=0) and MQA (β=1), tested at a tier where the binding constraint might be "K/V noise" rather than "K/V capacity."

## Scale evidence
- MQA at PaLM-540B (inference, validated as quality-preserving). GQA at Llama-2-7B/13B/70B (training, modest PPL win vs MQA, parity vs MHA). **Direct validation at ≥100M**.
- The gated interpolation is novel (no published paper at this exact form), but the underlying MQA/GQA mechanism is well-validated. Transfer-risk is **low**.

## Why it's worth a slot
The bet, in one sharp sentence: **at 0.94M, K/V sharing is a parameter-efficiency lever that the closed MHA-vs-GQA arch-sweep didn't test in gated form (it tested fixed group sizes only), and a per-head learnable blend lets the optimizer find the right amount of sharing per head without committing to MQA's full 75% K/V cut.** A null at 0.94M would close the MQA family for tiny tier and confirm that per-head K/V diversity is structurally important at our scale; a win would unlock the gated-KV-sharing axis for Phase-2 ≥135M where the parameter savings compound with larger d_model.
