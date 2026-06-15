---
id: 178-mqa-gated
status: needs-run
round: 1
updated: 2026-06-15T06:06:05Z
transfer-risk: low
plain: Test whether the per-head K/V gate moves off zero or collapses to zero — a probe, not a lever: if the optimizer never moves β, it mechanistically closes the GQA/MQA family for tiny tier with a causal explanation; if it does, the GQA arch-sweep's null was a fixed-group artifact, not a smooth-interp artifact.
---

# 178 — Gated MQA Probe (Per-Head Mix Between Head-Local and Shared K/V)

## Source
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (arXiv:1911.02150, 2019). MQA validated at T5/PaLM-class scale (~100M–540B) as quality-preserving; family is mature.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245, 2023). GQA interpolates MQA↔MHA via group size. Validated at Llama-2-7B/13B/70B.
- In-repo: `closed.md` line "MHA vs GQA, MLA, Tied QK" closed as null in the arch sweep; `043-mla` was taste-rejected citing the same closure. **The closed sweep tested fixed group sizes (H, 2, …) at tiny1m3m — it did not test a smooth, learnable per-head blend.** This idea is the probe the closed sweep cannot answer.

## Mechanism
Per head h, mix per-head and shared K/V with a learnable scalar β_h:
```
K_h = K_h_local + β_k_h * (K_shared − K_h_local)         (init β_k_h = 0)
V_h = V_h_local + β_v_h * (V_shared − V_h_local)         (init β_v_h = 0)
```
At init β=0, K_h = K_h_local exactly (bit-identical to MHA baseline at step 0). K_shared, V_shared are one `nn.Linear(d_model, d_model)` per block, broadcast across heads. β vectors are 4 floats per (K, V) per block — 8 params/block — negligible.

## Design sketch (unchanged from r1)
- `models/layers.py` — add `use_mqa_gated: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.mqa_gate_k = nn.Parameter(torch.zeros(n_heads))`, `self.mqa_gate_v = nn.Parameter(torch.zeros(n_heads))`; `self.W_K_shared`, `self.W_V_shared` as `nn.Linear(d_model, d_model, bias=False)`. Compute `K_h = K_h_local + β_k_h * (K_shared_broadcast − K_h_local)` (broadcasting across heads), same for V. Per-head W_K_h, W_V_h **stay allocated** through the run (needed for the head-local branch).
- `configs/llm_config.py` — add `use_mqa_gated: bool = False`. `Tiny1M3MMQAGatedConfig` with `use_mqa_gated: bool = True`.
- `models/llm.py` — thread `use_mqa_gated` into both `TransformerBlock` sites.
- **Config flag off by default** — baseline path is byte-identical (no K/V_shared projections are allocated when off).

## Param overhead (accepted)
- Gate vectors: H = 4 per (K, V) per block = 8 params/block, 96 total. Negligible.
- Shared K/V projections: 2 × d_model² = 8192 params/block × 12 blocks = 98,304 params (~10% of 0.94M model).
- Per-head W_K, W_V stay allocated through the run (we need them for the head-local branch; pruning them as β → 1 is a future optimization, out of scope for this probe).
- **Accepted cost**: ~10% extra params during the head-local phase. The point of the probe is the gate trajectory, not the loss number — paying 10% param overhead for a clean answer to "do the gates move?" is acceptable. The runner records per-head β values at the end of training so we can read the trajectory without paying for a second run.

## Scale evidence
- MQA at PaLM-540B (inference, quality-preserving). GQA at Llama-2 7B/13B/70B (training, modest PPL win over MQA, parity with MHA). **Direct validation at ≥100M** for the family.
- The gated-blend form is novel (no published paper at this exact β-form), but the underlying MQA/GQA mechanism is well-validated. Transfer-risk remains **low** for the mechanism; the probe's value at 0.94M is information, not generalization.

## Why it's worth a slot (r2 — probe framing)
The bet, in one sharp sentence: **after tiny1m3m training, the per-head β_h values are read out; we expect (a) at least one β_h to move measurably off zero in any block, OR (b) all β_h to collapse to (or stay at) zero.** Outcome (a) would mean the optimizer found a per-head K/V-sharing gradient that the closed arch-sweep's fixed group sizes missed — informative even if the val-loss delta is small. Outcome (b) would mean the closed arch-sweep's MHA-vs-GQA null has a mechanistic explanation (the optimizer doesn't want any sharing, so the convex interpolation can't help) and the GQA family is closed for tiny tier with causal evidence, not just empirical null.

This is a **probe, not a lever**. The success criterion is a clean answer to "do the gates move?", not a val-loss improvement. The val-loss column is recorded but not the deciding metric — a small win is interesting; a null is informative; a noisy half-move is a third outcome that itself teaches us something about the per-head loss surface.

Queue crowding acknowledged (179/180/181/182 are all attention-axis); this one earns its slot because it is the *only* idea in the queue that produces a mechanistic close-or-keep signal for the entire GQA family. The other four are loss-Δ-driven levers; this is a structural probe. They're complementary, not redundant.

## What the runner must record (additional metric)
Beyond val loss, record at the end of training: per-block, per-head final `β_k_h` and `β_v_h` values (4 × 12 × 2 = 96 scalars). Cost: a single tensor dump at end-of-run. This is the primary signal — the val-loss column is secondary.

## Plan

**Files**

- `models/layers.py` — add `use_mqa_gated: bool = False` to
  `MultiHeadAttention.__init__` and to `TransformerBlock.__init__`.
  When on, allocate:
  - `self.W_K_shared = nn.Parameter(zeros(n_kv_heads·d_k, d_model))`
  - `self.W_V_shared = nn.Parameter(zeros(n_kv_heads·d_k, d_model))`
  - `self.mqa_gate_k = nn.Parameter(zeros(n_kv_heads))`
  - `self.mqa_gate_v = nn.Parameter(zeros(n_kv_heads))`
  In `forward()`, after the K, V reshape (line ~2627), apply the
  per-KV-head blend: `K = K + β_k · (K_shared − K)`, same for V.
  K_shared is `F.linear(x, W_K_shared).reshape(B, T, n_kv_heads, d_k)`.
  All init is zero so the construction consumes no RNG, keeping
  the qkvo_proj random init aligned with the no-flag baseline
  (step-0 byte-identity). K_shared projects to `n_kv_heads·d_k`
  (the GQA-axis size), not `d_model` — this matches the head-
  local K, V layout at tiny1m3m where `n_kv_heads=2 ≠ n_heads=4`
  and the head-local K is per-KV-head; for non-GQA configs the
  two sizes are equal so the result is the same.
- `configs/llm_config.py` — add `Tiny1M3MMQAGatedConfig` subclass
  of `Tiny1M3MConfig` with `use_mqa_gated: bool = True`.
- `models/llm.py` — thread `self.use_mqa_gated = getattr(config,
  "use_mqa_gated", False)` into both TransformerBlock
  instantiations (YOCO upper-half and standard).

**Config flag**: `use_mqa_gated: bool` (default off).

**Step-0 byte-identical**: β_k = β_v = 0 ⇒ `K_mix = K_local`
exactly in fp32. The shared K, V is also zero-init so the
W_K_shared, W_V_shared matmul produces 0 regardless of input
(this also keeps the construction RNG-aligned with baseline).
Verified: `max-abs-diff = 0.0` between flag-on and flag-off
forward at step 0, same seed.

**Run command** (tiny1m3m, seed 42, baseline dataset):
```bash
python _arq_178-mqa-gated.py
```
which invokes `train_llm.main()` with
`--config_class __main__.C --seed 42
 --dataset_path processed_data/pretrain_1B --warmup false`.

**Reading the result**: from `autoresearch/records.jsonl`, the
treatment's `val_loss` at end-of-training vs the baseline
`Tiny1M3MConfig` (val 6.4216). The primary signal is the r2
probe framing above: read out the per-block per-head `β_k, β_v`
at end-of-run. The val-loss column is recorded but secondary.

