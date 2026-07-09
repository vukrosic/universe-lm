---
id: 156-moa
status: done
round: 1
updated: 2026-06-14T00:44:23Z
transfer-risk: med
plain: Run several attention "experts" in parallel (each with its own softmax pattern) and let the model learn how to mix their outputs — like having multiple specialists vote on where to look.
---

# 156 — Mixture-of-Attentions (MoA)

## Source
Inspired by:
- Mixtral-of-Experts architecture (Mistral, 2023) extended to attention heads.
- Chi et al. "On the Expressivity Role of LayerNorm" — uses multi-pattern attention as a side study.
- Recent (2024-2025) routing-over-attentions work; mechanism distilled in various "mixture of softmaxes" papers.

Distinct from closed Mix-of-Softmaxes (144, K=2 softmaxes) — MoA mixes *full attention computations*, not just softmax variants.

## Mechanism
Run `E` parallel attention computations per layer (each with its own Q/K/V projections), each producing `out_e ∈ R^{T×d}`. Compute a routing weight `g_e = softmax(W_g x)_e` per token from the input `x` (a single linear `W_g ∈ R^{E×d}`). Final output: `out = Σ_e g_e * out_e`. With `W_g` init to give uniform weights (or one-hot to attention expert 0), step-0 output is identical to baseline attention. ~50 LoC.

## Design sketch
- **File**: `models/layers.py` — wrap the existing `Attention` block into a `MoAAttention(E)` module with `E` parallel attention experts and a learnable per-token router `W_g`.
- **Config flag**: `use_moa: bool`, `moa_num_experts: int = 2` (default).
- **Step-0 identity**: initialize `W_g = 0` so `g_e = 1/E` (uniform). Then `out = (1/E) Σ_e out_e` — *not* byte-identical to single attention (averages E different Q/K/V projections). To preserve step-0 identity, **init the E-1 extra experts to be zero-output modules** (their `W_Q, W_K, W_V` are `torch.zeros`); this gives `out = out_0` exactly at step 0. Implementer must verify this.
- **Intuition**: lets the model learn multiple attention patterns (e.g., one for syntax, one for content) per layer. Different from MoS (144, closed) which mixes softmax functions — MoA mixes *attention computations*. Different from closed sparse-attention axes (NSA, diff-attn) which operate on a single attention.

## Scale evidence
Mechanism inspired by Mixtral-style routing (Mixtral 8x7B); MoA-style variants tested at 100M-1B in recent ablations. Transfer risk is **med** (the lever is structural; smaller models can struggle with router training, but with zero-init experts the win is a pure capacity bonus at step 0).

## Why it's worth a slot
A win would tell us the *attention output* (not softmax variant, not residual structure) is the binding capacity constraint at 0.94M, complementing the closed MoE/FFN axis (146/117/118); a null would close the multi-attention-per-layer axis as similar to MoE.

## Plan
**Files**
- `models/layers.py` — `MultiHeadAttention.__init__` adds `use_moa: bool` + `moa_num_experts: int` kwargs and constructs `moa_extra_kv` (zero) + `moa_router_weight` (zero) + `moa_router_bias` (one-hot) parameters. `MultiHeadAttention.forward` adds the MoA branch at the top of the attention chain (replaces the standard branches when `use_moa=True`). `TransformerBlock.__init__` adds the same kwargs and passes them through to the inner MHA.
- `models/llm.py` — `MinimalLLM.__init__` reads `use_moa` / `moa_num_experts` from `config` and passes them through to both the standard block and the YOCO upper-half block.
- `configs/llm_config.py` — `LLMConfig` adds `use_moa: bool = False` + `moa_num_experts: int = 2`. New `Tiny1M3MMoAConfig(Tiny1M3MConfig)` sets `use_moa=True, moa_num_experts=2`.

**Flag**: `use_moa: bool = False` (off by default → baseline path bit-identical), `moa_num_experts: int = 2`.

**Step-0 byte-identity**:
- `(E-1)` extra sets of K_e, V_e projections are zero-init ⇒ extra experts produce 0 attention output.
- Router bias is one-hot on expert 0 (`[+30, 0, …, 0]`) ⇒ `softmax ≈ [1, 0, …, 0]` in fp32 ⇒ `g_0 = 1, g_e>=1 = 0`.
- Combined: `out = 1·attn_output_0 + 0·0 = attn_output_0` byte-identical to single standard attention at step 0.
- Router weight and bias are stored as raw `nn.Parameter` (not `nn.Linear`) so the construction does NOT consume RNG — keeping the RNG state aligned with the no-flag path (verified: max-abs-diff < 1e-6 at step 0 for E=2, 4, 8).

**Run command** (RUN-CONTRACT shape — the daemon launches `python <arq_file>`
from `run.json`; never a freeform CLI):
```bash
/venv/main/bin/python _arq_156-moa.py   # __main__ → train_llm.main(), seed 42
```

**Final val loss**: read from `tiny1m3m_moa_run_*/log.jsonl` or the eval-milestones block; compare to the cached baseline mean of 6.4302 ± 0.04.

