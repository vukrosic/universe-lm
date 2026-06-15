---
id: 211-swiglu-alibi
author: claude-opus-4-8
status: running
round: 1
updated: 2026-06-15T12:49:56Z
transfer-risk: low
plain: Stack SwiGLU (the standard FFN in every modern open LLM — LLaMA, Mistral, Qwen, Gemma, PaLM) on top of the current ALiBi champion. ALiBi shapes attention over POSITIONS; SwiGLU rebuilds the FFN as a gated linear unit — a completely different sub-block. Unlike the last three attempts (208/209/210) which all touched attention and shared an axis with ALiBi and washed out, SwiGLU is on the FFN axis with ZERO overlap, and its effect size is large enough to clear the 0.04 noise band.
---

# 211 — SwiGLU FFN (Shazeer 2020) on the 175-ALiBi Champion

## Source
- Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202). SwiGLU is the FFN in LLaMA / Mistral / Qwen / Gemma / PaLM — the most battle-tested architectural FFN choice in the field.
- **In-repo:** `Tiny1M3MSwigluFFNConfig` / flag `use_swiglu_ffn` (`configs/llm_config.py:2582`). Zero-init gate, Shazeer 2/3 d_ff trick so FFN param count matches baseline within ~0.4%.

## Why this is the highest-EV record attempt now
The champion is **only** ALiBi (175, val 6.2403). The three most recent stacks all washed out because each touched attention and **shared an axis** with ALiBi:
- **208-value-residual** (NULL, Δ+0.019) — V/attention axis.
- **209-canon-conv** (NULL, Δ+0.012) — residual-stream locality, overlaps ALiBi's locality prior.
- **210-qk-layernorm** (NULL, Δ+0.021) — attention logit magnitude.

The lesson is twofold: (1) the second lever must share **no axis** with ALiBi, and (2) it must have an effect size **larger than the 0.04 noise band** — micro-levers are invisible at 92 steps. SwiGLU satisfies both: it is the **FFN** sub-block (zero attention overlap) and is a **large structural** change, not a micro-lever.

## Mechanism
Replace the FFN `down(act(W1·x))` with the gated linear unit:
```
y = down_proj( silu(W_gate·x) ⊙ (W_up·x) )
```
The gate matrix `W_gate` is **zero-init** ⇒ `silu(0)=0` ⇒ FFN output is exactly 0 at step 0 ⇒ residual carries only the attention sub-block ⇒ **byte-identical to the ALiBi champion at step 0** (clean ReZero-style start; the optimizer grows the gate). `d_ff` is scaled by the Shazeer 2/3 trick.

## Config (inline, no llm_config.py edit)
`_arq_211-swiglu-alibi.py` defines `@dataclass class C(Tiny1M3MAlibiConfig): use_swiglu_ffn = True` inline (the `@dataclass` decorator is required for the field override to take on the instance — the dataclass-inheritance pitfall from `_arq_161-dyt-temp.py`). Adds **no edit** to `configs/llm_config.py`, which the autopilot is concurrently editing.

## A/B design
- **Control**: `Tiny1M3MAlibiConfig` (champion, val 6.2403, band 0.04 — cache-authoritative, no re-measure).
- **Treatment**: inline `C` (`use_swiglu_ffn=True`).
- **PASS / WIN** (daemon gate): val < 6.2403 − 0.04 = **6.2003**.
- **NULL** band |Δ| < 0.04.
- Single seed (42); sub-noise INCONCLUSIVE per the one-seed-only rule.

Tier: tiny1m3m (0.94M, 12L, 4H, d_model=64), 92 update steps, seed 42, no warmup.

## Pre-run verification (done locally, claude-opus-4-8)
- `@dataclass` override took: `use_swiglu_ffn=True` on the instance ✓
- model builds: 947,568 params (alibi 949,104, Δ−1,536 = Shazeer 2/3 FFN reshape) ✓
- active lever: max-abs logit diff alibi vs treatment after 1 optimizer step = 0.0814 (not a no-op; the gate grows) ✓
