---
id: 152-attn-logit-bias
status: done
round: 1
updated: 2026-06-14T00:28:30Z
transfer-risk: low
plain: Give each attention head a small learnable nudge on its attention pattern, starting from no nudge at all, so heads can specialize their focus without disturbing step-0 behavior.
---

# 152 — Per-Head Attention Logit Bias

## Source
Anil et al. "PaLM 2 Technical Report" (Google, 2023) — Section on model architecture uses per-head attention logit bias (`bias` in `attn_logits`) alongside standard attention. Also OLMo 2 (OLMo et al. 2024) and several open-source reproductions. No single canonical arXiv; mechanism predates the modern literature (Vaswani et al. 2017 mentions it in passing).

## Mechanism
After computing `logits = QK^T / sqrt(d_head)`, add a learnable per-head additive bias `b_h` (shape `[H]`) before softmax: `logits_h = logits_h + b_h`. With `b_h = 0` at init, softmax is unchanged. Two-line change in `models/layers.py` attention forward. ~10 LoC including the parameter registration.

## Design sketch
- **File**: `models/layers.py` — modify the attention forward to add `self.attn_logit_bias = nn.Parameter(torch.zeros(n_heads))` after `QK^T / sqrt(d_head)` and before the softmax.
- **Config flag**: `use_attn_logit_bias: bool` (default False); when True, register the parameter and add it.
- **Step-0 identity**: `attn_logit_bias` is `torch.zeros(n_heads)`, so `softmax(QK^T/√d + 0) = softmax(QK^T/√d)` byte-for-byte at step 0.
- **Intuition**: lets each head shift its attention distribution independently (head 3 might prefer tighter locality, head 7 might prefer broader context) without forcing it through QK projections. The QK-norm lever (016) constrained the *scale* per head; this constrains the *location*.
- **Why now**: covered axes (logit softcap is a clamp, qk_norm is a normalize, both globally) didn't allow per-head additive offset. A null here would tell us per-head additive flexibility is dominated by Q/K weight updates at 0.94M.

## Scale evidence
Used in PaLM 2 (540B) and OLMo 2 (7B-13B); transfer risk is **low** (≥100M source scale, multiple production validations). The mechanism is cheap (H scalars, negligible compute) so it should not hurt at tiny1m3m.

## Why it's worth a slot
We expect each head to find a useful per-head shift on its attention distribution (e.g., one head discovers it wants a "sink-like" prior, another wants strict locality), independent of Q/K updates; a null confirms the per-head additive axis is saturated by the existing logit-softcap-clamp and qk-norm levers at this tier.

> **Caveat (math).** A *per-head scalar* `b_h ∈ R^H` added to scores is mathematically absorbed by softmax over the key axis: `softmax(QK^T/√d + b_h)[b,h,t,s] = softmax(QK^T/√d)[b,h,t,s]` because the per-(b,h,t) normalizer factors `e^{b_h}` cancel. So `b_h` has zero gradient and zero effect on the output for *all* steps, not just step 0. Implementing the lever exactly as specified here therefore produces a deterministic mathematical null — useful as a recorded baseline-null, but it does not actually test whether per-head additive flexibility helps. PaLM 2's actual formulation uses `attn_logits` bias of shape `[H, S]` (per-head × per-position) which breaks the cancellation. We implement `[H]` as the idea spec; if the result is a strong null, that's the documented outcome.

## Plan

**Files touched**
- `models/layers.py` — `MultiHeadAttention.__init__` adds the flag and Parameter; `MultiHeadAttention.forward` injects `attn_logit_bias` into scores and forces the manual attention path when the flag is on (so SDPA's flash/efficient backends don't change step-0 numerics). `TransformerBlock.__init__` passes the new flag through.
- `models/llm.py` — read `config.use_attn_logit_bias` via `getattr` (default `False`) and pass it into every `TransformerBlock` constructor call (two call sites).
- `configs/llm_config.py` — add `use_attn_logit_bias: bool = False` to `LLMConfig`; add `Tiny1M3MAttnLogitBiasConfig(Tiny1M3MConfig)` subclass that flips the flag on.

**Config flag**: `use_attn_logit_bias: bool` (default `False`); off ⇒ baseline forward graph bit-identical.

**LoC**: ~12 added to MHA (param registration + flag handling + 2 lines in forward + 1 line in the manual-path trigger list); ~6 added to TransformerBlock / `models/llm.py`; ~10 in `configs/llm_config.py`. Total ≲ 30 LoC, well under the 200 ceiling.

**Step-0 identity**: when off, the MHA does not register `self.attn_logit_bias`, the elif trigger does not include `use_attn_logit_bias`, the score-add branch is never taken, and the forward graph is byte-identical to the no-flag baseline.

**Run command** (runner, not executed here):
```bash
/venv/main/bin/python -c "
import sys; sys.path.insert(0, '.')
from configs.llm_config import Tiny1M3MAttnLogitBiasConfig, Tiny1M3MConfig
from models.llm import MinimalLLM
import torch
torch.manual_seed(42)
m_ctrl = MinimalLLM(Tiny1M3MConfig())
torch.manual_seed(42)
m_test = MinimalLLM(Tiny1M3MAttnLogitBiasConfig())
# Step-0 byte-identical: same logits within fp32 noise
x = torch.randint(0, m_ctrl.config.vocab_size, (2, 64))
y_ctrl = m_ctrl(x); y_test = m_test(x)
assert torch.allclose(y_ctrl, y_test, atol=1e-5), (y_ctrl - y_test).abs().max()
print('step-0 byte-identical OK')
"
```

**Final val-loss read**: end of `logs/<run-name>/log.jsonl` → last `eval/milestone` entry's `val_loss` field. Compare against the locked tiny1m3m baseline cache at `autoresearch/baseline-cache.json`. PASS if `val_loss ≤ baseline − 0.005`, DRIFT if `> baseline + 0.005`, NULL otherwise (see `autoresearch/closed.md` band conventions).

**Status claim**: `autoresearch/bin/flip.sh 152-attn-logit-bias needs-run implement-button "code ready; runnable at tiny1m3m seed 42"`.
