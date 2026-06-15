---
id: 146-sparse-ffn
status: done
round: 1
updated: 2026-06-13T20:32:25Z
transfer-risk: med
plain: Replace the dense FFN with a sparse mix-of-experts FFN where each token is sent to a single expert via a learned router.
---

# 146 — Sparse FFN (Switch-style)

## Source
Fedus, Zoph, Shazeer 2022, "Switch Transformers: Scaling to Trillion Parameter Models with Simplicity and Stability", Google, arXiv:2101.03961. https://arxiv.org/abs/2101.03961

## Mechanism
Replace the dense FFN with N parallel FFN "experts" and route each token to a single expert via a learned router (top-1 routing). This is the simplest form of sparse mixture-of-experts in the FFN position.
- `router_logits = W_router * x`  shape `[B*T, n_experts]`
- `expert_idx = argmax(router_logits, dim=-1)`  shape `[B*T]`
- For each token `i`: `output[i] = expert[expert_idx[i]](x[i])`

Capacity factor: each expert can hold at most `capacity = ceil(n_tokens / n_experts) * capacity_factor` tokens; overflowed tokens are skipped (passed through residual).

Identity at step 0: router weights init to 0 → argmax over uniform is arbitrary (by index in our impl). All tokens route to expert 0. Other experts are never used. Output = expert_0(x) for all tokens = standard FFN at step 0. ✓

## Design sketch (how it works + how to build it)
- Add a `SwitchFFN` module to `models/layers.py`: holds `n_experts` parallel `nn.Linear(d_model, d_ff)` + `nn.Linear(d_ff, d_model)` pairs, plus a `nn.Linear(d_model, n_experts, bias=False)` router. Forward: route, gather tokens per expert, run expert FFN, scatter back. ~120 LoC.
- Add `use_switch_ffn: bool = False`, `n_ffn_experts: int = 4`, `expert_capacity_factor: float = 1.25` to `configs/llm_config.py`.
- Identity at step 0: as above (all tokens → expert 0, output = single FFN). Note that switching from dense FFN to Switch FFN at flag-on multiplies the FFN params by `n_experts` (4×) — a sizeable param injection. So `use_switch_ffn=False` baseline is dense-FFN tiny1m3m, and `use_switch_ffn=True` is 4×-FFN-params tiny1m3m. The lever is "can the model use the 4× capacity to win val loss?".
- Why a real lever, not a hyperparam: the *routing* (top-1 hard routing) is a structural choice, not a hyperparameter. Different from 117-soft-moe (slot assignment, all experts always used) and 118-MoD (skip-routing). Three distinct MoE mechanisms, none closed at this axis.
- Targets baseline failure: dense FFN uses the same params for every token. Switch FFN can specialize experts on different token types (rare words, common words, code-like tokens, etc.) — at the cost of routing overhead and imbalance. The closed levers 117/118 nulled due to MoE-overhead-at-0.94M; Switch is the simplest MoE so the overhead is also minimal.

## Scale evidence
Paper trains 1.6T-param Switch Transformer (T5-based). 0.94M is 6+ orders of magnitude below the validated range. Independent replications show consistent (small) gains on small models when expert count is 2–4. Transfer risk: med — the FFN param injection is 4×, which is a *real* advantage even at 0.94M and might win on capacity alone.

## Plan

**Files changed**
- `models/switch_ffn.py` (NEW): `SwitchFFN` module — E parallel full-width experts + top-1 learned router. ~140 LoC.
- `models/layers.py`: import `SwitchFFN`; add `use_switch_ffn`, `n_ffn_experts`, `expert_capacity_factor` kwargs to `TransformerBlock.__init__`; select `SwitchFFN` (instead of dense FFN) when `use_switch_ffn=True`. ~10 LoC.
- `configs/llm_config.py`: add `use_switch_ffn: bool = False`, `n_ffn_experts: int = 4`, `expert_capacity_factor: float = 1.25` flags to `LLMConfig` (defaults match the idea spec). ~10 LoC.

**Flag**: `use_switch_ffn` (off by default — baseline dense FFN path is bit-identical when off).

**Identity at step 0**: `W_router` is zero-init; `argmax(0)` returns 0 for every row; all tokens route to expert 0; output = `expert_0(x)` = a standard dense FFN at step 0. Verified: `SwitchFFN.forward(x) == experts[0](x)` exactly at step 0.

**Capacity-factor caveat**: the paper's formula `ceil(N/E) * cf` yields paper_capacity < N when cf < E. At tiny1m3m N=4096, E=4, cf=1.25: paper_capacity=1280 — degenerate step-0 routing (all → expert 0) would drop 2816/4096 tokens. To keep step-0 byte-identical, we clamp `effective_capacity = max(N, paper_capacity)`. This means the cf knob is documentation at default values (cf <= E); users can lower cf below 1.0 to force truncation. The paper's drop mechanism is intact in code — the clamp just disables it at default settings.

**Run command** (mirrors `_arq_141-adabelief.py` template):
```bash
cd /root/universe-lm
python train_llm.py \
    --config_class configs.llm_config.Tiny1M3MConfig \
    --seed 42 \
    --dataset_path processed_data/pretrain_1B \
    --warmup false
# vs. treatment:
python _arq_146-sparse-ffn.py
```

where `_arq_146-sparse-ffn.py` is:
```python
from configs.llm_config import Tiny1M3MConfig
class C(Tiny1M3MConfig):
    use_switch_ffn: bool = True
    n_ffn_experts: int = 4
    expert_capacity_factor: float = 1.25
```

**Reading final val loss**: `--eval_milestones` includes the terminal milestone; the trainer prints `eval: step=… loss=…`. Take the loss at the last milestone. Compare to ctrl val on the same box, same session. NULL band |Δ| < 0.04 (two-ctrl bracket).

**Total LoC**: ~160 (SwitchFFN module + plumbing). Within the < 200 LoC budget.

## Why it's worth a slot
This is the *simplest* MoE mechanism — single-expert hard routing — and the closest to a "FFN with a routing head" extension. 117-soft-moe had slot-assignment overhead; 118-MoD had skip-router overhead. Switch FFN has neither — it's `argmax` plus gather/scatter. A null here (with 4× FFN params) would conclusively close the MoE axis at 0.94M; a win would be a real surprise and tell us sparse FFN can fire at tiny scale.
