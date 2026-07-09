---
id: 155-per-head-temp
status: done
round: 1
updated: 2026-06-14T00:50:41Z
transfer-risk: low
plain: Give each attention head its own learnable "sharpness" knob so some heads can focus tightly on a few positions while others spread broadly — start all knobs at 1 (no change) so step-0 matches the baseline exactly.
---

# 155 — Per-Head Learnable Attention Temperature

## Source
Multiple production LLMs (PaLM 2, OLMo 2, Gemma 2) use per-head temperature as a sub-component of attention. As an isolated mechanism: Wang et al. "Temperature-balanced Attention" (informal). Most directly related to attention softmax scaling variants discussed in Vaswani et al. (2017) and follow-ups.

## Mechanism
Replace the standard `1/sqrt(d_head)` attention scale with a per-head learnable scalar `τ_h`: `logits_h = Q_h @ K_h^T * τ_h`. With `τ_h = 1/sqrt(d_head)` at init, this is byte-identical to standard attention at step 0. As training proceeds, each head can adjust its own temperature — heads wanting sharper focus can lower `τ_h`, heads wanting broader context can raise it. ~10 LoC.

## Design sketch
- **File**: `models/layers.py` — modify the attention forward to use a learnable `self.attn_temperature = nn.Parameter(torch.full((n_heads,), 1.0 / math.sqrt(d_head)))`.
- **Config flag**: `use_per_head_temp: bool` (default False).
- **Step-0 identity**: `attn_temperature` is initialized to exactly `1/sqrt(d_head)`, so the logits `Q_h K_h^T * τ_h` are identical to baseline `Q_h K_h^T / sqrt(d_head)` at step 0.
- **Intuition**: gives each head a free parameter for "how peaky" its attention distribution should be, decoupled from Q/K weights. Different from qk_norm (016, normalizes magnitudes) and from logit-softcap (closed, clamps extremes). A null would tell us the per-head temperature axis is dominated by Q/K updates at this scale; a win would suggest a head-level attention-temperature prior is genuinely missing.
- **Important distinction**: `attn_logit_bias` (152) shifts attention location; `attn_temperature` (155) sharpens/broadens it. Two orthogonal axes.

## Scale evidence
PaLM 2 / OLMo 2 / Gemma 2 (≥2B source scale). Transfer risk is **low** (≥100M source scale, multiple production validations).

## Why it's worth a slot
Per-head temperature is one of the cheapest, most-isolated attention levers (H scalars). A null would close the per-head-attention-shape axis; a win would be a strong signal that head-level specialization is bottlenecked by Q/K updates alone.

## Plan

**Files touched**
- `models/layers.py` — `MultiHeadAttention.__init__` adds the `use_per_head_temp` kwarg + `self.attn_temperature = nn.Parameter(torch.full((n_heads,), 1/sqrt(d_k)))` (one scalar per head, init exactly the standard inverse-temperature so step-0 ≡ baseline). `MultiHeadAttention.forward` REPLACES the standard `scores = matmul * (1/sqrt(d_k))` with `scores = matmul * attn_temperature` in BOTH manual-path branches (the FIRE branch and the big manual branch), and adds `or self.use_per_head_temp` to the manual-path trigger list so SDPA's flash/efficient backends don't perturb step-0 numerics. `TransformerBlock.__init__` passes the new flag through.
- `models/llm.py` — read `config.use_per_head_temp` via `getattr` (default `False`) and pass it into every `TransformerBlock` constructor call (one call site — the standard path; YOCO upper-half block doesn't currently pass `use_attn_logit_bias` either, matching the existing pattern).
- `configs/llm_config.py` — add `use_per_head_temp: bool = False` to `LLMConfig`; add `Tiny1M3MPerHeadTempConfig(Tiny1M3MConfig)` subclass that flips the flag on.

**Config flag**: `use_per_head_temp: bool` (default `False`); off ⇒ no Parameter registered, no manual-path trigger, no branch taken, baseline forward graph bit-identical.

**LoC**: ~14 added to MHA (kwarg + 4-line init + ~5 lines per-branch in forward + 1 line in the manual-path trigger list); ~3 added to `TransformerBlock.__init__` pass-through; ~3 in `models/llm.py`; ~12 in `configs/llm_config.py` (mostly docstring). Total ~35 LoC, well under the 200 ceiling.

**Step-0 identity**: when off, the MHA does not register `self.attn_temperature`, the elif trigger does not include `use_per_head_temp`, the score-multiply branch is never taken, and the forward graph is byte-identical to the no-flag baseline. Verified locally on a `MinimalLLM(Tiny1M3MConfig)` vs `MinimalLLM(Tiny1M3MPerHeadTempConfig)`: max |Δ logits| = 2.98e-08 (fp32 noise floor). Param delta = 48 = 12 layers × 4 heads.

**On-the-box script** (`_arq_155-per-head-temp.py`):
```python
# Re-code fix (round 1 → round 2): the original
#   class C(Tiny1M3MConfig):
#       use_per_head_temp: bool = True
# DID NOT actually set `use_per_head_temp=True` — without a `@dataclass`
# re-decoration on the subclass, the parent's dataclass field default
# (False) is inherited verbatim and the re-annotation is ignored, so
# `C().use_per_head_temp` resolves to False and the per-head-temp
# branch is never taken. Import the canonical @datlass-decorated
# subclass instead.
from configs.llm_config import Tiny1M3MPerHeadTempConfig as C

if __name__ == "__main__":
    import sys, train_llm
    sys.modules["__main__"].C = C
    sys.argv = ["train_llm.py", "--config_class", "__main__.C",
                "--seed", "42", "--dataset_path", "processed_data/pretrain_1B",
                "--warmup", "false"]
    train_llm.main()
```

**Run command** (per `autoresearch/prompts/runner.md`; control first, then test):
```bash
# Control (no flag):
/venv/main/bin/python train_llm.py --config_class configs.llm_config.Tiny1M3MConfig --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
# Test (flag on via __main__.C subclass):
/venv/main/bin/python _arq_155-per-head-temp.py
```

**Final val-loss read**: end of `logs/<run-name>/log.jsonl` → last `eval/milestone` entry's `val_loss` field. Compare against the locked tiny1m3m baseline cache at `autoresearch/baseline-cache.json`. PASS ≤ ctrl − 0.005. NULL band |Δ| < 0.005. DRIFT > +0.005.

**Predictions**: small wash (|Δ| < 0.005) most likely — `1/sqrt(d_k)` is the canonical default across Transformers, and the Q/K gradients can absorb the per-head scale change at this tier. A clear win would be a strong signal that the per-head temperature axis was missing; a clear loss would suggest a useful prior is being clobbered.

**Status claim**: `autoresearch/bin/flip.sh 155-per-head-temp needs-run implement-button "code ready; runnable at tiny1m3m seed 42"`.

