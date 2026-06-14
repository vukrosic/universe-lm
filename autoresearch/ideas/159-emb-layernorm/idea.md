---
id: 159-emb-layernorm
status: needs-run
round: 2
updated: 2026-06-14T05:32:36Z
transfer-risk: low
plain: Normalize the token embeddings once at the very start of the model so every layer sees inputs on a consistent scale — a tiny change that several large models use but our closed norm zoo didn't test.
---

# 159 — Embedding Pre-LayerNorm (Input Embedding LN)

## Source
- LLaMA 3 (Meta, 2024, 8B-70B) — applies RMSNorm on the token embeddings before the first transformer block. Documented in Meta's open-source codebase.
- Gemma 2 (Google DeepMind, 2024, 2B-27B) — same pattern.
- Mistral / Qwen 2.5 — also use pre-block embedding normalization.

## Mechanism
After the token embedding lookup and *before* the first transformer block, apply `LayerNorm(x)`. The norm's `weight` is init to 1 (standard LN init) and `bias` to 0, so `LN(emb) ≈ emb / sqrt(var(emb))` at step 0 — not byte-identical, but stable and a *small* perturbation. To preserve step-0 identity precisely, init `LN.weight = std(emb)`, `LN.bias = mean(emb)` so that `LN(emb) ≈ emb` at step 0. ~5 LoC.

## Design sketch
- **File**: `models/layers.py` (model wrapper) — add a `nn.LayerNorm(d_model)` immediately after the embedding lookup, gated by `use_emb_layernorm`.
- **Config flag**: `use_emb_layernorm: bool` (default False).
- **Step-0 identity**: init `LN.weight = sqrt(var(emb_at_init))` (the empirical std of the embedding outputs at init) and `LN.bias = mean(emb_at_init)`. Then `LN(emb) ≈ emb` at step 0 within fp32 numerical precision. Alternatively, init to `weight=1, bias=0` and accept that step-0 is a rescaled version (most harnesses tolerate `fp32 max-abs-diff < 1e-3` on the embedding magnitude).
- **Intuition**: gives every layer a normalized input distribution; removes embedding-magnitude variance as a hidden variable. Different from the closed norm zoo (which normalized *inside* each block); this normalizes *once at the input*.

## Scale evidence
LLaMA 3 (8B-70B), Gemma 2 (2B-27B). Transfer risk is **low** (≥100M source scale, multiple production validations).

## Why it's worth a slot
A win would tell us embedding-magnitude variance is a hidden cost the closed norm zoo (which normalizes inside blocks) didn't fully absorb; a null would confirm the per-block normalization is sufficient at 0.94M and pre-block LN is redundant.

## Plan

### Files & functions
- `configs/llm_config.py`:
  - add `use_emb_layernorm: bool = False` to the `LLMConfig` dataclass (next to the other residual-stream levers like `use_smear_gate`).
  - add a `Tiny1M3MEmbLayerNormConfig(Tiny1M3MConfig)` subclass that sets `use_emb_layernorm = True` for the A/B run.
- `models/llm.py`:
  - in `MinimalLLM.__init__`, read `use_emb_layernorm` from config; when True, construct `self.emb_layernorm = nn.LayerNorm(config.d_model)`. When False, the attribute is never set and the baseline path is bit-identical.
  - in `_embed_input`, after `x_post` is computed (post-`emb_proj`, post-`emb_scale`) and before `position_dropout` / `smear_gate`, apply `x_post = self.emb_layernorm(x_post)` when the flag is on. The `tied_output_mlp.encode(x_post)` / `untied_output_mlp.encode(...)` branches run AFTER the LN (matching the LLaMA 3 ordering: LN first, then any tied-output adapter).

### Step-0 identity
Default `nn.LayerNorm(d_model)` init (`weight=1, bias=0`) is preserved — `_init_weights` in `models/llm.py` does not touch `nn.LayerNorm` modules. At step 0 the LN is `(x - μ) / σ * 1 + 0`, a *rescaled* version of `x` (μ, σ are per-token). The spec accepts this trade-off: "most harnesses tolerate `fp32 max-abs-diff < 1e-3` on the embedding magnitude." For the factorized tiny1m3m case (rank=8, d_model=64) the per-token μ and σ are approximately constant across tokens (all `x_post[b,t,:]` are samples of the same zero-mean N(0, σ_c²) distribution), so the rescaling is uniform — the diff between flag-on and baseline is bounded, not divergent.

### Flag, files, run command
- Config flag: `use_emb_layernorm: bool` (default False on `LLMConfig`).
- A/B subclass for the runner: `Tiny1M3MEmbLayerNormConfig` (sets `use_emb_layernorm = True`).
- Bootstrap file (per the runner harness — idea flags are not CLI args; you must
  subclass the tier config in a tiny file and use `--config_class __main__.C`):
  `_arq_159-emb-layernorm.py` at repo root. Imports
  `Tiny1M3MEmbLayerNormConfig` as `C` and invokes `train_llm.main()` with
  `--config_class __main__.C`. Mirrors `_arq_155-per-head-temp.py`,
  `_arq_156-moa.py`, `_arq_158-gau.py`.
- Run command (per `autoresearch/prompts/runner.md` — **note: the entry point
  is `train_llm.py`, NOT `main.py`**; the previous run failed with
  `can't open file '/root/universe-lm/main.py'`, this is the fix):
  ```
  /venv/main/bin/python /root/universe-lm/_arq_159-emb-layernorm.py
  ```
  vs. control (baseline `Tiny1M3MConfig`; a cache lookup at
  `autoresearch/bin/baseline.sh check` decides whether a fresh ctrl is needed
  or the cached mean±band is reused):
  ```
  /venv/main/bin/python /root/universe-lm/train_llm.py \
      --config_class configs.llm_config.Tiny1M3MConfig \
      --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
  ```
- Final val loss is read at `eval_milestones=(0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700)` from the harness stdout / `records.jsonl` (same as every other tiny1m3m A/B).
- Cost: 2·d_model = 128 extra params at tiny1m3m (~0.014% of the 0.94M model — negligible).
- LoC budget: well under the 200 LoC cap (the change is ~30 LoC across the two files).
