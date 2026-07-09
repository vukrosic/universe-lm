# Plan — 159 — Embedding Pre-LayerNorm

## Flag
- `use_emb_layernorm: bool` (default `False`) on `LLMConfig`.
  - Field: `configs/llm_config.py:63`.
  - A/B subclass: `Tiny1M3MEmbLayerNormConfig(Tiny1M3MConfig)` with `use_emb_layernorm: bool = True` at `configs/llm_config.py:1848`.

## Change
- `models/llm.py`:
  - In `MinimalLLM.__init__` (`models/llm.py:174-176`): read the flag via `getattr(config, "use_emb_layernorm", False)`; when on, build `self.emb_layernorm = nn.LayerNorm(config.d_model)`. The flag-OFF path never instantiates the LN module → baseline graph bit-identical.
  - In `_embed_input` (`models/llm.py:1200-1201`): after `x_post = tok * emb_scale` (or the factorized `emb_proj(tok) * emb_scale`) and *before* the tied/untied output-MLP encode branches, apply `x_post = self.emb_layernorm(x_post)` when the flag is on. LLaMA-3 ordering: pre-block norm first, then the tied/untied output-MLP encode.
  - The init comment block at `models/llm.py:1105-1119` documents the actual step-0 behavior: `nn.LayerNorm` defaults are `weight=1, bias=0` (left untouched by `_init_weights`), so step 0 is a uniform per-token rescaling `(x − μ(x)) / σ(x)`, not byte-identical. The spec accepts this trade-off (the factorized tiny1m3m per-token μ/σ are ~constant across tokens because every `x_post[b,t,:]` is a sample of the same zero-mean N(0, σ_c²) distribution, so the rescaling is uniform and bounded — not divergent).

## Control
- Treatment: `configs.llm_config.Tiny1M3MEmbLayerNormConfig` (flag on), seed 42, tier tiny1m3m, dataset `processed_data/pretrain_1B`, warmup off.
- Control (owned by the daemon): `configs.llm_config.Tiny1M3MConfig` (flag off), seed 42, same tier / dataset / warmup.
- Verdict: `autoresearch/bin/baseline.sh verdict` (mean ± band) on the Final Val Loss at `eval_milestones=(0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700)`.

## Cost
- Params: +2·d_model = +128 at tiny1m3m (~0.014% of the 0.94M model — negligible).
- FLOPs: 1·d_model per token per forward (a single LN pass at the embedding).
- Memory: 2·d_model extra weight bytes; no activation memory delta (LN is fused into the embedding pass).

## Run
- Artifact: `_arq_159-emb-layernorm.py` (repo root) defines top-level `class C(Tiny1M3MConfig): use_emb_layernorm = True` and dispatches `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/159-emb-layernorm/run.json` — `{"name": "159-emb-layernorm", "arq_file": "_arq_159-emb-layernorm.py", "job_timeout": "12m"}`.
- Daemon (`autoresearch/bin/queue-daemon.sh`): scp's the stub, runs the CPU build-smoke (`python _box_smoke.py _arq_159-emb-layernorm.py` → `SMOKE_OK`), then launches the run in the `arq` tmux.
- Pass/fail bar (copied from `idea.md`): treat as a single-axis A/B; a win = `Final Val Loss` below the control `mean − band` (per `baseline.sh verdict`); a null = within band; a crash/NaN = bounce back to `needs-recode` (still inside the round-2 recode budget).
- Recode origin (round 1): the prior run failed at the harness layer (`/venv/main/bin/python: can't open file '/root/universe-lm/main.py'`) because it shipped a `main.py`-style CLI invocation rather than the modern `_arq_<idea>.py` + `run.json` artifact shape. The code itself is correct; this gate only swaps the run descriptor and emits the stub the daemon actually consumes.
