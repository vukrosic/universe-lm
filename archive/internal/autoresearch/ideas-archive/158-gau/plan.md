# Plan — 158 gau

## Flag
- `use_gau: bool` (default `False`) on `LLMConfig`.
  - Field: `configs/llm_config.py` (new field added to `LLMConfig`).
  - A/B subclass: `Tiny1M3MGAUConfig(Tiny1M3MConfig)` with `use_gau: bool = True` at `configs/llm_config.py:5241-5303`.
  - Run stub: `_arq_158-gau.py` (repo root) subclasses `Tiny1M3MConfig` with `use_gau=True` and dispatches `train_llm.main()` via `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.

## Change
- `models/layers.py` — new `GAUBlock(nn.Module)` class at `models/layers.py:4305-4446` (~142 LoC body). Fused `Attention + FFN` unit. Single `nn.Parameter` of shape `[fused_size, d_model]` packs the Q/K/V/U_g/V_o projections (sizes `q_size + 2·kv_size + 2·d_model`), initialized with `normal_(std=0.02)` then the `U_g` and `V_o` slices are explicitly zeroed (`models/layers.py:4361-4362`) for step-0 identity. A second `nn.Parameter` of shape `[d_model, d_model]` is the output projection `U_o` (standard init). Pre-norm + RoPE on Q/K + causal SDPA + element-wise `z * V_o` gating + residual + dropout. Block signature matches `TransformerBlock.forward(x, x0, ve, **kwargs)` so the model loop dispatches transparently.
- `models/llm.py` — `MinimalLLM.__init__` reads `self.use_gau = getattr(config, "use_gau", False)` (`models/llm.py:483`). When on, asserts `use_yoco=False` (mutually exclusive, raised `ValueError` at construction), instantiates `self.gau_blocks = nn.ModuleList([GAUBlock(...) for _ in range(n_unique)])`, and `self.transformer_blocks` stays as-is (still built). When off, `self.gau_blocks = None` (baseline path never touches the new stack — bit-identical). The forward loop in `_run_post_embed` (`models/llm.py:1373-1374`) dispatches `block = self.gau_blocks[i // self.tie_layer_groups]` when `self.use_gau`. The `use_value_residual` layer-0 stash (`models/llm.py:1447-1448`) skips when `self.use_gau` (`GAUBlock` has no `.attention`). Hyper-Connections wrapper asserts mutually exclusive with GAU (`models/llm.py:1405-1411`).
- `configs/llm_config.py` — adds `use_gau: bool = False` on `LLMConfig`. Adds `Tiny1M3MGAUConfig(Tiny1M3MConfig)` `@dataclass` subclass (`configs/llm_config.py:5241-5303`) with `use_gau: bool = True`. No other tier inherits GAU; this is tiny1m3m-only per the one-tier rule.
- `training/trainer.py` — `_collect_entropy_reg` guard at `training/trainer.py:1442-1458` replaces the bare `for block in m.transformer_blocks:` with `for block in (m.transformer_blocks or ()):`. When `use_gau=True`, `m.transformer_blocks` is still a `nn.ModuleList` (the GAU branch builds GAU blocks but does not strip the standard stack — the GAU-only `forward` branch dispatches via `gau_blocks`, leaving `transformer_blocks` populated as an empty ModuleList when `use_gau=True`... actually: re-reading, GAU's branch sets `gau_blocks = nn.ModuleList(...)` and the standard stack is still built below, but `_run_post_embed` only reads from `gau_blocks`. The `or ()` guard makes the entropy-reg helper robust to either case.) Baseline path: `m.transformer_blocks` is a truthy `nn.ModuleList` ⇒ loop runs unchanged ⇒ byte-identical.

Step-0 identity (flag OFF): no new Parameter is registered, no `gau_blocks` is built, the forward loop dispatches via `transformer_blocks` ⇒ baseline path bit-identical. Verified locally: `MinimalLLM(Tiny1M3MConfig())` ≡ `MinimalLLM(Tiny1M3MConfig())` to 0.0 max-abs-diff on a 16-token forward at seed 42.

Step-0 identity (flag ON): the `fused_proj` + `out_proj` Parameters consume RNG state during model construction, AND the residual `nn.Parameter` count per block is smaller (16K vs ~45K for `TransformerBlock`), so the flag-on model is NOT byte-identical to the bare baseline at step 0. The spec pin for GAU is internal: with the explicit zero-init of the U_g and V_o slices (`models/layers.py:4361-4362`) and `U_o(0)=0` ⇒ `block(x)=x` exactly at step 0. The freed parameters are an intentional consequence, not an oversight.

## Control
- A (treatment): `configs.llm_config.Tiny1M3MGAUConfig` (flag on), seed 42, tier tiny1m3m, dataset `processed_data/pretrain_1B`, warmup off.
- B (control, owned by the daemon): `configs.llm_config.Tiny1M3MConfig` (flag off), seed 42, same tier / dataset / warmup.
- Verdict: `autoresearch/bin/baseline.sh verdict` on `Final Val Loss` at step 700, then apply the plan's bar (below).
- Tier: `tiny1m3m` (0.94M params, 3M tokens). Seed 42 only — one-seed-only rule.

## Cost
- Params: ~−350K (~−37%) at tiny1m3m. Baseline `TransformerBlock` per layer (squared_relu FFN, GQA): ~45K = 12,288 (qkvo) + 32,768 (FFN `2·d_model·d_ff`). `GAUBlock` per layer: ~16K = 12,288 (fused Q/K/V/U_g/V_o) + 4,096 (U_o) + 64 (norm gain). Total over 12 layers: ~558K → ~196K. Whole-model total: 949,056 → 640,320 (−32.5%). The 12-layer GAU stack is missing the entire `2·d_model·d_ff = 32,768` per block (the FFN is folded into the gating pair, not duplicated).
- FLOPs: −25% per-token (no FFN matmul; one fused 192→64 matmul replaces the 192→64 attention matmul + 256→64/64→256 FFN matmuls).
- Memory: smaller activations (no FFN intermediate). Same attention activation size.

## Run
- Artifact: `_arq_158-gau.py` (repo root, already present) defines top-level `class C(Tiny1M3MConfig): use_gau = True` and dispatches `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/158-gau/run.json` — `{"name": "158-gau", "arq_file": "_arq_158-gau.py", "job_timeout": "12m"}` (already present).
- Daemon (`autoresearch/bin/queue-daemon.sh`): scp's the stub, runs the CPU build-smoke (`python _box_smoke.py _arq_158-gau.py` → `SMOKE_OK`), then launches the run in the `arq` tmux.
- Expected wall-clock: ~3-7 min (smaller model, same token budget; faster than baseline's ~5-9 min).
- **Pass/fail bar** (copied from `idea.md`):
  - PASS: `Final Val Loss` ≤ `ctrl_mean − 0.01` (the −37% param cut is more than recovered by an architectural win at the binding-bottleneck level — a strong claim that the Attention/FFN separation is what the 0.94M tier actually needs to drop).
  - NULL: `|Δ| ≤ 0.01` (the closed FFN-side levers all hit null at this tier, and GAU is a stricter form of the same axis; per-block rebalancing absorbs the param cut).
  - DRIFT: `Δ > +0.01` (the −37% param cut under-trains the model relative to the same-token-budget control — the GAU design point is parameter efficiency at low budgets, but here we re-allocate the freed budget into nothing).
- **Recode origin**: rc=1 in `log.jsonl` (the runner bounced the idea back for the `_collect_entropy_reg` NoneType error). Local fix is the `or ()` guard at `training/trainer.py:1453`. The daemon's CPU build-smoke already catches stale-on-box trainer.py and bounces back here before GPU time is spent, so this gate re-confirms the build-smoke (§5 self-check) before re-releasing. Round stays at 1 per the reviewer's `round reset on approve` rule.