# Universe experiments plan

A handoff doc. Any agent (human or AI) should be able to pick one section and
ship it without asking the author for clarification. If something here is
ambiguous, fix the doc first, then implement.

## Ground rules

- One change per branch. Name: `experiment/<slug>` or `sweep/<slug>`.
- Every run produces: `runs/<branch>-<timestamp>/{config.json, train_log.txt, ppl.json, val_loss.csv}`.
- Compare against the **pinned baseline** in `BASELINE.md` (same data, same tokens, same seed).
- Release budget: **1B tokens per main release**, ~$3-5 on 3090.
- Sweep budget: **$2/week**, spent on 5M-param × 100M-token runs.
- Merge to `main` only if val loss ≤ baseline at the same token budget.

## Branch strategy

| Branch | Purpose | Lifetime |
|--------|---------|----------|
| `main` | Stable. Releases tagged from here. | Forever |
| `experiment/<slug>` | One architectural or training change. | Until merged or abandoned |
| `sweep/<slug>` | Runs many variants, writes a CSV, dies. | Days |
| `release/v0.X` | Frozen at release time. | Forever |

If a sweep proves a change works, open a clean `experiment/<slug>` branch with
just that change and a one-paragraph summary of the sweep result, and PR that
into `main`.

## Sweep harness (build this first — blocks everything else)

**Goal:** run N variants of a config in sequence, log their val loss curves
side by side, output a results CSV.

**Files to create:**

- `experiments/sweep.py` — CLI: `python -m experiments.sweep --config sweep_qk_norm.yaml`
- `experiments/sweeps/<name>.yaml` — declares baseline config + list of overrides
- `experiments/results/<sweep-name>.csv` — one row per variant: `variant, final_val_loss, ppl, train_tokens, wall_time_s, params`

**YAML schema:**

```yaml
base_preset: "5m"       # one of the configs in configs/llm_config.py
train_tokens: 100_000_000
seed: 42
variants:
  - name: baseline
    overrides: {}
  - name: qk_norm
    overrides: { use_qk_norm: true }
  - name: value_residual
    overrides: { use_value_residual: true }
```

**Behavior:**
1. For each variant, monkey-patch the config, run training, dump artifacts to `runs/<sweep>/<variant>/`.
2. After all variants finish, write `experiments/results/<sweep>.csv` and print a sorted table to stdout.
3. Honor `--dry-run` to print the plan without training.

**Implementation (done):** `experiments/sweep.py` exposes `run_sweep(yaml_path, dry_run=False)` and a CLI `python -m experiments.sweep --config <yaml> [--dry-run]`. Variants are trained by importing `train_minimal_llm` directly (not subprocess). Artifacts written per variant: `config.json`, `train_log.txt` (stdout captured), `val_loss.csv`, `final_metrics.json`. Results CSV at `experiments/results/<sweep>.csv`. `runs/` is gitignored.

## Experiment list

Each experiment below is a self-contained ticket. Hypothesis → change → method
→ success criterion. All initial sweeps run at **5M params × 100M tokens**
unless otherwise noted.

### EXP-1: muP (Maximal Update Parametrization)

**Hypothesis:** with muP, the optimal Muon LR found at 5M transfers to 70M
within ±20%. Without muP, it doesn't.

**Change:**
- Add `use_mup: bool = False` to `LLMConfig`.
- Add `mup_base_d_model: int = 128` (the "base width" we tune at).
- When `use_mup=True`:
  - Scale init std for hidden Linear layers by `sqrt(mup_base_d_model / d_model)`.
  - Scale Muon LR for hidden Linear weights by `mup_base_d_model / d_model`.
  - Embeddings, unembedding (tied), and norm layers unchanged.
- Reference: Yang et al. 2022, or the `mup` PyPI package as a reading aid (do not
  pull as a dependency, hand-roll ~30 lines).

**Files:** `configs/llm_config.py`, `models/llm.py` (init), `optimizers/muon.py`
(per-param-group LR scaling).

**Method:** LR sweep at 5M and 25M with `use_mup=True`. Confirm the LR that
minimizes val loss at 5M is within 20% of the optimum at 25M.

**Success:** LR transfer holds. If yes, all subsequent sweeps use muP and can
tune LR once at 5M.

**Branch:** `experiment/mup`

### EXP-2: Cosine LR schedule + warmup

**Hypothesis:** current `constant` schedule with `warmup_ratio=0` leaves
≥3% val loss on the table.

**Change:** in `train_llm.py`, when `schedule_type == "cosine"`, use cosine
decay from peak to 10% of peak over total steps, with linear warmup over
`warmup_ratio * total_steps`.

**Method:** at 25M × 500M tokens, run `constant` (current) vs `cosine` with
`warmup_ratio=0.02`.

**Success:** cosine ≥3% lower val loss. Make cosine the default if so.

**Branch:** `experiment/cosine-schedule`

### EXP-3: QK-norm

**Hypothesis:** RMSNorm on Q and K before attention stabilizes Muon at higher
LR, letting peak LR rise ~2× without divergence. Reduces val loss ~1-2%.

**Change:** in `models/layers.py`, attention block, after Q/K projection apply
`RMSNorm(d_k)` per head before computing logits.

**Method:** at 5M, sweep `{baseline, qk_norm}` × `{lr=current, lr=2×current}`.

**Success:** best `qk_norm` variant beats best baseline by ≥1% val loss.

**Branch:** `experiment/qk-norm`

### EXP-4: Value residuals

**Hypothesis:** carrying a value residual from layer 0 to later layers
reduces val loss ~3-5% (parameter-golf result).

**Change:** in `models/llm.py`, save the layer-0 V tensor and pass to later
blocks. Each block mixes `V_local = (1-α) * V_layer + α * V_0`, learnable α
per layer, init α=0.

**Method:** sweep at 5M, verify direction at 25M.

**Success:** ≥2% val loss reduction at 25M.

**Branch:** `experiment/value-residual`

### EXP-5: QK-Gain

**Hypothesis:** a learnable scalar gain on attention logits per head reduces
val loss ~1-3%.

**Change:** `models/layers.py` attention: `logits = gain * (Q @ K.T) / sqrt(d_k)`,
`gain` is `nn.Parameter` shape `(n_heads,)`, init 1.0.

**Method:** sweep at 5M.

**Success:** ≥1% val loss reduction.

**Branch:** `experiment/qk-gain`

**Existing work:** see `docs/qk_gain_paper/` (untracked — review before
starting).

### EXP-6: Width/depth sweep at fixed params

**Hypothesis:** current 22 layers × d=512 is off the Chinchilla-style optimum
for 88M params; a wider, shallower variant trains better.

**Change:** none, just config variants.

**Method:** at fixed ~70M params, sweep `{ (n_layers=12, d=640), (n_layers=16, d=576), (n_layers=22, d=512), (n_layers=28, d=448) }`
at 500M tokens.

**Success:** identify the variant within 1% of best, prefer the one with lowest
wall-clock (wider = faster).

**Branch:** `sweep/width-depth-70m`

### EXP-7: FlashAttention-2 integration

**Hypothesis:** drop-in 1.3-1.7× throughput on 3090, identical loss.

**Change:** replace SDPA call with `flash_attn_func` when on CUDA. Keep SDPA
path for MPS/CPU.

**Method:** train identical config with/without FA2 at 5M × 50M tokens.
Compare wall time and final val loss (should match within 0.5%).

**Success:** ≥1.3× throughput, val loss matches.

**Branch:** `experiment/flash-attn-2`

### EXP-8: Data swap — FineWeb-Edu vs cosmopedia

**Hypothesis:** FineWeb-Edu produces lower val loss per token than cosmopedia-v2
for chatbot-track pretraining.

**Change:** new `configs/dataset_config.py` preset `fineweb_edu_10b`.

**Method:** at 25M × 500M tokens, train on cosmopedia vs FineWeb-Edu. Eval val
loss on a *held-out FineWeb-Edu slice* (the target distribution).

**Success:** FineWeb-Edu wins on its own held-out val. (Cross-eval on cosmopedia
val also reported for transparency.)

**Branch:** `experiment/fineweb-edu`

### EXP-9: Parallel residuals (attn ∥ MLP)

**Hypothesis:** GPT-J-style parallel residuals give ~10% throughput, similar
val loss.

**Change:** `models/layers.py`, `TransformerBlock.forward`: change from
`x = x + attn(norm(x)); x = x + mlp(norm(x))` to
`x = x + attn(norm(x)) + mlp(norm(x))` with shared norm.

**Method:** at 25M × 500M tokens, baseline vs parallel.

**Success:** parallel within 1% val loss AND ≥1.05× throughput.

**Branch:** `experiment/parallel-residuals`

## Suggested order

EXP-7 (FlashAttention) and EXP-2 (cosine) first — free wins, low risk, ship to
main quickly. Then EXP-1 (muP) as enabler. Then EXP-3/4/5 sweeps in parallel
branches. EXP-6 (width/depth) once muP makes LR transfer reliable. EXP-8 (data)
can run anytime. EXP-9 (parallel residuals) last — biggest rewrite, smallest
expected win.

## Release flow

1. Cherry-pick wins from experiment branches into `main`.
2. Train at 70M × 1B tokens using current `main`.
3. Eval (ppl), sample, write `releases/v0.X/notes.md`.
4. Tag `v0.X`, push, upload to HuggingFace.
5. X post: "Universe v0.X: val loss A → B (-N%), changes: [list]."

If val loss didn't drop vs the last release, skip the release. Faking is not
allowed.

## Open questions

- muP with weight tying: does the unembedding need any LR adjustment? Verify on
  the muP paper before implementing.
- Sweep CSV format: do we want it readable by `pandas.read_csv` directly, or
  JSONL? Recommend CSV with a header row for now.
- Where to store the small-model sweep checkpoints? Default: gitignored under
  `runs/`. Only release-tagged runs get pushed.
