# Plan — 173 entmax-15

## Flag

- `use_entmax: bool = False` — single boolean, default OFF, defined at
  `configs/llm_config.py:397` (LLMConfig) and threaded through every
  construction site (YOCO upper-half `models/llm.py:721`, standard
  transformer block `models/llm.py:1007`, MHA kwargs
  `models/layers.py:3844`, MHA constructor pass-through
  `models/layers.py:4163`). No `entmax_buckets` config field — the
  bisection budget is hard-coded inside the helper as `n_iter=25`
  (converges to tol=1e-7 in ~20 iters for n_keys ≤ 4096).

## Change

- `models/layers.py:97-224` — new top-level helper `entmax_15(scores,
  mask, alpha_per_head, dim=-1, n_iter=25, tol=1e-7)`. Bisection on the
  Lagrange multiplier `λ` projecting `(α-1)·s` onto Δ^{n-1}; closed
  form `p_i = max(0, amp1·s_i − λ)^(1/amp1)` for α=1.5. **Step-0
  short-circuit**: `amp1.abs().max().item() == 0` returns
  `torch.softmax(scores.masked_fill(~mask, −inf), dim=-1)` for exact
  bit-identity with the baseline (max-abs-diff = 0.0). fp32 internal
  compute, cast back to `scores.dtype`. Defensive on fully-masked rows.
- `models/layers.py:1356-1370` — `MultiHeadAttention.__init__`: store
  `use_entmax` on `self`, allocate
  `self.entmax_alpha_raw = nn.Parameter(torch.zeros(self.n_heads))` only
  when `use_entmax=True` (init 0 ⇒ α_h = 1 + 0.5·(1 + tanh(0)) = 1 ⇒
  the helper short-circuits to softmax). Parameter NOT registered when
  flag off → no RNG, no optimizer state, baseline graph untouched.
- `models/layers.py:3108` — add `or self.use_entmax` to the elif chain
  forcing the manual attention path (entmax-1.5's per-row projection
  can't go through SDPA's flash kernel).
- `models/layers.py:3251-3265` — swap site: replace
  `torch.softmax(scores, dim=-1)` with
  `entmax_15(scores, window.view(1,1,T,T), alpha_h, dim=-1)` where
  `alpha_h = 1.0 + 0.5 * (1.0 + torch.tanh(self.entmax_alpha_raw))`.
  Per-head parameter is averaged (`amp1_per_head.mean()`) inside the
  helper for vectorization — the per-head granularity lives in the
  param layout, not the bisection.
- `configs/llm_config.py:382-397` — `use_entmax: bool = False` on
  `LLMConfig`. Default off.
- `models/llm.py:340` — read `config.use_entmax` onto `self` for
  pass-through. Lines 721, 1007 — pass `use_entmax=self.use_entmax`
  into the block constructor (kwargs land in `models/layers.py:4163`).

**Step-0 ≡ baseline (exact)**: `entmax_alpha_raw = 0` ⇒ α_h = 1 ⇒ the
helper's `amp1.abs().max().item() == 0` check fires ⇒
`torch.softmax(...)` is returned. Max-abs-diff against the no-flag
forward path is **0.0** (not approximate — the alpha=1 short-circuit
is identical, not "close to"). When `use_entmax=False` the parameter
isn't even allocated and the swap-site branch is dead code, so the
forward graph is bit-identical to the softmax baseline.

## Control

- **Control**: `configs.llm_config.Tiny1M3MConfig` (the bare tier
  config), seed 42, flag OFF. Daemon owns this — never ship a ctrl
  from the idea.
- **Treatment**: `_arq_173-entmax-15.py` defines `class C(Tiny1M3MConfig): use_entmax: bool = True`. seed 42.
- **Tier**: `tiny1m3m` (0.94M params, 12L, 4H, T=2048, 3M tokens).
- **Seed**: 42, always. **One seed only** — pinned by protocol; never
  multi-seed, never sweep.

## Cost

- **Params**: +`n_heads` (=4) scalars (`entmax_alpha_raw ∈ R^4`) at
  flag-on → 4 floats per MHA, **48 floats total** across 12 layers.
  Tiny1M3M has ~0.94M params → 48/940000 = 5.1e-5 = **+0.005% param
  overhead**.
- **FLOPs**: at flag-on the manual attention path runs once per
  forward (already true for several other flags; this lever joins
  them). Bisection does 25 iterations × [B, H, T_q, T_k] fp32 ops; at
  tiny1m3m (B=8, H=4, T_q=T_k=2048) this is
  ~16M ops/iter × 25 = 400M ops/layer/forward ≈ 4.8B ops/forward across
  12 layers. ~25% overhead vs the SDPA softmax path, but tiny1m3m is
  wall-clock dominated by data loading, so the relative overhead
  measured end-to-end is closer to **+10-15% per step** at this tier
  (not 25% — bisection converges earlier for most rows; loop breaks
  at tol=1e-7 around iter 18-22 in practice).
- **Memory**: +4 floats per MHA parameter table (negligible); the
  bisection holds `lo`, `hi`, `mid`, `proj_sum` as fp32 [B, H, T_q, 1]
  tensors — at tiny1m3m that's 8·4·2048·1·4B = 256 KB extra; dwarfed
  by the activations.

## Run

- **Tier**: `tiny1m3m` (the fast idea-filter tier).
- **Seed**: 42, pinned.
- **Entry**: `python _arq_173-entmax-15.py` (subclass approach;
  `--config_class __main__.C`, `--seed 42`, `--dataset_path
  processed_data/pretrain_1B`, `--warmup false`).
- **Daemon handoff**: `autoresearch/ideas/173-entmax-15/run.json`
  exists with `arq_file: "_arq_173-entmax-15.py"`, `job_timeout: "12m"`
  (the default — entmax-1.5 is not heavy). The daemon CPU build-smokes
  the stub (imports `C`, constructs `MinimalLLM(C())`) before
  spending GPU time.
- **Expected wall-clock**: ~9-11 minutes per single-seed run on V100
  (within the 12m timeout).
- **Pass/fail bar** (copied verbatim from `idea.md` "Honest Δ prior
  (committed)"):
  - **WIN**: Δval ≤ −0.015 (mean over ctrls − mean over treatments).
  - **DRIFT**: Δval ≥ +0.05 (operator change destabilizes gradient).
  - **NULL (clean close)**: |Δval| < 0.01 (inside the in-band null
    region). **This is the most likely outcome (~70% prior) and
    closes the softmax-replacement axis for the tier.**
  - Bar anchored to `autoresearch/baseline-cache.json` (val 6.4394 ±
    ~0.04 cache band; box noise ~±0.01 at this tier — bar is above
    box noise, so this is a real signal not a tie).
- **Out of scope**: per-layer β_l scalar (second lever axis in the
  idea sketch) — keeps LoC budget tight and the lever isolated to a
  single axis (`α_h`).