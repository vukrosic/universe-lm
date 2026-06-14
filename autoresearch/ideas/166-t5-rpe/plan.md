# Plan — 166 T5-Style Bucketed Relative Position Bias (RPE)

## Flag
- `use_t5_rpe: bool` (default `False`) + `t5_rpe_buckets: int = 32` on `LLMConfig`.
  - Field added at `configs/llm_config.py` next to `use_attn_logit_bias`
    (both are score-side additive per-head levers — same family).
- A/B subclass: `Tiny1M3MT5RPEConfig(Tiny1M3MConfig)` (newly added
  in `configs/llm_config.py`, `@dataclass`-decorated with
  `use_t5_rpe: bool = True`, `t5_rpe_buckets: int = 32`). Imported
  directly from `configs.llm_config` by `_arq_166-t5-rpe.py` — the
  dataclass-inheritance pitfall (159/155/161) precludes the bare
  `class C(Tiny1M3MConfig): use_t5_rpe: bool = True` annotation
  pattern.

## Change
- `configs/llm_config.py` — adds `use_t5_rpe: bool = False` and
  `t5_rpe_buckets: int = 32` next to the closed 152/155 knobs,
  with a comment block documenting the bucket function and the
  zero-init identity.
- `configs/llm_config.py` — adds `Tiny1M3MT5RPEConfig(Tiny1M3MConfig)`
  `@dataclass` subclass with `use_t5_rpe: bool = True` and
  `t5_rpe_buckets: int = 32`. This is the class the daemon
  imports as `C` for the build-smoke and run.
- `models/layers.py` — `MultiHeadAttention.__init__` accepts
  `use_t5_rpe: bool = False` and `t5_rpe_buckets: int = 32`
  (declared alongside the closed 152/155 knobs). When on,
  registers `self.rpe_bias = nn.Parameter(zeros(H, B))` and
  builds a precomputed `[max_seq_len, max_seq_len]` int64
  bucket-index matrix as a non-persistent buffer (no RNG, no
  serialization). When off, `self.rpe_bias = None` and
  `self._t5_rpe_bucket_idx = None` (stubs so attribute lookups
  are always safe).
- `models/layers.py` — In `forward()`, after the QK matmul +
  score-side tweaks (FIRE/CoPE/FoX/SSMax/per-head-temp/per-head-
  logit-bias/etc.) and AFTER the causal-mask `masked_fill(-1e9)`,
  adds the bucket-indexed bias to scores BEFORE softmax in BOTH
  branches:
  - **FIRE branch** (line ~2641): `bias = self.rpe_bias[:, bidx]`
    where `bidx = self._t5_rpe_bucket_idx[:T, :T].to(device)`,
    shape `[H, T, T]`. `scores = scores + bias.unsqueeze(0)`.
  - **Manual branch** (line ~2788): identical pattern; placed
    AFTER the masked_fill and BEFORE the Q5 talking-heads mix
    + softmax.
  Init `rpe_bias = 0` ⇒ `scores + 0` is bit-identical to the
  no-RPE baseline at step 0. Bucket function:
  `bucket(|i-j|) = floor(log2(|i-j|+1)).clamp_max(B-1)` — the
  canonical T5 parameterization.
- `models/layers.py` — adds `self.use_t5_rpe` to the manual-
  path trigger list at `models/layers.py:~2680` so SDPA's
  flash/efficient backends don't perturb step-0 numerics (the
  bucket-indexed bias cannot go through the fused kernel).
- `models/layers.py` — `TransformerBlock.__init__` accepts
  `use_t5_rpe: bool = False` + `t5_rpe_buckets: int = 32`
  pass-through kwargs and forwards them to the inner
  `MultiHeadAttention`.
- `models/llm.py` — `MinimalLLM.__init__` reads
  `self.use_t5_rpe = getattr(config, "use_t5_rpe", False)` and
  `self.t5_rpe_buckets = max(1, int(getattr(config,
  "t5_rpe_buckets", 32)))`. Pass-through into BOTH block
  constructor sites (YOCO upper-half at ~line 605 and standard
  TransformerBlock at ~line 825).

Step-0 identity (flag OFF): no `rpe_bias` parameter is
registered, no `_t5_rpe_bucket_idx` buffer is built, no branch
is taken — the existing baseline path is bit-identical.
Verified locally: `MinimalLLM(Tiny1M3MConfig())` ≡
`MinimalLLM(Tiny1M3MConfig())` to **max-abs-diff 0.0** on a
16-token forward at seed 42.

CPU build-smoke (the daemon's `MinimalLLM(C())` check):
- `MinimalLLM(Tiny1M3MConfig())` → 949,056 params ✓
- `MinimalLLM(Tiny1M3MT5RPEConfig())` → 950,592 params ✓
- Delta = +1,536 (one `nn.Parameter(zeros(H=4, B=32))` per
  block × 12 blocks = 1,536; +0.16% — negligible).

Step-0 identity (flag ON): `self.rpe_bias = zeros(H, B)` init ⇒
the added bias is exactly `0` in fp32 ⇒ `scores + 0 = scores` ⇒
the trt model produces bit-identical forward outputs to the
baseline at step 0. Verified: `sum(p.sum().item() for n, p in
trt.named_parameters() if "rpe_bias" in n) == 0.0` and the trt
forward path executes cleanly on a 16-token test input
(`output shape [1, 16, 49152]`, max abs 0.0517).

## Control
- A: `configs.llm_config.Tiny1M3MConfig` (seed 42, flag OFF) —
  bare tier config. The daemon owns this control via
  `autoresearch/bin/baseline.sh`.
- B: `_arq_166-t5-rpe.py` (seed 42, flag ON) — same tier,
  `use_t5_rpe=True`, `t5_rpe_buckets=32`. The `C` class is the
  build-smoke target.
- Tier: `tiny1m3m` (0.94M params, 3M tokens). Seed 42 only
  (one-seed-only rule).

## Cost
- Params: + `H × B = 4 × 32 = 128` per block × 12 blocks =
  +1,536 = +0.16% of 0.94M (verified by counting
  `MinimalLLM(C()).parameters()`).
- FLOPs: O(T²) bucket-index lookup once per forward (a single
  int64 gather) + O(H·T²) bias add to the score tensor. The
  manual-path attention already does O(H·T²) work for the
  matmul + softmax, so this is negligible.
- Memory: + `[max_seq_len, max_seq_len]` int64 buffer per
  block (8 KB at T=2048 — trivial).
- The lever adds NO new module on the model side — `rpe_bias`
  lives on each MHA, and `MinimalLLM` itself is unchanged in
  parameter count.

## Run
- Command (via daemon `bin/queue-daemon.sh`): `python
  _arq_166-t5-rpe.py` — invokes `train_llm.main()` with
  `--config_class __main__.C`, `--seed 42`, `--dataset_path
  processed_data/pretrain_1B`, `--warmup false`.
- Tier: tiny1m3m only (one-seed-only rule).
- Expected wall-clock: ~12 minutes (default `job_timeout: 12m`
  in `run.json`).
- Pass/fail bar (copied from `idea.md` §Design sketch and
  review.md): at tiny1m3m box noise is ≈ ±0.01 val loss; a
  real effect should clear ~−0.02 to be conclusive. PASS ≤
  ctrl − 0.02 (cached baseline mean 6.4346 ± 0.0458 after the
  162 fresh re-cache). NULL band |Δ| < 0.02. DRIFT > +0.02.
  A null closes the additive-logit-bias PE family at our tier
  and confirms the rotational family (RoPE/FIRE) is binding;
  a win would tell us the additive bias family is a usable
  lever at 0.94M orthogonal to the rotational family.
