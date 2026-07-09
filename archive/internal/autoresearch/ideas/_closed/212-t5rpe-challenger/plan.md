# Plan — 212 t5rpe-challenger

## Flag
`use_t5_rpe: bool = True`, `t5_rpe_buckets: int = 32` (on-box `Tiny1M3MT5RPEConfig`,
configs/llm_config.py:3139). Per-block `rpe_bias ∈ R^{H×B}` (zero-init) added to
attention scores, indexed by `bucket(|i-j|) = floor(log2(|i-j|+1)).clamp_max(B-1)`.
Default OFF on the parent `Tiny1M3MConfig` (configs/llm_config.py:146) — flag on
forces the **manual attention path** (no SDPA flash) so the bucket-indexed bias
is exact. See models/layers.py:2403 (`self.use_t5_rpe = use_t5_rpe`,
`self.rpe_bias = nn.Parameter(zeros(H, B))`) and models/layers.py:4708 / 4994
(where the bias is added to scores after qk norm).

## Change
- No code change. The on-box `Tiny1M3MT5RPEConfig` already exists; the
  `_arq_212-t5rpe-challenger.py` stub wraps it in a top-level `@dataclass C`
  (build-smoke target) and drives `train_llm.main()` with seed 42, dataset
  `processed_data/pretrain_1B`, `--warmup false`.
- Recode change: **`run.json` `job_timeout` bumped 12m → 35m** (the
  manual-attention path with bucket-indexed bias is significantly slower than
  SDPA — observed ETA at step 300/732 was 28m, so 12m was insufficient).

## Control
- **Bar to beat**: ALiBi champion val 6.2403, noise band 0.04 (cache-authoritative).
  PASS / WIN iff treatment val < 6.2003. NULL iff |Δ| < 0.04.
- Single seed (42). Sub-noise is INCONCLUSIVE per the one-seed-only rule.
- Daemon owns the ctrl (`Tiny1M3MConfig` baseline). No ctrl shipped in this idea.

## Cost
- params Δ: +1,488 (+0.16%) vs ALiBi (4×32 rpe_bias − 48 alibi slopes).
- FLOPs Δ: small (one extra `[H, T, T]` add per block per step). The
  *wall-clock* Δ is large because the bucket bias cannot use SDPA's flash
  kernel — observed step time ≈ 2.3 s / 100 steps vs ≈ 1.0 s / 100 steps on
  ALiBi (the latter rides SDPA).
- memory Δ: negligible (~1.5 KB total).

## Run
- Command (via daemon): `python _arq_212-t5rpe-challenger.py`
- Tier: tiny1m3m (0.94M, 12L, 4H, d_model=64), 92 update steps × 8 acc → 732
  optim steps total, seed 42, no warmup.
- Expected wall-clock: ~28 min (within 35m timeout).
- Pass/fail bar: copied from idea.md — PASS iff val < 6.2403 − 0.04 = **6.2003**.

## Self-check (run)
- `run.json` valid: `name`, `arq_file` (path relative to repo root), `job_timeout`.
- `_arq_212-t5rpe-challenger.py` defines top-level `C` (build-smoke target).
- `MinimalLLM(C())` constructs on CPU (the daemon's smoke).
- Flag OFF path (parent `Tiny1M3MConfig`) remains bit-identical to the no-RPE
  baseline — `use_t5_rpe=False` ⇒ `self.rpe_bias = None`, manual path is gated
  by `if self.use_t5_rpe:` and never taken (models/layers.py:4729).
- `__main__` block drives `train_llm.main()` with `--config_class __main__.C`,
  seed 42, dataset `processed_data/pretrain_1B`, `--warmup false`.