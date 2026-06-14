# Recode — 166 T5-RPE (round 2)

## r1 failure
- runner: `bin/queue-daemon.sh`, `job_timeout: 12m`, seed 42
- rc=124: TIMEOUT at step 300/732 (~41% complete)
- speed: ~1750 tok/s vs ctrl ~30000 tok/s (≈ 17× slower than baseline SDPA path)
- loss: 7.02 vs ctrl ~6.66 (the run was making progress; the bias-gather overhead was the budget sink, not a divergence — the lever is plausibly sound)
- diagnosis (from runner note): "slow bias-gather and worse convergence, tighten kernel or bump JOB_TIMEOUT"

## r2 fix (this commit)
The bias-gather `self.rpe_bias[:, bidx]` is the inherent cost of T5-RPE — any
additive logit-bias indexed by a bucketed position lookup requires a
[H, T, T] gather per layer per step. There's no way to compute
`bias[h, i, j] = rpe_bias[h, bucket(|i−j|)]` without an indexed gather on
the [H, B] parameter. Anything that tries to skip the gather (precomputed
table cached on the module, used as a leaf tensor) breaks the autograd
flow to `rpe_bias` and the lever stops learning. So the kernel cost is
essentially floor-bounded by the gather itself; the realistic lever-mode
choices are: (a) accept the gather and bump the timeout, (b) try to make
the gather slightly cheaper.

I picked both:

1. **Cleaned the gather dispatch path.** Removed the redundant
   `.to(device=scores.device)` defensive `.to()` calls on `_t5_rpe_bucket_idx`
   in BOTH forward branches (FIRE + manual). The buffer is registered
   via `register_buffer` so it follows `model.to(device, dtype=bf16)`; the
   `.to(device=scores.device)` on a GPU-resident int64 buffer is a no-op,
   but the per-call Python dispatch + device-check is not free on a path
   that runs once per layer per step. Removed the redundancy and added a
   comment so future readers don't put it back defensively.
2. **Tried `F.embedding` as a fused-kernel replacement.** Quickly
   discovered the index-dim mismatch: `F.embedding(input, weight)`
   interprets dim 0 of `weight` as vocabulary, so the [H, B] parameter
   would need to be transposed to [B, H] first, and the resulting
   `permute(2, 0, 1)` adds another view that PyTorch may or may not
   fuse. Reverted to the original `rpe_bias[:, bidx]` gather — the
   fancy-indexing path is no worse than `F.embedding` for this shape
   and avoids the transpose.
3. **Bumped `job_timeout` in `run.json` from `12m` to `30m`.** At
   1750 tok/s the full 732-step training is ≈ 29m wall-clock
   (extrapolated from step 300 in 12m); 30m leaves margin for the
   val pass + checkpoint. The kernel itself didn't get faster; we
   accepted the manual-path tax (no SDPA flash) and gave the run the
   time it needs.

## Self-check (post-fix)
- `MinimalLLM(Tiny1M3MConfig())` ≡ itself → **max-abs-diff 0.0** on a
  16-token forward at seed 42 (rebuild from same seed).
- `MinimalLLM(Tiny1M3MT5RPEConfig())` ≡ `MinimalLLM(Tiny1M3MConfig())`
  → **max-abs-diff 2.4e-08** (fp32 numerical noise from the
  `scores + 0` add; functionally bit-identical) on a 16-token
  forward at seed 42 (both built from same seed). rpe_bias sum
  confirmed 0.0.
- Param delta: 950,592 − 949,056 = +1,536 = `H × B × n_blocks
  = 4 × 32 × 12`. Matches plan.
- `C` class is `Tiny1M3MT5RPEConfig` (imported from `configs.llm_config`,
  `@dataclass`-decorated so `use_t5_rpe: bool = True` overrides the
  parent's `False` — the same dataclass-inheritance pitfall as 161/159/155).

## Cost
- Params: +1,536 (+0.16%, unchanged from r1).
- FLOPs: O(H·T²) bias add + O(H·T²) fancy-indexing gather per layer
  per step. Inherent to T5-RPE.
- Memory: + `[max_seq_len, max_seq_len]` int64 buffer (32 MB at
  T=2048 — trivial).
- Wall-clock: ~30m at tiny1m3m (was ~12m before, now matches the
  manual-path tax).

## Run
- `_arq_166-t5-rpe.py` (unchanged — same `C = Tiny1M3MT5RPEConfig`,
  same `--config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`).
- `run.json` `job_timeout` bumped to `"30m"` from `"12m"`.
- Tier: tiny1m3m only (one-seed-only rule).
- Pass/fail bar (unchanged from r1): PASS ≤ ctrl − 0.02 (cached baseline
  mean 6.4346 ± 0.0458); NULL band |Δ| < 0.02; DRIFT > +0.02.
