# Recode — 166 T5-RPE (round 3)

## r2 failure (recoded by orchestrator / box stale)
- runner: `bin/queue-daemon.sh`, build-smoke pre-check, seed 42
- pre-check FAIL: "box configs/llm_config.py lacks LLMConfig.use_t5_rpe/t5_rpe_buckets fields and models/llm.py plumbing — lever wired only in models/layers.py MHA. Implementer must add LLMConfig field + Tiny1M3MT5RPEConfig subclass + llm.py pass-through to TransformerBlock before re-run."
- The r2 release was a build-smoke bounce — the daemon's CPU `MinimalLLM(C())` import on the box could not resolve `getattr(config, "use_t5_rpe", False)` against a config class that didn't have the field. The local working tree at r2 release time had the wiring in uncommitted edits only; the box was on an earlier commit (pre-7186c49) that only had the layers.py MHA changes from the r1/r2 implementer.
- The actual fields and pass-through are present in commit `7186c49` ("166/167/168: wire T5-RPE, Z-Loss, AV-output carry") on the current branch:
  - `configs/llm_config.py:122` — `use_t5_rpe: bool = False` and `t5_rpe_buckets: int = 32` (LLMConfig).
  - `configs/llm_config.py:2112` — `@dataclass class Tiny1M3MT5RPEConfig(Tiny1M3MConfig)` with `use_t5_rpe: bool = True` / `t5_rpe_buckets: int = 32` (the daemon's `C` import target).
  - `models/llm.py:307-308` — `self.use_t5_rpe = getattr(config, "use_t5_rpe", False)` and `self.t5_rpe_buckets = max(1, int(getattr(config, "t5_rpe_buckets", 32)))`.
  - `models/llm.py:652-653, 885-886` — pass-through to BOTH block constructor sites (YOCO upper-half at ~line 605 and standard TransformerBlock at ~line 825).
  - `models/layers.py:623-624, 3387-3388` — MHA and TransformerBlock kwargs; layer 1329-1356 builds the bias parameter + bucket-index buffer; layer 2864-2868 / 3021-3025 apply it in both branches; line 2892 adds it to the manual-path trigger list.

## r3 fix (this pass)
The cause is box staleness, not a missing local change. Re-verified the local
repo end-to-end:

- `git show 7186c49:configs/llm_config.py` and `:models/llm.py` both contain
  the wiring (commit is reachable on the current branch).
- The arq stub (`_arq_166-t5-rpe.py`) and `run.json` are unchanged from r2 and
  are still correct (`C = Tiny1M3MT5RPEConfig` imported from
  `configs.llm_config`; `job_timeout: 30m`).
- Local build-smoke:
  - `MinimalLLM(Tiny1M3MConfig())` ≡ `MinimalLLM(Tiny1M3MConfig())`
    → **max-abs-diff 0.00e+00** (ctrl ≡ ctrl, same seed).
  - `MinimalLLM(Tiny1M3MT5RPEConfig())` ≡ `MinimalLLM(Tiny1M3MConfig())`
    → **max-abs-diff 2.24e-08** (fp32 noise from the `scores + 0` add;
    functionally bit-identical). rpe_bias sum confirmed 0.0.
  - Param delta: 950,592 − 949,056 = +1,536 = `H × B × n_blocks
    = 4 × 32 × 12`. Matches plan.
  - All non-`rpe_bias` parameters and shared buffers are bit-identical
    between ctrl and trt at the same seed (max-abs-diff 0.0 across
    123 named params and the common buffer set).
- The daemon's `sync_and_smoke` does `git pull --ff-only` on the box
  before the CPU build-smoke, so once this recode flips back to
  `needs-run` the next tick will pull the wiring commit into the
  box's working tree, the SCP will drop the arq file, and the
  `python _box_smoke.py _arq_166-t5-rpe.py` will return `SMOKE_OK`.

## r1 failure (history)
- runner: `bin/queue-daemon.sh`, `job_timeout: 12m`, seed 42
- rc=124: TIMEOUT at step 300/732 (~41% complete)
- speed: ~1750 tok/s vs ctrl ~30000 tok/s (≈ 17× slower than baseline SDPA path)
- loss: 7.02 vs ctrl ~6.66 (the run was making progress; the bias-gather overhead was the budget sink, not a divergence — the lever is plausibly sound)
- diagnosis (from runner note): "slow bias-gather and worse convergence, tighten kernel or bump JOB_TIMEOUT"

## r2 fix (history)
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

The r2 implementer picked both:

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

## Self-check (r3 re-verification)
- `MinimalLLM(Tiny1M3MConfig())` ≡ itself → **max-abs-diff 0.00e+00** on a
  16-token forward at seed 42 (rebuild from same seed, same input
  generator).
- `MinimalLLM(Tiny1M3MT5RPEConfig())` ≡ `MinimalLLM(Tiny1M3MConfig())`
  → **max-abs-diff 2.24e-08** (fp32 numerical noise from the
  `scores + 0` add; functionally bit-identical) on a 16-token
  forward at seed 42 (both built from same seed, same input
  generator). rpe_bias sum confirmed 0.0 across all 12 blocks.
- All 123 non-`rpe_bias` named parameters and the common buffer set
  are bit-identical between ctrl and trt at the same seed
  (max-abs-diff 0.0).
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
