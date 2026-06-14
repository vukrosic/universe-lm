---
id: 166-t5-rpe
status: running
round: 3
updated: 2026-06-14T10:35:40Z
transfer-risk: med
plain: Replace RoPE with T5-style bucketed relative-position bias on the attention logits — each pair of token positions gets a learnable additive bias from a small set of logarithmic distance buckets, initialized to zero so step-0 is bit-identical to the baseline.
---

# 166 — T5-Style Bucketed Relative Position Bias (RPE)

## Source
- Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, JMLR 2020, arXiv:1910.10683) — introduced the bucketed relative position bias (32 logarithmic distance buckets per direction, learned bias per bucket per head). Validated at T5-11B (T5-XXL).
- RoPE (Su et al. 2021) is the current autoregressive-LM standard; closed in our sweep as 500k-base winner. FIRE (Li et al. 2024, 009 WIN) is the depth-aware successor.
- ALiBi (Press et al. 2022) — linear distance bias with head-specific slopes. *Not* closed in our axis sweeps, but largely redundant with RoPE/FIRE for autoregressive LMs.
- T5-RPE has been re-used in BigBird (Zaheer et al. 2020), REALM (Guu et al. 2020), and LongT5 (Guo et al. 2021) — all validated at 100M+.
- Closed PE axes: NoPE, post-norm tying, RoPE base sweep, FIRE, CoPE.

Distinct from RoPE (rotary Q/K), ALiBi (linear slope), FIRE (depth-aware frequency). T5-RPE is *additive logit bias* on QK^T, bucketed by relative distance.

## Mechanism
Replace the rotary position embedding applied to Q and K with an *additive bias* on the QK^T logits, indexed by relative distance `|i − j|` via a logarithmic bucket function. The bias is a learned per-head per-bucket scalar:
```
buckets(i, j) = floor(log2(|i - j| + 1))   # 0..B-1, B=32
bias_h[buckets] = self.rpe_bias[h, buckets]  # (H, B)
logits = Q @ K^T / sqrt(d_head) + bias_h[buckets][None, None, :, :]
```
At init, `self.rpe_bias = nn.Parameter(torch.zeros(H, B))` ⇒ all biases are 0 ⇒ logits are bit-identical to the un-PEd baseline (QK^T/sqrt(d) with no rotation, no bias). The lever *adds* the RPE to whatever positional scheme is currently active (or replaces it — both are valid; starting as "added on top of un-rotated Q/K" gives the cleanest step-0 identity). ~30 LoC; +`H × B = 4 × 32 = 128` params per block (negligible).

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_t5_rpe: bool = False` and `t5_rpe_buckets: int = 32` kwargs. When on, registers `self.rpe_bias = nn.Parameter(torch.zeros(H, t5_rpe_buckets))`. In `forward`, *after* the QK matmul and *after* the causal mask, add `bias = self.rpe_bias[bucket_index_matrix]` to the logits before softmax. The bucket index matrix is a precomputed `(T, T)` tensor built once per call (cheap, T ≤ 2048). Optionally the lever can be implemented to *replace* the RoPE path (config flag selects one or the other).
- **Config flag**: `use_t5_rpe: bool = False`, `t5_rpe_buckets: int = 32` (default).
- **Step-0 identity**: `self.rpe_bias = nn.Parameter(torch.zeros(H, B))` init ⇒ all biases are 0 ⇒ added logit bias is 0 ⇒ logits are bit-identical to no-RPE baseline. The RoPE path can remain unchanged (RPE is added on top) or replaced via a separate config.
- **Intuition**: T5-RPE is a *logit-bias* positional encoding, structurally different from the *Q/K rotation* family (RoPE, FIRE, ALiBi). The hypothesis: an *additive* logit bias may capture distance-dependent attention priors that a *rotational* transform cannot. Specifically, T5-RPE can express "attend less to distant tokens" as a direct prior without needing a rotation to implicitly encode it. If T5-RPE wins alongside FIRE (009), the positional information is *over-determined* at this tier and the two encodings are partially redundant (info-value: how much extra positional info can be encoded). If T5-RPE wins *instead of* FIRE, the additive bias is the better mechanism at 0.94M.
- **Why now**: the closed PE axes (RoPE / NoPE / FIRE / CoPE) are all *Q/K-rotation* family. The additive-logit-bias family (T5-RPE, ALiBi) is largely unclosed at our tier. T5-RPE is the canonical, well-validated additive variant.

## Scale evidence
T5-XXL (11B parameters) used 32-bucket RPE on attention logits, achieving SOTA on SuperGLUE/GLUE/TriviaQA at release. BigBird, REALM, and LongT5 all re-used T5-RPE at 100M+. Transfer risk is **med**: T5-RPE is encoder-decoder-native and the autoregressive-LM case has less direct validation, but the mechanism is structurally simple and the bucket parameterization is known to work.

## Why it's worth a slot
A win would tell us the *additive logit bias* family of positional encodings works at autoregressive LMs at 0.94M, orthogonal to the *rotational Q/K* family (RoPE/FIRE) that dominates the closed axes. A null would close the additive-bias PE family at our tier and confirm the rotational family is the binding axis for autoregressive LMs. Either result is informative for future PE-level work.

## Plan (round 2 — re-code)

### Why recode
r1 (commit `0199a0e`) ran rc=124 TIMEOUT at step 300/732 (~1750 tok/s vs ctrl ~30000 tok/s). Lever is sound — r1 loss was making real progress (7.02 vs ctrl 6.66) — but the bias-gather `[H, B] → [H, T, T]` per layer per step is the manual-path tax we have to pay for an additive score-side bias. The pre-check on r2 caught a *plumbing* bug: the box had `models/layers.py` wiring but not the `LLMConfig` fields and not the `models/llm.py` pass-through, so the daemon's `MinimalLLM(C())` build-smoke could not even resolve `use_t5_rpe`/`t5_rpe_buckets`. This recode wires the missing pieces and the r1 fix (timeout bump, gather cleanup) is preserved.

### Files changed (this recode)
- `configs/llm_config.py` — adds `use_t5_rpe: bool = False` and `t5_rpe_buckets: int = 32` to `LLMConfig` (next to the other score-side knobs) and adds `@dataclass class Tiny1M3MT5RPEConfig(Tiny1M3MConfig): use_t5_rpe: bool = True, t5_rpe_buckets: int = 32`.
- `models/llm.py` — `MinimalLLM.__init__` reads `self.use_t5_rpe = getattr(config, "use_t5_rpe", False)` / `self.t5_rpe_buckets = max(1, int(getattr(config, "t5_rpe_buckets", 32)))` and pass-throughs both to BOTH `TransformerBlock` constructor sites (YOCO upper-half ~line 648 and standard block ~line 881).
- `models/layers.py` (already wired in r1, unchanged this recode) — `MultiHeadAttention.__init__` accepts the kwargs, registers `self.rpe_bias = nn.Parameter(zeros(H, B))` and the `[max_seq_len, max_seq_len]` int64 bucket-index buffer when on; `forward` adds `rpe_bias[:, bidx]` to scores after the causal mask in BOTH FIRE and manual branches; `self.use_t5_rpe` is in the manual-path trigger list at line ~2892 so SDPA's flash kernel doesn't perturb step-0 numerics; `TransformerBlock.__init__` pass-through forwards both kwargs to the inner MHA.
- `_arq_166-t5-rpe.py` (unchanged) — imports `Tiny1M3MT5RPEConfig` directly from `configs.llm_config` and invokes `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- `autoresearch/ideas/166-t5-rpe/run.json` — `job_timeout` bumped from `12m` to `30m` to cover the manual-path tax at ~1750 tok/s × 732 steps.

### Flag
- `LLMConfig.use_t5_rpe: bool = False` (default off → baseline path bit-identical)
- `LLMConfig.t5_rpe_buckets: int = 32` (T5-XXL's value; only buckets 0..11 are exercised at T ≤ 2048)
- `Tiny1M3MT5RPEConfig` overrides both to `True` / `32` via `@dataclass` re-decoration (the bare `class C(Tiny1M3MConfig): use_t5_rpe: bool = True` pattern is broken — see `_arq_166-t5-rpe.py` header for the dataclass-inheritance pitfall).

### Step-0 identity
- Flag OFF (baseline): no `rpe_bias` parameter is registered, no `_t5_rpe_bucket_idx` buffer is built, no branch is taken → existing baseline forward is bit-identical.
- Flag ON (lever): `self.rpe_bias = nn.Parameter(zeros(H, B))` init → `scores + 0 = scores` → trt forward ≡ baseline to fp32 noise.

Smoke test (local, seed 42, 16-token forward):
- `MinimalLLM(Tiny1M3MConfig())`: 949,056 params ✓
- `MinimalLLM(Tiny1M3MT5RPEConfig())`: 950,592 params ✓ — delta +1,536 = H × B × n_blocks = 4 × 32 × 12
- `baseline ≡ baseline`: max-abs-diff 0.0
- `baseline ≡ t5-rpe`: max-abs-diff 2.40e-08 (fp32 noise from the `scores + 0` add; functionally bit-identical)
- `sum(rpe_bias)` = 0.0

### Run
- Command (daemon `bin/queue-daemon.sh`): `python _arq_166-t5-rpe.py` → `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Tier: `tiny1m3m` (0.94M params, 3M tokens), seed 42 only.
- `job_timeout: 30m` (was 12m; bumped to cover the manual-path tax — ~1750 tok/s × 732 steps ≈ 29m).
- A: `configs.llm_config.Tiny1M3MConfig` (seed 42, flag OFF) — daemon-owned control via `autoresearch/bin/baseline.sh`. Cached mean 6.4346 ± 0.0458 (after 162 fresh re-cache).
- B: `_arq_166-t5-rpe.py` (seed 42, flag ON).
- Pass/fail bar (unchanged from r1 plan): PASS ≤ ctrl − 0.02. NULL band |Δ| < 0.02. DRIFT > +0.02.
- Val loss: read from `train_llm.main()`'s final eval (already used by the runner harness — no extra plumbing needed).
