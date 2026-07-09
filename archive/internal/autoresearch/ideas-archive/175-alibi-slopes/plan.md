# Plan — 175 alibi-slopes

## Flag
- `use_alibi_bias: bool = False` (default OFF) on
  `LLMConfig` — already declared at `configs/llm_config.py:1055`.
- `Tiny1M3MAlibiConfig(Tiny1M3MConfig)` with
  `use_alibi_bias: bool = True` — newly added at
  `configs/llm_config.py:2243-2301` (Block-1 cluster, after
  `Tiny1M3MT5RPEConfig` and before `Tiny1M3MDropConnectWOConfig`,
  per reviewer F3).

## Change
Single file edit; mechanism itself is already in `models/layers.py`.

- `configs/llm_config.py:2243` — new `@dataclass
  Tiny1M3MAlibiConfig(Tiny1M3MConfig)` with
  `use_alibi_bias: bool = True` and a docstring noting the
  sign-convention flip vs the ALiBi paper and the byte-identical
  step-0 guarantee.
- No edits to `models/layers.py`. The existing wiring at
  `models/layers.py:1068` (constructor arg),
  `:1972-1974` (init `nn.Parameter(torch.zeros(self.n_heads))` when
  flag is on), and `:3172-3176` (apply `scores -= m * (i − j)`
  pre-softmax in the manual attention path) is exactly what the
  experiment needs. The flag is also already in
  `LLMBackbone.__init__` and threaded through both
  `TransformerBlock(...)` sites at `models/llm.py:525, 781, 1069`.
  The `Screen10M20MAlibiBiasConfig` at
  `configs/llm_config.py:5068` proves the wiring works end-to-end.

Step-0 byte-identical: with `m_h = 0` for all heads (the default
init at `models/layers.py:1974`),
`scores -= 0 · (i − j) = scores` ⇒ softmax unchanged ⇒ AV
unchanged ⇒ residual+norm unchanged ⇒ block output unchanged ⇒
logits unchanged ⇒ loss unchanged ⇒ **max-abs-diff vs baseline
forward = 0.0** (verified via build-smoke §5).

## Control
- Control: `Tiny1M3MConfig` (baseline), val mean **6.4447 ± 0.0244**
  (14 cached measurements on box `5b8a7fea8963` per
  `autoresearch/baseline-cache.json`).
- Treatment: `Tiny1M3MAlibiConfig` (`use_alibi_bias=True`), seed 42.
- Tier: `tiny1m3m` (0.94M params / 12L / 4H / 3M tok).
- Box: 1 seed only — seed 42. Variance bracket = the two-ctrl
  protocol; we don't multi-seed.

## Cost
- Params: +4 scalars/block × 12 blocks = **+48 params** (+0.005% of
  ~0.94M total). Negligible.
- FLOPs: +1 add + 1 multiply per (b, h, t, s) in the manual
  attention path ⇒ **+1 O(B·H·T²·d_k) = +1 O(seq_len² · d_model)**
  per block. With seq_len ≤ 512 and d_model=64, that's < 0.01% of
  the attention FLOPs and ≪ 0.001% of total forward FLOPs.
- Memory: +1 `nn.Parameter` of `[H]` float32 per block ⇒ 16 bytes
  per block × 12 = 192 bytes total. Zero runtime activation
  overhead — the bias is fused into the existing score matmul
  (`scores -= m * diff.view(1, 1, T, T)` is a single broadcast-multiply).
- Path: forces the manual attention branch at
  `models/layers.py:3096-3104` (the elif), so SDPA flash kernels
  are not used. Manual path is ~5-10% slower per step at this
  size — the established cost for any score-side lever at tiny1m3m
  (same as 152, 155, 160, 166).

## Run
- Command (daemon's GPU last-mile):
  `/venv/main/bin/python _arq_175-alibi-slopes.py`
  (i.e. the standard `python <arq_file>` shape; `--config_class
  __main__.C` is wired inside the stub).
- Tier: **tiny1m3m** (default for `Tiny1M3MConfig`).
- Seed: **42** (one seed only, per pipeline policy).
- Dataset: `processed_data/pretrain_1B`.
- `--warmup false`.
- Expected wall-clock: ~3 min on RTX 3060 (manual-attention
  penalty, same as 152/166).
- Pass/fail bar (from `idea.md`):
  - Expected Δval ∈ [−0.025, −0.005].
  - **PASS** ≤ ctrl − 0.02 ⇒ trt_val ≤ **6.4247**.
  - **NULL** band: |Δ| < 0.02 (sub-noise inconclusive).
  - **DRIFT** (regression): trt_val > ctrl + 0.02 ⇒ close axis.
- Self-check: build-smoke (CPU `MinimalLLM(C())` constructs), then
  step-0 forward max-abs-diff vs `Tiny1M3MConfig` baseline must be
  0.0 within 1 ULP (it is — `m_h=0` ⇒ bias=0 ⇒ scores+0 ⇒
  softmax+0 ⇒ AV+0 ⇒ residual+0 ⇒ output+0).
