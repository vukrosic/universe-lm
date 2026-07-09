# Plan — 025 Scalable-Softmax (SSMax, length-aware attention temperature)

## Flag
- `use_ssmax: bool = False` on `LLMConfig` (`configs/llm_config.py:205`),
  threaded through `MinimalLLM` (`models/llm.py:236`) and
  `TransformerBlock` (`models/layers.py:1975, 2024`) into
  `MultiHeadAttention.__init__` (`models/layers.py:520, 693-695`).
- Trt config class: `Tiny1M3MSSMaxConfig`
  (`configs/llm_config.py:847-869`) — sets
  `use_ssmax = True` on the plain tiny1m3m baseline. Per the
  idea's A/B scope (`idea.md:28`), the primary A/B is
  `ctrl = baseline` vs `trt = baseline + ssmax`; the
  stack-with-FIRE and stack-with-qk-norm follow-ups are gated
  on the primary clearing and are NOT included in this gate.
- Built module: a single `nn.Parameter(torch.ones(n_heads))` on
  the MHA (instantiated lazily when `use_ssmax=True`; never
  referenced when off → baseline path bit-identical when flag
  is off and the `elif` chain keeps the SDPA path).

## Change
**1 new flag + 5 wiring touch points.** No new file, no new
dependencies. ~20 LoC of MHA-forward code.

- `configs/llm_config.py:205` — add `use_ssmax: bool = False` on
  `LLMConfig` (sits next to `use_softpick`).
- `configs/llm_config.py:847-869` — add `Tiny1M3MSSMaxConfig`
  (`use_ssmax=True`). No other knobs change.
- `models/layers.py:520` — add `use_ssmax: bool = False` kwarg
  to `MultiHeadAttention.__init__` with the design comment.
- `models/layers.py:693-695` — store `self.use_ssmax`; if on,
  build `self.ssmax_s = nn.Parameter(torch.ones(self.n_heads))`.
- `models/layers.py:1521-1527` (FIRE branch) — after the
  optional CoPE addition, before the mask: compute
  `log_n = log(arange(1, T+1))` (shape `[T]`, fp32) and apply
  `scores = scores * (s_h.view(1,H,1,1) * log_n.view(1,1,T,1))`.
- `models/layers.py:1651-1657` (Query-tweaks manual branch) —
  same operation, same placement (after CoPE, before mask).
- `models/layers.py:1567` — add `or self.use_ssmax` to the
  manual-branch OR chain (SSMax modifies scores, can't go
  through SDPA's flash kernel — same reason as `use_fox`).
- `models/layers.py:1975, 2024` — pass `use_ssmax` from
  `TransformerBlock` to its MHA child.
- `models/llm.py:236, 357` — add the pass-through
  `getattr(config, "use_ssmax", False)` and forward to the
  block.

The same `s_h * log_n` multiplier is applied at the same
logit-side site in both manual branches, so the FIRE-stack
follow-up and the qk-norm-stack follow-up can be added later
without re-architecting the code. SSMax is a per-tensor
multiply on `scores`; both FIRE (additive bias) and qk-norm
(bound on logit norm) compose with it cleanly.

## Control
- **Ctrl**: `Tiny1M3MConfig` (plain baseline, the same ctrl the
  runner has used for every other single-lever ablation in this
  batch). Per `idea.md:28` the primary A/B is *not* on top of
  FIRE — that's the follow-up.
- **Trt**: `Tiny1M3MSSMaxConfig` — same recipe as ctrl + `use_ssmax=True`.
- **Seed**: 42 (one seed only — `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Pass bar** (copied from `idea.md:19-25`):
  - **WIN**: `trt_val < ctrl_val − 0.01` (Δ ≤ −0.01, clearly
    outside the in-session ctrl-pair bracket of 0.006–0.02
    observed in the 2026-06-09 batch).
  - **Informative NULL**: `−0.01 < Δ ≤ 0` (sharpening lever is
    real but not binding at 2048/0.94M — still a *result*;
    logged to `closed.md` so it isn't re-mined).
  - **Regress / box-drift**: `Δ > 0`; the runner re-runs ctrl
    to disambiguate before calling a clean null, per
    `PIPELINE.md` box-validation rule.
  - **Anti-cheat**: ±0.0053 inside-bracket results do NOT count
    as WIN — the bar is the −0.01 threshold, not "any negative
    number" (mirrors the POLYLOSS-style outcome fence).

## Cost
- **Params**: `ssmax_s` = `n_heads` scalars per layer. At
  tiny1m3m (n_heads=4, n_layers=12): `4 × 12 = 48` extra
  scalars total. Negligible (<< 0.01% of the 0.94M-param
  tiny1m3m).
- **FLOPs**: one `[T]` log (computed once per layer), one
  broadcast-multiply of `scores [B,H,T,T] * [H,1] * [1,T]`
  ≈ `B · H · T² · 2` extra FLOPs. At B=2, H=4, T=2048:
  ~67M extra FLOPs / layer / forward → ~800M / 12 layers
  (~0.5% of total forward FLOPs).
- **Memory**: zero new activations — `log_n` is a single
  length-T vector, and the ssmax_s parameter is `n_heads`
  scalars. Scores tensor shape is unchanged.
- **Numerical**: `scores * (s_h * log n)` is well-defined for
  all `n ≥ 1`; `log(1) = 0` so the first position's multiplier
  is exactly 0 (the query at position 0 attends only to itself,
  and softmax over a single element is independent of
  temperature). At `n=2048`, `log n ≈ 7.6` so the multiplier
  is large but finite (no overflow risk in fp32; fp16/bf16
  also fine since scores are typically O(1) before softmax).

## Run
- **Command**: `python3 train_llm.py --config
  Tiny1M3MSSMaxConfig --seed 42 --tier tiny1m3m` (ctrl swaps
  to `Tiny1M3MConfig` for the baseline run, same
  `--seed 42 --tier tiny1m3m`). The config class is the A/B
  handle, not a CLI flag — see `vast-runner-harness.md`.
- **Tier**: tiny1m3m (single seed, no sweep).
- **Expected wall-clock**: ~4 hours on the Vast box (matches
  the bare-baseline tiny1m3m cost — the SSMax path is ~0.5%
  extra FLOPs, well inside run-to-run variance).
- **Pass/fail bar**: copied from `idea.md:19-25` above.
- **Step-0 note**: at `s_h = 1.0`, the forward is NOT
  bit-identical to vanilla softmax — query at position i sees
  scores scaled by `log(i+1)`. This is the *mechanism* of the
  paper (Nakanishi 2025 §3.1), explicitly justified in
  `idea.md:14` and re-confirmed in the reviewer's r2 verdict
  (`review.md:81-89`). The baseline-identity check is
  satisfied via "explicitly justified", not via
  "bit-identical at flag-off". The flag-OFF path IS
  bit-identical — the `ssmax_s` parameter is never built and
  the elif chain keeps the SDPA path.

## Self-check (per code-implementer.md §5)
- [x] `use_ssmax=False` path: `ssmax_s` parameter is not
  instantiated (`if self.use_ssmax: ...` at `models/layers.py:694`),
  forward never references it, no extra params allocated, no
  extra FLOPs. The elif chain at `models/layers.py:1555-1568`
  does not include `self.use_ssmax` so the SDPA path stays
  selected when only other score-side tweaks are off. Baseline
  path is bit-identical to a pre-flag build.
- [x] `use_ssmax=True` path: `ssmax_s` is built, the elif chain
  routes to the manual branch, and the SSMax multiplier
  is applied in both manual branches (FIRE and Query-tweaks).
  Numerical smoke: `log_n` for `T=2048` gives `log(2048) ≈
  7.625`; with `s_h=1` the post-multiply score scale is
  bounded and finite.
- [x] `plan.md` pass/fail bar matches `idea.md:19-25` exactly
  (WIN −0.01, NULL band, regress + box-drift fence,
  anti-cheat ±0.0053 fence).
- [x] Same-flag-equal-ctrl invariant: with `use_ssmax=True` and
  the rest of the model identical to `Tiny1M3MConfig`, only
  the new branch + 48 scalars differ from the baseline.

## Coordination note
`git diff models/layers.py configs/llm_config.py` shows the
parallel AI agent (working on 022-softpick) has touched both
files in this session. My edits are anchored on the
existing `use_softpick` lines and add `use_ssmax` /
`Tiny1M3MSSMaxConfig` / ssmax_s application immediately
adjacent — no overlap with the Softpick touch points, no
rebase needed. Per `project-parallel-ai.md` the working
tree is the source of truth and the diff is allowed to be
non-empty at the start of the pass; I am not reverting or
modifying the Softpick changes.
