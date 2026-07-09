# Review log for 175-alibi-slopes

## r1 — 2026-06-15 — verdict: approve

Source/machinery checks (all pass):
- **Source real and current**: ALiBi = Press, Smith, Lewis, ICLR 2022
  (arXiv:2108.12409); BLOOM-176B BigScience 2022 validation is the highest
  scale of any idea in the filed queue. "Learnable per-head slopes" is
  a smaller, real bet — not a re-pitch of the fixed geometric slopes.
- **Mechanism, not HP**: structural additive bias `scores -= m_h·(i-j)`
  on attention logits, per-head scalar per block (48 scalars total at
  0.94M/12L/4H). Init slope=0 ⇒ step-0 = baseline. Identity-at-step-0
  holds with the *direct linear* parameterization
  `nn.Parameter(torch.zeros(n_heads))` that the repo already uses at
  `models/layers.py:1901`. (The idea correctly flags that
  `softplus(s_raw)` would NOT be bit-identical; we use the direct
  linear init, so no concern.)
- **tiny1m3m only**: idea, plan, and pass/fail band all pinned to
  0.94M/3M tok. No screen/full-ladder references.
- **Not already closed**: closed.md entries for 152/155/160/166 are
  all *content-free* per-head scalars (free additive bias, scalar
  temperature, RMS gain, bucketed free bias). 175 is the
  *position-distance-structured* member of the family — the lever
  has a non-trivial axis (decay-vs-amplify along distance) that
  152/155/160/166 do not have. This is the structurally-distinct
  member, not a 5th derivative variant.
- **< 200 LoC**: implementation is even smaller than the idea claims.
  Mechanism is already wired in:
  - `models/layers.py:1899-1901` stores flag + creates
    `nn.Parameter(torch.zeros(n_heads))` slope (init 0, bit-identical).
  - `models/layers.py:3080-3082` applies the bias on scores pre-mask.
  - `models/layers.py:3005` (elif branch) forces the manual attention
    path when `use_alibi_bias` is on, so SDPA flash backends cannot
    perturb step-0 numerics (matches the 152/166 bit-identical
    guarantee).
  - `configs/llm_config.py:1039` declares `use_alibi_bias: bool = False`
    on `LLMConfig`.
  - `configs/llm_config.py:5052` already has
    `Screen10M20MAlibiBiasConfig(Screen10M20MConfig)` with
    `use_alibi_bias: bool = True` — proof the wiring works end-to-end.
  - `models/llm.py:517, 768, 1051` thread the flag through
    `LLMBackbone.__init__` and both `TransformerBlock(...)` call
    sites.
  - **Only missing piece**: a `Tiny1M3MAlibiConfig(Tiny1M3MConfig)`
    subclass with `use_alibi_bias: bool = True` (~3-5 LoC, mirroring
    `Tiny1M3MAttnLogitBiasConfig` / `Tiny1M3MT5RPEConfig`).
- **Falsifiable bar**: Δval ∈ [-0.005, -0.025] expected, NULL band
  |Δ| < 0.02. Box noise ~±0.01 val loss at this tier. Bar is tight
  but in line with the most recent precedents (166 used |Δ| < 0.02;
  155 used |Δ| < 0.01 against a 0.04 band). PASS ≤ ctrl − 0.02.
- **Transfer-risk justified**: `transfer-risk: low`. Scale-evidence
  section cites BLOOM-176B (highest scale of any filed idea). The
  lever is scale-free (48 scalars per block absorb at any tier).
  Tag matches the evidence — a "low" tag is appropriate for a
  mechanism validated at 176B that has no scale-sensitive axis.

Findings (actionable for reviser before code-gate hands off):
- **F1 (doc rotation)**: idea cites stale line numbers — `models/layers.py:1776-1777` (init), `:2957-2960` (apply), `:3973` (threading), and `configs/llm_config.py` references. Current locations are:
    - `models/layers.py:1899-1901` (init / parameter create)
    - `models/layers.py:3080-3082` (apply on scores)
    - `models/layers.py:3005` (manual-attention elif — needed for
      bit-identical step-0)
    - `models/layers.py:1041` (constructor arg on MHA)
    - `configs/llm_config.py:1039` (`LLMConfig.use_alibi_bias` field)
    - `configs/llm_config.py:5052` (existing `Screen10M20MAlibiBiasConfig` — proves the wiring works)
    - `models/llm.py:517, 768, 1051` (flag threading)
  The reviser should update the line references in `idea.md`'s
  Design Sketch section so the code-implementer doesn't grep the
  wrong lines. Mechanism is unchanged — this is purely a doc-fix.
- **F2 (sign convention note for the plan)**: the code at
  `models/layers.py:3080-3082` writes
  `scores = scores - m * (j - i)`. For causal positions (j ≤ i),
  `(j - i) ≤ 0`, so the effective bias on past tokens is
  `+m_h * (i - j)` — opposite sign from the ALiBi paper's
  `−s_h · (i − j)`. Since `m_h` is learnable and init to 0, the
  optimizer can learn either sign and the experiment is valid; the
  reviser should just note in the plan that the paper's
  `s_h = 1/2^(8k/H)` is irrelevant (we don't use fixed slopes) and
  that the only thing the optimizer sees is the unstructured
  per-head `m_h` axis. No code change needed — just a plan note so
  the code-implementer doesn't try to "port the paper's slopes
  exactly" and break step-0 bit-identical.
- **F3 (subclass name)**: the new subclass should be named
  `Tiny1M3MAlibiConfig` (matches the lever family naming: 152 →
  `Tiny1M3MAttnLogitBiasConfig`, 166 → `Tiny1M3MT5RPEConfig`, 175 →
  `Tiny1M3MAlibiConfig`). Do not name it `Tiny1M3MAlibiSlopesConfig`
  (the idea's slug includes "slopes" but the repo convention drops
  the description in favor of the mechanism name — see the Q1 alias
  `Screen10M20MAlibiBiasConfig` at line 5052, which uses "Bias"
  not "Slopes").

What the plan must contain (handed to code-implementer):
- Add `Tiny1M3MAlibiConfig(Tiny1M3MConfig)` with `use_alibi_bias:
  bool = True` to `configs/llm_config.py`, in the Block-1 cluster
  near the other per-head-attention-shape subclasses (after
  `Tiny1M3MT5RPEConfig` ~ line 2213, before `Tiny1M3MDropConnectWOConfig`
  ~ line 2228).
- No other code changes needed — all threading sites are already
  in place and the Screen10M20M variant already proves the wiring.
- Smoke test: build a `Tiny1M3MAlibiConfig()` model, run a single
  forward with init weights, confirm `max-abs-diff(output, baseline) = 0.0`
  within 1 ULP (slope=0 ⇒ scores+0 ⇒ softmax+0 ⇒ AV+0 ⇒ output+0).
- A/B vs `Tiny1M3MConfig` (val 6.4306 / cache 6.4216) at seed 42,
  3M tok. PASS = `trt_val ≤ 6.4216 − 0.02`. NULL band = `|Δ| < 0.02`
  sub-noise inconclusive. Report plan: mean ± std of step-0 val,
  step-100 val, final val, final train, val_acc, and
  max-abs-diff(step-0 forward output, baseline forward output).

Sign convention in the published ALiBi paper uses `−s_h · (i − j)`.
The repo implementation uses `+m_h · (i − j)` (sign-flipped). With
learnable `m_h` init 0, this is mathematically equivalent and the
optimizer can recover ALiBi's behavior by learning `m_h < 0` for
the decay case. Plan should note this and not attempt to "fix" the
sign in code (it would break step-0 bit-identical if implemented
incorrectly, and the sign convention is irrelevant to the
experiment's validity).
