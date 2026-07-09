# 172-per-head-rope-base — review log

## r1 — 2026-06-14 — verdict: approve

Source, mechanism, scope, dedup, implementability all check out. Step-0
byte-identity is verifiable in the existing code. Findings below are
plan-stage guidance, not blockers — they go to the code-implementer.

### Confirmations (no fix needed)

- **Source real, current.** RoFormer (Su et al., arXiv:2104.09864, Neurocomputing 2024)
  is the canonical RoPE paper. NTK-aware scaling (bloc97, 2023) and YaRN
  (Peng et al., arXiv:2309.00071) are real and cited accurately. Per-head
  learnable RoPE base is the principled *head-specialization* extension of
  the closed-axis global-base sweep (`closed.md:22`, "RoPE base sweep —
  500k winner"). No fabricated citations.
- **Mechanism is a mechanism, not a hyperparameter.** Architectural change:
  one learned scalar `head_scale[h] = exp(per_head_rope_log[h])` per head
  (init 0 → exp(0)=1.0), multiplied into the per-head frequency spectrum.
  Step-0 byte-identical to the global 500k-base baseline (verified at
  `models/layers.py:1753-1754` init and `:2031-2034` use; `head_scale=1.0`
  default branch in the same code path keeps the `use_per_head_rope_base=False`
  case bit-identical too). Not an LR/init-constant lever.
- **Not a closed-axis duplicate.** Per-head *frequency* scaling (172) is
  mathematically distinct from per-head *temperature* (155-per-head-temp →
  null at tiny1m3m), per-layer *temperature* (161-dyt-temp → null), and
  rebased-attn (154-rebased-attn → WIN record-break, K/V pre-softmax
  rebase). None of the closed-by-the-loop entries duplicate this lever.
  The closest relative is the closed global "RoPE base sweep — 500k winner"
  axis, which 172 takes the next step on.
- **Implementable in < 200 LoC.** Mechanism is already built and wired:
  `models/layers.py:1753-1754` (init), `:2031-2034` (use), `:2828/:2851`
  (forward-graph integration), `:3429/:3839` (constructor); the flag is
  already plumbed through `models/llm.py:508/:745/:1015`. The only
  implementation work is a single config subclass (`Tiny1M3MPerHeadRopeConfig`)
  modeled on the existing `Tiny1M3MQKNormConfig` at `configs/llm_config.py:2334`
  (or `:4902` for the existing `Screen10M20MPerHeadRopeBaseConfig`).
- **Transfer-risk: med, correctly tagged.** RoFormer scales to 100B+
  (canonical); per-head learnable extension is principled (every LM has
  per-head Q/K/V) but not directly validated at ≥100M. Closest analog
  `partial_rotary_p` is validated at 1B+. Citation matches the tag.
- **One seed (42) only.** Idea pins to tiny1m3m, seed 42; no multi-seed
  request. Confirmed.

### Plan-stage guidance (actionable by code-implementer)

- **Tighten the pass/fail bar.** Idea states "expected Δval ≈ -0.003 to
  -0.012". At tiny1m3m box noise is ±0.01 val loss; the lower bound sits
  below the noise floor. Pin to the three-zone convention used elsewhere
  in this loop:
    - **WIN** — trt vs ctrl Δval < -0.01 (and trt beats both paired ctrl
      slots, per the existing two-ctrl bracket convention).
    - **Informative null** — |Δval| < 0.01 between trt and *both* ctrl
      slots. Closes the per-head RoPE frequency axis at 0.94M.
    - **DRIFT** — Δval > +0.01 (wrong-sign, lever hurts).
  This is consistent with 162-q-only-norm's "pass bar -0.005 missed by
  0.0007" closure pattern.
- **Step-0 byte-identity is mandatory.** Plan must include an explicit
  fp32 max-abs-diff < 1e-6 check against `Tiny1M3MConfig(rope_base=500000)`
  ctrl on the full forward (not just the RoPE module). The lever inherits
  the existing `_per_token_rope_log` path at `models/layers.py:2033-2037`
  — that path must stay `None` for 172 (don't accidentally turn it on).
- **Use rope_base=500000 as the ctrl, NOT rope_base=10000.** The 172
  subclass must set `rope_base: int = 500000` (the closed-winner) and the
  ctrl must be `Tiny1M3MConfig(rope_base=500000)`, not the default 10000.
  This isolates the per-head learning effect from the global-base-sweep
  winner; without it, the run conflates two axes.
- **Param-overhead note.** 48 scalars (+0.005% of 0.94M). Gradient signal
  per head may be weak at 0.94M/12L. Plan should treat a clean null as
  expected (mirrors 155-per-head-temp at this tier) and a small WIN
  (Δ < -0.01) as the surprise that justifies a Phase-2 re-test at ≥135M.
- **Family-fit flag (carry from taste).** Recent attention-positioning
  cluster: 154-rebased-attn (WIN), 155-per-head-temp (null), 161-dyt-temp
  (null). 172 is on the *frequency* axis (different from temperature and
  rebasing) but in the same family. Plan should report `head_scale[h]`
  values at the end of training (per-head max, min, std) so the result
  is interpretable even on null — a null with `head_scale` close to 1.0
  across all heads means the lever learned "stay near baseline"; a null
  with `head_scale` spread to e.g. [0.7, 1.4] means the lever learned
  useful specialization but val-loss didn't move (informative re-evaluate
  at larger tier).
- **Config wiring scope.** Add `Tiny1M3MPerHeadRopeConfig(Tiny1M3MConfig)`
  to `configs/llm_config.py` mirroring the existing
  `Screen10M20MPerHeadRopeBaseConfig` at `:4902`:
    - `use_per_head_rope_base: bool = True`
    - `rope_base: int = 500000`
  Model on `Tiny1M3MQKNormConfig` at `:2334` for Tiny1M3M-style docstring
  ("A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306)…
  add a `Tiny1M3MConfig(rope_base=500000)` companion ctrl if you want to
  isolate the per-head effect from the base-sweep axis"). No edits to
  `models/layers.py` or `models/llm.py` — mechanism already wired.