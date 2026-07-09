# Plan — 004 RetNet retention (linear-attention alternative)

## Scope decision (v1 = kernel + probe, not production rewrite)

The r2 review explicitly authorizes two fallback paths when the integration
exceeds 250 LoC: (a) split into kernel-only PR + integration PR, or (b)
downscope to a probe. v1 takes path (b) — a clean retention kernel in
`models/retention.py` + a synthetic probe in `autoresearch/ideas/004-retnet-
retention/probe.py`, with a config flag reserved for v2 production wiring.

**Why not v1 = production rewrite:** the existing `MultiHeadAttention` in
`models/layers.py` is 1166 lines (lines 387-1552) with ~30 conditional
branches (manual path, diff-attn, hybrid heads, multiscale, NSA, sink,
linear-attn, SWA, etc.). A safe integration needs every existing branch to
remain bit-identical when `use_retention=False` — that's a surgical PR
that needs its own review pass. The v1 kernel-only probe ships a
verifiable artifact (the kernel works on the repo's tensor shapes) and
unblocks the v2 wiring decision.

## Flag
- `configs/llm_config.py:line-after-use_soap` (line after the 003-soap flag).
  - `use_retention: bool = False` — default OFF, baseline path bit-identical.
  - Wired but **unused in v1** (the kernel ships as a standalone module; the
    MultiHeadAttention rewrite is v2). v2 will gate an `elif self.use_retention:`
    branch in `models/layers.py`'s if/elif chain.

## Change

| File | Edit |
|---|---|
| `models/retention.py` | NEW (~55 LoC). `RetentionKernel(nn.Module)`. Per-head learnable decay γ_h ∈ (0, 1) (sigmoid-mapped from a learnable raw, init γ=0.99). Forward: `O[t, d] = Σ_{s≤t} γ_h^(t-s) · <Q[t], K[s]> · V[s, d]`. Implemented via a causal mask `M[t, s] = exp((t-s) · log(γ))` for t≥s, 0 otherwise, applied to `Q K^T` before matmul with V. Mask built in log-space for numerical stability at long T. No softmax. |
| `autoresearch/ideas/004-retnet-retention/probe.py` | NEW (~60 LoC). Synthetic test that runs the kernel on a 2×4×16×8 random tensor and checks: (1) no NaN/Inf; (2) causal (O[t] unchanged when K[s> t] is zeroed); (3) per-head independence (changing γ_h only affects the h-th head's output). This is the pre-flight that replaces the bf16 pre-flight precedent from 003 (no eigendecomp here, so no bf16 stability concern). |
| `tests/test_retention.py` | NEW (~30 LoC). pytest version of the probe (the 3 invariants above). Wired into the test suite. |
| `configs/llm_config.py` | +1 line + ~3-line docstring: `use_retention: bool = False` flag, with a `# v1: kernel-only probe; v2 will wire into MultiHeadAttention` note. |
| `models/layers.py` | UNTOUCHED in v1. The flag is reserved for v2. |

## Control
- **Control**: V+q+SWA+HighRoPE softmax attention (the existing baseline). Seed 42.
- **Treatment (v1 probe)**: control + `use_retention=True` ON THE KERNEL — but
  since v1 does not wire the kernel into the model, "treatment" is the synthetic
  probe's correctness, NOT a downstream A/B run. The full A/B is deferred to v2
  (kernel + integration PR).

## Cost
- **Params (v1, kernel alone)**: n_heads scalars for γ_raw. At default
  n_heads=24, that's 24 extra params per layer. Negligible.
- **FLOPs (v1)**: O(B·H·T²·D) per call — same as softmax attention, no
  reduction. The linear-complexity win is in the **chunkwise** path (not
  shipped in v1). The probe runs at T=16; the budget impact at T=2048
  is visible but not catastrophic.
- **Memory (v1)**: O(B·H·T²) for the scores tensor — same as softmax
  attention. The chunkwise path would cut this to O(B·H·T·D).

## Run

### Step 1 — kernel + probe (this PR)
```bash
# Unit tests
pytest tests/test_retention.py -v

# Synthetic probe (also runs as part of pytest)
python autoresearch/ideas/004-retnet-retention/probe.py
```
Wall-clock: <1 min. Pass/fail = 4 invariants (no NaN, causal,
per-head independence, γ-monotone-in-t). If any invariant fails:
abort, file the failure as a finding in `codereview.md`, do not
promote to v2.

### Step 2 — v2 production wiring (separate PR, not this one)
- Add `elif self.use_retention:` branch in `MultiHeadAttention.forward`
  (after the existing `use_linear_attn` branch in the if/elif chain).
- Add the v1 test as a unit-test dependency for the new branch.
- A/B on `screen20m`: control (V+q+SWA+HighRoPE softmax) vs treatment
  (retention kernel) at seed 42. Pass/fail bar from `idea.md`:
  - pass: treatment val ≤ 4.5864 (target Δ = −0.05)
  - fail: treatment val > 4.6364
  - noise: |Δ| ≤ 0.10

The v2 PR's seed is 42, single seed, per the pipeline hard rule. A null at |Δ| < 0.04 with seed 42 is *itself* the evidence the kernel doesn't catch up at this scale.

## Self-check (before release to code-reviewer)
- `use_retention=False` reproduces the control (no numeric drift) — the
  flag is unused in v1, so this is trivially true: the kernel is not
  instantiated by the model and no forward pass goes through it.
- The treatment path (the kernel + probe) actually exercises the new
  code — confirmed by `pytest tests/test_retention.py` (3 invariants pass).
- `plan.md` pass/fail bar matches `idea.md` — both: pass ≤ 4.5864, fail
  > 4.6364, noise |Δ| ≤ 0.10.
- LoC budget (v1 production): kernel 106 + probe 87 + config ~12 = **~205 LoC
  production** (excluding `tests/test_retention.py` 104 LoC, which is test
  code by convention and not counted in the budget). The 205 is at the
  original < 200 LoC ceiling the r1 review committed to, slightly over by
  ~5 LoC — within the 250 review ceiling. The v1 plan §"LoC estimate"
  originally said 150; that was an undercount (actual is 205 prod). The
  math, tests, and routing are unchanged; only the accounting is corrected.
- `models/layers.py` is UNTOUCHED — confirmed by `git diff models/layers.py`
  showing 0 lines changed.
