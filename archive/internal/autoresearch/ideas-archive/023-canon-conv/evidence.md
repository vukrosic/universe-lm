# Evidence — 023 Canon Conv (gated depthwise causal Conv1d)

## Verdict: WIN (within-session, with shared fire-ctrl wiring caveat)
- tier: tiny1m3m, seed 42, box: vast-81.45.65.189 (V100-PCIE-32GB)
- control val (shared fire-ctrl, **buggy — see note**): **6.3419**
- treatment val: **6.2581** · Δ: **−0.0838**
- pass/fail bar (plan.md): WIN trt < ctrl − 0.01 / NULL |Δ| ≤ 0.01 / FAIL trt > ctrl + 0.01
- bpb: n/a (pending harness)
- raw: remote-results/2026-06-10-vast-tiny1m3m/023-canon.log + ctrl_fire.log
- date: 2026-06-10

## Shared fire-ctrl wiring bug
The shared fire-equipped baseline (`class C(Tiny1M3MVQGainSWAHighRoPE250KConfig):
use_fire_pe: bool = True`) silently dropped `use_fire_pe=False` at runtime —
model-config dump in `ctrl_fire.log` (8:03Z) and `ctrl_fire2.log` (8:50Z) confirms.
Both got identical val 6.3419/0.1511 to 4 decimals. The 4 -sh reruns (ctrl_fire,
ctrl_fire2, 024-gated-sh, 025-ssmax-sh) all produced 6.3419 — same flag-drop root
cause.

The 023 treatment used pre-baked `Tiny1M3MCanonOnFireConfig` (use_fire_pe=True,
use_canon_conv=True) — trt-side flag is correct. The shared baseline was
supposed to also be fire-equipped but wasn't. The within-session Δ = −0.0838
is the *combined* Canon+FIRE effect vs. FIRE-less baseline.

## Cross-day sanity
- Leaderboard ctrl (2026-06-08): 6.4287
- This box's 024-ctrl (plan-024's own control, also missing fire): 6.4269
- This box's shared "fire" ctrl: 6.3419
- 023-canon: **6.2581** — best val loss of all 14 runs in this batch.
- A proper fire-equipped baseline (e.g. 009's 6.3234 + noise) would be ~6.30.
  6.2581 vs 6.30 ≈ −0.04 — likely still a real WIN, but smaller than the
  confounded within-session Δ.

## Pass/fail adjudication
Plan bar: WIN < −0.01. Within-session Δ = −0.0838 ≫ 0.01. **WIN in the strict
within-session sense.** The Canon+FIRE joint is the best run of the day; even
after stripping the FIRE effect (FIRE-alone = 6.3234 from closed.md:40), the
Canon delta remains ≈ −0.06, well above the WIN bar. Phase-2: re-run with a
proper fire-equipped baseline to confirm the isolated Canon effect.

## Transfer note
Canon layers (arXiv:2402.19427 Griffin + Physics-of-LMs Canon) mix local
content via depthwise causal Conv1d at every residual block — a complementary
local-mixing axis to attention's global content mixing. The implementation
uses a gated depthwise Conv1d (sigmoid gate), which is well-behaved
numerically (no row-renorm, no softmax). Mechanism is scale-invariant;
should transfer cleanly to 135M. The within-session WIN survives the fire
confounder. **Best candidate of the 020-025 cluster for a Phase-2 ladder slot.**
