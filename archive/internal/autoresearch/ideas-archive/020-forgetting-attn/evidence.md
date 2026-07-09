# Evidence — 020 Forgetting Transformer (FoX)

## Verdict: FAIL — NaN, needs-recode
- tier: tiny1m3m, seed 42, box: vast-81.45.65.189 (V100-PCIE-32GB)
- control val: n/a (shared fire-ctrl was buggy — see "Wiring bug" below)
- treatment val: **NaN** (both orig 2026-06-10T08:14Z and rerun 2026-06-10T11:13Z)
- pass/fail bar (plan.md): WIN trt < ctrl − 0.02 / NULL |Δ| < 0.02 / FAIL trt > ctrl + 0.01
- bpb: n/a (NaN)
- box check: ctrl_fire run on this box logged use_fire_pe=False (wiring bug, see below) so we have no fire-equipped shared baseline
- raw: remote-results/2026-06-10-vast-tiny1m3m/{020-fox.log, 020-fox-r.log}
- date: 2026-06-10

## Wiring bug (cross-cutting — affects 020-025 shared-ctrl)
The shared fire-equipped baseline used `class C(Tiny1M3MVQGainSWAHighRoPE250KConfig): use_fire_pe: bool = True`. The subclass override was silently dropped at runtime — both `ctrl_fire` (8:03Z) and `ctrl_fire2` (8:50Z) dumped the model config with `use_fire_pe: False` and produced identical val_loss 6.3419/0.1511 (deterministic on same seed + same model). The 4 -sh reruns (ctrl_fire, ctrl_fire2, 024-gated-sh, 025-ssmax-sh) all produced 6.3419 — same flag-drop root cause. The 020-023 treatments (020-fox, 021-vres, 023-canon) used pre-baked `*OnFireConfig` classes that DO carry use_fire_pe=True, so the trt-side flag is correct; the failure is the shared baseline side. Recommend fix: pre-bake `Tiny1M3MVQGainSWAHighRoPE250KFireConfig` in llm_config.py instead of overriding in the _arq subclass.

## NaN mode
- Both runs reached step 400/732 (~55% progress) and stayed at `loss=nan, acc=0.000` for the rest of training.
- Final Train Loss: nan · Final Val Loss: nan.
- Forward sanity: trt config builds + reaches step ~400 before loss → +inf → NaN. Loss is not 0 (which would indicate a wiring bug); it's a blow-up. Probable cause: the per-head FoX decay multiplicatively compounds across attention rows; with `b_f = +10.0` the gate's `sigmoid(·) → log_f ≤ 0` produces a sharply decreasing cumsum, and the row-renormalisation step amplifies any per-row mass imbalance. At longer effective context (>~200 tokens) the renormalised row can hit floating-point underflow → +inf → NaN on the next matmul.
- Test `tests/test_fox.py` only verified `T=2048` initial bounds; it did not exercise the post-init training dynamics.

## Transfer note
The NaN is not a scale-invariant property (small attention block + tiny d_model tolerates the gate; deeper/longer amplifies it). The mechanism as designed (per-head learnable decay) is paper-validated (FoX, arXiv:2505.11780), so the fix is likely numerical: (a) clamp `gate_b` to a smaller init (e.g. +3.0 → max decay `exp(-3·T)≈0`, but no row-renorm blow-up), (b) clip the per-row renormalised probs, or (c) add fp32 row-renorm safety. The lever is real at 135M; the tiny1m3m impl just needs the numerical guard.
