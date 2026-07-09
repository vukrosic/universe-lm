# Evidence — 160 — Per-Head RMS Gain on Attention Value Output

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03)
- baseline: cached mean=6.4504 ±0.0558 (box 5b8a7fea8963, measured 2026-06-14T05:39:17Z, n=7)
- treatment val: 6.4481   Δ vs baseline: -0.0023  (well inside band; inside the plan's own |Δ|<0.005 null window)
- bpb: n/a (pending harness)
- pass/fail bar (from plan.md): PASS ≤ ctrl − 0.005 (= 6.4454) → not met; NULL band |Δ| < 0.005 → met (Δ = -0.0023); DRIFT > +0.005 → not triggered
- box check: baseline mean 6.4504 vs leaderboard ctrl ~6.41 (within noise band — no drift sentinel triggered)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (logs: 160-rms-gain-per-head_52674.log)
- date: 2026-06-14

## Transfer note
Per-head V-output gain (Gemma-2 / Qwen-2.5 mechanism) is a *no-op at step 0* (gain init 1.0) and a small per-head scalar at convergence (+48 params). At 0.94M the gradient signal is too weak to push the gains away from 1 in 3M tokens — the Δ is well inside the box variance and the plan's own null band. This is consistent with 016-qk-norm (WIN, pre-softmax QK axis) but suggests the *post-AV* magnitude axis is redundant given the W_O projection that follows. Mechanism is real at scale (head output magnitude control is a known large-model lever) but invisible at 0.94M — the network simply has no pressure to use the gain in 3M tokens. Re-test at 135M: the gradient per token increases ~140× and the head-gain axis should become binding.
