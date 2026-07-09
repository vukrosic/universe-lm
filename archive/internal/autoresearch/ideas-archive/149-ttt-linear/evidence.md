# Evidence — 149 TTT-Linear (Sun et al. 2024)

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674
- baseline: cached mean=6.4302 ±0.04 (box 5b8a7fea8963, measured 2026-06-14)
- treatment val: 6.4303   Δ vs baseline: +0.0001
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ ctrl − 0.01; NULL band |Δ| < 0.01; DRIFT > +0.01  → **met (NULL band)**
- box check: cached mean 6.4302 vs leaderboard ctrl 6.4306 (within noise)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 149-ttt-linear_52674.log)
- date: 2026-06-14

## Transfer note
Per-input closed-form fast-weight linear (`TTTLinear`, Newton-step on
`||W·x − x||²`) sits inside the FFN up-projection. At 0.94M / 92 update steps
the lever's headline capacity-multiplier bet does not survive: the model has
no signal to ramp `ttt_lr` away from its init-zero value, so the fast path
stays pinned to `F.linear(x, weight)` (byte-identical to a vanilla `nn.Linear`)
for the entire run — the extra matmul is paid but the adaptation is never
activated. The closed-form gradient requires a non-zero reconstruction
error to be useful; with the target distribution still very high-rank at
92 steps there is error, but the optimizer never sees a gradient toward
`ttt_lr > 0` because the gradient flow through a near-zero scalar is
suppressed. The lever pays its cost only after the model has enough signal
to learn a non-zero `ttt_lr` — that signal does not materialize at this
tier. At larger scale (≥135M, longer horizon) the closed-form path may
actually engage, but Phase-1 transfer bet is null: the mechanism is
*latent* — present in the model, never used — which closes the
input-conditioned-weight axis at 0.94M.
