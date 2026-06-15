# Evidence — 109 KDA Channel Gate

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast-1.208.108.242 (1.208.108.242:52649, RTX 3060, sm_86)
- control val: 6.4037
- treatment val: 6.4091
- ctrl2 val: 6.4078
- delta vs ctrl: +0.0054
- delta vs ctrl2: +0.0013
- control gap: 0.0041
- two-ctrl rule: treatment does not beat both ctrls (loses to both by 0.0054 and 0.0013, both inside the +0.01 NULL band) → NULL
- pass/fail bar (idea.md): Δ ≤ −0.01 = WIN; |Δ| < 0.01 = NULL; Δ > +0.01 = DRIFT. Δ = +0.0054 sits inside the NULL band.
- leaderboard line: leaderboard ctrl 6.4306; box check = −0.0269 (this run's ctrl 6.4037 vs leaderboard 6.4306; within ~0.04 box noise — box is OK, slightly faster than the leaderboard snapshot)
- bpb: n/a (pending harness)
- raw: remote-results/2026-06-13-vast-tiny1m3m/results.json
- logs: remote-results/2026-06-13-vast-tiny1m3m/arq-109/{ctrl_52649.log,109-kda-channel-gate_52649.log,ctrl2_52649.log}
- date: 2026-06-13

## Cost
ctrl 949,056 params, trt 949,824 params — diff 768 (= 12 layers × 4 heads × 16 d_k), matches the idea.md prediction (~0.08% over baseline). Bounded 2·σ(g) gate (one per (head, channel) of V), zero-init ⇒ 2·σ(0)=1.0 at step 0.

## Transfer note
The bounded 2·σ(g) per-channel V-gate is the *only* form of the per-channel V axis that wasn't already in the closed-#001-cautious-Muon-era sweep. The closed `use_value_channel_gate` (unbounded 1+g) was already saturated at this scale; the KDA framing replaces unbounded with bounded 2·σ so each channel can only amplify/dampen within (0, 2). A clean null says the per-channel-V axis has been fully explored at tiny1m3m — the bounded parametrization has nothing to add. The diagonal-decay interpretation (KDA's Γ = diag(γ_1,…,γ_d)) is faithful to the softmax-attention V stream, but the available headroom at 0.94M is zero. At 135M the lever may grow if attention concentration is a load-bearing problem, but the leverage is at most the closed V-channel axis at this tier. No transfer.
