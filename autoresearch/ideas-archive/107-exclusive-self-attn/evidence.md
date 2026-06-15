# Evidence - 107 Exclusive Self Attention

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast-1.208.108.242 (1.208.108.242:52649, RTX 3060, sm_86)
- control val: 6.4163
- treatment val: 6.4047
- ctrl2 val: 6.3763
- delta vs ctrl: -0.0116
- delta vs ctrl2: +0.0284
- control gap: 0.0400
- two-ctrl rule: treatment does not beat both ctrls by > gap -> NULL
- pass/fail bar (idea.md): delta <= -0.01
- leaderboard line: ctrl 6.4306; box check = -0.0144 vs leaderboard ctrl
- raw: remote-results/2026-06-13-vast-tiny1m3m/results.json
- logs: remote-results/2026-06-13-vast-tiny1m3m/arq-107/{ctrl_52649.log,107-exclusive-self-attn_52649.log,ctrl2_52649.log}
- date: 2026-06-13

Exclusive self-attention helps against the first control, but the second control is stronger. Under the bracket rule this is a NULL, not a WIN.
