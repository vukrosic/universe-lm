# Evidence — 003 soap

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.4191 — n=1 (single seed, per ONE-SEED-ONLY protocol)
- control bracket: ctrl=6.4078, ctrl2=6.4072 (gap 0.0006 — very tight this batch)
- Δ vs ctrl: +0.0113 (treatment is *worse*); Δ vs ctrl2: +0.0119 (worse)
- two-ctrl rule: treatment loses to *both* ctrls → NULL (wrong sign)
- raw: ~/arq/logs/{003-soap.log,ctrl.log,ctrl2.log} on box (batch 2026-06-09T10:39–10:51Z)
- date: 2026-06-09

Notes:
- First two attempts OOM'd: `torch.eye(d_out=49152)` = 9 GiB preconditioner on
  the vocab-sized params. Fixed by `MAX_PRECONDITIONER_DIM=2048` → AdamW
  fallback for any dim > 2048 (optimizers/soap.py). Third attempt ran clean.
- Caveat from that fix: at tiny1m3m the embedding/lm_head (vocab=49152) are the
  bulk of params and now take the plain-AdamW path, so SOAP only preconditions
  the small transformer matrices. The benefit SOAP is supposed to provide is
  largely bypassed at this tier — a fairer test of SOAP would need a tier where
  the preconditioned 2D blocks dominate. As-run at tiny1m3m: no gain.
