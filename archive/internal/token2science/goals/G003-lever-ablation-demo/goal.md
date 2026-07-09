# G003 - Does the candidate lever lower validation loss?

**Question:** does adding `good_lever` to the model lower validation loss
compared to an identical control without it?

**Metric:** `val_loss` (lower is better)
**Pass-bar:** `val_loss < 6.37`
**Baseline:** the control arm (`T301`, no lever) under the same mock backend.

This is a mock-backed study: experiments are answered by the deterministic
testbench oracle instead of a GPU, so the A/B runs instantly and reproduces
exactly. It exists to demonstrate a full research loop - hypothesis, control
vs treatment, replication across seeds, and a written finding - end to end
without compute.

## Tasks
- `T301` - control arm (no lever).
- `T302` - treatment arm (`good_lever`).
