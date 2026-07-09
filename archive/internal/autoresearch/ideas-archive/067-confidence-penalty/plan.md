# Plan — 067 Confidence penalty

## Goal
Test whether the entropy bonus on the output distribution beats the gold-only softening family at tiny1m3m.

## Control
- `Tiny1M3MConfig`
- seed 42
- same data order and step count

## Treatment
- `Tiny1M3MConfPenaltyConfig` with `conf_penalty_beta = 0.01`
- trainer keeps the loss train-only and evaluation stays plain CE

## Pass bar
- Win if `trt <= ctrl - 0.01`
- Null if `|trt - ctrl| < 0.005`
- Fail if `trt >= ctrl + 0.01`

## Checks
- Confirm the aux term is train-only
- Verify the existing config wiring is used, not a new code path
- Keep the softmax architecture bit-identical
