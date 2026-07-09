# Plan — 058 Fixup

## Goal
Test whether the existing zero-init residual flag helps tiny1m3m at 6 layers.

## Control
- `Tiny1M3MConfig`
- `zero_init_resid=False`
- Two control runs for variance bracketing, both seed 42

## Treatment
- `Tiny1M3MConfig`
- `zero_init_resid=True`
- Same seed 42

## Pass bar
- Win if `trt <= min(ctrl1, ctrl2) - 0.005`
- Null if `|trt - ctrl| < 0.01`
- Fail if `trt >= ctrl + 0.01`

## Checks
- Confirm both residual zero-init sites are actually zeroed:
  - `qkv[qkv_size:]`
  - `ffn.down_proj.weight`
- Log the step-0 delta between ctrl and treatment to confirm the wiring starts identically.

