# Plan — 051 ScaleNorm

## Goal
Replace each residual-stream vector gain with a single scalar gain and test whether the reduced norm expressivity helps tiny1m3m.

## Control
- `Tiny1M3MConfig`
- seed 42
- same data order and step count

## Treatment
- `Tiny1M3MScaleNormConfig`
- `norm_type="scalenorm"`
- all other settings unchanged

## Pass bar
- Win if `trt <= ctrl - 0.005`
- Null if `|trt - ctrl| < 0.005`
- Fail if `trt >= ctrl + 0.005`

## Checks
- Confirm step-0 output matches the baseline when `weight=1`
- Verify both residual norms route through the scalar-gain norm
- Keep all other config fields identical to the baseline
