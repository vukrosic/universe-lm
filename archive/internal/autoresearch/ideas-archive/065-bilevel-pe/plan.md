# Plan — 065 bilevel-pe

## Goal
Test whether a fixed-stride two-band RoPE decomposition stacked on FIRE improves tiny1m3m loss.

## Control
- `009-fire-pe` style FIRE-equipped control
- seed 42
- same model/data schedule

## Treatment
- fixed stride `S = 64`
- two-band rotary frequencies with `g_intra = g_inter = 1.0` at init
- stack on FIRE, do not replace it

## Pass bar
- Win if `trt <= 6.3184`
- Null if `|trt - 6.3234| <= 0.005`
- Fail if `trt > 6.3234`

## Checks
- Verify step-0 is bit-identical to the FIRE control
- Confirm the stride segmentation is deterministic and does not use text metadata
- Log whether the per-band gates move off 1.0 during training

