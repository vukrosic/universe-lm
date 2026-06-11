# Plan — 038 SWAN

## Goal
Replace Muon orthogonalization on matrix params with SWAN whitening and test whether the stateless preprocessing holds up at tiny1m3m.

## Control
- current Muon-on-matrix / AdamW-on-scalars routing
- seed 42
- same data order and step count

## Treatment
- same routing, but matrix grads use SWAN preprocessing instead of Muon
- 1-D, norms, embeddings stay on AdamW

## Pass bar
- Win if `trt <= ctrl - 0.01`
- Null if `|trt - ctrl| < 0.005`
- Fail if `trt >= ctrl + 0.01`

## Checks
- Confirm the matrix/non-matrix routing stays identical except for the matrix updater
- Log whitening strength over the first 100 steps
- Keep the same LR and momentum values from the current Muon config
- Verify the SWAN optimizer stays stateless
