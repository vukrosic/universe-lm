#!/usr/bin/env python3
"""Ablation interpretation frame — predict E3/E4 outcomes before the runs land."""

import json

print("""
=== ABLATION PREDICTIONS (E3/E4 at 8M rung) ===

E3: deepnet_ab (α + canonical β init)
  Hypothesis: β adds no flattening on top of α (just rescales magnitude 10×, optimizer absorbs)
  Init-probe finding: deepnet_ab cv=0.020 vs deepnet cv=0.011 (β only rescales, no extra flattening)
  Prediction: deepnet_ab ≈ deepnet ≈ baseline on the ladder (null stacking)
  Expected: all three within ±0.005, identical bounce pattern late

E4: rezero (learned scalar α init 0) + layerscale (learned per-channel γ init 1e-4)
  Hypothesis: the whole residual-damping family is Muon-redundant
  Theory: Muon's per-matrix orthogonalization already supplies the per-layer balancing
  Prediction: rezero ≈ layerscale ≈ deepnet ≈ baseline (family-wide null)
  Expected: all four arms indistinguishable (within band), confirming that optimization-stability
            levers are closed by Muon + RMSNorm at ≤30 layers

=== CONFIRMATION THRESHOLDS ===
  NULL confirmed: |Δ| ≤ 0.005 (deepnet's margin)
  Null questioned: 0.005 < |Δ| ≤ 0.02 (borderline, needs 3-seed confirm)
  Real finding: |Δ| > 0.02 (clears screen, warrants investigation)

=== INTERPRETATION LOGIC ===
  If E3 confirms (deepnet_ab ≈ deepnet):
    → β's canonical pairing is NOT required; α alone is sufficient for the null effect
    → implication: canonical DeepNet (Wang 2022) paired α+β is overkill; α's mechanism
      is gradient uniformity, but Muon already handles it
  
  If E4 confirms (rezero ≈ layerscale ≈ deepnet ≈ baseline):
    → the whole residual-damping family (fixed-scalar, learned-scalar, learned-perchannel)
      is indistinguishable — NOT because they're all bad, but because Muon erases the
      distinction; the choice of damping mechanism is moot when Muon normalizes anyway
    → research takeaway: at modest depths (L≤30) with Muon + RMSNorm, optimization-stability
      mechanisms do NOT bend the scaling exponent; the lever must be in attention/long-context

=== NEXT STEPS (after ablations land) ===
  1. Fit L(N) = E + A·N^(-α) on all 6 points (baseline + deepnet + 4 ablations, 8M only)
     — check if all exponents collapse to same α (vs. baseline slope being unique)
  2. Update DEEPNET-SYNTHESIS.md with E3/E4 results + final interpretation
  3. Write DEEPNET-RESEARCH.md closure: "Why deepnet is not the release lever"
  4. Pivot research direction: wire the long-context ladder arms (RoPE-base, QK-norm, diff-attn)
     — these are the mechanisms Muon does NOT replace
""")
