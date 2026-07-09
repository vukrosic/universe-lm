"""Synthetic probe for the RetNet retention kernel.

Runs the same 3 invariants as `tests/test_retention.py` but as a
standalone script (no pytest needed). Use as the v1 pre-flight — the
3-seed protocol and screen20m A/B are deferred to v2 (kernel + integration
PR). See `autoresearch/ideas/004-retnet-retention/plan.md`.

Usage: `python autoresearch/ideas/004-retnet-retention/probe.py`
"""
import math
import os
import sys

# Probe is at autoresearch/ideas/004-retnet-retention/probe.py — make the
# repo root importable so `from models.retention import ...` works.
# Path: probe → 004-retnet-retention → ideas → autoresearch → repo_root.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from models.retention import RetentionKernel


def main() -> int:
    torch.manual_seed(42)

    # Build a small kernel + inputs.
    kernel = RetentionKernel(d_k=8, n_heads=4, init_gamma=0.9)
    Q = torch.randn(2, 4, 16, 8)
    K = torch.randn(2, 4, 16, 8)
    V = torch.randn(2, 4, 16, 8)

    # Invariant 1: no NaN/Inf.
    O = kernel(Q, K, V)
    assert torch.isfinite(O).all(), f"[FAIL] NaN/Inf in output: shape {O.shape}"
    print(f"[OK]  no NaN/Inf — output shape {tuple(O.shape)}")

    # Invariant 2: causal — zero K at positions t >= T/2; O[t] for
    # t < T/2 must be unchanged (the kernel only sums s <= t).
    T = K.size(2)
    split = T // 2
    K_clipped = K.clone()
    K_clipped[:, :, split:, :] = 0
    O_clipped = kernel(Q, K_clipped, V)
    past_diff = (O[:, :, :split, :] - O_clipped[:, :, :split, :]).abs().max().item()
    assert past_diff < 1e-5, f"[FAIL] causal leak: max abs diff = {past_diff}"
    future_diff = (O[:, :, split:, :] - O_clipped[:, :, split:, :]).abs().max().item()
    assert future_diff > 1e-3, (
        f"[FAIL] future outputs identical — test is vacuous: {future_diff}"
    )
    print(
        f"[OK]  causal — past Δ = {past_diff:.2e}, "
        f"future Δ = {future_diff:.4f} (vacuous-check OK)"
    )

    # Invariant 3: per-head independence.
    kernel_b = RetentionKernel(d_k=8, n_heads=4, init_gamma=0.9)
    with torch.no_grad():
        kernel_b.gamma_raw[2] = kernel_b.gamma_raw[2] + 3.0
    O_b = kernel_b(Q, K, V)
    for h in (0, 1, 3):
        d = (O[:, h] - O_b[:, h]).abs().max().item()
        assert d < 1e-5, f"[FAIL] head {h} changed when only head 2's γ was perturbed: {d}"
    d2 = (O[:, 2] - O_b[:, 2]).abs().max().item()
    assert d2 > 1e-3, f"[FAIL] head 2 did not change: {d2}"
    print(f"[OK]  per-head independence — heads 0/1/3 identical, head 2 Δ = {d2:.4f}")

    # Extra sanity: γ is per-head, learnable, in (0, 1).
    gammas = torch.sigmoid(kernel.gamma_raw)
    assert (gammas > 0).all() and (gammas < 1).all(), (
        f"[FAIL] γ out of (0,1): {gammas}"
    )
    print(f"[OK]  γ in (0, 1) — values {[f'{g:.4f}' for g in gammas.tolist()]}")

    print()
    print("ALL 3 INVARIANTS PASS — kernel ready for v2 production wiring")
    return 0


if __name__ == "__main__":
    sys.exit(main())
