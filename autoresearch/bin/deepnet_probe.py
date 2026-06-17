#!/usr/bin/env python3
"""deepnet_probe.py — E5 understanding (no training): measure how DeepNet-a bounds
the residual-stream magnitude across depth.

DeepNet-a scales every block's sublayer output by a = (2*n_layers)^(-1/2) before
the residual add. Wang et al. 2022 (Thm 1) claim this bounds the residual stream's
growth to O(1) across depth instead of the ~sqrt(L) growth of an un-scaled pre-norm
stack. This probe verifies that claim DIRECTLY on the real model at init: build the
network with and without `use_deepnet_alpha`, forward a random batch, and read off
the per-layer residual-stream RMS. No data, no training, no GPU — CPU-only.

Width is held FIXED (d_model=128) and DEPTH is varied, so the effect we see is the
depth-dependent residual bounding, not a width/N artifact (previews E2). See
autoresearch/DEEPNET-RESEARCH.md.

  python3 autoresearch/bin/deepnet_probe.py
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from configs.llm_config import Ladder8M155MConfig
from models.llm import MinimalLLM

DEPTHS = [4, 8, 16, 30]   # 30 = the 135M release target's depth
BATCH, SEQ = 2, 64
SEED = 0


def make_config(n_layers, flags=None):
    """Fixed-width (d=128) config at a given depth, with arbitrary flag overrides."""
    flags = flags or {}
    fields = [("n_layers", int, dataclasses.field(default=n_layers))]
    fields += [(k, type(v), dataclasses.field(default=v)) for k, v in flags.items()]
    C = dataclasses.make_dataclass("C", fields, bases=(Ladder8M155MConfig,))
    return C()


# arm -> flag overrides (mirrors run_rung.py ARM_FLAGS; all D002-safe)
ARMS = {
    "baseline":   {},
    "deepnet":    {"use_deepnet_alpha": True},
    "deepnet_ab": {"use_deepnet_alpha": True, "use_deepnet_beta_init": True},
}


def per_layer_grad_norm(cfg):
    """Forward a random batch + CE loss on random targets, backward, return the
    per-block total gradient norm. The PROFILE across layers (not the absolute
    scale) is the signal: DeepNet's bounded update (Wang Thm 1) should make the
    per-layer grad norms more UNIFORM than the baseline."""
    torch.manual_seed(SEED)
    model = MinimalLLM(cfg).train()
    vocab = getattr(cfg, "vocab_size", None) or 49152
    ids = torch.randint(0, vocab, (BATCH, SEQ))
    tgt = torch.randint(0, vocab, (BATCH, SEQ))
    out = model(ids)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    norms = []
    for blk in model.transformer_blocks:
        sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in blk.parameters() if p.grad is not None)
        norms.append(sq ** 0.5)
    return norms


def per_layer_rms(cfg):
    """Build the model, forward a random batch, return per-block residual RMS."""
    torch.manual_seed(SEED)
    model = MinimalLLM(cfg).eval()
    blocks = model.transformer_blocks
    rms = []

    def hook(_m, _inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        rms.append(float(x.detach().float().pow(2).mean().sqrt()))

    handles = [b.register_forward_hook(hook) for b in blocks]
    vocab = getattr(cfg, "vocab_size", None) or 49152
    ids = torch.randint(0, vocab, (BATCH, SEQ))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    return rms


def main():
    print("=== DeepNet-a residual-stream bounding (step-0, d_model=128, random batch) ===")
    print("Per-block residual RMS; 'grow' = last/first. DeepNet should keep grow ~flat.\n")
    print(f"{'L':>3} {'alpha':>7} | {'baseline first->last (grow)':>34} | {'deepnet first->last (grow)':>34}")
    print("-" * 84)
    for L in DEPTHS:
        a = (2.0 * L) ** -0.5
        base = per_layer_rms(make_config(L))
        deep = per_layer_rms(make_config(L, {"use_deepnet_alpha": True}))
        bg = base[-1] / base[0] if base[0] else float("nan")
        dg = deep[-1] / deep[0] if deep[0] else float("nan")
        print(f"{L:>3} {a:>7.3f} | {base[0]:>8.3f} -> {base[-1]:>8.3f} ({bg:>5.2f}x) | "
              f"{deep[0]:>8.3f} -> {deep[-1]:>8.3f} ({dg:>5.2f}x)")
    print("\nReading: if baseline 'grow' rises with L while deepnet stays ~flat, the")
    print("mechanism (depth-bounded residual stream) is confirmed empirically — the")
    print("structural reason DeepNet-a is a depth-driven (H1) candidate, not just an")
    print("intercept tweak. sqrt(L) reference grow:  " +
          "  ".join(f"L{L}:{math.sqrt(L):.2f}" for L in DEPTHS))

    # --- Gradient-uniformity probe (the update-side / optimization claim) ---
    Lg = DEPTHS[-1]  # target depth (30) — where bounded-update should matter most
    print(f"\n=== Per-layer GRADIENT-norm uniformity at L={Lg} (random CE loss, 1 backward) ===")
    print("DeepNet's bounded update (Thm 1) should make per-layer grad norms more")
    print("UNIFORM. spread = max/min, cv = std/mean across the L blocks (lower = flatter).\n")
    print(f"{'arm':>12} | {'spread (max/min)':>16} | {'cv (std/mean)':>14} | first/last block grad")
    print("-" * 78)
    for arm, flags in ARMS.items():
        g = per_layer_grad_norm(make_config(Lg, flags))
        gmin = min(x for x in g if x > 0) or 1e-12
        spread = max(g) / gmin
        mean = sum(g) / len(g)
        var = sum((x - mean) ** 2 for x in g) / len(g)
        cv = (var ** 0.5) / mean if mean else float("nan")
        print(f"{arm:>12} | {spread:>16.2f} | {cv:>14.3f} | {g[0]:.3e} .. {g[-1]:.3e}")
    print("\nReading: lower spread/cv = flatter per-layer update profile. If deepnet (and")
    print("more so deepnet_ab) flattens it vs baseline, that's the optimization-side")
    print("mechanism the forward-RMS probe can't see — and the reason E3 (alpha vs")
    print("alpha+beta) could move even though alpha's forward effect is mild.")


if __name__ == "__main__":
    main()
