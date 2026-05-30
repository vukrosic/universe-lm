#!/usr/bin/env python3
"""Gain vs final val_loss for all models."""

import json, matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

# ToyConfig (32k tokens)
toy = sorted([
    (2.0, 9.3025), (3.0, 9.3119), (1.5, 9.3163), (4.0, 9.3506),
    (5.0, 9.3600), (1.0, 9.3775), (0.0, 9.3769),
], key=lambda x: x[0])
gains_t, vl_t = zip(*toy)
ax.plot(gains_t, vl_t, "o-", color="#e17055", linewidth=2, markersize=8,
        label="ToyConfig (3.2M, 32k tokens)")

# ToySmallConfig (1M tokens)
small = sorted([
    (3.0, 6.4175), (4.0, 6.4238), (2.0, 6.4322), (1.0, 6.4444), (0.0, 6.4675),
], key=lambda x: x[0])
gains_s, vl_s = zip(*small)
ax.plot(gains_s, vl_s, "s-", color="#4ecdc4", linewidth=2, markersize=8,
        label="ToySmallConfig (6.7M, 1M tokens)")

ax.set_xlabel("qk_gain", fontsize=12)
ax.set_ylabel("Final Val Loss", fontsize=12)
ax.set_title("Gain vs Final Val Loss", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/vukrosic/my-life/llm-research-kit-scaling/runs/qk_gain_vs_loss.png", dpi=150)
print("Saved: runs/qk_gain_vs_loss.png")