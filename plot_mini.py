#!/usr/bin/env python3
"""Plot Mini10M20M baseline vs gain=4.0 training curves."""

import json, matplotlib.pyplot as plt

baseline = json.load(open('/tmp/toy_medium_baseline.json'))
gain4 = json.load(open('/tmp/toy_medium.json'))

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(baseline['steps'], baseline['val_losses'], 'o-',
        color='#2d3436', linewidth=2, markersize=5,
        label=f'baseline (gain=0.0, final={baseline["final_vl"]:.4f})')
ax.plot(gain4['steps'], gain4['val_losses'], 's-',
        color='#d63031', linewidth=2, markersize=5,
        label=f'gain=4.0 (final={gain4["final_vl"]:.4f})')

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Validation Loss", fontsize=12)
ax.set_title("Mini10M20MConfig (10M, 20M tokens): baseline vs gain=4.0", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/vukrosic/my-life/llm-research-kit-scaling/runs/mini10m20m/qk_gain_mini.png", dpi=150)
print("Saved: runs/mini10m20m/qk_gain_mini.png")