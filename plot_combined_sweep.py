#!/usr/bin/env python3
"""Combined qk_gain sweep — all 12 variants, step-number X-axis."""

import json
import matplotlib.pyplot as plt

# Local baseline (step-aligned)
local = {
    "gain": "baseline",
    "final_vl": 9.3769,
    "steps": [0, 1, 2, 3, 4, 5, 6, 7],
    "val_losses": [10.8125, 10.75375, 10.716875, 10.63375, 10.4775, 10.199375, 9.7925, 9.376875],
}

# Load remote results
with open("/tmp/qk_sweep_low.json") as f:
    low = json.load(f)
with open("/tmp/qk_sweep_high.json") as f:
    high = json.load(f)

all_results = low + high

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    0.0: "#888888",
    0.1: "#4ecdc4",
    0.2: "#45b7d1",
    0.5: "#96ceb4",
    1.0: "#ffeaa7",
    1.5: "#fdcb6e",
    2.0: "#e17055",
    3.0: "#d63031",
    4.0: "#e84393",
    5.0: "#a29bfe",
    6.0: "#6c5ce7",
    8.0: "#74b9ff",
}

# Local baseline
ax.plot(local["steps"], local["val_losses"], "o-",
        color="#2d3436", linewidth=2.5, markersize=6,
        label=f'baseline (local, final=9.377)', zorder=10)

# Remote results
for r in all_results:
    g = r["gain"]
    steps = list(range(len(r["val_losses"])))
    color = colors.get(g, "black")
    marker = "s" if g >= 3.0 else "o"
    ls = "--" if g >= 3.0 else "-"
    ax.plot(steps, r["val_losses"], f"{marker}{ls}",
            color=color, linewidth=1.5, alpha=0.8,
            label=f"gain={g} (final={r['final_vl']:.4f})")

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Validation Loss", fontsize=12)
ax.set_title("Toy Sweep: qk_gain (8 steps, 32k tokens) — step-number aligned", fontsize=13)
ax.set_xticks(range(8))
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/Users/vukrosic/my-life/llm-research-kit-scaling/runs/toy/qk_gain_all.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

# Summary table
print(f"\n{'gain':<8} {'final_vl':>10} {'vs_local':>12}")
print("-"*32)
sorted_res = sorted(all_results, key=lambda x: x["final_vl"])
for r in sorted_res:
    delta = r["final_vl"] - 9.3769
    sign = "+" if delta > 0 else ""
    print(f"{r['gain']:<8.1f} {r['final_vl']:>10.4f} {sign}{delta:.4f}")
best = sorted_res[0]
print(f"\n★ Best: gain={best['gain']} → {best['final_vl']:.4f} (Δ vs local baseline: {best['final_vl']-9.3769:.4f})")