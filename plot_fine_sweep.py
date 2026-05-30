#!/usr/bin/env python3
"""Fine gain sweep + combined plot for ToyConfig."""

import json, matplotlib.pyplot as plt

with open("/tmp/qk_sweep_fine.json") as f:
    results = json.load(f)

baseline_vl = [10.8125, 10.75375, 10.716875, 10.63375, 10.4775, 10.199375, 9.7925, 9.376875]

fig, ax = plt.subplots(figsize=(9, 6))

colors = {
    0.75: "#a29bfe",
    1.6: "#00b894",
    1.7: "#0984e3",
    1.8: "#00cec9",
    1.9: "#e17055",
    2.0: "#d63031",
    2.1: "#fd79a8",
    2.2: "#6c5ce7",
    2.3: "#00b894",
    2.5: "#fdcb6e",
    3.5: "#636e72",
}

# Baseline
ax.plot(range(8), baseline_vl, "o-", color="#2d3436",
        linewidth=2.5, markersize=6, label=f'baseline (final=9.377)', zorder=10)

# All gains (sorted by gain value)
for r in sorted(results, key=lambda x: x["gain"]):
    g = r["gain"]
    steps = list(range(len(r["val_losses"])))
    color = colors.get(g, "black")
    ax.plot(steps, r["val_losses"], "s--", color=color, linewidth=1.5, alpha=0.8,
            label=f"gain={g} (final={r['final_vl']:.4f})")

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Validation Loss", fontsize=12)
ax.set_title("ToyConfig Fine Sweep: qk_gain around optimal (8 steps, 32k tokens)", fontsize=12)
ax.set_xticks(range(8))
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = "/Users/vukrosic/my-life/llm-research-kit-scaling/runs/toy/qk_gain_fine.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

# Print results sorted by final_vl
print(f"\n{'gain':<8} {'final_vl':>10} {'vs_baseline':>12} {'time':>8}")
print("-"*40)
sorted_res = sorted(results, key=lambda x: x["final_vl"])
for r in sorted_res:
    delta = r["final_vl"] - 9.3769
    sign = "+" if delta > 0 else ""
    print(f"{r['gain']:<8.2f} {r['final_vl']:>10.4f} {sign}{delta:.4f}   {r['wall']:.1f}s")