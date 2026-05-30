#!/usr/bin/env python3
"""Plot ToySmall 1M token sweep results."""

import json, matplotlib.pyplot as plt

with open("/tmp/toy_small_sweep.json") as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(10, 6))

colors = {0.0: "#aaaaaa", 1.0: "#4ecdc4", 2.0: "#e17055", 3.0: "#d63031", 4.0: "#e84393"}

for r in results:
    g = r["gain"]
    steps = list(range(len(r["val_losses"])))
    ax.plot(steps, r["val_losses"], "o-",
            color=colors.get(g, "black"), linewidth=2,
            label=f"gain={g} (final={r['final_vl']:.4f})")

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Validation Loss", fontsize=12)
ax.set_title("ToySmall (~6.7M params, 1M tokens): qk_gain sweep", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/Users/vukrosic/my-life/llm-research-kit-scaling/runs/toy_small/qk_gain_sweep.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

print(f"\n{'gain':<8} {'final_vl':>10}")
print("-"*20)
for r in sorted(results, key=lambda x: x["final_vl"]):
    print(f"{r['gain']:<8.1f} {r['final_vl']:>10.4f}")