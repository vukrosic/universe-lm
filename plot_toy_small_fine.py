#!/usr/bin/env python3
"""Plot ToySmall fine sweep + combined with prior results."""

import json, matplotlib.pyplot as plt

with open("/tmp/toy_small_fine.json") as f:
    fine = json.load(f)
with open("/tmp/toy_small_sweep.json") as f:
    prior = json.load(f)

baseline_vl = [7.6009, 7.1887, 6.9909, 6.8506, 6.7809, 6.7075, 6.6616, 6.6003, 6.5722, 6.5403, 6.5172, 6.4813]

# Combined all gains
all_gains = {}
for r in prior:
    all_gains[r["gain"]] = r
for r in fine:
    all_gains[r["gain"]] = r

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    0.0: "#aaaaaa", 1.0: "#74b9ff", 2.0: "#fdcb6e", 2.5: "#e17055",
    3.0: "#d63031", 3.5: "#fd79a8", 4.0: "#e84393", 4.5: "#a29bfe",
    5.5: "#00b894", 6.5: "#636e72",
}

for g, r in sorted(all_gains.items()):
    ax.plot(range(len(r["val_losses"])), r["val_losses"], "o-",
            color=colors.get(g, "black"), linewidth=2, markersize=5,
            label=f"gain={g} (final={r['final_vl']:.4f})")

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Validation Loss", fontsize=12)
ax.set_title("ToySmallConfig (6.7M, 1M tokens): fine gain sweep", fontsize=13)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/vukrosic/my-life/llm-research-kit-scaling/runs/toy_small/qk_gain_fine.png", dpi=150)
print("Saved: runs/toy_small/qk_gain_fine.png")

print(f"\n{'gain':<8} {'final_vl':>10}")
print("-"*20)
for g, r in sorted(all_gains.items(), key=lambda x: x[1]["final_vl"]):
    print(f"{g:<8.1f} {r['final_vl']:>10.4f}")