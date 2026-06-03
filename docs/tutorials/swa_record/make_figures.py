"""Waterfall of val loss vs method — the winning lineage (screen20m, natural end).

Run: python docs/tutorials/swa_record/make_figures.py

All numbers are seed=42 natural-end screen runs (step 4,883) from LEADERBOARD.md.
This is the chain with the most cumulative drops, ending at the SWA record (#51).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# ---- the winning lineage (screen20m, natural end, seed 42) ----------------
labels = ["control", "+ V-embed", "+ q_gain", "+ SWA\n(window=512)"]
vals   = [4.7984,     4.7728,      4.6797,     4.6700]
refs   = ["baseline", "#29",       "#39",      "#51  ← record"]

deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]

BLUE, GREY, GREEN = "#2563eb", "#9ca3af", "#16a34a"

fig, ax = plt.subplots(figsize=(9.5, 5.6))

x = range(len(vals))
bar_colors = [GREY, BLUE, BLUE, GREEN]
bars = ax.bar(x, vals, width=0.62, color=bar_colors, zorder=3,
              edgecolor="white", linewidth=1.2)

# zoom y so the drops are visible
ax.set_ylim(4.62, 4.83)
ax.set_yticks([4.65, 4.70, 4.75, 4.80])

# value label on each bar
for xi, v, r in zip(x, vals, refs):
    ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom",
            fontsize=11, weight="bold")
    ax.text(xi, 4.625, r, ha="center", va="bottom", fontsize=8.5,
            color="#555")

# drop arrows + delta labels between bars
for i, d in enumerate(deltas):
    x0, x1 = i, i + 1
    y0, y1 = vals[i], vals[i + 1]
    ax.annotate("", xy=(x1, y1), xytext=(x1, y0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
    ax.text(x1 + 0.34, (y0 + y1) / 2, f"{d:+.4f}",
            ha="left", va="center", fontsize=10, color=GREEN, weight="bold")
    # faint guide line carrying previous level across
    ax.plot([x0, x1], [y0, y0], color="#cbd5e1", lw=1.0, ls="--", zorder=2)

total = vals[-1] - vals[0]
ax.set_title("How each method dropped val loss — the winning lineage",
             fontsize=14, weight="bold", pad=14)
ax.text(0.5, 4.815, f"10M params · 20M tokens · seed 42 · cumulative {total:+.4f}",
        fontsize=10, color="#444")

ax.set_ylabel("validation loss  (lower is better)", fontsize=11)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eee", zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
path = os.path.join(OUT, "val_loss_waterfall.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("wrote", path)
