"""Generate the figures for the value-embeddings (#29) tutorial.

Run: python docs/tutorials/value_embeddings/make_figures.py
Data are the step-4000 schedule-matched 10M screen results (seed=42).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

steps = [500, 1000, 2000, 4000]
ve    = [6.4059, 5.8800, 5.3375, 4.9381]   # + value embeddings (#29)
ctrl  = [6.3972, 5.8853, 5.3856, 5.0078]   # control = emb-factor-depth champion
delta = [c - v for c, v in zip(ctrl, ve)]  # positive => value-embed is better

BLUE, GREY, RED, GREEN = "#2563eb", "#9ca3af", "#dc2626", "#16a34a"

# ---- Figure 1: validation-loss curves -------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps, ctrl, "o-", color=GREY, lw=2, label="control (emb-factor-depth)")
plt.plot(steps, ve, "o-", color=BLUE, lw=2, label="+ value embeddings (#29)")
for s, v in zip(steps, ve):
    plt.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                 xytext=(0, -15), fontsize=8, color=BLUE, ha="center")
plt.xscale("log")
plt.xticks(steps, [str(s) for s in steps])
plt.xlabel("training step (log scale)")
plt.ylabel("validation loss")
plt.title("10M screen — value embeddings vs control (lower is better)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/loss_curves.png", dpi=140)
plt.close()

# ---- Figure 2: the widening gap -------------------------------------------
plt.figure(figsize=(8, 5))
colors = [RED if d < 0 else GREEN for d in delta]
bars = plt.bar([str(s) for s in steps], delta, color=colors, width=0.6)
plt.axhline(0, color="k", lw=0.8)
plt.axhline(0.01, color=GREY, ls="--", lw=1)
plt.text(3.4, 0.013, "noise band (~0.01)", fontsize=8, color=GREY, ha="right")
for b, d in zip(bars, delta):
    plt.annotate(f"{d:+.4f}", (b.get_x() + b.get_width() / 2, d),
                 textcoords="offset points", xytext=(0, 6 if d >= 0 else -14),
                 ha="center", fontsize=9)
plt.xlabel("training step")
plt.ylabel("gain vs control  (control − value-embed)")
plt.title("The gap widens as the zero-init projection trains")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/delta_vs_control.png", dpi=140)
plt.close()

# ---- Figure 3: architecture schematic -------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.0))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)


def box(x, y, w, h, text, fc, ec="#374151"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(x1, y1, x2, y2, text=None, color="#374151", dy=0.16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, color=color, lw=1.5))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=8,
                ha="center", color=color)


ax.text(5, 5.6, "Value embeddings — feed token identity straight into the attention values",
        ha="center", fontsize=11, weight="bold")

box(0.2, 2.6, 1.5, 0.8, "token ids", "#fde68a")
box(2.2, 2.6, 2.2, 0.8, "token embedding\n(vocab × r = 48)", "#bfdbfe")
arrow(1.7, 3.0, 2.2, 3.0)

# main path (unchanged)
box(5.3, 4.1, 2.3, 0.8, "emb_proj → x\nresidual stream", "#bbf7d0")
arrow(4.4, 3.3, 5.3, 4.3, "main path (unchanged)")
box(8.0, 4.1, 1.7, 0.8, "attention\n(Q, K, V)", "#fecaca")
arrow(7.6, 4.5, 8.0, 4.5)

# value-embed path (new, blue)
box(5.3, 1.0, 2.3, 0.8, "W  (zero-init)\nvalue-embed proj", "#bfdbfe", ec=BLUE)
arrow(4.4, 2.7, 5.3, 1.5, "ve = same table, reused", color=BLUE)
box(7.9, 1.0, 1.9, 0.8, "V += W·ve", "#fecaca", ec=BLUE)
arrow(7.6, 1.4, 7.9, 1.4, color=BLUE)
arrow(8.85, 1.8, 8.85, 4.1, "values", color=BLUE)

ax.text(5, 0.25,
        "One embedding table feeds BOTH the residual stream and (via a zero-init W) the values — only ~55k extra params.",
        ha="center", fontsize=8, color="#555")
plt.tight_layout()
plt.savefig(f"{OUT}/architecture.png", dpi=140)
plt.close()

print("wrote 3 figures to", OUT)
