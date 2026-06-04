"""Figures for the Q-gain / K-gain lesson.

Concept plots (illustration, no training data):
  A. sharpness.png      — same scores, three temperatures: soft / baseline / sharp
  B. before_after.png   — every head same sharpness vs each head its own
  C. redundancy.png     — (1+a)(1+b): the ridge of (a,b) pairs that tie

Result plots (real run histories from runs/*/metrics.json):
  1. loss_curves.png    — val loss vs step, all six runs (full + tail zoom)
  2. final_val_bars.png — final val loss, alone family vs stacked family

Run from repo root:  python docs/tutorials/qk_gain/make_figures.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


# ============================================================ concept plots
# These use no training data. They build intuition before the results.

# ---- HERO: one head, one dial, soft -> focused -------------------------
# The simplest possible picture of the whole lesson.
hraw = np.array([2.0, 1.6, 1.2, 1.0, 0.7, 0.5, 0.3, 0.1])
hpos = np.arange(len(hraw))
figH, (axlo, axhi) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
axlo.bar(hpos, softmax(hraw * 0.4), color="#ff7f0e", width=0.7)
axlo.set_title("dial low  ->  spread out", fontsize=13)
axlo.set_xlabel("token position")
axlo.set_ylabel("attention")
axlo.set_ylim(0, 1)
axhi.bar(hpos, softmax(hraw * 2.4), color="#1f77b4", width=0.7)
axhi.set_title("dial high  ->  focused", fontsize=13)
axhi.set_xlabel("token position")
axhi.set_ylim(0, 1)
figH.suptitle("Q-gain adds one dial per head: turn attention from spread to focused",
              fontsize=14)
# arrow between the two panels
figH.text(0.5, 0.45, "→", ha="center", va="center", fontsize=40, color="#444")
figH.tight_layout(rect=(0, 0, 1, 0.93))
figH.savefig(os.path.join(OUT, "hero.png"), dpi=140)
print("wrote hero.png")

# ---- A: same scores, three temperatures --------------------------------
# One head, eight candidate positions, one fixed set of raw scores.
# Q-gain multiplies the scores by (1 + g). Watch the softmax change shape.
raw = np.array([2.0, 1.6, 1.2, 1.0, 0.7, 0.5, 0.3, 0.1])
pos = np.arange(len(raw))
settings = [(-0.6, "soft  (1+g = 0.4)", "#ff7f0e"),
            (0.0,  "baseline  (1+g = 1.0)", "#888888"),
            (1.2,  "sharp  (1+g = 2.2)", "#1f77b4")]

figA, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (g, title, c) in zip(axes, settings):
    probs = softmax(raw * (1 + g))
    ax.bar(pos, probs, color=c, width=0.7)
    ax.set_title(title)
    ax.set_xlabel("token position")
    ax.set_ylim(0, 1)
    ax.set_xticks(pos)
axes[0].set_ylabel("attention weight")
figA.suptitle("Same scores, one knob: (1 + g) sets how peaked the head looks",
              fontsize=12)
figA.tight_layout()
figA.savefig(os.path.join(OUT, "sharpness.png"), dpi=140)
print("wrote sharpness.png")

# ---- B: before vs after, across heads ----------------------------------
# "Concentration" = the largest attention weight a head puts on one token.
# Before: every head is forced to the same temperature -> same concentration.
# After: each head learns its own (1 + g) -> each picks its own concentration.
np.random.seed(0)
n_heads = 6
base_scores = [np.sort(np.random.uniform(0, 2, 8))[::-1] for _ in range(n_heads)]
before = [softmax(s * 1.0).max() for s in base_scores]                  # same temp
learned_g = np.array([-0.5, 1.4, 0.2, -0.2, 0.9, 1.8])                  # per head
after = [softmax(s * (1 + g)).max() for s, g in zip(base_scores, learned_g)]

figB, (axb, axa) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
axb.bar(range(n_heads), before, color="#888888", width=0.6)
axb.set_title("Before: one fixed temperature\nevery head equally peaked")
axb.set_xlabel("head"); axb.set_ylabel("concentration (max weight)")
axb.set_ylim(0, 1)
axa.bar(range(n_heads), after, color="#1f77b4", width=0.6)
axa.set_title("After Q-gain: each head sets its own\nsome sharp, some soft")
axa.set_xlabel("head")
axa.set_ylim(0, 1)
figB.tight_layout()
figB.savefig(os.path.join(OUT, "before_after.png"), dpi=140)
print("wrote before_after.png")

# ---- B2: the actual per-head shapes, before vs after -------------------
# Same six heads, drawn as their full attention curves (sorted, descending).
# Before: one shared temperature -> the curves bunch into one shape.
# After: each head's own (1 + g) -> the curves fan out, sharp to soft.
x8 = np.arange(8)
head_colors = plt.cm.viridis(np.linspace(0, 0.9, n_heads))
figB2, (b2l, b2r) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for i, s in enumerate(base_scores):
    b2l.plot(x8, np.sort(softmax(s * 1.0))[::-1], color=head_colors[i],
             marker="o", ms=3, label=f"head {i}")
    b2r.plot(x8, np.sort(softmax(s * (1 + learned_g[i])))[::-1],
             color=head_colors[i], marker="o", ms=3, label=f"head {i}")
b2l.set_title("Before: shared temperature\ncurves bunch into one shape")
b2l.set_xlabel("rank (1st, 2nd, ... attended token)")
b2l.set_ylabel("attention weight")
b2l.set_ylim(0, 1)
b2r.set_title("After Q-gain: per-head g\ncurves fan out, sharp to soft")
b2r.set_xlabel("rank (1st, 2nd, ... attended token)")
b2r.set_ylim(0, 1)
b2r.legend(frameon=False, fontsize=8, ncol=2)
figB2.tight_layout()
figB2.savefig(os.path.join(OUT, "before_after_shapes.png"), dpi=140)
print("wrote before_after_shapes.png")

# ---- C: the (1+a)(1+b) redundancy ridge --------------------------------
a = np.linspace(-0.5, 1.5, 200)
b = np.linspace(-0.5, 1.5, 200)
A, B = np.meshgrid(a, b)
prod = (1 + A) * (1 + B)

figC, ax = plt.subplots(figsize=(6.6, 5.4))
cf = ax.contourf(A, B, prod, levels=20, cmap="viridis")
# the iso-line where the product equals 1.56 — every point on it is the SAME model
ax.contour(A, B, prod, levels=[1.56], colors="white", linewidths=2.5)
pts = [(0.20, 0.30), (0.56, 0.00), (0.00, 0.56)]
ax.scatter(*zip(*pts), color="red", zorder=5, s=55)
for (pa, pb) in pts:
    ax.annotate(f"({pa},{pb})", (pa, pb), textcoords="offset points",
                xytext=(6, 6), color="white", fontsize=9)
ax.set_xlabel("a  (Q-gain)")
ax.set_ylabel("b  (K-gain)")
ax.set_title("(1+a)(1+b): the white line is one model\nthree red points = identical attention")
figC.colorbar(cf, ax=ax, label="score multiplier")
figC.tight_layout()
figC.savefig(os.path.join(OUT, "redundancy.png"), dpi=140)
print("wrote redundancy.png")


# ====================================================== result plots (real)
# name -> (run dir, color, leaderboard final val at 3-seed mean where it matters)
RUNS = {
    "control":  ("s_ctrl_full",  "#888888"),
    "q_gain":   ("s_qgain_full", "#1f77b4"),
    "k_gain":   ("s_kgain_full", "#ff7f0e"),
    "q+k":      ("s_qkgain_full", "#2ca02c"),
    "V+q":      ("s_vqgain_full", "#9467bd"),
    "V+q+k":    ("s_vqkgain_full", "#d62728"),
}


def load(run_dir):
    with open(os.path.join(ROOT, "runs", run_dir, "metrics.json")) as f:
        h = json.load(f)["history"]
    return h["steps"], h["val_losses"]


curves = {name: load(rd) for name, (rd, _) in RUNS.items()}
colors = {name: c for name, (_, c) in RUNS.items()}

# ---------------------------------------------------------------- figure 1
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

for name in RUNS:
    steps, vals = curves[name]
    axL.plot(steps, vals, color=colors[name], label=name, lw=1.8)
axL.set_title("Full training (all six runs)")
axL.set_xlabel("step")
axL.set_ylabel("val loss")
axL.legend(frameon=False)
axL.grid(alpha=0.25)

# tail zoom — where the runs actually separate
for name in RUNS:
    steps, vals = curves[name]
    pts = [(s, v) for s, v in zip(steps, vals) if s >= 2000]
    xs, ys = zip(*pts)
    axR.plot(xs, ys, color=colors[name], label=name, lw=2.0, marker="o", ms=3)
axR.set_title("Tail zoom (step >= 2000)")
axR.set_xlabel("step")
axR.set_ylabel("val loss")
axR.legend(frameon=False)
axR.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "loss_curves.png"), dpi=140)
print("wrote loss_curves.png")

# ---------------------------------------------------------------- figure 2
# Final val loss. The "alone" family is single-seed; the V+q / V+q+k bars
# carry their 3-seed leaderboard means so the anti-additivity is honest.
finals_seed42 = {name: curves[name][1][-1] for name in RUNS}
mean_3seed = {"V+q": 4.6815, "V+q+k": 4.6949}  # leaderboard 3-seed means

fig2, ax = plt.subplots(figsize=(9, 5))
order = ["control", "q_gain", "k_gain", "q+k", "V+q", "V+q+k"]
xs = range(len(order))
heights = [mean_3seed.get(n, finals_seed42[n]) for n in order]
bars = ax.bar(xs, heights, color=[colors[n] for n in order], width=0.62)

ctrl = finals_seed42["control"]
for x, n, h in zip(xs, order, heights):
    delta = h - ctrl
    tag = f"{h:.4f}\n({delta:+.4f})" if n != "control" else f"{h:.4f}"
    ax.text(x, h + 0.004, tag, ha="center", va="bottom", fontsize=9)

ax.axhline(ctrl, color="#888888", ls="--", lw=1, alpha=0.7)
ax.set_xticks(list(xs))
ax.set_xticklabels(order)
ax.set_ylabel("final val loss  (lower = better)")
ax.set_title("Final val loss vs control  (V+q / V+q+k = 3-seed mean)")
ax.set_ylim(min(heights) - 0.05, ctrl + 0.04)
ax.grid(axis="y", alpha=0.25)

fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "final_val_bars.png"), dpi=140)
print("wrote final_val_bars.png")
