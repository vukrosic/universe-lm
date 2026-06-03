"""Generate the figures for the Q/K/V-embeddings family tutorial (#29-#32).

Run: python docs/tutorials/qkv_embeddings/make_figures.py

All numbers are the seed=42 natural-end screen runs, taken from
docs/youtube-architecture-ablation-log.md (§10, §12, §13). The 4,000-step
control (5.0078) is the only control we have; the natural-end control is
still pending, so we never draw a delta-vs-control at step 4,882.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# ---- data (natural-end runs, seed=42) -------------------------------------
steps = [500, 1000, 4000, 4882]
vq   = [6.0992, 5.6015, 4.7875, 4.7428]   # V+Q  (#32)
v    = [6.4059, 5.8800, 4.9381, 4.7728]   # V    (#29)
q    = [6.1853, 5.6941, 4.8607, 4.8159]   # Q    (#30)  4k = milestone history
k    = [6.1641, 5.6813, 4.8722, 4.8228]   # K    (#31)
ctrl = [6.3972, 5.8853, 5.0078, None]     # control — natural-end is PENDING

BLUE, GREY, RED, GREEN = "#2563eb", "#9ca3af", "#dc2626", "#16a34a"
PURPLE, ORANGE = "#7c3aed", "#ea580c"

# ===========================================================================
# Figure 1 — the injection architecture (HOW)
# ===========================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.0))
ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 7)


def box(x, y, w, h, text, fc, ec="#374151", fs=9, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, text=None, color="#374151", dy=0.18, fs=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, color=color, lw=1.6))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, fontsize=fs,
                ha="center", color=color)


ax.text(5.5, 6.6, "One shared token embedding, injected into Q, K and V",
        ha="center", fontsize=12, weight="bold")

# shared source
box(0.2, 3.0, 1.6, 0.9, "token ids", "#fde68a")
box(2.2, 3.0, 2.3, 0.9, "ve  =  raw token emb\n(vocab × r=48, shared)", "#bfdbfe")
arrow(1.8, 3.45, 2.2, 3.45)

# the residual / main path label
ax.text(3.35, 2.55, "also feeds the residual stream (unchanged)",
        ha="center", fontsize=7.5, color="#555", style="italic")

# three zero-init projections
box(5.2, 5.1, 2.2, 0.8, "Wq  (zero-init)", "#ddd6fe", ec=PURPLE)
box(5.2, 3.05, 2.2, 0.8, "Wk  (zero-init)", "#ddd6fe", ec=ORANGE)
box(5.2, 1.0, 2.2, 0.8, "Wv  (zero-init)", "#ddd6fe", ec=BLUE)
arrow(4.5, 3.6, 5.2, 5.5, color=PURPLE)
arrow(4.5, 3.45, 5.2, 3.45, color=ORANGE)
arrow(4.5, 3.3, 5.2, 1.4, color=BLUE)

# injection points
box(8.0, 5.1, 2.6, 0.8, "Q += Wq·ve\n→ RoPE + norm", "#ede9fe", ec=PURPLE, fs=8.5)
box(8.0, 3.05, 2.6, 0.8, "K += Wk·ve\n→ RoPE + norm", "#fff7ed", ec=ORANGE, fs=8.5)
box(8.0, 1.0, 2.6, 0.8, "V += Wv·ve\n(raw — no RoPE)", "#eff6ff", ec=BLUE, fs=8.5)
arrow(7.4, 5.5, 8.0, 5.5, color=PURPLE)
arrow(7.4, 3.45, 8.0, 3.45, color=ORANGE)
arrow(7.4, 1.4, 8.0, 1.4, color=BLUE)

ax.text(5.5, 0.25,
        "Each projection starts at zero → step 0 is bit-for-bit the baseline. "
        "Total cost V+Q+K ≈ 276k params (~3.6%).",
        ha="center", fontsize=8, color="#555")
plt.tight_layout()
plt.savefig(f"{OUT}/architecture.png", dpi=140)
plt.close()

# ===========================================================================
# Figure 2 — what each position means (WHY)
# ===========================================================================
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4)
ax.text(5.5, 4.05, "Same signal, three jobs — what the token identity biases",
        ha="center", fontsize=12, weight="bold")

box(0.4, 2.2, 2.4, 1.0, "Q  (query)\n“what I look for”", "#ede9fe", ec=PURPLE)
box(4.3, 2.2, 2.4, 1.0, "K  (key)\n“what I advertise”", "#fff7ed", ec=ORANGE)
box(8.2, 2.2, 2.4, 1.0, "V  (value)\n“what I contribute”", "#eff6ff", ec=BLUE)

arrow(2.8, 2.7, 4.3, 2.7, "score = Q·Kᵀ", dy=0.16)
arrow(6.7, 2.7, 8.2, 2.7, "softmax → weights", dy=0.16)

ax.text(1.6, 1.7, "Q-embed: bias how\ntoken i looks others up", ha="center",
        fontsize=8, color=PURPLE)
ax.text(5.5, 1.7, "K-embed: bias how\ntoken j is matched", ha="center",
        fontsize=8, color=ORANGE)
ax.text(9.4, 1.7, "V-embed: bias what\ntoken j hands back", ha="center",
        fontsize=8, color=BLUE)

ax.text(5.5, 0.55,
        "A deep transformer works on a context-mixed hidden state — a token's raw "
        "identity gets diluted with depth.\nThese levers re-inject “I am token #X” at "
        "every layer, in whichever of the three slots you choose.",
        ha="center", fontsize=8.5, color="#374151")
plt.tight_layout()
plt.savefig(f"{OUT}/attention_roles.png", dpi=140)
plt.close()

# ===========================================================================
# Figure 3 — the family loss curves + crossover (WHAT WORKS)
# ===========================================================================
plt.figure(figsize=(9, 5.5))
# control only where we have it (no natural-end control yet)
cs = [s for s, c in zip(steps, ctrl) if c is not None]
cv = [c for c in ctrl if c is not None]
plt.plot(cs, cv, "o--", color=GREY, lw=2, label="control (4k only — natural end pending)")
plt.plot(steps, k,  "o-", color=ORANGE, lw=2, label="K-embed (#31)")
plt.plot(steps, q,  "o-", color=PURPLE, lw=2, label="Q-embed (#30)")
plt.plot(steps, v,  "o-", color=GREEN,  lw=2, label="V-embed (#29)")
plt.plot(steps, vq, "o-", color=BLUE,   lw=2.6, label="V+Q (#32)")

plt.annotate("4.7428", (4882, 4.7428), textcoords="offset points",
             xytext=(8, 2), fontsize=8, color=BLUE)
plt.annotate("K/Q lead\nthe warmup", (700, 5.95), fontsize=8, color="#555", ha="center")
plt.annotate("V (then V+Q)\nwin the endgame", (4400, 5.25), fontsize=8,
             color="#555", ha="center")

plt.xscale("log")
plt.xticks(steps, [str(s) for s in steps])
plt.xlabel("training step (log scale)")
plt.ylabel("validation loss")
plt.title("10M screen — token-identity-into-attention family (lower is better)")
plt.legend(loc="upper right", fontsize=8.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/family_curves.png", dpi=140)
plt.close()

# ===========================================================================
# Figure 4 — honest delta-vs-control, at the 4k tier (WHAT WORKS)
# ===========================================================================
c4k = 5.0078
runs   = ["V+Q (#32)", "Q (#30)", "K (#31)", "V (#29)"]
vals4k = [4.7875, 4.8607, 4.8722, 4.9381]
gains  = [c4k - x for x in vals4k]   # positive => better than control
colors = [BLUE, PURPLE, ORANGE, GREEN]

plt.figure(figsize=(8.5, 5))
bars = plt.bar(runs, gains, color=colors, width=0.62)
plt.axhline(0, color="k", lw=0.8)
plt.axhline(0.01, color=GREY, ls="--", lw=1)
plt.text(3.4, 0.02, "noise band (~0.01)", fontsize=8, color=GREY, ha="right")
for b, g in zip(bars, gains):
    plt.annotate(f"+{g:.4f}", (b.get_x() + b.get_width() / 2, g),
                 textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
plt.ylabel("gain vs control  (control − run, at step 4,000)")
plt.title("All four beat the 4k control — V+Q by the most (Q≈K, inside noise)")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/gain_vs_control_4k.png", dpi=140)
plt.close()

print("wrote 4 figures to", OUT)
