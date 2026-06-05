"""3:4 portrait poster (Chinese) for the RoPE-base tutorial - for Douyin.

Run: python docs/tutorials/rope_base/make_poster_cn.py

Draws a compact version of the tiny-sweep curve. Intentionally omits external
call-to-action lines from README.cn.md.
Writes images/poster_cn.png at 1080x1440 (3:4).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "images")
CJK = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

BLUE, GREY, GREEN, INK = "#2563eb", "#64748b", "#16a34a", "#111827"
LIGHT, GREEN_LIGHT, BLUE_LIGHT = "#f8fafc", "#ecfdf5", "#eff6ff"
LINE = "#dbe3ee"

# 3:4 -> 1080x1440 at 240 dpi -> 4.5 x 6.0 inches
fig = plt.figure(figsize=(4.5, 6.0), dpi=240)
fig.patch.set_facecolor("white")


def text(x, y, s, size, color=INK, weight="normal", ha="left"):
    fig.text(x, y, s, fontproperties=CJK, fontsize=size, color=color,
             weight=weight, ha=ha, va="top")


def box(x, y, w, h, fc, ec=LINE, lw=0.9):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=fig.transFigure,
    )
    fig.add_artist(patch)


# Title
text(0.07, 0.955, "RoPE base", 18.5, INK, "bold")
text(0.07, 0.905, "不一定 10,000 最好", 18.5, INK, "bold")
text(0.07, 0.845, "一个 tiny sweep 里，250k 跑出了最低验证损失。", 9.2, GREY)
text(0.07, 0.816, "它不是新模块，只是一个可以顺手扫的整数旋钮。", 9.2, GREY)

# Chart
text(0.07, 0.765, "tiny sweep：约 1M 参数，训练 3M tokens", 9.3, INK, "bold")
ax = fig.add_axes([0.12, 0.465, 0.79, 0.27])
bases = [125, 250, 375, 500, 750]
losses = [6.3650, 6.3506, 6.3656, 6.3694, 6.3769]
ax.plot(bases, losses, color="#94a3b8", lw=2.3, zorder=1)
ax.scatter(bases, losses, s=54, color=BLUE, zorder=3)
ax.scatter([250], [6.3506], s=72, color=GREEN, zorder=4)
ax.annotate(
    "最低点 250k",
    xy=(250, 6.3506),
    xytext=(360, 6.3552),
    color=GREEN,
    fontsize=9,
    fontproperties=CJK,
    weight="bold",
    arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.6),
)
for x, y in zip(bases, losses):
    if x == 250:
        ax.text(x - 20, y + 0.0023, f"{y:.4f}", ha="right", va="bottom",
                fontsize=7.8, color=INK, fontweight="bold")
    else:
        ax.text(x, y + 0.0015, f"{y:.4f}", ha="center", va="bottom",
                fontsize=7.8, color=INK, fontweight="bold")
ax.set_xlim(90, 785)
ax.set_ylim(6.346, 6.3815)
ax.set_xticks(bases)
ax.set_xticklabels([f"{b}k" for b in bases], fontsize=8)
ax.set_yticks([6.35, 6.36, 6.37, 6.38])
ax.set_yticklabels(["6.35", "6.36", "6.37", "6.38"], fontsize=8)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#94a3b8")
ax.tick_params(axis="both", colors="#334155", length=3, pad=4)
ax.set_xlabel("RoPE base", fontsize=8, color=GREY, labelpad=5)
ax.set_ylabel("验证损失", fontsize=8, color=GREY, labelpad=5, fontproperties=CJK)

# Result strip
box(0.07, 0.300, 0.25, 0.08, LIGHT)
box(0.375, 0.300, 0.25, 0.08, GREEN_LIGHT, "#bbf7d0")
box(0.68, 0.300, 0.25, 0.08, LIGHT)
text(0.095, 0.362, "125k", 8.3, GREY, "bold")
text(0.095, 0.334, "6.3650", 12.2, INK, "bold")
text(0.40, 0.362, "250k", 8.3, GREEN, "bold")
text(0.40, 0.334, "6.3506", 12.2, GREEN, "bold")
text(0.70, 0.362, "750k", 8.3, GREY, "bold")
text(0.70, 0.334, "6.3769", 12.2, INK, "bold")

# Takeaways
box(0.07, 0.118, 0.86, 0.14, BLUE_LIGHT, "#bfdbfe")
text(0.10, 0.231, "结论", 9.5, BLUE, "bold")
text(0.10, 0.199, "RoPE base 只是一个整数。", 11.2, INK, "bold")
text(0.10, 0.163, "不增加参数，不改变 tensor shape，", 9.4, INK)
text(0.10, 0.135, "但可能移动验证损失。", 9.4, INK)

fig.savefig(os.path.join(OUT, "poster_cn.png"),
            facecolor="white", bbox_inches=None, pad_inches=0)
print("wrote", os.path.join(OUT, "poster_cn.png"))
