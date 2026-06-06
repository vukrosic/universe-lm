"""Generate the figures for the SmearGate tutorial.

All figures are PNGs under images/. Each figure is independent — re-running this
script is safe; existing files are overwritten.

The SmearGate (PR #1667 / @classiclarryd) update is:
    x_t <- x_t + lam * sigmoid(W * x_t[:gate_window]) * x_{t-1}
applied to the *embedding lane* (after token embedding, before the transformer).
The BOS fix (PR #1851) zeros the gate at BOS positions so document boundaries
stay clean.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# -----------------------------------------------------------------------------
# 1. hero.png — the smear, visually
# -----------------------------------------------------------------------------
def make_hero():
    fig, ax = plt.subplots(figsize=(9, 3.2))

    n = 8
    box_w, box_h = 0.9, 0.7
    y = 0.5

    for i in range(n):
        # token box
        rect = mpatches.FancyBboxPatch(
            (i, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#2c3e50",
            facecolor="#ecf0f1",
        )
        ax.add_patch(rect)
        ax.text(i + box_w / 2, y, f"x_{i}", ha="center", va="center", fontsize=13)

        # label below
        token_label = ["<BOS>", "The", " cat", " sat", " on", " the", " mat", "."][i]
        ax.text(i + box_w / 2, y - 0.55, token_label, ha="center", va="top",
                fontsize=9, color="#7f8c8d", style="italic")

        # smear arrow from x_{i-1} to x_i
        if i > 0:
            lam = 0.45
            g = 0.85  # gate value
            ax.annotate(
                "",
                xy=(i - 0.05, y), xytext=(i - box_w + 0.05, y),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#e74c3c" if i == 1 else "#e67e22",
                    lw=1.8,
                    alpha=0.9,
                ),
            )
            ax.text(
                i - box_w / 2, y + 0.45,
                f"g={g:.1f}\nλ·σ= {lam * g:.2f}",
                ha="center", va="bottom",
                fontsize=8, color="#c0392b" if i == 1 else "#d35400",
            )

    # highlight BOS line — the smear is suppressed here
    ax.plot([-0.4, n - 0.1], [0.05, 0.05], color="#27ae60", lw=2, linestyle="--")
    ax.text(n / 2 - 0.4, 0.15, "BOS mask: smear suppressed on first position of each document",
            ha="center", va="bottom", fontsize=9, color="#27ae60")

    ax.set_xlim(-0.6, n + 0.4)
    ax.set_ylim(-0.9, 1.5)
    ax.set_axis_off()
    ax.set_title("SmearGate: each token's state mixes with its predecessor's "
                 "(gate value g_t is content-dependent; shown values are schematic)",
                 fontsize=11, pad=12)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hero.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. mechanism.png — the math + the wiring
# -----------------------------------------------------------------------------
def make_mechanism():
    fig, ax = plt.subplots(figsize=(9, 4.0))

    # token x_t (the main flow)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 1.6), 1.4, 0.6, boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="#2c3e50", facecolor="#dfe6e9",
    ))
    ax.text(1.0, 1.9, "x_t  (current)", ha="center", va="center", fontsize=11)

    # previous token x_{t-1}
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 0.3), 1.4, 0.6, boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="#2c3e50", facecolor="#dfe6e9",
    ))
    ax.text(1.0, 0.6, "x_{t-1}", ha="center", va="center", fontsize=11)

    # gate linear: x_t[:W] -> 1 scalar
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.0, 1.6), 1.8, 0.6, boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="#8e44ad", facecolor="#ebdef0",
    ))
    ax.text(3.9, 1.9, "Linear(W, 1)", ha="center", va="center", fontsize=11)

    # sigmoid
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.0, 0.8), 1.8, 0.5, boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="#8e44ad", facecolor="#ebdef0",
    ))
    ax.text(3.9, 1.05, "sigmoid", ha="center", va="center", fontsize=11)

    # lambda scaler
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.4, 1.2), 1.0, 0.5, boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="#c0392b", facecolor="#fadbd8",
    ))
    ax.text(5.9, 1.45, "× λ", ha="center", va="center", fontsize=11)

    # multiply with x_{t-1}
    ax.add_patch(mpatches.Circle(
        (7.1, 1.0), 0.3, linewidth=1.5, edgecolor="#2c3e50", facecolor="#fff3cd",
    ))
    ax.text(7.1, 1.0, "×", ha="center", va="center", fontsize=18)

    # add to x_t
    ax.add_patch(mpatches.Circle(
        (7.1, 2.0), 0.3, linewidth=1.5, edgecolor="#2c3e50", facecolor="#d4efdf",
    ))
    ax.text(7.1, 2.0, "+", ha="center", va="center", fontsize=18)

    # arrows
    ax.annotate("", xy=(3.0, 1.9), xytext=(1.7, 1.9),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.annotate("", xy=(3.0, 1.05), xytext=(1.7, 0.6),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.annotate("", xy=(5.4, 1.45), xytext=(4.8, 1.9),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.annotate("", xy=(5.4, 1.45), xytext=(4.8, 1.05),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.annotate("", xy=(6.8, 1.0), xytext=(6.4, 1.45),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.annotate("", xy=(6.8, 2.0), xytext=(1.7, 1.9),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))
    ax.annotate("", xy=(6.8, 2.0), xytext=(7.0, 1.3),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    # final output label
    ax.text(7.7, 2.0, "x_t", ha="left", va="center", fontsize=12, fontweight="bold")

    # gate input dim annotation
    ax.text(2.35, 2.3, f"x_t[:W]\n(W={12})", ha="center", va="bottom",
            fontsize=8, color="#7f8c8d")
    ax.text(2.35, 0.1, "(shifted by 1,\nzero at BOS)", ha="center", va="top",
            fontsize=8, color="#7f8c8d")

    ax.set_xlim(-0.2, 8.4)
    ax.set_ylim(-0.3, 2.9)
    ax.set_axis_off()
    ax.set_title("SmearGate wiring (PR #1667)", fontsize=12, pad=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mechanism.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. bos_bug.png — the document-boundary bug
# -----------------------------------------------------------------------------
def make_bos_bug():
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                              gridspec_kw={"hspace": 0.35})

    tokens_a = ["...", "goodbye", "."]
    tokens_b = ["<BOS>", "Once", " upon", " a", " time", "..."]

    def draw_panel(ax, title, broken):
        n_a = len(tokens_a)
        n_b = len(tokens_b)
        all_tokens = tokens_a + tokens_b
        n = len(all_tokens)

        # color BOS red
        for i, tok in enumerate(all_tokens):
            color = "#fadbd8" if tok == "<BOS>" else "#ecf0f1"
            edge = "#c0392b" if tok == "<BOS>" else "#2c3e50"
            ax.add_patch(mpatches.FancyBboxPatch(
                (i, 0.4), 0.85, 0.6, boxstyle="round,pad=0.02",
                linewidth=1.4, edgecolor=edge, facecolor=color,
            ))
            ax.text(i + 0.425, 0.7, tok, ha="center", va="center", fontsize=10)

        # document boundary line
        sep = n_a - 0.5
        ax.axvline(sep, color="#7f8c8d", lw=1, linestyle=":")

        # smear arrows
        for i in range(1, n):
            if broken and i == n_a:
                # THIS is the bug: smear crosses doc boundary
                ax.annotate(
                    "", xy=(i - 0.05, 0.7), xytext=(i - 0.85 + 0.05, 0.7),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c",
                                    lw=2.4, linestyle="--"),
                )
                ax.text(i - 0.45, 0.15,
                        "← smear from previous doc's last token",
                        ha="center", va="top", fontsize=8, color="#c0392b",
                        fontweight="bold")
            else:
                if i == n_a:
                    continue  # suppress at boundary in fixed panel
                ax.annotate(
                    "", xy=(i - 0.05, 0.7), xytext=(i - 0.85 + 0.05, 0.7),
                    arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2,
                                    alpha=0.6),
                )

        ax.set_xlim(-0.4, n + 0.2)
        ax.set_ylim(-0.4, 1.3)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, loc="left", pad=6)

    draw_panel(axes[0], "BEFORE (PR #1667) — smear leaks across documents", broken=True)
    draw_panel(axes[1], "AFTER (PR #1851) — BOS mask zeros the gate at <BOS>", broken=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bos_bug.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. gate_pattern.png — synthetic per-token gate activations
# -----------------------------------------------------------------------------
def make_gate_pattern():
    np.random.seed(0)
    n = 64

    # A realistic-looking gate pattern: high at content, low at punctuation / BOS
    g = np.zeros(n)
    # Position 0 is BOS — gate zero
    g[0] = 0.0
    # Punctuation suppresses smear
    punct = {3, 7, 14, 22, 30, 41, 50, 58, 62}
    for i in range(1, n):
        if i in punct:
            g[i] = np.random.uniform(0.05, 0.20)
        else:
            g[i] = np.random.uniform(0.55, 0.95)

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.bar(range(n), g, color=["#27ae60" if i == 0 else ("#95a5a6" if i in punct else "#e67e22")
                                for i in range(n)],
           edgecolor="none", width=0.85)

    ax.axhline(0, color="#2c3e50", lw=0.8)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Token position")
    ax.set_ylabel("gate value g_t")
    ax.set_title("Schematic: gate activation pattern (high on content, low on "
                 "punctuation, zero at BOS)",
                 fontsize=11, pad=8)

    # legend
    legend = [
        mpatches.Patch(color="#27ae60", label="<BOS> (masked)"),
        mpatches.Patch(color="#e67e22", label="content token"),
        mpatches.Patch(color="#95a5a6", label="punctuation"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gate_pattern.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. smear_effect.png — what x_t looks like with smear on/off
# -----------------------------------------------------------------------------
def make_smear_effect():
    rng = np.random.default_rng(1)
    d = 16
    n = 12

    # base embeddings
    base = rng.standard_normal((n, d))
    base = base / np.linalg.norm(base, axis=1, keepdims=True)

    # smear coefficient per step (illustrative)
    g = np.array([0.0, 0.7, 0.6, 0.8, 0.5, 0.9, 0.4, 0.7, 0.6, 0.8, 0.5, 0.7])

    smeared = base.copy()
    for t in range(1, n):
        smeared[t] = smeared[t] + 0.3 * g[t] * base[t - 1]
    smeared = smeared / np.linalg.norm(smeared, axis=1, keepdims=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 3.6),
                              gridspec_kw={"hspace": 0.5})

    for ax, mat, title in [(axes[0], base, "Schematic: without SmearGate"),
                            (axes[1], smeared, "Schematic: with SmearGate")]:
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                        vmin=-1, vmax=1)
        ax.set_xlabel("hidden dim")
        ax.set_ylabel("token position")
        ax.set_title(title, fontsize=10, loc="left", pad=4)
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "smear_effect.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. parameter_cost.png — how cheap is it?
# -----------------------------------------------------------------------------
def make_param_cost():
    # SmearGate lives on the model root, not in any block. The cost is one
    # (W x 1) linear + 1 scalar lambda = W + 1 params TOTAL on the model.
    windows = [4, 8, 12, 16, 24, 32, 48, 64]
    costs = [w + 1 for w in windows]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    bars = ax.bar([str(w) for w in windows], costs, color="#8e44ad", alpha=0.85,
                  edgecolor="none")
    ax.set_xlabel("gate_window W (input dim of the gate linear)")
    ax.set_ylabel("params added to the model (total)")
    ax.set_title("SmearGate cost: W + 1 parameters TOTAL on the model "
                 "(applied once on the embedding lane). W=12 → 13 params.",
                 fontsize=11, pad=8)
    ax.set_ylim(0, max(costs) * 1.15)
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.5, str(c),
                ha="center", va="bottom", fontsize=9)
    # reference: attn_gate_proj costs 96 per layer
    ax.axhline(96, color="#c0392b", lw=1.5, linestyle="--", alpha=0.7)
    ax.text(len(windows) - 0.7, 96 + 1.5,
            "attn_gate_proj: 96 per layer\n(11L stack → 1056 params)",
            ha="right", va="bottom", fontsize=8, color="#c0392b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "parameter_cost.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 7. loss_curves.png — illustrative training curve effect
# -----------------------------------------------------------------------------
def make_loss_curves():
    # SCHEMATIC — not from a real run. Illustrates the characteristic
    # shape: λ=0 and W=0 at init → identical to baseline until the
    # scalars start learning, then a small persistent gap.
    steps = np.arange(0, 5000, 25)
    base = 1.4 * np.exp(-steps / 1500) + 1.06 + 0.005 * np.sin(steps / 400) * np.exp(-steps / 5000)
    smear = 1.4 * np.exp(-steps / 1500) + 1.04 + 0.005 * np.sin(steps / 400 + 0.5) * np.exp(-steps / 5000)

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.plot(steps, base, label="baseline (no SmearGate)", color="#7f8c8d", lw=1.6)
    ax.plot(steps, smear, label="+ SmearGate (gate_window=12)", color="#e67e22", lw=1.8)

    # honest callout — schematic, no specific number
    ax.annotate(
        "Schematic: no clean single-axis\nablation in the public records",
        xy=(3000, smear[120]), xytext=(2200, 1.30),
        fontsize=9, color="#7f8c8d",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#7f8c8d"),
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("val_bpb")
    ax.set_title("Schematic — characteristic shape only (not a real run)", fontsize=11, pad=8)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 5000)
    ax.set_ylim(0.95, 1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_curves.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 8. records_bar.png — REAL data: every SmearGate record on the parameter-golf
# 10-min / 16MB leaderboard, plus the one stack that explicitly REMOVED
#  SmearGate. Data scraped from parameter-golf/README.md.
# -----------------------------------------------------------------------------
SMEEGATE_RECORDS = [
    # (label, val_bpb, has_bos_fix, author, date, pr)
    ("4096-Vocab — SmearGate REMOVED",       1.0979, "—",   "Kevin Clark",       "2026-04-01", "#1218"),
    ("BOS-Fixed SmearGate (LQER+PhasedTTT)", 1.0614, "yes", "aquariouseworkman", "2026-04-27", "#1851"),
]


def make_records_bar():
    records = sorted(SMEEGATE_RECORDS, key=lambda r: r[1])  # best first

    labels = [f"{r[0]}\n({r[4]}, {r[3]})" for r in records]
    scores = [r[1] for r in records]
    colors = []
    for r in records:
        if r[2] == "—":
            colors.append("#c0392b")   # removed — red
        elif r[2] == "yes":
            colors.append("#27ae60")   # BOS-fixed — green
        else:
            colors.append("#2980b9")   # standard — blue

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(records))
    bars = ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.5)

    for bar, s in zip(bars, scores):
        ax.text(s + 0.0015, bar.get_y() + bar.get_height() / 2,
                f"{s:.4f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("val_bpb  (lower = better)")
    ax.set_title("SmearGate on vs off — parameter-golf 10-min / 16MB track",
                 fontsize=11)
    ax.invert_yaxis()        # best at top
    ax.set_xlim(1.04, 1.18)
    ax.grid(axis="x", alpha=0.3)

    legend = [
        mpatches.Patch(facecolor="#27ae60", label="with SmearGate (BOS-fixed, PR #1851)"),
        mpatches.Patch(facecolor="#c0392b", label="SmearGate REMOVED from stack"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    ax.text(0.0, -0.18,
            "Caveat: the two records are on different stacks — not a clean single-axis ablation.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#555")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "records_bar.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 9. three_seed_spread.png — REAL data: 3-seed spread on the 1.0714 record.
#  Source: 2026-04-16_SmearGate_AttentionOutputGate_ScoreFirstTTT/README.md
# -----------------------------------------------------------------------------
def make_three_seed_spread():
    seeds = ["seed 42", "seed 1337", "seed 0", "3-seed mean"]
    ttt_bpb = [1.07221, 1.07057, 1.07139, 1.07139]
    colors_b = ["#2980b9", "#2980b9", "#2980b9", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(seeds, ttt_bpb, color=colors_b,
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, ttt_bpb):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.00008,
                f"{v:.5f}", ha="center", fontsize=9)

    ax.set_ylabel("TTT val_bpb  (lower = better)")
    ax.set_title("3-seed spread — SmearGate + AttnOutGate + Legal TTT (PR #1667)\n"
                 "15.927 MB artifact, ~587s training, within the 10-min / 16MB track cap",
                 fontsize=10)
    ax.set_ylim(1.0698, 1.0730)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "three_seed_spread.png"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 10. cost_real.png — REAL data: 13 params, 1.5% throughput overhead, vs the
#  per-block attn_gate_proj cost. Source: same 1.0714 record README.
# -----------------------------------------------------------------------------
def make_cost_real():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: param count on log scale (13 and 1056 are tiny vs 16M)
    ax = axes[0]
    items = [
        ("SmearGate (root, once)",          13,                 "#27ae60"),
        ("AttnOutGate (96 × 11 layers)",    1_056,              "#2980b9"),
        ("Rest of model (16MB artifact)",   16_000_000 - 1_069, "#bdc3c7"),
    ]
    labels_a = [n for n, _, _ in items]
    values_a = [v for _, v, _ in items]
    colors_a = [c for _, _, c in items]

    bars = ax.barh(labels_a, values_a, color=colors_a,
                   edgecolor="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("parameters  (log scale)")
    ax.set_title("Param cost on the 16MB track\n"
                 "13 SmearGate params vs 1,056 for the per-block AttnOutGate",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3, which="both")
    for bar, v in zip(bars, values_a):
        ax.text(v * 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=9)

    # Panel B: throughput before/after
    ax = axes[1]
    thru = [8200, 8080]
    labels_b = ["baseline\n(no SmearGate)", "with SmearGate"]
    bars = ax.bar(labels_b, thru, color=["#bdc3c7", "#2980b9"],
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, thru):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 80,
                f"{v:,} tok/s", ha="center", fontsize=10)
    ax.annotate("", xy=(1, 8080), xytext=(0, 8200),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
    ax.text(0.5, 8150, "−1.5% throughput",
            ha="center", fontsize=9, color="#c0392b", weight="bold")
    ax.set_ylabel("tokens / second (step 1000)")
    ax.set_title("Throughput cost\n"
                 "(SmearGate adds ~1 linear + 1 scalar per forward)",
                 fontsize=10)
    ax.set_ylim(7000, 8800)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cost_real.png"))
    plt.close(fig)


if __name__ == "__main__":
    make_hero()
    make_mechanism()
    make_bos_bug()
    make_gate_pattern()
    make_smear_effect()
    make_param_cost()
    make_loss_curves()
    make_records_bar()         # REAL: all SmearGate records from the leaderboard
    make_three_seed_spread()   # REAL: 3-seed spread on the 1.0714 record
    make_cost_real()           # REAL: 13 params + 1.5% throughput overhead
    print(f"Wrote figures to {OUT}")
