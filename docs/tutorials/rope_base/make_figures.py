"""Figures for the RoPE-base tutorial.

Run: python docs/tutorials/rope_base/make_figures.py

Writes 5 PNGs into images/. All run numbers are seed=42 final val_loss read
from results/tiny1m_0604/*.json (final_metrics.val_loss). Only two base
values are grounded in runs: the ~1M tier was fully swept (optimum 250k) and
the ~10M best recipe used 500k (a single recipe, NOT a sweep). No 10M base
sweep exists yet, so fig_scales plots best-base-vs-scale rather than a
fabricated 10M loss U-curve.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "images")
RES = os.path.join(HERE, "..", "..", "..", "results", "tiny1m_0604")
os.makedirs(OUT, exist_ok=True)

BLUE, GREY, GREEN, RED, ORANGE = "#2563eb", "#9ca3af", "#16a34a", "#dc2626", "#ea580c"


def load(name):
    """Return (final_val_loss, history_dict) for a results JSON stem."""
    d = json.load(open(os.path.join(RES, name + ".json")))
    return d["final_metrics"]["val_loss"], d.get("history", {})


# ---------------------------------------------------------------------------
# Data: tiny tier sweep (V-embed + q-gain + SWA512), seed 42
# ---------------------------------------------------------------------------
TINY = [
    (125_000, "tiny1m_vqgain_swa_rope125k_full"),
    (250_000, "tiny1m_vqgain_swa_rope250k_full"),   # winner
    (375_000, "tiny1m_vqgain_swa_rope375k_full"),
    (500_000, "tiny1m_vqgain_swa_highrope_full"),
    (750_000, "tiny1m_vqgain_swa_rope750k_full"),
]
tiny_bases = [b for b, _ in TINY]
tiny_vals = [load(n)[0] for _, n in TINY]
tiny_hist = {b: load(n)[1] for b, n in TINY}
tiny_win = int(np.argmin(tiny_vals))


# ===========================================================================
# FIG 1 — hero U-curve (tiny tier)
# ===========================================================================
def fig_hero():
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    x = np.arange(len(tiny_bases))
    colors = [GREEN if i == tiny_win else BLUE for i in range(len(x))]
    ax.plot(x, tiny_vals, color="#94a3b8", lw=2, zorder=2)
    ax.scatter(x, tiny_vals, s=180, c=colors, zorder=3,
               edgecolor="white", linewidth=1.5)

    for xi, v in zip(x, tiny_vals):
        ax.text(xi, v + 0.0016, f"{v:.4f}", ha="center", va="bottom",
                fontsize=10.5, weight="bold")
    bw = tiny_bases[tiny_win]
    ax.annotate(f"optimum = {bw//1000}k",
                xy=(tiny_win, tiny_vals[tiny_win]),
                xytext=(tiny_win + 0.7, tiny_vals[tiny_win] + 0.006),
                fontsize=11, color=GREEN, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b//1000}k" for b in tiny_bases], fontsize=11)
    ax.set_xlabel("RoPE base", fontsize=11)
    ax.set_ylabel("validation loss  (lower is better)", fontsize=11)
    ax.set_title("RoPE base forms a U-curve with an interior optimum",
                 fontsize=14, weight="bold", pad=12)
    ax.text(0.02, 0.04, "tiny tier · 0.94M params · 3M tokens · seed 42 · "
            "V-embed + q-gain + SWA(512)",
            transform=ax.transAxes, fontsize=9.5, color="#555")
    ax.set_ylim(min(tiny_vals) - 0.004, max(tiny_vals) + 0.010)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee")
    ax.set_axisbelow(True)
    _save(fig, "hero_ucurve.png")


# ===========================================================================
# FIG 2 — training loss curves (tiny sweep), with tail-zoom inset
# ===========================================================================
def fig_curves():
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(tiny_bases)))
    for (b, c) in zip(tiny_bases, cmap):
        h = tiny_hist[b]
        steps, vl = h["steps"], h["val_losses"]
        lw = 2.6 if b == tiny_bases[tiny_win] else 1.6
        ax.plot(steps, vl, color=c, lw=lw,
                label=f"{b//1000}k" + ("  ← best" if b == tiny_bases[tiny_win] else ""))
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("validation loss", fontsize=11)
    ax.set_title("Same training, different RoPE base — separation appears at the tail",
                 fontsize=13.5, weight="bold", pad=12)
    ax.legend(title="RoPE base", fontsize=9.5, title_fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eee")
    ax.set_axisbelow(True)

    # tail-zoom inset
    axin = ax.inset_axes([0.52, 0.45, 0.45, 0.48])
    for (b, c) in zip(tiny_bases, cmap):
        h = tiny_hist[b]
        steps, vl = np.array(h["steps"]), np.array(h["val_losses"])
        m = steps >= 400
        lw = 2.6 if b == tiny_bases[tiny_win] else 1.6
        axin.plot(steps[m], vl[m], color=c, lw=lw, marker="o", ms=4)
    axin.set_title("final-step zoom", fontsize=9)
    axin.tick_params(labelsize=8)
    axin.grid(color="#eee")
    axin.set_axisbelow(True)
    ax.indicate_inset_zoom(axin, edgecolor="#bbb")
    _save(fig, "loss_curves.png")


# ===========================================================================
# FIG 3 — optimum scales with model size (tiny vs screen20m)
# ===========================================================================
def fig_scales():
    # Honest: only TWO base values are grounded in runs.
    #   ~1M params : full sweep → optimum 250k (real U-curve, see hero fig)
    #   ~10M params: single best recipe used 500k (NOT swept — sweep pending)
    # Plot best-base-vs-scale, not fabricated loss U-curves.
    params = [0.94, 10.0]          # millions
    best = [250_000, 500_000]
    swept = [True, False]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(params, best, color="#94a3b8", lw=2, ls="--", zorder=2)
    for p, b, s in zip(params, best, swept):
        ax.scatter([p], [b], s=240, zorder=3,
                   c=(GREEN if s else "white"), edgecolor=GREEN, linewidth=2.4)
        tag = "swept optimum" if s else "single recipe\n(sweep pending)"
        ax.annotate(f"{b//1000}k", (p, b), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    fontsize=12, weight="bold")
        ax.annotate(tag, (p, b), xytext=(0, -34),
                    textcoords="offset points", ha="center",
                    fontsize=9, color="#555")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 18)
    ax.set_ylim(1.6e5, 8e5)
    ax.set_xticks(params)
    ax.set_xticklabels(["~1M", "~10M"], fontsize=11)
    ax.set_yticks([250_000, 500_000])
    ax.set_yticklabels(["250k", "500k"], fontsize=11)
    ax.set_xlabel("model size", fontsize=11)
    ax.set_ylabel("best RoPE base", fontsize=11)
    ax.set_title("Best observed base rises with scale — but the 10M point is one recipe,\n"
                 "not a sweep",
                 fontsize=13, weight="bold", pad=12)
    ax.text(0.5, -0.2,
            "1M: full base sweep (real U-curve). 10M: best single-seed recipe "
            "used 500k; a full 10M base sweep is still needed.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#888")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eee", which="both")
    ax.set_axisbelow(True)
    _save(fig, "optimum_scales.png")


# ===========================================================================
# FIG 4 — mechanism: rotation wavelength per dimension, base 10k vs 500k
# ===========================================================================
def fig_mechanism():
    d_k = 24
    seq_len = 2048
    i = np.arange(0, d_k // 2)            # 12 frequency pairs
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for base, c, lab in [(10_000, RED, "base = 10k"),
                         (500_000, GREEN, "base = 500k")]:
        theta = base ** (-2 * i / d_k)        # θ_0 = 1 for ANY base
        period = 2 * np.pi / theta            # tokens per full rotation
        n_clean = int((period > seq_len).sum())
        ax.plot(i, period, color=c, lw=2.4, marker="o", ms=6,
                label=f"{lab}  ({n_clean}/12 non-wrapping)")

    # shade the region where a dim completes >1 full turn across the context
    # (period < seq_len) — those dims alias, so they can't cleanly encode
    # long-range distance.
    ax.axhspan(1, seq_len, color="#fee2e2", alpha=0.6, zorder=0)
    ax.axhline(seq_len, color=BLUE, lw=1.4, ls="--")
    ax.text(0.1, seq_len * 0.62, f"period < context ({seq_len}): wraps / aliases",
            color="#b91c1c", fontsize=9, weight="bold")
    ax.text(0.1, seq_len * 1.5, "period > context: non-wrapping → clean long-range",
            color=GREEN, fontsize=9, weight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("rotation dimension  i  (0 = fastest, 11 = slowest)", fontsize=11)
    ax.set_ylabel("rotation period  (tokens / full cycle, log)", fontsize=11)
    ax.set_title("What the base does: it sets how many dimensions wrap\n"
                 "within the context (θ₀ is fixed at ~6 tokens for any base)",
                 fontsize=12.5, weight="bold", pad=12)
    ax.legend(fontsize=10, frameon=False, loc="lower right", title="more base → longer periods")
    ax.set_xticks(i)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eee", which="both")
    ax.set_axisbelow(True)
    _save(fig, "mechanism_wavelength.png")


# ===========================================================================
# FIG 5 — record waterfall ending at HighRoPE
# ===========================================================================
def fig_waterfall():
    labels = ["control", "+ V-embed", "+ q_gain", "+ SWA(512)", "+ RoPE 500k"]
    vals = [4.7984, 4.7728, 4.6797, 4.6700, load("s_vqgain_swa_highrope_full")[0]]
    refs = ["baseline", "#29", "#39", "#51", "record"]
    deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    x = np.arange(len(vals))
    bar_colors = [GREY, BLUE, BLUE, BLUE, GREEN]
    ax.bar(x, vals, width=0.6, color=bar_colors, zorder=3,
           edgecolor="white", linewidth=1.2)
    ax.set_ylim(4.60, 4.83)
    for xi, v, r in zip(x, vals, refs):
        ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom",
                fontsize=10.5, weight="bold")
        ax.text(xi, 4.605, r, ha="center", va="bottom", fontsize=8.5, color="#555")
    for i, dlt in enumerate(deltas):
        x1, y0, y1 = i + 1, vals[i], vals[i + 1]
        ax.annotate("", xy=(x1, y1), xytext=(x1, y0),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
        ax.text(x1 + 0.32, (y0 + y1) / 2, f"{dlt:+.4f}",
                ha="left", va="center", fontsize=9.5, color=GREEN, weight="bold")
        ax.plot([i, x1], [y0, y0], color="#cbd5e1", lw=1.0, ls="--", zorder=2)
    # highlight the RoPE drop
    ax.text(4, 4.79, "HighRoPE = last\nbig lever in the ladder",
            ha="center", fontsize=9.5, color=GREEN, weight="bold")
    total = vals[-1] - vals[0]
    ax.set_title("RoPE base = 500k closes the screen20m record lineage",
                 fontsize=14, weight="bold", pad=12)
    ax.text(0.0, 4.818, f"10M params · 20M tokens · seed 42 · cumulative {total:+.4f}",
            fontsize=10, color="#444")
    ax.set_ylabel("validation loss", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee")
    ax.set_axisbelow(True)
    _save(fig, "record_waterfall.png")


def _save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, HERE))


if __name__ == "__main__":
    fig_hero()
    fig_curves()
    fig_scales()
    fig_mechanism()
    fig_waterfall()
