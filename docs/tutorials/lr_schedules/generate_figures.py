"""
Generate intuition figures for the LR-schedule tutorial.

Why each figure exists:
  1. lr_schedules.png      - the two schedules side by side (what we changed)
  2. loss_noise_floor.png  - the symptom: constant stagnates, decay keeps dropping
  3. sgd_trajectory.png    - the cause, in 2D: constant orbits the minimum, decay spirals in
  4. marble_bowl.png       - the cause, in 1D: step size sets a "loss floor"

All figures are illustrative (synthetic) - they exist to build intuition, not to
report the real run. The real run lives in real_run.png.

Run:  python generate_figures.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#1f77b4"   # constant LR
ORANGE = "#ff7f0e" # warmup + decay
plt.rcParams.update({"figure.dpi": 130, "font.size": 12, "axes.grid": True,
                     "grid.alpha": 0.3})


# ----------------------------------------------------------------------------
# 1. The two learning-rate schedules
# ----------------------------------------------------------------------------
def fig_lr_schedules():
    steps = np.linspace(0, 1, 1000)          # fraction of training
    peak = 0.024
    warmup = 0.02                            # 2% of steps

    constant = np.full_like(steps, peak)

    decay = np.where(
        steps < warmup,
        peak * steps / warmup,               # linear warmup
        peak * (1 - (steps - warmup) / (1 - warmup)),  # linear decay to zero
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(steps, constant, color=BLUE, lw=2.5, label="constant LR")
    ax.plot(steps, decay, color=ORANGE, lw=2.5, label="warmup + decay-to-zero")
    ax.set_xlabel("training progress (fraction of steps)")
    ax.set_ylabel("learning rate")
    ax.set_title("The only thing we changed: the LR schedule")
    ax.set_ylim(0, peak * 1.15)
    ax.legend()
    ax.annotate("same peak LR", xy=(0.02, peak), xytext=(0.18, peak * 1.07),
                arrowprops=dict(arrowstyle="->", color="gray"), color="gray")
    ax.annotate("decays to exactly 0", xy=(1.0, 0), xytext=(0.62, peak * 0.25),
                arrowprops=dict(arrowstyle="->", color="gray"), color="gray")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lr_schedules.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. The symptom: loss curves (synthetic, mirrors the real shape)
# ----------------------------------------------------------------------------
def fig_loss_noise_floor():
    t = np.linspace(0, 1, 1000)
    # both drop fast; constant asymptotes to a higher floor than decay
    loss_const = 5.00 + 5.8 * np.exp(-15 * t)
    loss_decay = 4.55 + 6.25 * np.exp(-13 * t)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(t, loss_const, color=BLUE, lw=2.5, label="constant LR")
    ax.plot(t, loss_decay, color=ORANGE, lw=2.5, label="warmup + decay-to-zero")

    ax.axhline(5.00, color=BLUE, ls="--", lw=1, alpha=0.6)
    ax.annotate("noise floor: it stops improving\nbut isn't done learning",
                xy=(0.7, 5.0), xytext=(0.42, 6.4),
                arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE)
    ax.annotate("keeps descending\nas LR shrinks",
                xy=(0.85, loss_decay[850]), xytext=(0.55, 4.0),
                arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE)

    ax.set_xlabel("training progress")
    ax.set_ylabel("validation loss")
    ax.set_title("Same model, same data — only the schedule differs")
    ax.set_ylim(3.6, 8.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_noise_floor.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. The cause in 2D: noisy SGD on a bowl
#    constant LR -> orbits the minimum forever (a cloud)
#    decaying LR -> spirals into the minimum (settles)
# ----------------------------------------------------------------------------
def _run_sgd(lr_schedule, n=400, seed=0):
    rng = np.random.default_rng(seed)
    a, b = 1.0, 4.0                  # anisotropic bowl: grad = [a*x, b*y]
    p = np.array([3.2, 2.6])         # start
    sigma = 1.2                      # gradient noise
    traj = [p.copy()]
    for i in range(n):
        lr = lr_schedule(i / n)
        grad = np.array([a * p[0], b * p[1]])
        noise = rng.normal(0, sigma, size=2)
        p = p - lr * (grad + noise)
        traj.append(p.copy())
    return np.array(traj)


def fig_sgd_trajectory():
    a, b = 1.0, 4.0
    xs = np.linspace(-4, 4, 300)
    ys = np.linspace(-3.5, 3.5, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (a * X**2 + b * Y**2)

    peak = 0.12
    warmup = 0.05
    const_sched = lambda f: peak
    def decay_sched(f):
        if f < warmup:
            return peak * f / warmup
        return peak * (1 - (f - warmup) / (1 - warmup))

    traj_c = _run_sgd(const_sched, seed=1)
    traj_d = _run_sgd(decay_sched, seed=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, traj, title, col in [
        (axes[0], traj_c, "constant LR: orbits the minimum forever", BLUE),
        (axes[1], traj_d, "decay-to-zero: spirals in and settles", ORANGE),
    ]:
        ax.contour(X, Y, Z, levels=12, cmap="Greys", alpha=0.5, linewidths=0.8)
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=0.9, alpha=0.8)
        ax.scatter([0], [0], marker="*", s=220, color="red", zorder=5,
                   label="true minimum")
        ax.scatter(traj[0, 0], traj[0, 1], s=40, color="black", zorder=5,
                   label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], s=60, color=col,
                   edgecolor="black", zorder=6, label="final")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("parameter 1")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("parameter 2")
    fig.suptitle("Why constant LR can't settle: a fixed step size keeps overshooting",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sgd_trajectory.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. The cause in 1D: a "marble in a bowl" and the loss floor
# ----------------------------------------------------------------------------
def fig_marble_bowl():
    x = np.linspace(-2.2, 2.2, 400)
    loss = x**2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    # left: constant LR -> wide resting band -> high loss floor
    r_c = 1.25
    ax = axes[0]
    ax.plot(x, loss, color="gray", lw=2)
    ax.axvspan(-r_c, r_c, color=BLUE, alpha=0.15)
    ax.axhline(r_c**2, color=BLUE, ls="--", lw=1.5)
    ax.annotate("step too big to settle:\nmarble rattles in this band",
                xy=(0, r_c**2), xytext=(-2.1, 3.0),
                arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE, fontsize=10)
    ax.text(0, r_c**2 + 0.12, "loss floor", color=BLUE, ha="center", fontsize=10)
    for xb in (-r_c, -0.4, 0.7, r_c):
        ax.scatter(xb, xb**2, color=BLUE, s=45, zorder=5)
    ax.set_title("constant LR", fontsize=12)
    ax.set_xlabel("parameter")
    ax.set_ylabel("loss")

    # right: decay -> band shrinks -> settles at the bottom
    ax = axes[1]
    ax.plot(x, loss, color="gray", lw=2)
    for r, alpha in [(1.25, 0.06), (0.7, 0.10), (0.3, 0.18)]:
        ax.axvspan(-r, r, color=ORANGE, alpha=alpha)
    ax.scatter(0, 0, color=ORANGE, s=80, edgecolor="black", zorder=6)
    ax.annotate("LR shrinks -> band shrinks ->\nmarble settles at the bottom",
                xy=(0, 0), xytext=(-2.1, 3.0),
                arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, fontsize=10)
    ax.set_title("warmup + decay-to-zero", fontsize=12)
    ax.set_xlabel("parameter")

    fig.suptitle("The step size sets a loss floor; shrinking it lets the marble reach the bottom",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "marble_bowl.png", bbox_inches="tight")
    plt.close(fig)


GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
BROWN = "#8c564b"


# ----------------------------------------------------------------------------
# 5. What is a learning rate? Step size: too small / just right / too big
# ----------------------------------------------------------------------------
def fig_lr_step_size():
    x = np.linspace(-2.7, 2.7, 400)
    loss = x**2
    start = 2.0
    configs = [
        ("too small", 0.05, BLUE, "barely moves — very slow"),
        ("just right", 0.40, GREEN, "settles in a few steps"),
        ("too big", 1.05, RED, "overshoots — diverges"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, (title, lr, col, note) in zip(axes, configs):
        ax.plot(x, loss, color="gray", lw=2, zorder=1)
        p = start
        pts = [p]
        for _ in range(9):
            p = p - lr * (2 * p)          # grad of x**2 is 2x
            pts.append(p)
            if abs(p) > 2.7:
                break
        pts = np.array(pts)
        ax.plot(pts, pts**2, "o-", color=col, lw=1.5, ms=6, zorder=3)
        ax.scatter([0], [0], marker="*", s=200, color="black", zorder=4)
        ax.set_title(f"learning rate {title}  (lr={lr})", fontsize=11)
        ax.set_xlabel("parameter")
        ax.text(0.5, 0.93, note, transform=ax.transAxes, ha="center",
                fontsize=10, color=col)
        ax.set_ylim(-0.4, 6.5)
        ax.set_xlim(-2.9, 2.9)
    axes[0].set_ylabel("loss")
    fig.suptitle("A learning rate is your step size on the loss surface", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lr_step_size.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. The schedule zoo: constant vs the decaying family
# ----------------------------------------------------------------------------
def fig_schedule_zoo():
    s = np.linspace(0, 1, 1000)
    warm = 0.05
    rampup = np.clip(s / warm, 0, 1)
    after = np.clip((s - warm) / (1 - warm), 0, 1)
    decay_start = 0.8

    constant = np.ones_like(s)
    step = np.where(s < 0.5, 1.0, np.where(s < 0.75, 0.3, 0.1))
    invsqrt = np.minimum(s / warm, np.sqrt(warm / np.maximum(s, 1e-6)))
    cosine = rampup * 0.5 * (1 + np.cos(np.pi * after))
    linear = rampup * (1 - after)
    wsd = np.where(s < warm, s / warm,
                   np.where(s < decay_start, 1.0,
                            1 - (s - decay_start) / (1 - decay_start)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s, constant, lw=2, color="#7f7f7f", label="constant")
    ax.plot(s, step, lw=2, color=PURPLE, label="step decay")
    ax.plot(s, invsqrt, lw=2, color=BROWN, label="inverse-sqrt (original Transformer)")
    ax.plot(s, cosine, lw=2, color=BLUE, label="warmup + cosine")
    ax.plot(s, wsd, lw=2.5, color=GREEN, label="warmup-stable-decay (WSD)")
    ax.plot(s, linear, lw=3, color=ORANGE, label="warmup + linear decay-to-zero (this tutorial)")
    ax.set_xlabel("training progress")
    ax.set_ylabel("learning rate (fraction of peak)")
    ax.set_title("The schedule zoo: constant vs the decaying family")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "schedule_zoo.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. Why warmup: a fresh model blows up on a full first step
# ----------------------------------------------------------------------------
def fig_warmup_why():
    steps = np.arange(0, 200)
    no_warm = 7.5 * np.exp(-steps / 40) + 4.6
    spikes = np.array([0, 3.0, 5.5, 4.0, 6.2, 3.5, 2.0, 1.2, 0.6, 0.3, 0.1, 0.0])
    no_warm[:len(spikes)] += spikes
    warm = 7.0 * np.exp(-steps / 45) + 4.55

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(steps, no_warm, color=RED, lw=2, label="no warmup (full LR from step 1)")
    ax.plot(steps, warm, color=ORANGE, lw=2.5, label="with warmup")
    ax.axvspan(0, len(spikes), color=ORANGE, alpha=0.12)
    ax.annotate("a full step on a fresh model\ncan blow the loss up",
                xy=(4, no_warm[4]), xytext=(55, no_warm[4] + 1.2),
                arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=10)
    ax.text(len(spikes) / 2, 13.5, "warmup\nwindow", ha="center",
            color="#b35900", fontsize=9)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.set_title("Why warmup: ease the LR up so a fresh model doesn't diverge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "warmup_why.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. LR as temperature: constant-LR SGD samples a cloud, width grows with LR
# ----------------------------------------------------------------------------
def fig_lr_temperature():
    def sample_positions(lr, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        p = 0.0
        xs = []
        for i in range(n):
            p = p - lr * (2 * p + rng.normal(0, 1.0))   # bowl loss = x**2
            if i > 2000:
                xs.append(p)
        return np.array(xs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.linspace(-2.2, 2.2, 400)
    ax1.plot(x, x**2, color="gray", lw=2)
    ax1.scatter([0], [0], marker="*", s=200, color="red", zorder=5)
    ax1.set_title("Constant-LR SGD never sits at the minimum —\nit samples a cloud around it")
    ax1.set_xlabel("parameter")
    ax1.set_ylabel("loss")

    for lr, col in [(0.30, RED), (0.12, ORANGE), (0.03, BLUE)]:
        xs = sample_positions(lr)
        ax2.hist(xs, bins=80, density=True, alpha=0.55, color=col, label=f"lr = {lr}")
    ax2.set_title("...and the cloud's width IS a 'temperature' set by the LR")
    ax2.set_xlabel("parameter value")
    ax2.set_ylabel("how often the model sits here")
    ax2.legend(title="constant LR")
    fig.suptitle("Learning rate = temperature: high LR = wide cloud, low LR = tight. "
                 "Decaying to zero = cooling into the minimum.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lr_temperature.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 9. WSD: the loss plateaus while LR is high, then drops sharply in the decay
# ----------------------------------------------------------------------------
def fig_wsd_decay_drop():
    s = np.linspace(0, 1, 1000)
    warm, decay_start = 0.05, 0.8
    lr = np.where(s < warm, s / warm,
                  np.where(s < decay_start, 1.0,
                           1 - (s - decay_start) / (1 - decay_start)))
    base = 4.6 + 2.5 * np.exp(-6 * s)
    extra = 0.32 * np.clip((s - decay_start) / (1 - decay_start), 0, 1)
    loss = base - extra

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(s, lr, color=GREEN, lw=2.5)
    ax1.axvspan(decay_start, 1.0, color=GREEN, alpha=0.12)
    ax1.set_ylabel("learning rate")
    ax1.set_title("Warmup-Stable-Decay (WSD): loss drops sharply in the decay phase")
    ax1.text(0.42, 0.6, "stable", ha="center", color="gray")
    ax1.text(0.9, 0.55, "decay", ha="center", color=GREEN)

    ax2.plot(s, loss, color=ORANGE, lw=2.5)
    ax2.axvspan(decay_start, 1.0, color=GREEN, alpha=0.12)
    ax2.annotate("plateaus while LR is high",
                 xy=(0.6, loss[600]), xytext=(0.28, loss[600] + 0.5),
                 arrowprops=dict(arrowstyle="->", color="gray"), color="gray")
    ax2.annotate("sharp drop\nwhen LR decays",
                 xy=(0.97, loss[970]), xytext=(0.55, loss[970] + 0.7),
                 arrowprops=dict(arrowstyle="->", color=GREEN), color=GREEN)
    ax2.set_xlabel("training progress")
    ax2.set_ylabel("validation loss")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wsd_decay_drop.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_lr_schedules()
    fig_loss_noise_floor()
    fig_sgd_trajectory()
    fig_marble_bowl()
    fig_lr_step_size()
    fig_schedule_zoo()
    fig_warmup_why()
    fig_lr_temperature()
    fig_wsd_decay_drop()
    print("wrote figures to", FIG_DIR)
