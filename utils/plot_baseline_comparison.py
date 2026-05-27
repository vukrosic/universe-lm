import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def token_axis(metrics: dict) -> list[float]:
    history = metrics["history"]
    steps = history["steps"]
    actual_steps = max(metrics.get("actual_steps", steps[-1]), 1)
    total_tokens = metrics.get("train_tokens", 1)
    tokens_per_step = total_tokens / actual_steps
    return [step * tokens_per_step for step in steps]


def plot_single(metrics_path: Path, out_path: Path, title: str) -> None:
    data = load_metrics(metrics_path)
    steps = data["history"]["steps"]
    val_losses = data["history"]["val_losses"]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, val_losses, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Training steps")
    plt.ylabel("Validation loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_comparison(current_path: Path, baseline_path: Path, out_path: Path, title: str) -> None:
    current = load_metrics(current_path)
    baseline = load_metrics(baseline_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    views = [
        ("steps", "Training steps", lambda d: d["history"]["steps"]),
        ("tokens", "Tokens seen", token_axis),
        ("time", "Elapsed time (minutes)", lambda d: [t / 60.0 for t in d["history"]["elapsed_times"]]),
    ]

    for ax, (_, xlabel, x_fn) in zip(axes, views):
        ax.plot(x_fn(current), current["history"]["val_losses"], marker="o", linewidth=2, label="5m")
        ax.plot(x_fn(baseline), baseline["history"]["val_losses"], marker="o", linewidth=2, label="25m")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline runs separately and together.")
    parser.add_argument("--five_m", required=True, type=Path, help="Path to 5m metrics.json")
    parser.add_argument("--twenty_five_m", required=True, type=Path, help="Path to 25m metrics.json")
    parser.add_argument("--output_dir", default="plots/baseline_compare", type=Path, help="Output directory")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_single(
        args.five_m,
        out_dir / "5m_val_loss_steps.png",
        "5M compiled baseline: validation loss vs steps",
    )
    plot_single(
        args.twenty_five_m,
        out_dir / "25m_val_loss_steps.png",
        "25M compiled baseline: validation loss vs steps",
    )
    plot_comparison(
        args.five_m,
        args.twenty_five_m,
        out_dir / "5m_vs_25m_steps_tokens_time.png",
        "5M vs 25M compiled baselines",
    )


if __name__ == "__main__":
    main()
