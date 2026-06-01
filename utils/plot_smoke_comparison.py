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


def time_axis(metrics: dict) -> list[float]:
    return [t / 60.0 for t in metrics["history"]["elapsed_times"]]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runs must be in LABEL=PATH format")
    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multiple smoke runs on shared axes.")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Add a run as LABEL=PATH_TO_METRICS_JSON. Repeat for multiple runs.",
    )
    parser.add_argument("--output", type=Path, default=Path("plots/smoke_comparison.png"))
    parser.add_argument("--title", type=str, default="Smoke run comparison")
    args = parser.parse_args()

    runs = [(label, load_metrics(path)) for label, path in args.run]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [
        ("steps", "Training steps", lambda d: d["history"]["steps"]),
        ("tokens", "Tokens seen", token_axis),
        ("time", "Elapsed time (minutes)", time_axis),
    ]

    for ax, (_, xlabel, x_fn) in zip(axes, views):
        for label, metrics in runs:
            ax.plot(
                x_fn(metrics),
                metrics["history"]["val_losses"],
                marker="o",
                linewidth=2,
                label=label,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
