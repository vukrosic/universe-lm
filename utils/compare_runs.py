
import json
import matplotlib.pyplot as plt
import sys
import os

def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparisons(baseline_path, experiment_path, output_dir="."):
    baseline_metrics = load_metrics(baseline_path)
    experiment_metrics = load_metrics(experiment_path)

    base_hist = baseline_metrics.get("history", {})
    exp_hist = experiment_metrics.get("history", {})

    if not base_hist or not exp_hist:
        print("Error: Could not find 'history' in one or both metrics files.")
        return

    # Extract steps
    # Keys found: ['steps', 'val_losses', 'val_aux_losses', 'val_accuracies', 'val_perplexities', 'elapsed_times', 'learning_rates']
    base_steps = base_hist.get("steps", [])
    exp_steps = exp_hist.get("steps", [])

    # Plot 1: Loss Comparison
    plt.figure(figsize=(10, 6))
    
    # Val Loss
    if "val_losses" in base_hist:
        plt.plot(base_steps, base_hist["val_losses"], label="Baseline Val Loss", linestyle="--", linewidth=2)
    if "val_losses" in exp_hist:
        plt.plot(exp_steps, exp_hist["val_losses"], label="Experiment Val Loss", linestyle="-", linewidth=2)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Validation Loss Comparison: Baseline vs Experiment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "comparison_loss.png"))
    print(f"Saved {os.path.join(output_dir, 'comparison_loss.png')}")
    plt.close()

    # Plot 2: Accuracy and Perplexity Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy
    if "val_accuracies" in base_hist:
        ax1.plot(base_steps, base_hist["val_accuracies"], label="Baseline Accuracy", linestyle="--")
    if "val_accuracies" in exp_hist:
        ax1.plot(exp_steps, exp_hist["val_accuracies"], label="Experiment Accuracy", linestyle="-")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Validation Accuracy Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Perplexity
    if "val_perplexities" in base_hist:
        ax2.plot(base_steps, base_hist["val_perplexities"], label="Baseline Perplexity", linestyle="--")
    if "val_perplexities" in exp_hist:
        ax2.plot(exp_steps, exp_hist["val_perplexities"], label="Experiment Perplexity", linestyle="-")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Validation Perplexity Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_metrics.png"))
    print(f"Saved {os.path.join(output_dir, 'comparison_metrics.png')}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_runs.py <baseline_metrics_path> <experiment_metrics_path>")
        sys.exit(1)

    baseline_path = sys.argv[1]
    experiment_path = sys.argv[2]
    
    plot_comparisons(baseline_path, experiment_path)
