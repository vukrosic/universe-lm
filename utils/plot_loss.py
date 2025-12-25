import matplotlib.pyplot as plt
import json
import argparse
import os

def plot_loss(metrics_file, output_file, title="Validation Loss Curve", baseline_file=None):
    if not os.path.exists(metrics_file):
        print(f"Error: File not found {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    if 'history' not in data:
        print("Error: 'history' key not found in metrics file")
        return
    
    history = data['history']
    steps = history.get('steps', [])
    val_losses = history.get('val_losses', [])
    
    if not steps or not val_losses:
        print("Error: incomplete data in history")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot baseline if provided
    if baseline_file and os.path.exists(baseline_file):
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            if 'history' in baseline_data:
                b_history = baseline_data['history']
                b_steps = b_history.get('steps', [])
                b_val_losses = b_history.get('val_losses', [])
                if b_steps and b_val_losses:
                    plt.plot(b_steps, b_val_losses, marker='', linestyle='--', color='orange', linewidth=2, label='Baseline Loss', alpha=0.8)
                    print(f"   ✓ Baseline comparison: {len(b_steps)} data points from {baseline_file}")
        except Exception as e:
            print(f"Warning: Failed to load baseline file: {e}")

    plt.plot(steps, val_losses, marker='o', linestyle='-', color='b', linewidth=1.5, label='Current Run Loss')
    
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Annotate points for current run
    for i, (x, y) in enumerate(zip(steps, val_losses)):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validation loss curve")
    parser.add_argument("metrics_file", help="Path to metrics.json")
    parser.add_argument("output_file", help="Path to save the plot image")
    parser.add_argument("--title", default="Validation Loss - 8M Tokens Benchmark", help="Plot title")
    parser.add_argument("--baseline_file", default=None, help="Path to baseline metrics.json to compare against")
    
    args = parser.parse_args()
    
    plot_loss(args.metrics_file, args.output_file, args.title, args.baseline_file)

