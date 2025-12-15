import subprocess
import sys
import os
import shutil
import glob
import json
import matplotlib.pyplot as plt

# Hardcoded configurations for quick sweep
CONFIGS = [
    {"muon_lr": 0.01, "adamw_lr": 0.0015},
    {"muon_lr": 0.02, "adamw_lr": 0.003},
    {"muon_lr": 0.04, "adamw_lr": 0.006},
]

STEPS = 200

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        sys.exit(1)

def analyze_results():
    print("Analyzing sweep results...")
    
    # Find all metrics.json files in checkpoints/sweep_lr_*
    metric_files = glob.glob("checkpoints/sweep_lr_*/metrics.json")
    
    if not metric_files:
        print("No sweep results found in checkpoints/")
        return

    results = {}
    
    # Load data
    for f in metric_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
            # Extract run name from path
            run_name = os.path.basename(os.path.dirname(f))
            
            # Store history
            results[run_name] = data['history']
            print(f"Loaded {run_name}: Final Val Loss = {data['final_metrics']['val_loss']:.4f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # Plot
    if not results:
        print("No valid results to plot.")
        return

    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for i, (name, history) in enumerate(sorted(results.items())):
        steps = history['steps']
        losses = history['val_losses']
        
        color = colors[i % len(colors)]
        plt.plot(steps, losses, f'{color}-o', label=name, markersize=4)
        
        # Mark best
        min_loss = min(losses)
        min_idx = losses.index(min_loss)
        plt.plot(steps[min_idx], min_loss, f'{color}*', markersize=10)

    plt.title("Learning Rate Sweep - Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = "sweep_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Comparison plot saved to {out_path}")

def main():
    print("Starting Quick Learning Rate Sweep...")
    
    print("Cleaning up previous sweep checkpoints...")
    for sweep_dir in glob.glob("checkpoints/sweep_lr_*"):
        print(f"Removing {sweep_dir}")
        shutil.rmtree(sweep_dir)

    for cfg in CONFIGS:
        muon_lr = cfg["muon_lr"]
        adamw_lr = cfg["adamw_lr"]
        
        # Consistent naming
        if muon_lr == 0.02:
            suffix = "1.0x"
        elif muon_lr < 0.02:
            suffix = "0.5x"
        else:
            suffix = "2.0x"
            
        experiment_name = f"sweep_lr_{suffix}"
        
        print(f"\n{'='*50}")
        print(f"Starting Experiment: {experiment_name}")
        print(f"Muon LR: {muon_lr}")
        print(f"AdamW LR: {adamw_lr}")
        print(f"{'='*50}\n")
        
        cmd = (
            f"python train_moe.py "
            f"--muon_lr {muon_lr} "
            f"--adamw_lr {adamw_lr} "
            f"--max_steps {STEPS} "
            f"--experiment_name {experiment_name}"
        )
        
        run_command(cmd)
        
        # Cleanup large files
        ckpt_dir = os.path.join("checkpoints", experiment_name)
        for fname in ["final_model.pt", "model.pt"]:
            p = os.path.join(ckpt_dir, fname)
            if os.path.exists(p):
                print(f"Removing checkpoint {p} to save space...")
                os.remove(p)

    print("\nSweep Completed!")
    
    # Run analysis internally
    analyze_results()

if __name__ == "__main__":
    main()
