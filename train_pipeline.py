import argparse
import subprocess
import os
import sys

def run_command(cmd):
    print(f"🚀 Running: {cmd}")
    # Use Popen to stream output
    process = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with code {process.returncode}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_tokens", type=int, default=100_000_000)
    args = parser.parse_args()
    
    # Paths
    pretrain_data = f"./processed_data/pretrain_mix_{args.pretrain_tokens}"
    sft_data = "./processed_data/sft_mix"
    
    # Check data exists
    if not os.path.exists(pretrain_data):
        print(f"❌ Pretraining data not found at {pretrain_data}")
        print("   Please run: python data/prepare_mix_data.py --target_tokens {args.pretrain_tokens}")
        return
        
    if not os.path.exists(sft_data):
        print(f"❌ SFT data not found at {sft_data}")
        print("   Please run: python data/prepare_sft_data.py")
        return
        
    print("="*60)
    print("STAGE 1: PRE-TRAINING (Next-Token Prediction)")
    print("="*60)
    
    # We use our custom config file logic. 
    # train_llm.py doesn't accept a --config argument yet, it hardcodes config loading.
    # To properly support switching configs without editing code, we should likely 
    # modify train_llm.py to accept a config module/class name or use a environment variable.
    # OR, we can just use the command line overrides since our configs differ mostly in args 
    # that are already exposed (lr, batch_size, Compile).
    # BUT, PretrainConfig and SFTConfig might have different defaults not fully exposed.
    
    # Let's use a Python script trick: we will run train_llm.py but passing our config object 
    # via a wrapper or by making train_llm.py import dynamic config.
    
    # EASIER APPROACH for this task:
    # Modify train_llm.py to import the config class specified by --config_class arg.
    # I will assume we add this capability next.
    
    cmd_stage1 = (
        f"python train_llm.py "
        f"--config_class configs.pretrain_config.PretrainConfig "
        f"--dataset_path {pretrain_data} "
        f"--experiment_name stage1_pretrain_100m "
        f"--max_steps 1000 " # Explicitly set steps appropriate for 100M data (Batch 4 * 12 * 2048 = 100k/step -> 1000 steps)
        # Actually 100M / 100k = 1000 steps. Perfect.
    )
    
    try:
        run_command(cmd_stage1)
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        return

    print("\n" + "="*60)
    print("STAGE 2: SUPERVISED FINE-TUNING (Instruction Tuning)")
    print("="*60)
    
    # Checkpoint path from Stage 1
    # Checkpoints are saved in checkpoints/stage1_pretrain_100m/final_model.pt
    ckpt_path = "checkpoints/stage1_pretrain_100m/final_model.pt"
    
    if not os.path.exists(ckpt_path):
         print("❌ Stage 1 checkpoint not found!")
         return
         
    # SFT Run
    # We need to tell train_llm.py to resume_from or load_weights_from this checkpoint.
    # Currently train_llm.py doesn't have a simple "load weights but start new run" flag?
    # trainer.py train_moe_model supports resuming if checkpoint exists in output_dir.
    # But here we want to load weights into a NEW experiment.
    # We'll likely need to add a --load_checkpoint argument to train_llm.py.
    
    cmd_stage2 = (
        f"python train_llm.py "
        f"--config_class configs.sft_config.SFTConfig "
        f"--dataset_path {sft_data} "
        f"--experiment_name stage2_sft "
        f"--load_checkpoint {ckpt_path} " 
        f"--max_steps 500 " # 50k samples / (4*12) = ~1000 steps for 1 epoch. Let's do 500 for safety/speed test.
    )
    
    try:
        run_command(cmd_stage2)
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        return
        
    print("\n✅ Pipeline Complete!")

if __name__ == "__main__":
    main()
