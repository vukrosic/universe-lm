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
    
    cmd_stage1 = (
        f"python train_llm.py "
        f"--config_class configs.pretrain_config.PretrainConfig "
        f"--dataset_path {pretrain_data} "
        f"--experiment_name stage1_pretrain_100m "
        f"--max_steps 1000"
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
    ckpt_path = "checkpoints/stage1_pretrain_100m/final_model.pt"
    
    if not os.path.exists(ckpt_path):
         print("❌ Stage 1 checkpoint not found!")
         return
         
    cmd_stage2 = (
        f"python train_llm.py "
        f"--config_class configs.sft_config.SFTConfig "
        f"--dataset_path {sft_data} "
        f"--experiment_name stage2_sft "
        f"--load_checkpoint {ckpt_path} "
        f"--max_steps 500"
    )
    
    try:
        run_command(cmd_stage2)
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        return
        
    print("\n✅ Pipeline Complete!")

if __name__ == "__main__":
    main()
