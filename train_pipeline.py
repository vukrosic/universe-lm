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
    # SFT Data Strategy
    # Target: 20,000 samples (approx 6-10M tokens) is usually plenty for Chat
    # Constraint: For very small models (e.g. 1M pre-train), we can't train on 6M SFT tokens.
    # Cap SFT tokens at ~10% of pre-train tokens.
    
    TARGET_SFT_SAMPLES = 20_000
    AVG_TOKENS_PER_SAMPLE = 300 # rough estimate
    
    # 10% of pretrain tokens / tokens_per_sample
    dynamic_cap = (args.pretrain_tokens * 0.10) / AVG_TOKENS_PER_SAMPLE
    sft_samples = min(TARGET_SFT_SAMPLES, int(dynamic_cap))
    sft_samples = max(sft_samples, 100) # Ensure minimum 100 samples
    
    print(f"🧠 SFT Strategy: Target={TARGET_SFT_SAMPLES}, Dynamic Cap={int(dynamic_cap)}")
    print(f"   -> Selected SFT Samples: {sft_samples}")

    pretrain_data = f"./processed_data/pretrain_mix_{args.pretrain_tokens}"
    # Use a specific directory for this size of SFT data to avoid collisions
    sft_data_dir = f"./processed_data/sft_{sft_samples}_samples"
    sft_data = f"{sft_data_dir}/sft_mix"
    
    # Check data exists
    if not os.path.exists(pretrain_data):
        print(f"❌ Pretraining data not found at {pretrain_data}")
        print(f"   Please run: python data/prepare_mix_data.py --target_tokens {args.pretrain_tokens}")
        # Build command hint roughly:
        # run_command(f"python data/prepare_mix_data.py --target_tokens {args.pretrain_tokens}") 
        return
        
    if not os.path.exists(sft_data):
        print(f"⚠️ SFT data not found at {sft_data}. Generating...")
        cmd_gen_sft = f"python data/prepare_sft_data.py --max_samples {sft_samples} --output_dir {sft_data_dir}"
        try:
            run_command(cmd_gen_sft)
        except Exception as e:
            print(f"❌ Failed to generate SFT data: {e}")
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
