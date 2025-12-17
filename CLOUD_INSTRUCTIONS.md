# 🫐 Blueberry-Nano: Cloud Training Guide

This guide explains how to train the **Blueberry-Nano (151M)** model on a fresh cloud instance (e.g., Runpod, Lambda, AWS) using the pre-processed data from Hugging Face.

## 1. Setup Environment
First, clone the repo and install dependencies.

```bash
# Clone your repo (assuming you pushed code to git, or upload it manually)
git clone https://github.com/YourUsername/5-dollar-llm
cd 5-dollar-llm

# Install requirements
pip install -r requirements.txt
pip install huggingface_hub
```

## 2. Download Data
Instead of generating 1B tokens from scratch (which takes hours), download the pre-shuffled, pre-tokenized dataset we uploaded to Hugging Face.

```python
# Create a script: download_data.py
from datasets import load_dataset
import os

# 1. Download Pretraining Data
print("Downloading 1B Pretraining Data...")
ds_pretrain = load_dataset("vukrosic/blueberry-1B-pretrain")
# Save to disk so the trainer can read it as a folder
ds_pretrain.save_to_disk("processed_data/pretrain_mix_1000000000")

# 2. Download SFT Data
print("Downloading SFT Data...")
ds_sft = load_dataset("vukrosic/blueberry-1B-sft")
ds_sft.save_to_disk("processed_data/sft_mix")

print("✅ Data Ready!")
```

Run it:
```bash
python3 download_data.py
```

## 3. Start Training

### 1B Token Pretraining
Run the training script pointing to the downloaded data.

```bash
# Optimized for RTX 4090 (24GB VRAM)
python train_llm.py \
    --config_class configs.llm_config.Blueberry24GBConfig \
    --dataset_path processed_data/pretrain_mix_1000000000
```

### SFT (Fine-Tuning)
After pretraining finishes, fine-tune on the instruction data.

```bash
python train_llm.py \
    --config_class configs.llm_config.Blueberry24GBConfig \
    --dataset_path processed_data/sft_mix \
    --load_checkpoint checkpoints/moe_training/final_model.pt \
    --train_tokens 10000000 \
    --experiment_name sft_run
```
