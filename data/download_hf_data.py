from datasets import load_dataset
import os

# 1. Download Pretraining Data
print("Downloading 1B Pretraining Data...")
ds_pretrain = load_dataset("vukrosic/blueberry-1B-pretrain")
# Save to disk so the trainer can read it as a folder
os.makedirs("processed_data", exist_ok=True)
ds_pretrain.save_to_disk("processed_data/pretrain_mix_1000000000")


print("✅ Data Ready!")

