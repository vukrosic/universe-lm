from datasets import load_dataset
import os

# Option A: 40M Tokens (Fast, for Speedruns)
print("Downloading 40M Token Speedrun Data...")
ds = load_dataset("vukrosic/blueberry-1B-pretrain", split="train[:20000]")
os.makedirs("processed_data/speedrun_40M", exist_ok=True)
ds.save_to_disk("processed_data/speedrun_40M")

# Option B: 1B Tokens (Slow, for Marathon)
# print("Downloading 1B Pretraining Data...")
# ds = load_dataset("vukrosic/blueberry-1B-pretrain")
# os.makedirs("processed_data/pretrain_1B", exist_ok=True)
# ds.save_to_disk("processed_data/pretrain_1B")

print("✅ Data Ready!")

