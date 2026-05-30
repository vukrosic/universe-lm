#!/usr/bin/env python3
"""Run ToyMediumConfig with gain=4.0."""

import sys, json, time
sys.path.insert(0, "/workspace/universe-lm-qk-pilot")

from configs.llm_config import Mini10M20MConfig
from training.trainer import train_minimal_llm
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from torch.utils.data import DataLoader
import torch

GAIN = 4.0
SEED = 42

c = ToyMediumConfig()
c.qk_gain_init = GAIN

data_cfg = DataConfig(dataset_path="auto", seq_length=c.max_seq_len,
                      num_samples=50, cache_dir="./hf_cache")
tokenizer = setup_tokenizer(data_cfg)
c.vocab_size = tokenizer.vocab_size

from train_llm import prepare_datasets
train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True,
                          num_workers=0, generator=g)
val_loader = DataLoader(val_ds, batch_size=c.batch_size, shuffle=False,
                         num_workers=0)

start = time.time()
result = train_minimal_llm(c, train_loader, val_loader, output_dir=None)
wall = time.time() - start

r = {
    "gain": GAIN,
    "final_vl": result["metrics"]["val_loss"],
    "steps": result["history"]["steps"],
    "val_losses": result["history"]["val_losses"],
    "elapsed": result["history"].get("elapsed_times", []),
    "wall": wall,
}

print(f"\n=== qk_gain={GAIN} ===")
print(f"  final val_loss: {r['final_vl']:.4f}")
print(f"  wall time: {r['wall']:.1f}s")

with open("/tmp/toy_medium.json", "w") as f:
    json.dump(r, f, indent=2)
print("Saved to /tmp/toy_medium.json")