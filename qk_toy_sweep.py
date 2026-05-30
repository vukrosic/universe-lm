#!/usr/bin/env python3
"""Quick toy qk_gain sweep — no YAML, just a loop."""

import sys, json, time
from pathlib import Path

sys.path.insert(0, "/workspace/universe-lm-qk-pilot")

from configs.llm_config import ToyConfig
from training.trainer import train_minimal_llm
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from torch.utils.data import DataLoader
import torch

GAINS = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
SEED = 42

def run_toy(qk_init):
    c = ToyConfig()
    c.qk_gain_init = qk_init

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

    return {
        "gain": qk_init,
        "final_vl": result["metrics"]["val_loss"],
        "steps": result["history"]["steps"],
        "val_losses": result["history"]["val_losses"],
        "elapsed": result["history"].get("elapsed_times", []),
        "wall": wall,
    }

if __name__ == "__main__":
    results = []

    for g in [0.0] + GAINS:  # 0.0 = no qk_gain (baseline)
        print(f"\n=== qk_gain={g} ===")
        r = run_toy(g)
        results.append(r)
        print(f"  final val_loss: {r['final_vl']:.4f}")

    print("\n" + "="*50)
    print(f"{'gain':<8} {'final_vl':>10} {'wall_s':>8}")
    print("-"*50)
    for r in sorted(results, key=lambda x: x["gain"]):
        print(f"{r['gain']:<8.1f} {r['final_vl']:>10.4f} {r['wall']:>8.1f}")
    print("="*50)

    with open("/workspace/qk_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /tmp/qk_sweep_results.json")