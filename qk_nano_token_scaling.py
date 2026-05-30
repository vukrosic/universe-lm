#!/usr/bin/env python3
"""Efficient token scaling: 7 gain runs, each training once to 1M tokens.

qk_gain doesn't affect training gradients, but training is from-scratch per gain run.
We train 7 models (one per gain) to 1M tokens, then compare final val_loss at each
gain's actual optimal step for each token count. Eval milestones at 32k, 100k, 300k, 1M.
"""

import sys, json, time
sys.path.insert(0, "/workspace/universe-lm-qk-pilot")

from configs.llm_config import Nano3M32KConfig
from training.trainer import train_minimal_llm
from data.loader import setup_tokenizer
from configs.dataset_config import DataConfig
from torch.utils.data import DataLoader
import torch

GAINS = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5]
CHECKPOINT_TOKENS = [32_768, 100_000, 300_000, 1_000_000]
CHECKPOINT_STEPS = [t // 4096 for t in CHECKPOINT_TOKENS]
SEED = 42

def run_gain(gain, max_tokens):
    c = Nano3M32KConfig()
    c.qk_gain_init = gain
    c.train_tokens = max_tokens
    c.batch_size = 4
    c.eval_milestones = tuple(CHECKPOINT_STEPS)

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

    steps = result["history"]["steps"]
    val_losses = result["history"]["val_losses"]

    # Map step → val_loss, then extract at each checkpoint
    step_to_vl = dict(zip(steps, val_losses))
    row = {"gain": gain, "wall": wall, "checkpoints": {}}
    for t, s in zip(CHECKPOINT_TOKENS, CHECKPOINT_STEPS):
        if s in step_to_vl:
            row["checkpoints"][t] = step_to_vl[s]

    return row

if __name__ == "__main__":
    results = []
    for gain in GAINS:
        print(f"\n=== gain={gain} ===")
        r = run_gain(gain, 1_000_000)
        results.append(r)
        for t, vl in sorted(r["checkpoints"].items()):
            print(f"  {t:,} tokens → val_loss={vl:.4f}")
        print(f"  wall: {r['wall']:.1f}s")

    with open("/tmp/nano_token_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print matrix
    print("\n" + "="*70)
    print("Gain vs Tokens — val_loss at each checkpoint")
    print("="*70)
    header = f"{'gain':<8}"
    for t in CHECKPOINT_TOKENS:
        header += f" {t:>10,}"
    print(header)
    print("-"*70)
    for r in sorted(results, key=lambda x: x["gain"]):
        row = f"{r['gain']:<8.1f}"
        for t in CHECKPOINT_TOKENS:
            vl = r["checkpoints"].get(t, None)
            row += f" {vl:>10.4f}" if vl is not None else f" {'N/A':>10}"
        print(row)