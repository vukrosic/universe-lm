# LLM Research Kit

A high-performance codebase for LLM research, pretraining, and optimization: testing new architectures, optimizers, or training.

- Modular transformer with GQA, RoPE, and RMSNorm
- Muon optimizer alongside AdamW
- Training script, flexible configuration

- `models/`: Transformer layers and components (RoPE, RMSNorm, Multi-Head Attention).
- `optimizers/`: Muon optimizer (outperforms AdamW and all others).
- `training/`: Core trainer logic and utilities.
- `configs/`: Hyperparameter and dataset configurations.
- `utils/`: Logging, plotting, and helper functions.

## 🏁 Speedruns

Race to the **lowest val loss at a fixed token budget.** Clone, train, report — beat the record.

| Rung | Size | Tokens | ~Time (1 GPU) | ~$/entry |
|---|---|---|---|---|
| Small | 25M | 0.5B | ~4 min | ~$1 |
| Flagship | 135M | 3B | ~35 min | ~$5–10 |

Pinned for every entry: `seed=42`, bf16. See the [**leaderboard**](LEADERBOARD.md), the [rules](docs/PLAN.md), and [how to enter](CONTRIBUTING.md).

## 🚀 Getting Started

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset

The simplest path is:

```bash
python data/download_hf_data.py
```

If you are training on a remote GPU, start the run inside `tmux` so the job keeps running after you disconnect.

### Option A: 1B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
print('✅ Full Data Ready!')
"
```

### Option B: 2B tokens
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 2B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-2B-pretrain')
os.makedirs('processed_data/pretrain_2B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_2B')
print('✅ Full Data Ready!')
"
```

### Option C: Quick Start (40M Tokens)
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 40M Token Subset...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain', split='train[:20000]')
os.makedirs('processed_data/speedrun_40M', exist_ok=True)
ds.save_to_disk('processed_data/speedrun_40M')
print('✅ Speedrun Data Ready!')
"
```

## 🧠 LLM Architecture

Default is an **88M parameter** transformer LLM, you can modify configs.

- **Layers**: 22 Transformer blocks.
- **Hidden Dimension (`d_model`)**: 512.
- **Feed-Forward Dimension (`d_ff`)**: 2048.
- **Attention System**:
  - 8 Query heads, 4 Key-Value heads (**Grouped Query Attention**).
  - Rotary Positional Embeddings (**RoPE**).
  - Fused QKVO projection for optimized compute.
  - QK-Normalization for training stability.
- **Normalization**: Pre-norm **RMSNorm**.
- **Activation**: **Squared ReLU** (Primer-style).
- **Vocab Size**: 49,152.
- **Sequence Length**: 2048 tokens.

### Optimization Highlights
- **Weight Tying**: Shared weights between token embeddings and the LM head.
- **Muon Support**: Architecture optimized for the Muon optimizer's orthogonal updates.
- **Efficiency**: Designed for `torch.compile` compatibility and mixed-precision (BF16) training.
