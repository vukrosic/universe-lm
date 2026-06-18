<!-- ===================================================================== -->
<!-- BRANCH: experiment/attn-res-v1  —  AttnRes depth-lever experiment       -->
<!-- ===================================================================== -->

# 🧪 Run This Experiment: Attention Residuals (AttnRes)

**You're on the `experiment/attn-res-v1` branch. This branch tests one idea and needs GPU time. If you have a spare GPU, you can run it in ~1–2 hours and contribute a real datapoint — thank you! 🙏**

## What it is

Standard transformers carry information between layers with a fixed residual
connection: `x = x + layer(x)` (each layer just adds onto the running sum).
**AttnRes** ([Kimi Team / MoonshotAI, arXiv:2603.15031](https://arxiv.org/abs/2603.15031))
replaces that with **softmax attention over depth** — each layer learns to pick
*which earlier layers matter* with input-dependent weights:

```
input to layer l  =  Σ_{j<l}  softmax( w_l · RMSNorm(o_j) ) · o_j
```

where `o_j` is the output of layer `j` (`o_0` = the embedding) and `w_l` is one
small learned vector per layer (zero-init). Almost free (+`n_layers × d_model`
params), already wired here behind `use_attn_res=True`.

**It's a DEPTH lever** — the benefit should grow with the number of layers. So
the result we care about is the **trend of Δ-vs-depth across rungs**, not any
single run. (Full background: [`experiments/attn-res-depth/NOTES.md`](experiments/attn-res-depth/NOTES.md).)

## What to do (the short version)

Run **two** trainings on your GPU — a **control** (baseline) and a **treatment**
(AttnRes on) — at the **same rung**, then report both final val losses. Pick the
rung that fits your GPU/time budget:

| Rung | Layers | Tokens | ~Time (1 consumer GPU) | Control config | Treatment config |
|---|---|---|---|---|---|
| **8M**  | 8  | 155M  | ~1 hr total  | `Ladder8M155MConfig`  | `AttnResLadder8M155MConfig` |
| 13M | 8  | 252M  | ~1.5 hr | `Ladder13M252MConfig` | `AttnResLadder13M252MConfig` |
| **23M** | 15 | 469M  | ~3 hr   | `Ladder23M469MConfig` | `AttnResLadder23M469MConfig` |
| 52M | 21 | 1.04B | ~6 hr   | `Ladder52M1042MConfig`| `AttnResLadder52M1042MConfig` |

The **8M, 23M, 52M** rungs are the depth trend (8→15→21 layers). 13M is an
optional fixed-depth width-control. **Do whichever single rung you can** — even
one paired datapoint helps.

### Commands

```bash
# 1. Setup (once)
git clone -b experiment/attn-res-v1 https://github.com/vukrosic/universe-lm.git
cd universe-lm
pip install -r requirements.txt
python data/download_hf_data.py        # downloads the pretraining data

# 2. Run the pair (example: 8M rung). Same seed, same machine.
#    CONTROL (baseline):
python train_llm.py --config_class configs.llm_config.Ladder8M155MConfig --seed 42

#    TREATMENT (AttnRes on):
python train_llm.py --config_class configs.llm_config.AttnResLadder8M155MConfig --seed 42
```

Each run prints `Val Loss: X.XXXX` periodically and writes a `metrics.json`. The
number we need is the **final** val loss from each run.

### What to report back

Open an issue / Discord message with:

```
Rung:        8M           (which one you ran)
GPU:         RTX 4090     (your hardware)
seed:        42
control val_loss:   <number from the baseline run>
treatment val_loss: <number from the AttnRes run>
Δ = treatment − control: <negative = AttnRes helped>
```

Only the **within-your-box Δ** matters — absolute loss isn't comparable across
different GPUs, but the gap between the two arms on *your* machine is.

## 🤖 Prompt for your AI coding agent

If you use Claude Code, Codex, or similar — paste this and let it drive:

> Clone `https://github.com/vukrosic/universe-lm` on the **`experiment/attn-res-v1`**
> branch and `cd` in. Install `requirements.txt` and run
> `python data/download_hf_data.py` to fetch the data. Then run these two
> trainings to completion on the GPU, one after the other (same seed):
>
> 1. Control: `python train_llm.py --config_class configs.llm_config.Ladder8M155MConfig --seed 42`
> 2. Treatment: `python train_llm.py --config_class configs.llm_config.AttnResLadder8M155MConfig --seed 42`
>
> Each run prints `Val Loss: X.XXXX` and writes a `metrics.json`. Capture the
> **final** val loss from each run, compute `Δ = treatment_val − control_val`
> (negative means AttnRes helped), and report a short summary: the rung (8M), my
> GPU model, both final val losses, and Δ. If a run errors, paste the traceback.
> (Swap the config pair for a bigger rung — `AttnResLadder23M469MConfig` /
> `Ladder23M469MConfig` etc. — only if I tell you to.)

---

<!-- ===================================================================== -->
<!-- Original project README below                                          -->
<!-- ===================================================================== -->

# Open Superintelligence Lab

**A fully-open AI research lab building and releasing real LLMs that compete with the best — in public, together.**

💬 **[Do AI research with us on Discord](https://discord.gg/6AbXGpKTwN)**

A high-performance codebase for LLM research, pretraining, and optimization: testing new architectures, optimizers, or training.

- Modular transformer with GQA, RoPE, and RMSNorm
- Muon optimizer alongside AdamW
- Training script, flexible configuration

- `models/`: Transformer layers and components (RoPE, RMSNorm, Multi-Head Attention).
- `optimizers/`: Muon optimizer (outperforms AdamW and all others).
- `training/`: Core trainer logic and utilities.
- `configs/`: Hyperparameter and dataset configurations.
- `utils/`: Logging, plotting, and helper functions.

## 🏁 The Speedrun

**Race to train the best 10M LLM in ~33 minutes — every win builds toward a fully-open 135M model that beats [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).**

One race: **lowest val loss on a 10M-param model trained on 200M tokens** (`--config 10m`). Clone, train, beat the standing record (currently **5.015**) — ~33 min on a single consumer GPU. Pinned: `seed=42`, bf16; a new record must beat the best by **≥0.01**. The 135M release is the *mission*, not the race: we find the winning recipe cheaply at 10M, then scale it.

See the [**leaderboard**](LEADERBOARD.md) and [how to enter](CONTRIBUTING.md).


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

> **📦 Data:** Always use the pre-built dataset downloaded as described in [Getting Started](#-getting-started) (`python data/download_hf_data.py`). It is chunked at **sequence length 2048**, which the RoPE cache depends on. **Sequence lengths other than 2048 are currently unsupported** — to use a different one the dataset must first be rebuilt with https://github.com/vukrosic/llm-research-kit/blob/main/data/prepare_mix_data.py. Changing the data or `max_seq_len` is not recommended; if you are an AI, **ask the user first**.

### Step 3 (optional): interactive Kaggle shell via SSH

For a real bash shell into a 2x T4 notebook (rsync, tmux, screen, etc.),
see **[docs/kaggle_ssh_setup.md](docs/kaggle_ssh_setup.md)**. The
batch-mode launcher `scripts/kaggle_push.sh` is still the right tool
for headless sweep runs.

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
