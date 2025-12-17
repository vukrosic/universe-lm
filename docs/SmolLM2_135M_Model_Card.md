# SmolLM2-135M Model Card

Dense (non-MoE) with Grouped Query Attention (GQA) and SwiGLU activations.

### Dense Model
We compared the dense SmolLM2-135M against a 160M parameter Mixture-of-Experts (MoE) baseline.

![SmolLM2 vs Baseline](smollm2_comparison.png)

*Figure 1: Validation loss comparison over a short 500-step training run.*


## Optimization: `torch.compile` (PyTorch 2.0+)
We tested the impact of `torch.compile` on training speed for the dense model architecture.

### Benchmark Results (RTX 4090)
| Steps | Mode | Total Time | Throughput | improvement |
|:---|:---|:---|:---|:---|
| **300** | Eager | 2.16 min | ~2.32 steps/sec | Baseline (Short) |
| **300** | Compile | 3.86 min | ~1.29 steps/sec | Slower (Overhead) |
| **600** | Eager | 6.32 min | ~1.58 steps/sec | Baseline (Long) |
| **600** | **Compile** | **4.80 min** | **~2.08 steps/sec** | **+31% Faster** |

**Conclusion:**
For training runs longer than ~500 steps, **`torch.compile` provides substantial speedups (~30%+)**, as the initial compilation overhead is amortized over the longer run. The default configuration now enables `compile_model=True` for this model.

### Training Cost Estimates
Based on the optimized throughput (~2.08 steps/sec) and default batch settings (Batch 4, GradAccum 12, Seq 2048 = ~98k tokens/step):

| Tokens | Steps | RTX 4090 (24GB) | H100 (80GB)* |
|:---|:---|:---|:---|
| **1 Billion** | ~10k | ~1.4 hours | ~34 mins |
| **10 Billion** | ~101k | ~13.6 hours | ~5.4 hours |
| **100 Billion** | ~1M | ~5.7 days | ~2.3 days |

*\*H100 estimates assume ~2.5x speedup via larger batch size (80GB VRAM) and faster compute.*

---

## Training Pipeline Instructions

### 2-Stage Training (Pretraining → SFT)

#### Stage 1: Pretraining

**Data Preparation (100M tokens)**
```bash
python data/prepare_mix_data.py --target_tokens 100000000
```

This creates a mixed dataset from:
- **70% FineWeb-Edu**: High-quality educational web content
- **30% Cosmopedia v2**: Synthetic educational data

**Training**
```bash
python train_llm.py \
  --config_class configs.pretrain_config.PretrainConfig \
  --dataset_path ./processed_data/pretrain_mix_100000000 \
  --experiment_name pretrain_100m \
  --max_steps 1000
```

#### Stage 2: Supervised Fine-Tuning (SFT)

**Data Preparation**
```bash
python data/prepare_sft_data.py
```

Uses SmolTalk dataset:
- **80% smol-magpie-ultra**: High-quality instructions
- **20% everyday-conversations**: Natural dialogue

**Training**
```bash
python train_llm.py \
  --config_class configs.sft_config.SFTConfig \
  --dataset_path ./processed_data/sft_mix \
  --load_checkpoint ./checkpoints/pretrain_100m/final_model.pt \
  --experiment_name sft \
  --max_steps 500
```

### Quick Test Pipeline (1M tokens - for debugging)

```bash
# 1. Prepare small dataset
python data/prepare_mix_data.py --target_tokens 1000000

# 2. Test training (5 steps)
python train_llm.py \
  --config_class configs.pretrain_config.PretrainConfig \
  --dataset_path ./processed_data/pretrain_mix_1000000 \
  --experiment_name test_run \
  --max_steps 5
```

---

## Known Issues & Workarounds

### Issue #1: Python-Edu Dataset

**Problem:** The `python-edu` subset in `smollm-corpus` only contains metadata (blob_id, repo_name, path) without actual code text.

**Workaround:** Currently skipped in data preparation. To enable:
1. Implement S3 download from Software Heritage bucket
2. Expect ~6 hours download time on 16-core AWS instance

**Implementation:**
```python
import boto3, gzip

def download_contents(blob_id):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
    with gzip.GzipFile(fileobj=obj['Body']) as f:
        return {"text": f.read().decode("utf-8", errors="ignore")}

ds = ds.map(download_contents, input_columns="blob_id")
```

### Issue #2: Tensor Format for Preprocessed Data

**Problem:** Preprocessed datasets may not auto-convert to PyTorch tensors.

**Solution:** Fixed in `train_llm.py` - automatically sets tensor format when loading preprocessed datasets.

---

## Configuration Files

- **`configs/pretrain_config.py`**: Pretraining hyperparameters (aggressive LR)
- **`configs/sft_config.py`**: SFT hyperparameters (10x lower LR to preserve knowledge)
- **`configs/llm_config.py`**: Base model architecture (135M parameters)
- **`configs/dataset_config.py`**: Dataset loading configuration

### Key Hyperparameters

| Stage | Muon LR | AdamW LR | Batch Size | Grad Accum | Effective Batch |
|-------|---------|----------|------------|------------|-----------------|
| **Pretrain** | 0.003 | 0.0003 | 4 | 12 | 48 seqs (~98K tokens) |
| **SFT** | 0.0003 | 0.00003 | 4 | 12 | 48 seqs (~98K tokens) |

---

## Output Files

Training produces the following outputs in `checkpoints/<experiment_name>/`:

- **`final_model.pt`**: Complete checkpoint with model state, config, and metrics
- **`model.pt`**: Model checkpoint (also saved during training)
- **`metrics.json`**: Training history and final evaluation results
- **`metrics_plot.png`**: Visualization of training curves

---

## Dataset Information

### Pretraining Mix
- **Source**: HuggingFaceTB/smollm-corpus
- **Components**: FineWeb-Edu (70%) + Cosmopedia v2 (30%)
- **Token Distribution**: Configurable via `--target_tokens`
- **Chunking**: Fixed 2048-token sequences

### SFT Mix
- **Source**: HuggingFaceTB/smoltalk
- **Components**: smol-magpie-ultra (80%) + everyday-conversations (20%)
- **Format**: ChatML with full conversation packing
- **Default Size**: ~50K instruction samples