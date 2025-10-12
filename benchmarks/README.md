# Benchmarks

Centralized benchmark suite for evaluating language models across experiments.

## Available Benchmarks

### ARC-Challenge
Grade-school level science question answering (multiple choice).

- **Dataset**: AI2 Reasoning Challenge (Challenge Set)
- **Metric**: Accuracy
- **Splits**: train (1,119), validation (299), test (1,172)

### HellaSwag
Commonsense reasoning via sentence completion (4 choices).

- **Dataset**: HellaSwag
- **Metric**: Accuracy (lower perplexity = better)
- **Splits**: train, validation, test

## Usage

### Single Model Evaluation

```bash
# ARC-Challenge
python benchmarks/arc_challenge.py \
    --checkpoint experiments/exp7_hybrid_deltanet_ablation/checkpoints/best_model.pt

# HellaSwag
python benchmarks/hellaswag.py \
    --checkpoint experiments/exp7_hybrid_deltanet_ablation/checkpoints/best_model.pt

# Custom split and sample limit
python benchmarks/arc_challenge.py \
    --checkpoint exp7/checkpoints/best_model.pt \
    --split validation \
    --max-samples 100
```

### Compare Multiple Models

```bash
# Compare exp6 vs exp7
python benchmarks/compare_models.py \
    experiments/exp6_gated_deltanet_training/checkpoints/best_model.pt \
    experiments/exp7_hybrid_deltanet_ablation/checkpoints/best_model.pt

# Quick comparison (100 samples per benchmark)
python benchmarks/compare_models.py \
    --checkpoints exp6/checkpoints/best_model.pt exp7/checkpoints/best_model.pt \
    --max-samples 100

# Only ARC-Challenge
python benchmarks/compare_models.py \
    --checkpoints exp6/checkpoints/best_model.pt exp7/checkpoints/best_model.pt \
    --benchmarks arc
```

## Output

Results are saved to JSON files:
- Single benchmarks: `<experiment>/results/<benchmark>_<split>_results.json`
- Comparisons: `benchmark_comparison_results.json` (or custom path via `--output`)

## Performance Baselines

### ARC-Challenge (validation set)

| Model | Accuracy |
|-------|----------|
| Random | 25% |
| GPT-3 | ~50-60% |
| GPT-4 | ~80-85% |

### HellaSwag (validation set)

| Model | Accuracy |
|-------|----------|
| Random | 25% |
| GPT-3 | ~60-70% |
| GPT-4 | ~85-90% |

## Adding New Benchmarks

1. Create `benchmarks/<benchmark_name>.py`
2. Use `common.load_model_from_checkpoint()` for model loading
3. Return results dict with standard fields: `total_samples`, `correct`, `accuracy`, `accuracy_percent`
4. Add to `compare_models.py` if needed

## Module Structure

```
benchmarks/
├── __init__.py              # Package init
├── common.py                # Shared utilities (model loading, device setup)
├── arc_challenge.py         # ARC-Challenge benchmark
├── hellaswag.py             # HellaSwag benchmark
├── compare_models.py        # Multi-model comparison
└── README.md                # This file
```

