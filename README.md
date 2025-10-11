# Blueberry LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install -r requirements.txt

python train_moe.py
```

## About

Purpose of this repository is to research better, faster, smarter LLMs.

This repository contains cutting-edge language model experiments and architectures. We believe scientists do their best work when given freedom to explore, so this is a space for your independent research and discovery.

Fork this repository, create a new experiment in `experiments/` folder, then create a pull request to merge it back.

## Experiments

> Each experiment below is validated on a specific git tag. 
> Later commits may introduce breaking changes. 
> To run an experiment with correct version of the repo, checkout its validated tag using: `git checkout <tag-name>`

| Experiment | Researcher | Validated Tag | Research Question | Key Findings | Tutorials |
|------------|-----------|---------------|-------------------|--------------|-----------|
| *Your experiments will be added here* | | | | | |
| [Exp5: Batch Size vs Sequence Length](experiments/exp5_batch_vs_seqlen_ablation/) | Vuk Rosić [YouTube](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g) [GitHub](https://github.com/vukrosic) | | Should you use big batches with short sequences, or small batches with long sequences? | A balanced approach to batch size and sequence length (e.g., 26x1024) is most effective. It outperforms both large batches with short sequences and small batches with long sequences. While large batches may seem to train faster, longer sequences ultimately provide better learning. | [📺 Video](https://youtu.be/bu5dhaLmr7E) [📝 Article](https://opensuperintelligencelab.com/learn/large-language-models/batch-size-vs-sequence-length/) |
| [Exp4: AMP vs FP32 on T4](experiments/exp4_amp_fp32_t4/) | Sumner Marston [GitHub](https://github.com/Summykai) | | When should you use mixed precision (FP16) vs full precision (FP32) on an NVIDIA T4 GPU? | On a T4 GPU, mixed precision (AMP) is up to 2x faster than full precision (FP32) when processing over 600-1,000 tokens at once. Larger models benefit from AMP with fewer tokens, while smaller models require more tokens to see a speedup. | - |
| [Exp3: PLASA + GDN Hybrid](experiments/exp3_plasa_gdn_hybrid/) | Overtimepog [GitHub](https://github.com/overtimepog) | `git checkout experiments-v1.0` | 1. Can per-layer adaptive sparse attention (PLASA) with progressive sparsity scheduling improve upon the uniform sparse attention tested in Exp1? <br><br> 2. Does the PROGRESSIVE_SPARSE schedule align with transformer layer hierarchy (dense early layers, aggressive sparse middle layers, moderate sparse late layers)? <br><br> 3. Which combination produces the best efficiency-performance tradeoff across 11 patterns (pure architectures + PLASA hybrids + Original hybrids)? | A full Per-Layer Adaptive Sparse Attention (PLASA) architecture is optimal. It significantly improves validation loss, accuracy, and training speed compared to hybrid models or full attention. The results also confirm that applying progressively sparser attention to the middle layers of a transformer is a highly effective strategy. | - |
| [Exp1: DSA + GDN Hybrid](experiments/exp1_dsa_gdn_hybrid/) | Vuk Rosić [YouTube](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g) [GitHub](https://github.com/vukrosic) | `git checkout experiments-v1.0` | 1. Can replacing full attention with DeepSeek Sparse Attention (DSA) improve the efficiency and performance of a hybrid attention architecture that combines full attention and Gated DeltaNet (GDN)? <br><br> 2. Which combination of attention mechanisms across layers produces the best efficiency-performance tradeoff: (1) Full Attention + GDN, (2) DSA + GDN, (3) DSA only, or (4) Full Attention only? |Using DeepSeek Sparse Attention (DSA) in a hybrid model with Gated DeltaNet (GDN) provides faster initial training. However, full attention seems to achieve better performance over longer training runs. The optimal combination of attention mechanisms still requires more research. | [📺 Video](https://youtu.be/kAEPS_AUGy8) [📝 Article](https://opensuperintelligencelab.com/blog/deepseek-sparse-attention/) |

## Getting Started

1. **Fork this repository** - Click the "Fork" button at the top right of this page to create your own copy
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/blueberry-llm.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Read `CONTRIBUTING.md` for contribution guidelines
5. Create your own experiment and merge it
6. Explore the `experiments/` folder for ongoing research and inspiration
7. Once you finish with your research, create a pull request to merge it back to this repo

## Philosophy

We don't prescribe what to research. Instead, we provide:
- Freedom to explore interesting ideas
- Infrastructure to test hypotheses
- A collaborative environment for learning

## Structure

- **`experiments/`** - Research experiments with their own documentation
- **`models/`** - Model architectures and implementations (DeepSeek, Qwen3-Next)
- **`training/`** - Training scripts and utilities
- **`configs/`** - Configuration files

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.