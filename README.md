# Blueberry LLM 2

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install -r requirements.txt
```

## About

Purpose of this repository is to research better, faster, smarter LLMs.

This repository contains cutting-edge language model experiments and architectures. We believe scientists do their best work when given freedom to explore, so this is a space for your independent research and discovery.

Fork this repository, create a new experiment in `experiments/` folder, then create a pull request to merge it back.

## Experiments

| Experiment | Researcher | Research Question | Key Findings |
|------------|-----------|-------------------|--------------|
| [Exp1: DSA + GDN Hybrid](experiments/exp1_dsa_gdn_hybrid/) | Vuk Rosić [YouTube](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g) [GitHub](https://github.com/vukrosic) | 1. Can replacing full attention with DeepSeek Sparse Attention (DSA) improve the efficiency and performance of a hybrid attention architecture that combines full attention and Gated DeltaNet (GDN)? <br><br> 2. Which combination of attention mechanisms across layers produces the best efficiency-performance tradeoff: (1) Full Attention + GDN, (2) DSA + GDN, (3) DSA only, or (4) Full Attention only? | 1. Trains faster in the beginning, but full attention seems to surpass it with more training. Future work is to investigate this further. <br><br> 2. Currently L → F → F → L (Gated Deltanet → Full Attention → Full Attention → Gated Deltanet). Future work is to investigate this further. |
| *Your experiments will be added here* |

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Read `CONTRIBUTING.md` for contribution guidelines
3. Create your own experiment and merge it
4. Explore the `experiments/` folder for ongoing research and inspiration
5. Check "Future Work" sections in experiments for inspiration

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