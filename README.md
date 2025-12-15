# Looking for a name for this LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

### Goals

1. Create a network of contributors and open lab structure to develop the best LLMs.
2. Get more compute for a large-scale LLM project that will compete with the best LLMs.

## Quick Start

```bash
pip install -r requirements.txt


python train_moe.py
```

## Debugging

To run a quick debug session with a tiny model (useful for verifying the training loop or checking for errors on any hardware):

```bash
python debug_moe.py
```

This uses a `DebugMoEConfig` with a very small model size and runs for only 100 steps.


## Hyperparameter Tuning

To optimize training for your specific hardware or architecture changes, you can use the quick sweep tool:

```bash
python experiments/quick_sweep.py
```

This script will:
1.  Train the model with 3 different learning rates (0.5x, 1.0x, 2.0x) for 200 steps each.
2.  Clean up large checkpoint files automatically, keeping only metrics.
3.  Generate a comparison plot (`sweep_comparison.png`) to help you choose the best learning rate.

You can then apply the best learning rate in `configs/moe_config.py`.


## Baselines

We maintain baseline performance metrics to track improvements. All experiments should aim to surpass these baselines.

### 24GB GPU Baseline (GPU24GBMoEModelConfig)

Run on a single 24GB VRAM GPU (e.g., RTX 3090/4090).

- **Validation Loss**: 4.0977
- **Validation Accuracy**: 31.90%
- **Perplexity**: 60.20

![GPU 24GB Baseline Metrics](baselines/gpu_24gb/metrics_plot.png)

Trained on 1x4090.

If your experiment runs for more time, doesn't mean it's worse, you may be using weaker hardware. We will test it.

Full baseline results (metrics and plots) are stored in `baselines/gpu_24gb/`.


## Getting Started

1. **Fork this repository** - Click the "Fork" button at the top right of this page to create your own copy
    > **Note**: If you cannot fork the repository (e.g., because you already have a fork), you can clone it locally, create a new empty repository on GitHub, and push the code there. You may ask ChatGPT for help.
2. Clone your fork: `git clone URL_HERE`
3. Install dependencies: `pip install -r requirements.txt`
4. Read `CONTRIBUTING.md` for contribution guidelines
5. Create your own experiment and merge it
6. Explore the `experiments/` folder for ongoing research and inspiration
7. Once you finish with your research, create a pull request to merge it back to this repo

## Contributing

1. Pick a topic / task from [issues](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/issues) (issues are general name for tasks), carefully read it and understand it
2. Fork the repo
3. Clone it and implement the experiment, follow README
4. Once finished, report it back on Discord / Issue, we may merge your full code or just final results, depending on the task

See `CONTRIBUTING.md` for guidelines on how to contribute to this project