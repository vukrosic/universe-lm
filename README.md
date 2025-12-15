# 5-Dollar LLM (looking for a better name)

> Training the best possible LLM from scratch for $5.

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

---

## 🚀 Getting Started

To get started with development, follow these steps to set up your environment.

1. **Fork this repository** - Click the "Fork" button at the top right to create your own copy.
2. **Clone your fork**:
   ```bash
   git clone FORK_URL_HERE
   cd 5-dollar-llm
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
You may clone it with yout IDE as well.

## ⚡ Quick Start

Once installed, you can train a model using our default configurations.

### Train on Custom Hardware (Recommend 24GB+ VRAM)
To train the main MoE model (configured for a single 24GB GPU like an RTX 3090/4090):

```bash
python train_moe.py
```

### Debugging (Any Hardware)
To verify the training loop or check for errors on any hardware (including CPU/MPS), run the debug session:

```bash
python debug_moe.py
```
*Runs a `DebugMoEConfig` with a tiny model for 100 steps.*

## 📊 Baselines

We maintain baseline performance metrics to track improvements. All experiments should aim to surpass these benchmarks.

### 24GB GPU Baseline (GPU24GBMoEModelConfig)
*Hardware: Single Nvidia RTX 4090 (24GB)*

> You may train on other hardware.

| Metric | Value |
| :--- | :--- |
| **Validation Loss** | 4.0977 |
| **Validation Accuracy** | 31.90% |
| **Perplexity** | 60.20 |

![GPU 24GB Baseline Metrics](baselines/gpu_24gb/metrics_plot.png)

*Full baseline results are stored in `baselines/gpu_24gb/`.*

## 🤝 Contributing

We welcome all contributions!

1. **Pick a task**: Check the [Issues](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/issues) tab.
2. **Implement**: Fork, clone, and work on your experiment.
3. **Report**: Share results on Discord or the Issue thread. We may merge the full code or just the results.

Please read `CONTRIBUTING.md` for detailed guidelines.

### Goals
1. Create a network of contributors and open lab structure to develop state-of-the-art LLMs.
2. Secure compute resources for a larger-scale LLM project to compete with top LLMs.