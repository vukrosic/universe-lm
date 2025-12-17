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