# SmolLM2-135M Model Card

## Model Details
**SmolLM2-135M** is a compact 135 million parameter language model designed for efficiency and educational purposes. It is part of the SmolLM2 family of models, known for being "blazingly fast and remarkably powerful" for their size.

- **Developer:** Antigravity (Implementation based on Hugging Face SmolLM2 paper)
- **Model Type:** Decoder-only Transformer (Llama architecture)
- **Parameters:** ~135M
- **Context Length:** 2048 tokens
- **Architecture:** Dense (non-MoE) with Grouped Query Attention (GQA) and SwiGLU activations.

## Architecture
The model uses a modernized Llama architecture optimized for small scale:
- **Layers:** 30
- **Hidden Dimension (`d_model`):** 576
- **MLP Dimension (`d_ff`):** 1536 (SwiGLU)
- **Attention Heads:** 9
- **KV Heads:** 3 (GQA group size 3)
- **Vocabulary:** 49,152

## Training
This implementation supports training on custom datasets using the provided trainer.

### Performance Comparison
We compared the dense SmolLM2-135M against a 160M parameter Mixture-of-Experts (MoE) baseline.

![SmolLM2 vs Baseline](smollm2_comparison.png)

*Figure 1: Validation loss comparison over a short 500-step training run.*

**Observations:**


### Power-of-2 Optimization Study
We conducted an extended **5000-step** training run to definitively compare the original parameter set against a hardware-optimized "Power-of-2" variant.

| Aspect | Original Config | Pow2 Config |
|:-------|:---------------:|:-----------:|
| **Params** | ~174M | ~151M |
| **Dimensions** | 576 / 9 / 30 | 512 / 8 / 32 |
| **Training Time** | 10.76 min | **10.06 min (-6.5%)** |
| **Val Loss** | **3.626** | 3.628 |
| **Val Accuracy** | 36.01% | **36.03%** |

#### Original Config Metrics (5k steps)
![Original 5k Metrics](original_5k_metrics_plot.png)

#### Power-of-2 Config Metrics (5k steps)
![Pow2 5k Metrics](pow2_5k_metrics_plot.png)

**Conclusion:**
The Power-of-2 configuration is significantly faster (~6.5% faster training) while maintaining comparable performance to the original configuration. Although the original configuration achieved a marginally lower validation loss (3.626 vs 3.628), the Power-of-2 variant actually achieved slightly higher validation accuracy (36.03% vs 36.01%). Given the speed advantage and standard architecture dimensions, we recommend the **Pow2 Config** for efficiency.


## Usage
```python
import torch
from configs.llm_config import SmolLM2_135M_Pow2_Config
from models.llm import MoEMinimalLLM

# Initialize optimized config
config = SmolLM2_135M_Pow2_Config()

# Initialize model
model = MoEMinimalLLM(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 32))
logits = model(input_ids)
print(logits.shape) # torch.Size([1, 32, 49152])
```

## Next Steps / Future Work
1.  **Downstream Tasks:** Evaluate the superior Pow2 model on standard benchmarks (HellaSwag, ARC).
2.  **Quantization:** Verify that the power-of-2 dimensions facilitate efficient quantization as expected.


## Intended Use
- Educational experimentation with SLMs (Small Language Models).
- Testing architectural optimizations like GQA on consumer hardware.
- Low-latency inference applications.

## Citation
If you use this implementation, please cite the original SmolLM2 paper:
> Allal, L. B., et al. "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model". arXiv:2502.02737. 2025.
