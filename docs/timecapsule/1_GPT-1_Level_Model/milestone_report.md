# Milestone Report: GPT-1 Level Model (Blueberry-Nano)

**Date**: December 17, 2025  
**Project**: 5-Dollar-LLM  
**Milestone**: 151M parameters trained on 1 Billion tokens.

## 🚀 Model Identity: Blueberry-Nano
The Blueberry-Nano represents our first major leap into training "real" language models from scratch on consumer hardware. With 151M parameters, it sits in the same class as the original OpenAI GPT-1 (117M), but benefits from modern architectural improvements.

### 🏗️ Architecture Details
- **Parameters**: 151,000,000
- **Layers**: 32
- **Embedding Dimension (d_model)**: 512
- **Attention**: Grouped Query Attention (GQA) with 4 KV heads and 8 Query heads.
- **Context Length**: 2048 tokens
- **Vocabulary**: 49,152 (SmolLM2-135M Tokenizer)
- **Features**: Mixture of Experts (MoE) support (disabled for this dense run), RoPE embeddings, RMSNorm.

## 📊 Training Environment & Performance
- **Hardware**: Single RTX 4090 (24GB)
- **Training Time**: 156.80 minutes (approx. 2.6 hours)
- **Tokens Processed**: 1,000,000,000 (1B)
- **Optimizer**: Muon & AdamW
- **Precision**: AMP (Automatic Mixed Precision)

### Final Metrics
| Metric | Value |
|--------|-------|
| **Validation Loss** | 3.1940 |
| **Validation Accuracy** | 40.19% |
| **Validation Perplexity** | 24.38 |

![Training Plot](metrics_plot.png)

## 🔮 Inference Hall of Fame (Time Capsule)
These samples capture the model's "personality" and capabilities at the 1B token milestone.

### The Good: Creative Storytelling
> **Prompt**: "Once upon a time"  
> **Output**: "Once upon a time, in a small town named Harmonyville, there lived two best friends named Sammy and Sally. They loved exploring their town and learning new things every day..."

### The Strange: Concept Confusion
> **Prompt**: "Explain backpropagation"  
> **Output**: Defines it as "increasing the probability of a given feature... such as the size of a block of cells." It also invented a new term: **"Neuralization"**.

### The Hallucination: Historical Loops
> **Prompt**: "A poem about a lonely robot on Mars"  
> **Output**: Reverted to a repetitive list of German physicists from 1962-1973.

## 📜 Historical Context (GPT-1 Comparison)
- **Hugging Face Model**: [vukrosic/Blueberry-Nano-151M](https://huggingface.co/vukrosic/Blueberry-Nano-151M)

In 2018, OpenAI's GPT-1 changed the world with:
- **Parameters**: 117M
- **Training Data**: ~5GB (BookCorpus)
- **Training Time**: ~1 month on 8 GPUs

Our **Blueberry-Nano** (151M) reached similar complexity in **under 3 hours** on a single consumer GPU, thanks to modern efficiencies like Flash Attention, GQA, and optimized training pipelines. This demonstrates the democratization of large-scale model training.

---
*Created by the Open Superintelligence Lab.*
