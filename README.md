# Blueberry LLM 2

Open superintelligence lab - we do open research. Learn real AI research done in real companies.

## Latest Experiment Results

### Experiment 1: DSA + GDN Hybrid Attention Variants

**Training Configuration:**
- 4 layers, 128 hidden dimension, ~14M parameters
- 4 experts, top-2 routing, MoE every 2 layers
- 1000 training steps per pattern
- 8 different attention pattern combinations tested

**Results Summary:**

| Rank | Pattern | Type | Val Loss | Val Acc | Val PPL | Time |
|------|---------|------|----------|---------|---------|------|
| 🏆 1 | Sandwich | Original | 5.4012 | 36.25% | 221.67 | 75.6s |
| 🥈 2 | Linear First | Original | 5.4712 | 35.84% | 237.74 | 85.4s |
| 🥉 3 | Alternating | Original | 5.5215 | 32.41% | 250.02 | 78.7s |
| 4 | Full First | Original | 5.9077 | 27.22% | 367.86 | 84.6s |
| 5 | DSA Sandwich | DSA | 6.3442 | 22.83% | 569.18 | 84.7s |
| 6 | DSA Linear First | DSA | 6.3947 | 21.60% | 598.67 | 84.0s |
| 7 | DSA Alternating | DSA | 6.5257 | 20.08% | 682.44 | 83.3s |
| 8 | DSA Full First | DSA | 6.7721 | 16.65% | 873.12 | 83.3s |

**Category Comparison:**

| Category | Avg Val Loss | Avg Val Acc | Avg Val PPL | Avg Time |
|----------|--------------|-------------|-------------|----------|
| **Original** (Full + Linear Attention) | 5.5754 | 32.93% | 269.32 | 81.1s |
| **DSA** (DeepSeek Sparse + Linear) | 6.5092 | 20.29% | 680.85 | 83.8s |
| **Difference** | +0.9338 | -12.64% | +411.53 | +2.8s |

### Training Loss Curves

#### All 8 Patterns Comparison
![Loss Comparison - All Patterns](experiments/exp1_dsa_gdn_hybrid/results/loss_comparison.png)

#### Original vs DSA Average
![Loss Comparison - Category Averages](experiments/exp1_dsa_gdn_hybrid/results/loss_comparison_average.png)

### Key Findings

1. **Original architectures (Full + Linear Attention) outperform DSA variants** by a significant margin
   - ~17% lower validation loss on average
   - ~12.6% higher accuracy
   - ~60% lower perplexity

2. **Best performing pattern: Sandwich (L → F → F → L)**
   - Lowest validation loss: 5.4012
   - Highest accuracy: 36.25%
   - Best perplexity: 221.67

3. **Linear attention placement matters**
   - Linear attention at the beginning and end (sandwich pattern) works best
   - Having all full attention first (Full First) performs worst among original patterns

4. **DSA patterns show consistent underperformance**
   - All 4 DSA patterns perform worse than all 4 original patterns
   - DSA adds minimal training time overhead (~2.8s average)
   - The sparse attention mechanism may require longer training or different hyperparameters

### Pattern Notation
- **F** = Full Attention
- **L** = Linear Attention (Gated DeltaNet)
- **D** = DeepSeek Sparse Attention

### Experiment Details

See [experiments/exp1_dsa_gdn_hybrid/README.md](experiments/exp1_dsa_gdn_hybrid/README.md) for more details.
