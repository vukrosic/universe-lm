# Recommended Experiments to Break the 6.7 Loss Record

This plan outlines several experiments inspired by the `modded-nanogpt` speedrun techniques to reach a training loss of 6.7 faster than the current record of 22.24s.

## User Review Required

> [!IMPORTANT]
> Some of these changes are architectural and might affect the model's behavior beyond just convergence speed. However, they are standard in the speedrunning community.

## Proposed Experiments

### 1. Architectural Tweaks

#### **Zero-Initialize Residual Projections**
Initialize the weight matrix of the last linear layer in each residual branch to exactly zero. In our codebase, this corresponds to:
*   `MultiHeadAttention.w_o`
*   `SwiGLUFeedForward.down_proj` (or `Expert.linear2` if using MoE)
This ensures the model starts as an identity mapping, allowing gradients to flow more easily in the early steps.

#### **ReLU² Squared Activation**
Replace the `SwiGLU` activation with `ReLU(x)**2`. While `SwiGLU` is generally better for final quality, `ReLU²` is faster to compute and has been shown to converge very aggressively in short speedruns.

#### **Untie Embeddings**
Currently, the `token_embedding` and `lm_head` are tied. Try untying them. This increases the parameter count slightly (without making the model "bigger" in terms of layers/dim) and allows the model to learn the output distribution more flexibly.

---

### 2. Optimization Tuning

#### **Higher Muon LR with Short Warmup**
The record uses `LR 0.015` with `Warmup 0`. Try:
*   `muon_lr: 0.02` or `0.025`
*   `warmup_ratio: 0.005` (very short warmup to stabilize the high LR)

#### **Muon Momentum Tuning**
The default is `0.95`. Try `0.98` for faster accumulation of gradients across steps, or `0.9` if the training is unstable at high LRs.

#### **Disable Gradient Clipping**
The trainer currently clips gradients at `1.0`. Removing clipping can speed up wall-clock time and sometimes allows the model to take larger, beneficial steps early on.

---

### 3. Training & Data Efficiency

#### **Increase Effective Batch Size**
If hardware permits, try increasing the `batch_size` or `gradient_accumulation_steps`. While more tokens per step might seem slower per-step, it often results in fewer steps to reach the target loss.

#### **Dropout to Zero**
Ensure `dropout: 0.0` is used. For a short speedrun to 6.7 loss, regularization is likely counter-productive.

## Verification Plan

### Automated Tests
*   Run a short training run (e.g., 50 steps) with each change to ensure stability and verify that loss is decreasing faster than the baseline.
*   Baseline: Use the record config (`LR 0.015, Warmup 0, Constant, GradAcc 1`).

### Manual Verification
*   Monitor wall-clock time to 6.7 loss using the `train_llm.py` script and compare with the 22.24s record.
