# 🫐 Blueberry-Nano Speedrun Challenge

## The "100M Dash" Standard
To allow rapid iteration, all experimental runs will use the **100M Token** benchmark.
*   **Duration**: ~15 Minutes (on RTX 4090).
*   **Command**: `python train_llm.py --train_tokens 100000000`

---

## Categories

### 1. 🏎️ Fastest Training On 100M Tokens (Any% Speedrun)
**Goal**: Lowest **Time to Finish 100M Tokens**.
*   **Constraint**: Model parameters must remain **~151M (±5%)**.
*   **Relaxed Rules**: You DO NOT need to use the exact default config. You can change `d_model`, `layers`, or use custom kernels/fused ops, as long as the parameter count stays similar and the model **converges** (Loss < 4.5 @ 100M).
*   **Focus**: Throughput, MFU, Kernel optimization.

### 2. 🧠 Lowest Val Loss At 100M Tokens (100% Completion)
**Goal**: Lowest **Validation Loss** after 100M Tokens.
*   **Constraint**: Runtime must be under **20 Minutes** (on 4090 or equivalent).
*   **Focus**: Architecture (RoPE, GQA), Hyperparameters, Scheduler, Curriculum.

### 3. ⚡ Fastest To 3.0 Val Loss (Glitchless)
**Goal**: Reach **Validation Loss < 3.0** in the shortest time.
*   **Time Limit**: **4 Hours** (Strict).
*   **Dataset**: Must use official `blueberry-1B-pretrain`.
*   **Metric**: Wall-clock time to hit the target.

### 4. 🏅 The 1B Marathon (World Record)
**Goal**: Best overall model trained on the full 1B dataset.
*   **Time Limit**: **4 Hours** (Strict).
*   **Dataset**: Must use official `blueberry-1B-pretrain`.
*   **Metric**: Final Eval Loss & Downstream Benchmarks.

---

## 🤝 Compute Sponsorship & Verification
**Don't have a 4090? We've got you.**

1.  **Verification**: If you optimize the code but can't verify the exact time, submit a Pull Request. We will run your code on our standardized hardware to confirm the record!
2.  **Sponsorship**: If you have a great idea (e.g., a new architecture) but no GPU, open an Issue/Ticket. If the idea looks promising, we will run the experiment for you and credit you.

## 📝 How to Submit
1.  Run your experiment.
2.  Take a screenshot of the loss curve/metrics.
3.  Open a Pull Request with your code changes and add your result to `LEADERBOARD.md`.
