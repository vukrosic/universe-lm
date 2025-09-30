Research questions:
1. Why do we need extra weight for indexer score?



# DeepSeek Sparse Attention

Prerequisites: Attention Mechanism

**📺 Recommended Video Resource:** For a comprehensive understanding of attention mechanisms and DeepSeek's Multihead Latent Attention, watch this video: [https://youtu.be/TfEG0TwueTs](https://youtu.be/TfEG0TwueTs)

*   **If you're new to attention mechanisms:** Start from the beginning of the video
*   **If you want to focus on DeepSeek's Multihead Latent Attention (MLA):** Jump to 38:53 or use this direct link: [https://youtu.be/TfEG0TwueTs?t=2333](https://youtu.be/TfEG0TwueTs?t=2333)
*   **Note:** I will explain MLA again in this article / video, but I recommend watching both for better understanding.

Standard Transformers use an "attention" mechanism where every new token being generated looks back at all the previous tokens in the sequence.

This is computationally very expensive. If you have a sequence of length L, the complexity is O(L²), meaning the computation and memory required grow quadratically.

Doubling the text length from 10,000 to 20,000 tokens doesn't just double the cost—it quadruples it. This makes processing very long documents (like books or large codebases) prohibitively slow and expensive.

Instead of having each token attend to all previous tokens, DeepSeek Sparse Attention (DSA) intelligently selects a small, fixed-size subset (k) of the most relevant previous tokens to attend to. This changes the complexity from O(L²) to O(L * k), which is much more manageable since k is a small constant (e.g., 2048) and L can be very large (e.g., 128,000).

DSA is made of two main components:

The lightning indexer will perform full attention between every token but it's a lot smaller and faster attenion - ReLU actionvation which is very fast and a lot smaller dimension of key and query.

#### Component 1: The Lightning Indexer

This is a fast and lightweight mechanism whose only job is to figure out which past tokens are important for the current token.

*   **How it works:** For the current token (`h_t`), the indexer quickly calculates an "index score" (`I_t,s`) for every previous token (`h_s`). This score represents the predicted relevance of token `s` to token `t`.
*   **Formula `(1)`:** The formula 1 is essentially a simplified attention calculation. It uses its own small set of queries (`q_I`) and keys (`k_I`) to compute these scores.
*   **Why it's "Lightning":** It's designed for speed. It uses a simple `ReLU` activation function and can be run with low-precision numbers (FP8), making it computationally very cheap, even though it still technically looks at all previous tokens (an `O(L²)` operation, but a very, very fast one).

### 1. The Formulas Explained (The "What")

The paper provides two key formulas that describe this two-step process.

#### **Formula (1): The Lightning Indexer**

`I_t,s = Σ H_I (j=1) [w_t,j^I ⋅ ReLU(q_t,j^I ⋅ k_s^I)]`

This formula calculates the **index score** (`I_t,s`), which represents the "relevance" of a past token `s` to the current token `t`. Let's break it down:

*   `I_t,s`: The final importance score. A higher score means token `s` is more important for token `t`.
*   `h_t` and `h_s`: These are the vector representations (hidden states) of the current token (`t`) and a previous token (`s`).
*   `q_t,j^I` and `k_s^I`: These are special, lightweight **query** and **key** vectors created just for the indexer (indicated by the `I` superscript). They are derived from `h_t` and `h_s` respectively.
*   `q_t,j^I ⋅ k_s^I`: This is a dot product, the fundamental operation in attention. It measures the similarity or compatibility between the query and the key.
*   `ReLU(...)`: A simple activation function (Rectified Linear Unit). It's very fast to compute. If the dot product is negative, it becomes 0; otherwise, it stays the same.
*   `w_t,j^I`: An additional weight, also derived from the query token `h_t`. It acts as a learned gate or importance factor for each indexer head `j`.
*   `Σ ...`: This sums the results across all the indexer's heads (`H^I`). The indexer has only a few heads to keep it fast.

**In simple terms:** The Lightning Indexer is a mini, simplified attention mechanism. Its only job is to quickly calculate a relevance score for every pair of tokens without doing the full, expensive attention computation.

#### **Formula (2): The Main Attention Calculation**

`u_t = Attn(h_t, {c_s | I_t,s ∈ Top-k(I_t,:)})`

This formula describes how the final output (`u_t`) is computed after the selection is done.

*   `u_t`: The final output hidden state for the current token `t`.
*   `Attn(...)`: This represents the main, powerful attention mechanism (in this case, Multi-Query Attention).
*   `h_t`: The query from the current token.
*   `{c_s | I_t,s ∈ Top-k(I_t,:)}`: This is the most important part. It means: "Use the set of key-value entries `c_s` **only if** their corresponding index score `I_t,s` (calculated in Formula 1) is among the `top-k` highest scores for the current token `t`."

**In simple terms:** The main attention mechanism is told to ignore almost all previous tokens and focus *only* on the handful of key-value entries that the Lightning Indexer identified as most important.

#### Component 2: The Fine-grained Token Selection
This component is simple: it takes all the index scores calculated by the Lightning Indexer and picks the `top-k` highest scores.

*   **Function:** It acts as a gatekeeper. It tells the main, powerful attention mechanism: "You don't need to look at all 100,000 previous tokens. I've found the 2,048 most important ones for you. Just look at these."

The final attention output (`u_t`) is then calculated by the main attention module, but only using the current token's query and the `k` key-value pairs that were selected.

### Step 3: How The Model Was Trained

They didn't train this model from scratch. They cleverly adapted an existing, powerful model (**DeepSeek-V3.1-Terminus**) that was already trained on long contexts. The training happened in several stages.

#### Stage 1: Continued Pre-Training (Two Phases)

1.  **Dense Warm-up Stage:**
    *   **Goal:** To teach the brand-new Lightning Indexer what "important" tokens look like.
    *   **Method:** They froze the main model and kept the standard (dense) attention active. They then trained *only* the Lightning Indexer. The indexer's objective was to make its importance scores match the attention scores from the powerful, pre-trained main model. They used a KL-divergence loss, which is a way of measuring how similar two probability distributions are. In essence, they told the indexer: "Learn to predict what the main model *would have* paid attention to." This phase was very short (1,000 steps).

2.  **Sparse Training Stage:**
    *   **Goal:** To adapt the entire model to work with the sparse attention pattern.
    *   **Method:** They "switched on" the `top-k` selector, making the attention sparse. They unfroze the main model and trained everything together.
        *   The **main model** was trained on its usual task: predicting the next word (language modeling loss). It had to learn to perform well with only the limited context provided by the selector.
        *   The **Lightning Indexer** continued to be trained with the KL-divergence loss to align with the main model's attention, but now only on the selected `k` tokens.
    *   This was the main training phase (15,000 steps, using 943.7 billion tokens).

#### Stage 2: Post-Training
After the pre-training was done, they fine-tuned the model for specific tasks (like coding, math, reasoning, and following instructions) using Reinforcement Learning (RL). Crucially, they used the **exact same data and methods** as they did for the original DeepSeek-V3.1-Terminus model. This ensures a fair comparison between the dense and sparse models.