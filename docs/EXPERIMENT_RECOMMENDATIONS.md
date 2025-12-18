# Things to try
Most of these are from the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)
FlexAttention, Value Embeddings, and Logit Softcapping

#### **ReLU² Squared Activation**
Replace the `SwiGLU` activation with `ReLU(x)**2`. While `SwiGLU` is generally better for final quality, `ReLU²` is faster to compute and has been shown to converge very aggressively in short speedruns.

#### **Untie Embeddings**
Currently, the `token_embedding` and `lm_head` are tied. Try untying them. This increases the parameter count slightly (without making the model "bigger" in terms of layers/dim) and allows the model to learn the output distribution more flexibly.