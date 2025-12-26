# Multimodal Image Generation (Autoregressive)

This implementation adds a "Hard Mode" multimodal image generation capability to the LLM, built from scratch without pre-trained weights.

## Roadmap

### 1. Visual Tokenizer (VQ-VAE)
Before the LLM can understand images, we must train a VQ-VAE to compress 128x128 images into discrete "visual words".
- **Script:** `train_vqvae.py`
- **Goal:** Learn a codebook of 1024 visual tokens.
- **Output:** `checkpoints/vqvae_epoch_10.pt`

### 2. Data Preparation
Convert an image-text dataset (like Pokemon-BLIP) into an interleaved sequence of tokens:
`[BOS] text [SEG_START] image_tokens [SEG_END] [EOS]`
- **Script:** `data/prepare_multimodal_data.py`
- **Output:** `processed_data/multimodal_data.jsonl`

### 3. Training the LLM
Train the `MinimalLLM` (Llama-style architecture) on the interleaved data.
- **Script:** `train_multimodal_llm.py`
- **Optimization:** Uses the **Muon** optimizer for high-efficiency single-GPU training.
- **Architecture:** 8 layers, 512 hidden size (approx. 40M parameters).

### 4. Inference
Generate a new image from a text prompt.
- **Script:** `generate_multimodal.py --prompt "A fire-breathing dragon"`
- **Output:** `generated_pokemon.png`

## How to Run

```bash
# 1. Train the VQ-VAE (Visual Dictionary)
python train_vqvae.py

# 2. Prepare the multimodal dataset
python data/prepare_multimodal_data.py

# 3. Train the Multimodal LLM
python train_multimodal_llm.py

# 4. Generate an image
python generate_multimodal.py --prompt "A cute water pokemon"
```
