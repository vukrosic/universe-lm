import argparse
import os
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from models.vqvae import VQVAE
from configs.multimodal_config import MultimodalConfig

def prepare_multimodal_data(args):
    config = MultimodalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load VQ-VAE
    vq_model = VQVAE(num_embeddings=config.image_vocab_size).to(device)
    if os.path.exists(args.vqvae_path):
        print(f"Loading VQ-VAE from {args.vqvae_path}")
        vq_model.load_state_dict(torch.load(args.vqvae_path, map_location=device))
    else:
        print("Warning: VQ-VAE checkpoint not found. Using random weights for tokenization logic demonstration.")
    vq_model.eval()
    
    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # 3. Dataset
    print(f"Loading dataset {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    
    img_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    output_data = []
    
    print("Processing samples...")
    for i, sample in enumerate(dataset):
        if i >= args.max_samples:
            break
            
        # Text tokens
        caption = sample["text"]
        text_ids = tokenizer.encode(caption, add_special_tokens=True)
        
        # Image tokens
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_tensor = img_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            indices = vq_model.encode(img_tensor) # [1, 1024, 1] or similar
            indices = indices.view(-1).cpu().numpy()
            
        # Offset image tokens
        image_tokens = (indices + config.image_token_offset).tolist()
        
        # Interleave
        # [BOS] text [SEG_START] image_tokens [SEG_END] [EOS]
        # BOM is usually included in text_ids if add_special_tokens=True
        full_sequence = text_ids + [config.seg_start_id] + image_tokens + [config.seg_end_id]
        
        # We need to make sure it fits in max_seq_len
        if len(full_sequence) <= config.max_seq_len:
            # For simplicity, we just take the sequence as is. 
            # In a real training, we'd pad or pack.
            output_data.append({"input_ids": full_sequence, "labels": full_sequence})
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} samples")

    # Save to JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "multimodal_data.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in output_data:
            json.dump(entry, f)
            f.write("\n")
            
    print(f"✅ Saved {len(output_data)} samples to {jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_path", type=str, default="checkpoints/vqvae_epoch_10.pt")
    parser.add_argument("--dataset_name", type=str, default="lambdalabs/pokemon-blip-captions")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output_dir", type=str, default="./processed_data")
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()
    prepare_multimodal_data(args)
