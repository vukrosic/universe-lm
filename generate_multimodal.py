import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.llm import MinimalLLM
from models.vqvae import VQVAE
from configs.multimodal_config import MultimodalConfig
from PIL import Image
import os

def generate(prompt, model_path, vqvae_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MultimodalConfig()
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    # 2. Load LLM
    model = MinimalLLM(config).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Load VQ-VAE
    vq_model = VQVAE(num_embeddings=config.image_vocab_size).to(device)
    if os.path.exists(vqvae_path):
        vq_model.load_state_dict(torch.load(vqvae_path, map_location=device))
    vq_model.eval()
    
    # 4. Prepare Prompt
    text_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # Append <seg_start> to force image generation start
    input_ids = torch.tensor(text_ids + [config.seg_start_id], device=device).unsqueeze(0)
    
    # 5. Generate Autoregressively
    print(f"Generating image for prompt: {prompt}")
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(config.num_image_tokens):
            logits = model(input_ids)
            last_logits = logits[:, -1, :]
            
            # Simple greedy sampling for now
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            if next_token.item() == config.seg_end_id:
                break
    
    # 6. Extract and Decode Image Tokens
    # Remove offset to get back to VQ codebook range
    image_tokens = [t - config.image_token_offset for t in generated_tokens if config.image_token_offset <= t < config.image_token_offset + config.image_vocab_size]
    
    if len(image_tokens) == 0:
        print("Error: No image tokens generated.")
        return
        
    print(f"Decoded {len(image_tokens)} tokens.")
    
    # Pad if necessary to match expected 32x32 = 1024
    if len(image_tokens) < 1024:
        image_tokens += [0] * (1024 - len(image_tokens))
    image_tokens = image_tokens[:1024]
    
    tokens_tensor = torch.tensor(image_tokens, device=device).unsqueeze(1)
    
    with torch.no_grad():
        # Our VQ-VAE expects 1024 tokens for 128x128 image (32x32 grid)
        decoded_img = vq_model.decode(tokens_tensor, 32, 32)
        
    # 7. Save Image
    # Denormalize
    decoded_img = (decoded_img + 1) / 2
    decoded_img = decoded_img.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    decoded_img = (decoded_img * 255).astype("uint8")
    
    img = Image.fromarray(decoded_img)
    img.save("generated_pokemon.png")
    print("✅ Image saved to generated_pokemon.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A friendly green dragon")
    parser.add_argument("--model_path", type=str, default="checkpoints/multimodal_llm_epoch_5.pt")
    parser.add_argument("--vqvae_path", type=str, default="checkpoints/vqvae_epoch_10.pt")
    args = parser.parse_args()
    
    generate(args.prompt, args.model_path, args.vqvae_path)
