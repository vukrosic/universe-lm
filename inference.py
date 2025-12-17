#!/usr/bin/env python3
"""
Inference script for trained language models.
Supports interactive text generation with the trained model.
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from models.llm import MoEMinimalLLM
from configs.llm_config import Blueberry80GBConfig, Blueberry24GBConfig, DebugConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Try to infer config - default to Blueberry80GBConfig
        print("⚠️  Config not found in checkpoint, using Blueberry80GBConfig")
        config = Blueberry80GBConfig()
    
    # Initialize model
    model = MoEMinimalLLM(config).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle compiled models (remove _orig_mod. prefix)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    model.eval()
    
    # Load tokenizer
    tokenizer_name = getattr(config, 'tokenizer_name', 'HuggingFaceTB/SmolLM2-135M')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return model, tokenizer, config


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda"
):
    """Generate text from a prompt."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # Get logits - handle both tuple and tensor returns
        outputs = model(generated_ids, return_aux_loss=False)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Check for EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def interactive_mode(model, tokenizer, config, args):
    """Run interactive inference mode."""
    print("\n" + "="*70)
    print("🤖 Interactive Inference Mode")
    print("="*70)
    print(f"Model: {config.name if hasattr(config, 'name') else 'Blueberry-Nano'}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Device: {args.device}")
    print("\nGeneration settings:")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-k: {args.top_k}")
    print(f"  - Top-p: {args.top_p}")
    print("\nType 'quit' or 'exit' to stop.")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("\n📝 Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\n🔮 Generating...\n")
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )
            
            print(f"💬 Output:\n{generated}\n")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference script for trained LLMs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/moe_training/final_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for generation (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    print(f"\n📦 Loading model from {args.checkpoint}...")
    model, tokenizer, config = load_model(args.checkpoint, args.device)
    print("✅ Model loaded successfully!\n")
    
    if args.prompt:
        # Single prompt mode
        print(f"📝 Prompt: {args.prompt}\n")
        print("🔮 Generating...\n")
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        print(f"💬 Output:\n{generated}\n")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, config, args)


if __name__ == "__main__":
    main()
