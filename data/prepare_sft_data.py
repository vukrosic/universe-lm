import argparse
import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def apply_chat_template(example, tokenizer):
    """
    Formats the conversation using the tokenizer's chat template.
    Expects example['messages'] to be a list of dicts: [{'role': 'user', 'content': ...}, ...]
    """
    messages = example['messages']
    # Use standard chat template (ensure tokenizer has one set, or use default ChatML)
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        # Fallback to manual ChatML if apply_chat_template fails/isn't set
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n" # Only if we were prompting generation, but for training we want full text
        # Actually for training we just want the full string.
    
    return {"text": formatted}

def prepare_sft_data(args):
    print(f"🚀 Preparing SFT data (max {args.max_samples} samples)...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # 1. Load Data
    # SmolTalk has multiple configs (subsets)
    # subset 1: smol-magpie-ultra (High quality instructions)
    print("Loading smol-magpie-ultra...")
    ds_magpie = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train")
    
    # subset 2: everyday-conversations (Natural chat)
    print("Loading everyday-conversations...")
    ds_chat = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
    
    # 2. Select Subsets
    # Shuffle and take top N
    max_magpie = min(len(ds_magpie), int(args.max_samples * 0.8))
    max_chat = min(len(ds_chat), int(args.max_samples * 0.2))
    
    ds_magpie = ds_magpie.shuffle(seed=42).select(range(max_magpie))
    ds_chat = ds_chat.shuffle(seed=42).select(range(max_chat))
    
    # 3. Format
    print("Formatting with ChatML...")
    
    # Combine
    combined_samples = []
    
    # Process Magpie
    for sample in ds_magpie:
         # Magpie structure is usually 'messages' column
         combined_samples.append(apply_chat_template(sample, tokenizer)['text'])
         
    # Process Chat
    for sample in ds_chat:
         combined_samples.append(apply_chat_template(sample, tokenizer)['text'])
    
    print(f"Total samples: {len(combined_samples)}")
    
    # 4. Tokenize
    print("Tokenizing...")
    all_token_ids = []
    for text in combined_samples:
        ids = tokenizer.encode(text, add_special_tokens=True)
        # For SFT, we ideally want to train on (Prompt + Response), masking Prompt loss.
        # But for simple LLM training we often just train on everything with packing.
        # Given "SmolLM2" style, we will just Pack them similar to pre-training for efficiency.
        # However, strictly SFT masking is better. 
        # For this implementation, we will use PACKING (next token prediction on full sequence)
        # to mesh well with our existing trainer which expects standard packed input_ids.
        all_token_ids.extend(ids)
    
    print(f"Total tokens collected: {len(all_token_ids):,}")
    
    # 5. Save
    output_path = os.path.join(args.output_dir, "sft_mix")
    os.makedirs(output_path, exist_ok=True)
    
    chunk_size = 2048
    total_length = len(all_token_ids)
    total_length = (total_length // chunk_size) * chunk_size
    reshaped_ids = np.array(all_token_ids[:total_length]).reshape(-1, chunk_size)
    
    print(f"Packed into {len(reshaped_ids):,} sequences.")
    
    final_ds = Dataset.from_dict({
        "input_ids": reshaped_ids,
        "labels": reshaped_ids 
    })
    
    print(f"Saving to {output_path}...")
    final_ds.save_to_disk(output_path)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="Output directory")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Tokenizer")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples for testing")
    
    args = parser.parse_args()
    prepare_sft_data(args)
