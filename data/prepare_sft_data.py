import argparse
import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def process_messages_with_masking(messages, tokenizer):
    """
    Tokenizes messages strictly ensuring ChatML format.
    Returns input_ids and labels (masked user tokens).
    """
    input_ids = []
    labels = []
    
    # Check if messages is None or empty
    if not messages:
        return [], []

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if not role or not content:
            continue
            
        # Format: <|im_start|>role\ncontent<|im_end|>\n
        # We manually tokenize each part to ensure control
        
        # 1. Header: <|im_start|>role\n
        header_text = f"<|im_start|>{role}\n"
        header_ids = tokenizer.encode(header_text, add_special_tokens=False)
        
        # 2. Content
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        
        # 3. Footer: <|im_end|>\n
        footer_text = "<|im_end|>\n"
        footer_ids = tokenizer.encode(footer_text, add_special_tokens=False)
        
        # Combine
        part_input_ids = header_ids + content_ids + footer_ids
        
        # Build Labels
        if role == "assistant":
            # Assistant: Train on content and footer (to learn when to stop)
            # Mask header? Usually yes, we prompt with header.
            # Label = [-100]*header + content + footer
            part_labels = [-100] * len(header_ids) + content_ids + footer_ids
        else:
            # User/System: Mask everything
            part_labels = [-100] * len(part_input_ids)
            
        input_ids.extend(part_input_ids)
        labels.extend(part_labels)
        
    return input_ids, labels

def prepare_sft_data(args):
    print(f"🚀 Preparing SFT data (Assist-Only Prediction, max {args.max_samples} samples)...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Check special tokens
    if "<|im_start|>" not in tokenizer.get_vocab():
        print("Warning:Tokenizer might not support ChatML special tokens directly. Ensure they are added if needed.")
    
    # 1. Load Data
    print("Loading smol-magpie-ultra...")
    ds_magpie = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train")
    
    # 2. Select Subset
    ds_magpie = ds_magpie.shuffle(seed=42).select(range(min(len(ds_magpie), args.max_samples)))
    
    # 3. Process
    print("Processing and Tokenizing...")
    
    all_input_ids = []
    all_label_ids = []
    
    count = 0
    for sample in ds_magpie:
        # Expect 'messages' list
        msgs = sample.get('messages', [])
        ids, labs = process_messages_with_masking(msgs, tokenizer)
        
        all_input_ids.extend(ids)
        all_label_ids.extend(labs)
        
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count} conversations...", end='\r')
            
    print(f"\nTotal tokens collected: {len(all_input_ids):,}")
    
    # 4. Packing
    chunk_size = 2048
    total_length = len(all_input_ids)
    # Truncate to multiple of chunk_size
    n_chunks = total_length // chunk_size
    valid_len = n_chunks * chunk_size
    
    print(f"Packing into {n_chunks} sequences of {chunk_size}...")
    
    packed_input_ids = np.array(all_input_ids[:valid_len]).reshape(-1, chunk_size)
    packed_labels = np.array(all_label_ids[:valid_len]).reshape(-1, chunk_size)
    
    # 5. Save
    output_path = os.path.join(args.output_dir, "sft_mix")
    os.makedirs(output_path, exist_ok=True)
    
    final_ds = Dataset.from_dict({
        "input_ids": packed_input_ids,
        "labels": packed_labels 
    })
    
    print(f"Saving to {output_path}...")
    final_ds.save_to_disk(output_path)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="Output directory")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Tokenizer")
    parser.add_argument("--max_samples", type=int, default=50000, help="Max samples") # Default to reasonable amount for 1B run info
    
    args = parser.parse_args()
    prepare_sft_data(args)
