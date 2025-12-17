import argparse
import os
import random
import json
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def process_stream(name, dataset, tokenizer, target_tokens, f_handle, chunk_size=2048):
    print(f"  Processing {name} for {target_tokens:,} tokens...")
    iterator = iter(dataset)
    current_tokens = 0
    token_buffer = []
    
    while current_tokens < target_tokens:
        try:
            text = next(iterator)['text']
            ids = tokenizer.encode(text, add_special_tokens=True)
            token_buffer.extend(ids)
            current_tokens += len(ids)
            
            # Flush complete chunks
            while len(token_buffer) >= chunk_size:
                chunk = token_buffer[:chunk_size]
                json.dump({"input_ids": chunk, "labels": chunk}, f_handle)
                f_handle.write("\n")
                token_buffer = token_buffer[chunk_size:]
                
            if current_tokens % 1_000_000 < 5_000:
                print(f"    {name}: {current_tokens:,} / {target_tokens:,}", end='\r')
                
        except StopIteration:
            print(f"    {name} ran out of data!")
            break
            
    print(f"    {name}: Collected {current_tokens:,} tokens.")
    return current_tokens

def prepare_pretraining_data(args):
    print(f"🚀 Preparing pre-training data (Target: {args.target_tokens:,} tokens) [Sequential + Shuffle]...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    output_path = os.path.join(args.output_dir, f"pretrain_mix_{args.target_tokens}")
    jsonl_path = os.path.join(args.output_dir, "temp_mix.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ratios - 70% FineWeb-Edu, 30% Cosmopedia
    target_fw = int(args.target_tokens * 0.7)
    target_cosmo = int(args.target_tokens * 0.3)
    
    with open(jsonl_path, "w") as f:
        # 1. FineWeb-Edu
        ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
        process_stream("FineWeb-Edu", ds_fineweb, tokenizer, target_fw, f)
        
        # 2. Cosmopedia
        ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
        process_stream("Cosmopedia", ds_cosmo, tokenizer, target_cosmo, f)
        
        # 3. Python-Edu removed as requested
        
    print("\nLoading JSONL...")
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    
    print("Global Shuffling...")
    ds = ds.shuffle(seed=42)
    
    print(f"Saving to {output_path}...")
    ds.save_to_disk(output_path)
    
    print("Cleaning up...")
    os.remove(jsonl_path)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens", type=int, default=100_000_000, help="Number of tokens to prepare")
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="Output directory")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Tokenizer")
    
    args = parser.parse_args()
    prepare_pretraining_data(args)
