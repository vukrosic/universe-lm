#!/usr/bin/env python3
"""
Tokenizer Benchmark Script

Compares different tokenizers for pretraining efficiency:
- Vocab size (affects embedding layer parameters)
- Bytes/token ratio (tokenization efficiency)
- Tokenization speed
- Sequence length distribution

Usage:
    python utils/tokenizer_benchmark.py
    python utils/tokenizer_benchmark.py --tokenizers "meta-llama/Llama-3.2-1B,gpt2"
"""

import argparse
import time
import os
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class TokenizerStats:
    name: str
    vocab_size: int
    total_tokens: int
    total_bytes: int
    bytes_per_token: float
    tokens_per_second: float
    embedding_params: int  # for d_model=512
    estimated_speedup: float  # relative to baseline


# Default tokenizers to compare
DEFAULT_TOKENIZERS = [
    "HuggingFaceTB/SmolLM2-135M",  # Current baseline (vocab 49,152)
    "meta-llama/Llama-3.2-1B",     # Llama 3 (vocab 128,256) - requires auth
    "meta-llama/Llama-2-7b-hf",    # Llama 2 (vocab 32,000) - requires auth
    "mistralai/Mistral-7B-v0.1",   # Mistral (vocab 32,000)
    "gpt2",                         # GPT-2 (vocab 50,257)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # TinyLlama (vocab 32,000)
]


def get_sample_texts(num_samples: int = 1000) -> List[str]:
    """Get sample texts for benchmarking."""
    try:
        from datasets import load_dataset
        print(f"Loading {num_samples} sample texts from FineWeb-Edu...")
        ds = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "fineweb-edu-dedup",
            split="train",
            streaming=True
        )
        texts = []
        for i, sample in enumerate(ds):
            if i >= num_samples:
                break
            texts.append(sample['text'])
        print(f"Loaded {len(texts)} texts")
        return texts
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using synthetic texts...")
        return [
            "The quick brown fox jumps over the lazy dog. " * 50
            for _ in range(num_samples)
        ]


def benchmark_tokenizer(tokenizer_name: str, texts: List[str], d_model: int = 512) -> TokenizerStats:
    """Benchmark a single tokenizer."""
    from transformers import AutoTokenizer
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {tokenizer_name}")
    print('='*60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None
    
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}")
    
    # Tokenize all texts and measure time
    total_tokens = 0
    total_bytes = 0
    
    start_time = time.perf_counter()
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        total_tokens += len(tokens)
        total_bytes += len(text.encode('utf-8'))
    elapsed = time.perf_counter() - start_time
    
    bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 0
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
    
    # Calculate embedding parameters (embedding + lm_head)
    embedding_params = vocab_size * d_model * 2  # embedding + lm_head
    
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total bytes: {total_bytes:,}")
    print(f"  Bytes/token: {bytes_per_token:.2f}")
    print(f"  Tokens/sec: {tokens_per_second:,.0f}")
    print(f"  Embedding params (d={d_model}): {embedding_params:,} ({embedding_params/1e6:.1f}M)")
    print(f"  Tokenization time: {elapsed:.2f}s")
    
    return TokenizerStats(
        name=tokenizer_name,
        vocab_size=vocab_size,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        bytes_per_token=bytes_per_token,
        tokens_per_second=tokens_per_second,
        embedding_params=embedding_params,
        estimated_speedup=1.0  # Will be calculated relative to baseline
    )


def calculate_speedup_estimates(stats: List[TokenizerStats], baseline_name: str) -> None:
    """Calculate estimated speedups relative to baseline."""
    baseline = next((s for s in stats if s.name == baseline_name), None)
    if not baseline:
        baseline = stats[0]
    
    print(f"\n{'='*60}")
    print(f"Speedup Estimates (relative to {baseline.name})")
    print('='*60)
    
    for s in stats:
        # Factors affecting speed:
        # 1. Fewer embedding params = faster forward/backward
        # 2. Fewer tokens for same content = fewer training steps
        
        param_ratio = baseline.embedding_params / s.embedding_params
        token_ratio = baseline.total_tokens / s.total_tokens
        
        # Combined estimate (simplified)
        # More tokens = slower (need more steps for same data)
        # More params = slower (bigger model)
        s.estimated_speedup = param_ratio * token_ratio
        
        print(f"\n{s.name}:")
        print(f"  Param ratio: {param_ratio:.2f}x")
        print(f"  Token ratio: {token_ratio:.2f}x")
        print(f"  Estimated speedup: {s.estimated_speedup:.2f}x")


def print_summary_table(stats: List[TokenizerStats]) -> None:
    """Print summary comparison table."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print('='*80)
    
    # Sort by estimated speedup
    stats_sorted = sorted(stats, key=lambda x: x.estimated_speedup, reverse=True)
    
    print(f"{'Tokenizer':<45} {'Vocab':>10} {'B/Tok':>8} {'Embed(M)':>10} {'Speedup':>10}")
    print('-'*80)
    
    for s in stats_sorted:
        short_name = s.name.split('/')[-1] if '/' in s.name else s.name
        print(f"{short_name:<45} {s.vocab_size:>10,} {s.bytes_per_token:>8.2f} {s.embedding_params/1e6:>10.1f} {s.estimated_speedup:>10.2f}x")
    
    print('-'*80)
    print("\nRECOMMENDATION:")
    best = stats_sorted[0]
    print(f"  Best for speed: {best.name}")
    print(f"  - Vocab size: {best.vocab_size:,}")
    print(f"  - Estimated speedup: {best.estimated_speedup:.2f}x")
    
    # Find best bytes/token (most efficient encoding)
    best_efficiency = max(stats, key=lambda x: x.bytes_per_token)
    print(f"\n  Most efficient encoding: {best_efficiency.name}")
    print(f"  - Bytes/token: {best_efficiency.bytes_per_token:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokenizers for pretraining")
    parser.add_argument(
        "--tokenizers",
        type=str,
        default=None,
        help="Comma-separated list of tokenizer names to benchmark"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of text samples to use"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model dimension for embedding size calculation"
    )
    args = parser.parse_args()
    
    # Get tokenizers to benchmark
    if args.tokenizers:
        tokenizer_names = [t.strip() for t in args.tokenizers.split(',')]
    else:
        tokenizer_names = DEFAULT_TOKENIZERS
    
    print("="*60)
    print("TOKENIZER BENCHMARK FOR PRETRAINING")
    print("="*60)
    print(f"Tokenizers to test: {len(tokenizer_names)}")
    print(f"Sample texts: {args.num_samples}")
    print(f"Model dimension: {args.d_model}")
    
    # Load sample texts
    texts = get_sample_texts(args.num_samples)
    
    # Benchmark each tokenizer
    results = []
    for name in tokenizer_names:
        stats = benchmark_tokenizer(name, texts, args.d_model)
        if stats:
            results.append(stats)
    
    if not results:
        print("No tokenizers successfully benchmarked!")
        return
    
    # Calculate speedups
    calculate_speedup_estimates(results, "HuggingFaceTB/SmolLM2-135M")
    
    # Print summary
    print_summary_table(results)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
To test a tokenizer in training:

1. Prepare data with new tokenizer:
   python data/prepare_mix_data.py \\
       --tokenizer_name "meta-llama/Llama-3.2-1B" \\
       --target_tokens 10000000

2. Run training:
   python train_llm.py --train_tokens 8000000

3. Compare time and val_loss with baseline.
""")


if __name__ == "__main__":
    main()

