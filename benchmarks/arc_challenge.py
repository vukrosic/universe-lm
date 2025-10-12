"""
ARC-Challenge Benchmark
Evaluates models on grade-school level science question answering

Usage:
    python benchmarks/arc_challenge.py --checkpoint experiments/exp7_*/checkpoints/best_model.pt
    python benchmarks/arc_challenge.py --checkpoint exp7/checkpoints/best_model.pt --split test
    python benchmarks/arc_challenge.py --checkpoint exp7/checkpoints/best_model.pt --max-samples 100
"""

import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from common import load_model_from_checkpoint, get_device_and_dtype


def compute_choice_loglikelihood(model, tokenizer, question, choice, device='cuda'):
    """
    Compute log-likelihood of a choice given the question
    Higher log-likelihood = more likely answer
    """
    full_text = f"Question: {question}\nAnswer: {choice}"
    
    inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    
    if input_ids.shape[1] < 2:
        return float('-inf')
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            log_likelihood = -loss.item()
    
    return log_likelihood


def evaluate_arc_sample(model, tokenizer, sample, device='cuda'):
    """Evaluate a single ARC sample"""
    question = sample['question']
    choices_text = sample['choices']['text']
    choices_label = sample['choices']['label']
    correct_label = sample['answerKey']
    
    log_likelihoods = []
    for choice_text in choices_text:
        log_lik = compute_choice_loglikelihood(model, tokenizer, question, choice_text, device)
        log_likelihoods.append(log_lik)
    
    predicted_idx = np.argmax(log_likelihoods)
    predicted_label = choices_label[predicted_idx]
    is_correct = (predicted_label == correct_label)
    
    return predicted_label, correct_label, is_correct, log_likelihoods


def evaluate_arc(model, tokenizer, split='validation', max_samples=None, device='cuda'):
    """Evaluate model on ARC-Challenge dataset"""
    print(f"\nLoading ARC-Challenge dataset ({split} split)...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    results_list = []
    
    for sample in tqdm(dataset, desc="Evaluating ARC-Challenge"):
        predicted_label, correct_label, is_correct, log_likelihoods = evaluate_arc_sample(
            model, tokenizer, sample, device
        )
        
        if is_correct:
            correct += 1
        total += 1
        
        results_list.append({
            'id': sample['id'],
            'question': sample['question'],
            'predicted': predicted_label,
            'correct': correct_label,
            'is_correct': is_correct,
            'log_likelihoods': [float(ll) for ll in log_likelihoods],
            'choices': sample['choices']['text'],
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        'dataset': 'ARC-Challenge',
        'split': split,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_percent': accuracy * 100,
        'samples': results_list,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on ARC-Challenge')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='validation',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: auto-generated)')
    args = parser.parse_args()
    
    print("="*70)
    print("ARC-Challenge Benchmark")
    print("="*70)
    
    # Setup device and dtype
    device, dtype = get_device_and_dtype()
    print(f"Device: {device}")
    
    # Load model
    model, config, tokenizer = load_model_from_checkpoint(
        args.checkpoint, device=device, dtype=dtype
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("Running Evaluation")
    print("="*70)
    
    results = evaluate_arc(
        model, tokenizer,
        split=args.split,
        max_samples=args.max_samples,
        device=device
    )
    
    # Add checkpoint info to results
    results['checkpoint_path'] = str(args.checkpoint)
    results['model_info'] = {
        'hidden_size': getattr(config, 'hidden_size', 'N/A'),
        'num_layers': getattr(config, 'num_hidden_layers', 'N/A'),
        'num_heads': getattr(config, 'num_attention_heads', 'N/A'),
    }
    
    # Print results
    print("\n" + "="*70)
    print("ARC-Challenge Results")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {results['split']}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy_percent']:.2f}%")
    print("="*70)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_path = Path(args.checkpoint)
        exp_dir = checkpoint_path.parent.parent
        results_dir = exp_dir / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        output_path = results_dir / f"arc_challenge_{args.split}_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print example predictions
    print("\n" + "="*70)
    print("Example Predictions (first 3)")
    print("="*70)
    
    for i, sample in enumerate(results['samples'][:3]):
        print(f"\n[{i+1}] Q: {sample['question'][:80]}...")
        print(f"    Choices: {sample['choices']}")
        print(f"    Predicted: {sample['predicted']} | Correct: {sample['correct']}")
        print(f"    {'✓ CORRECT' if sample['is_correct'] else '✗ WRONG'}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

