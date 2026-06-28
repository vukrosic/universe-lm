"""
Compare multiple model checkpoints on benchmarks
Runs ARC-Challenge and HellaSwag on multiple checkpoints and compares results

Usage:
    python benchmarks/compare_models.py \
        experiments/exp6_*/checkpoints/best_model.pt \
        experiments/exp7_*/checkpoints/best_model.pt
    
    python benchmarks/compare_models.py \
        --checkpoints exp6/checkpoints/best_model.pt exp7/checkpoints/best_model.pt \
        --benchmarks arc hellaswag \
        --max-samples 100
"""

import torch
import argparse
import json
from pathlib import Path
from tabulate import tabulate
import sys

from common import (
    load_model_from_checkpoint,
    load_hf_model,
    model_size_info,
    get_device_and_dtype,
)
from arc_challenge import evaluate_arc
from hellaswag import evaluate_hellaswag


def compare_checkpoints(checkpoints, benchmarks=['arc', 'hellaswag'],
                        max_samples=None, hf_baselines=None):
    """Compare universe-lm checkpoints (and optional HuggingFace baselines such
    as SmolLM2-135M) on selected benchmarks."""

    device, dtype = get_device_and_dtype()
    print(f"Device: {device}\n")

    # Build a unified work list: ('checkpoint', path) and ('hf', model_name).
    targets = [('checkpoint', c) for c in (checkpoints or [])]
    targets += [('hf', m) for m in (hf_baselines or [])]

    all_results = []

    for kind, ref in targets:
        print("="*70)
        print(f"Evaluating: {ref}")
        print("="*70)

        # Load model (checkpoint or HF baseline)
        try:
            if kind == 'hf':
                model, config, tokenizer = load_hf_model(ref, device=device, dtype=dtype)
                exp_name = ref.split('/')[-1]
                ref_str = ref
            else:
                ref = Path(ref)
                model, config, tokenizer = load_model_from_checkpoint(
                    ref, device=device, dtype=dtype
                )
                exp_name = ref.parent.parent.name
                ref_str = str(ref)
        except Exception as e:
            print(f"❌ Error loading {ref}: {e}")
            continue

        result = {
            'checkpoint': ref_str,
            'exp_name': exp_name,
            'model_info': model_size_info(config),
        }

        # Run ARC-Challenge
        if 'arc' in benchmarks:
            print("\n" + "-"*70)
            print("Running ARC-Challenge...")
            print("-"*70)
            arc_results = evaluate_arc(
                model, tokenizer,
                split='validation',
                max_samples=max_samples,
                device=device
            )
            result['arc_accuracy'] = arc_results['accuracy_percent']
            result['arc_correct'] = arc_results['correct']
            result['arc_total'] = arc_results['total_samples']
        
        # Run HellaSwag
        if 'hellaswag' in benchmarks:
            print("\n" + "-"*70)
            print("Running HellaSwag...")
            print("-"*70)
            hellaswag_results = evaluate_hellaswag(
                model, tokenizer,
                split='validation',
                max_samples=max_samples,
                device=device
            )
            result['hellaswag_accuracy'] = hellaswag_results['accuracy_percent']
            result['hellaswag_correct'] = hellaswag_results['correct']
            result['hellaswag_total'] = hellaswag_results['total_samples']
        
        all_results.append(result)
        print("\n")
    
    return all_results


def print_comparison_table(results):
    """Print comparison table"""
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON")
    print("="*70)
    
    # Prepare table data
    headers = ['Experiment', 'Model Size', 'ARC-Challenge', 'HellaSwag']
    rows = []
    
    for r in results:
        model_size = f"{r['model_info']['hidden_size']}d, {r['model_info']['num_layers']}L"
        
        arc_score = f"{r.get('arc_accuracy', 0):.2f}% ({r.get('arc_correct', 0)}/{r.get('arc_total', 0)})" if 'arc_accuracy' in r else 'N/A'
        hellaswag_score = f"{r.get('hellaswag_accuracy', 0):.2f}% ({r.get('hellaswag_correct', 0)}/{r.get('hellaswag_total', 0)})" if 'hellaswag_accuracy' in r else 'N/A'
        
        rows.append([
            r['exp_name'],
            model_size,
            arc_score,
            hellaswag_score
        ])
    
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare multiple models on benchmarks')
    parser.add_argument('checkpoints', nargs='*',
                        help='Paths to model checkpoints')
    parser.add_argument('--checkpoints', dest='checkpoint_list', nargs='+',
                        help='Alternative way to specify checkpoints')
    parser.add_argument('--benchmarks', nargs='+', default=['arc', 'hellaswag'],
                        choices=['arc', 'hellaswag'],
                        help='Which benchmarks to run')
    parser.add_argument('--hf-baselines', dest='hf_baselines', nargs='+', default=None,
                        help='HuggingFace models to benchmark alongside, e.g. '
                             'HuggingFaceTB/SmolLM2-135M (the model we race to beat)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per benchmark (None = all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for comparison results')
    args = parser.parse_args()

    # Get checkpoint paths
    checkpoint_paths = args.checkpoints or args.checkpoint_list

    if not checkpoint_paths and not args.hf_baselines:
        print("Error: specify at least one checkpoint or --hf-baselines model")
        parser.print_help()
        sys.exit(1)

    print("="*70)
    print("MODEL BENCHMARK COMPARISON")
    print("="*70)
    print(f"Checkpoints: {len(checkpoint_paths or [])}")
    if args.hf_baselines:
        print(f"HF baselines: {', '.join(args.hf_baselines)}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    if args.max_samples:
        print(f"Samples per benchmark: {args.max_samples}")
    else:
        print(f"Samples per benchmark: All (full evaluation)")
    print()

    # Run comparison
    results = compare_checkpoints(
        checkpoint_paths,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        hf_baselines=args.hf_baselines,
    )
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('benchmark_comparison_results.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Comparison results saved to: {output_path}\n")


if __name__ == "__main__":
    main()

