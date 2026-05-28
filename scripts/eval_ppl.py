"""Compute perplexity on a text corpus for a Universe checkpoint.

Uses sliding-window evaluation over a chosen HF dataset (default: Wikitext-2).
No HuggingFace model-format required — works on raw .pt checkpoints.

Usage:
    python scripts/eval_ppl.py \\
        --checkpoint checkpoints/v0.0/model.pt \\
        --dataset wikitext --subset wikitext-2-raw-v1 --split test \\
        --stride 512
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from scripts.generate import load_model
from training.device import resolve_device


@torch.no_grad()
def perplexity(model, tokenizer, text: str, device, stride: int) -> float:
    seq_len = model.config.max_seq_len
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)
    n = enc.size(1)
    if n < 2:
        raise SystemExit("text too short")

    nll_sum, token_count = 0.0, 0
    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + seq_len, n)
        trg_len = end - prev_end
        ids = enc[:, begin:end]
        if ids.size(1) < 2:
            break

        logits = model(ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous().clone()
        # Mask out positions before the new region we're scoring
        mask_until = ids.size(1) - trg_len - 1
        if mask_until > 0:
            shift_labels[:, :mask_until] = -100

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid = (shift_labels != -100).sum().item()
        nll_sum += loss.item()
        token_count += valid
        prev_end = end
        if end == n:
            break

    return math.exp(nll_sum / max(token_count, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model, _ = load_model(args.checkpoint, device)

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    text = "\n\n".join(x for x in ds["text"] if x.strip())
    ppl = perplexity(model, tokenizer, text, device, args.stride)

    result = {
        "dataset": f"{args.dataset}/{args.subset}",
        "split": args.split,
        "stride": args.stride,
        "perplexity": ppl,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
