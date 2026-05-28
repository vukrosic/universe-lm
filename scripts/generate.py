"""Generate text from a Universe checkpoint.

Usage:
    python scripts/generate.py \\
        --checkpoint checkpoints/v0.0/model.pt \\
        --prompt "Once upon a time" \\
        --max-new-tokens 100

Loads the raw .pt checkpoint produced by train_llm.py (model_state_dict + config).
Tokenizer is whichever HF tokenizer trained the data (read from config or arg).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from training.device import resolve_device


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MinimalLLM, LLMConfig]:
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = blob["config"]
    model = MinimalLLM(config).to(device).eval()

    state_dict = blob["model_state_dict"]
    # Strip torch.compile's _orig_mod. prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model, config


@torch.no_grad()
def generate(
    model: MinimalLLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_seq = model.config.max_seq_len

    for _ in range(max_new_tokens):
        input_ids = ids[:, -max_seq:]
        logits = model(input_ids)[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model, _ = load_model(args.checkpoint, device)

    text = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        device,
    )
    print(text)


if __name__ == "__main__":
    main()
