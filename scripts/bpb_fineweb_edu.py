"""Compute FineWeb-Edu held-out bits-per-byte (BPB) for a causal LM.

Tokenizer-independent scalar used as the A/B decision metric in
docs/plans/beat-smollm2-135m.md. Pinned to the protocol in
docs/plans/benchmark-protocol.md §5.

Held-out slice: first 5 MB of UTF-8 text from HuggingFaceFW/fineweb-edu,
config sample-10BT, train split (FineWeb-Edu ships no validation split;
this prefix is deterministic per the dataset's parquet layout and matches
the modded-nanogpt convention).

Usage:
    python scripts/bpb_fineweb_edu.py \\
        --model HuggingFaceTB/SmolLM2-135M \\
        --output results/baseline-smollm2-135m/bpb.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id or local path (e.g. HuggingFaceTB/SmolLM2-135M)")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset",  default="sample-10BT")
    ap.add_argument("--split",   default="train")
    ap.add_argument("--max_bytes", type=int, default=5 * 1024 * 1024,
                    help="Hold out this many UTF-8 bytes of the prefix.")
    ap.add_argument("--seq_len",   type=int, default=2048)
    ap.add_argument("--stride",    type=int, default=1024,
                    help="Sliding-window stride; non-overlap is conservative.")
    ap.add_argument("--output",    required=True,
                    help="Path to write JSON result.")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    # 1. Materialize the held-out prefix.
    ds = load_dataset(args.dataset, name=args.subset, split=args.split,
                      streaming=True)
    buf, n_bytes = [], 0
    for ex in ds:
        buf.append(ex["text"])
        n_bytes += len(ex["text"].encode("utf-8"))
        if n_bytes >= args.max_bytes:
            break
    text = "\n\n".join(buf)
    total_bytes = len(text.encode("utf-8"))

    # 2. Sliding-window NLL over the text.
    enc = tok(text, return_tensors="pt", truncation=False).input_ids[0]
    n = enc.size(0)
    if n < 2:
        print("ERROR: text too short after tokenization", file=sys.stderr)
        return 1

    nll_nats, counted = 0.0, 0
    L, stride = args.seq_len, args.stride
    with torch.no_grad():
        for begin in range(0, n - 1, stride):
            end = min(begin + L, n)
            ids = enc[begin:end].unsqueeze(0).to(device)
            out = model(ids).logits
            shift_logits = out[:, :-1, :].float()
            shift_labels = ids[:, 1:]
            loss_sum = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            nll_nats += loss_sum.item()
            counted += (end - begin - 1)
            if end == n:
                break

    bpb = nll_nats / math.log(2) / total_bytes
    mean_nll_per_token = nll_nats / max(counted, 1)
    ppl = math.exp(mean_nll_per_token)

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "max_bytes": args.max_bytes,
        "actual_bytes": total_bytes,
        "tokens_evaluated": counted,
        "nll_nats": nll_nats,
        "bpb": bpb,
        "ppl": ppl,
        "seq_len": L,
        "stride": stride,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
