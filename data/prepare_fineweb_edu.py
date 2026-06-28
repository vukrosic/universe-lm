"""Prepare a FineWeb-Edu pretraining shard in the exact on-disk format the
trainer expects, so we can swap data without touching the rest of the pipeline.

Why this exists
---------------
The README names DATA as the single biggest un-pulled lever for beating
SmolLM2-135M: SmolLM2's edge comes largely from FineWeb-Edu / DCLM filtering.
The current repo trains on a pre-tokenized `vukrosic/blueberry-*` mix. This
script builds an equivalent shard straight from `HuggingFaceFW/fineweb-edu`
so we can A/B the data lever on equal footing.

Fairness rationale
------------------
To keep the SmolLM2 comparison honest, this MUST match the existing tokenizer
and sequence length exactly:
  * tokenizer = HuggingFaceTB/SmolLM2-135M  (vocab 49152) — same tokens the
    SmolLM2 baseline uses, so loss/BPB stay comparable.
  * seq_len   = 2048                        — the RoPE cache hard-requires 2048
    (see train_llm.py: prep_metadata.json `max_seq_len` guard).
Chunking mirrors data/loader.py:tokenize_and_chunk exactly (concatenate
tokenized docs, reshape into (n, 2048) blocks, drop the partial tail).

On-disk format (must match what the loader reads back)
------------------------------------------------------
Saved via `Dataset.save_to_disk(out_dir)` as a SINGLE Dataset (not a dict)
with two columns, each row exactly 2048 tokens:
  * input_ids : List[int]   (length 2048)
  * labels    : List[int]   (== input_ids.copy())
train_llm.py loads this with `load_from_disk`, sets torch format on
["input_ids", "labels"], and auto-splits 90/10 into train/val. A
`prep_metadata.json` with {"max_seq_len": 2048} is written alongside so the
trainer's RoPE-cache guard validates instead of erroring.

Run example
-----------
    python data/prepare_fineweb_edu.py --target-tokens 1_000_000_000 \
        --out processed_data/pretrain_fineweb_edu

Note: streams from HF (no full download); still pulls ~target-tokens worth of
text, so run on a box with bandwidth/disk. NOT run by default.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# Pinned to match the SmolLM2 baseline and the RoPE cache. Do not change these
# without re-checking train_llm.py's max_seq_len guard and the loader.
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
SEQ_LEN = 2048
DATASET_PATH = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"  # ~10B-token deduped sample; we take a prefix
TEXT_COLUMN = "text"


def build_blocks(target_tokens: int, out_dir: str, cosmopedia_frac: float) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cosmopedia_frac > 0.0:
        # TODO(data-mix): optionally interleave HuggingFaceTB/cosmopedia-v2
        # synthetic docs at this fraction before chunking, to mirror the
        # SmolLM2 web+synthetic recipe. Left out of the core path on purpose
        # (keep this script small): wire a second streaming load_dataset and
        # round-robin documents into the token buffer below. Until then this
        # flag is recorded in prep_metadata.json but not yet applied.
        print(
            f"[warn] --cosmopedia-frac={cosmopedia_frac} requested but the mix "
            "path is a TODO; proceeding with 100% FineWeb-Edu."
        )

    print(f"Streaming {DATASET_PATH} ({DATASET_CONFIG}) for ~{target_tokens:,} tokens...")
    ds = load_dataset(DATASET_PATH, name=DATASET_CONFIG, split="train", streaming=True)

    # Concatenate-and-chunk, mirroring data/loader.py:tokenize_and_chunk.
    # add_special_tokens=True prepends the SmolLM2 BOS/EOS per document, exactly
    # like the existing loader does (loader.py:71).
    token_buffer: list[int] = []
    blocks: list[np.ndarray] = []
    docs_seen = 0

    for example in ds:
        ids = tokenizer(
            example[TEXT_COLUMN],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        token_buffer.extend(ids)
        docs_seen += 1

        # Drain full blocks out of the buffer as soon as we have them.
        while len(token_buffer) >= SEQ_LEN:
            block = np.asarray(token_buffer[:SEQ_LEN], dtype=np.int64)
            blocks.append(block)
            del token_buffer[:SEQ_LEN]
            if len(blocks) * SEQ_LEN >= target_tokens:
                break
        if len(blocks) * SEQ_LEN >= target_tokens:
            break

    # The final partial block is dropped (matches loader.py's tail truncation).
    n_tokens = len(blocks) * SEQ_LEN
    print(f"Built {len(blocks):,} blocks ({n_tokens:,} tokens) from {docs_seen:,} docs.")
    if not blocks:
        raise RuntimeError("No full 2048-token blocks produced; target too small?")

    input_ids = np.stack(blocks)  # shape (n_blocks, 2048)
    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids.tolist(),
            "labels": input_ids.tolist(),  # labels == input_ids (causal LM)
        }
    )

    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)

    # RoPE-cache guard: train_llm.py reads this and refuses to train if it
    # disagrees with the configured seq_length.
    with open(os.path.join(out_dir, "prep_metadata.json"), "w") as f:
        json.dump(
            {
                "max_seq_len": SEQ_LEN,
                "tokenizer_name": TOKENIZER_NAME,
                "source": f"{DATASET_PATH}:{DATASET_CONFIG}",
                "target_tokens": target_tokens,
                "actual_tokens": n_tokens,
                "n_blocks": len(blocks),
                "cosmopedia_frac": cosmopedia_frac,
            },
            f,
            indent=2,
        )
    print(f"Saved dataset + prep_metadata.json to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare FineWeb-Edu shard for universe-lm.")
    ap.add_argument(
        "--target-tokens",
        type=int,
        default=1_000_000_000,
        help="Approx number of packed tokens to produce (default 1B).",
    )
    ap.add_argument(
        "--out",
        default="processed_data/pretrain_fineweb_edu",
        help="Output dir read back by train_llm.py via load_from_disk.",
    )
    ap.add_argument(
        "--cosmopedia-frac",
        type=float,
        default=0.0,
        help="Fraction of synthetic cosmopedia-v2 to mix in (mix path is a TODO).",
    )
    args = ap.parse_args()
    build_blocks(args.target_tokens, args.out, args.cosmopedia_frac)


if __name__ == "__main__":
    main()
