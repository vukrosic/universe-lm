# Pinned benchmark protocol — `beat-smollm2-135m`

Status: Phase 0 protocol (2026-06-10). Frozen once the baseline SmolLM2-135M rerun
completes. All future A/B comparisons use these exact versions and commands.

## 1. Harness pin

| | |
|---|---|
| Repo | `https://github.com/EleutherAI/lm-evaluation-harness` |
| Tag | `v0.4.12` |
| Commit | `6d642546f4688648fced259eb3302efd36ece5af` |
| Release date | 2025-05-11 |
| Python | 3.10+ (3.12 verified) |
| Required extras | `lm_eval[hf]` (transformers, datasets, accelerate) |

v0.4.12 is the current stable release as of 2026-06; the next major (v0.5) is
not yet cut. Re-pin in `requirements.txt` to `lm-eval @ git+https://github.com/
EleutherAI/lm-evaluation-harness@6d642546f4688648fced259eb3302efd36ece5af`
so a future tag cannot silently change task behavior. v0.4.x task YAMLs are
considered stable; bumping to a different v0.4.x is a one-line PR after
re-running the baseline.

## 2. Pinned task list

All under `v0.4.12`. Win condition is ≥5 of 6 0-shot tasks. MMLU is reported
but not in the win condition (135M is near chance).

| # | Task name | Dataset (HF) | Split | Fewshot | Metric | Task version |
|---|---|---|---|---|---|---|
| 1 | `hellaswag` | `Rowan/hellaswag` | validation | 0 | acc_norm | 1.0 |
| 2 | `arc_easy` | `allenai/ai2_arc` (config `ARC-Easy`) | test | 0 | acc_norm | 1.0 |
| 3 | `arc_challenge` | `allenai/ai2_arc` (config `ARC-Challenge`) | test | 0 | acc_norm | 1.0 |
| 4 | `piqa` | `baber/piqa` | validation | 0 | acc_norm | 1.0 |
| 5 | `winogrande` | `allenai/winogrande` (config `winogrande_xl`) | validation | 0 | acc | 1.0 |
| 6 | `openbookqa` | `allenai/openbookqa` (config `main`) | test | 0 | acc_norm | 1.0 |
| 7 | `commonsense_qa` | `tau/commonsense_qa` | validation | 0 | acc | (no `metadata.version` in YAML — pin via harness commit) |
| 8 | `mmlu` (group) | `cais/mmlu` | test (dev→5-shot) | 5 | acc | task 1.0, group 2 |

Notes:
- We report `acc_norm` (length-normalized) for the multiple-choice-with-
  variable-length-completions tasks (HellaSwag, ARC, PIQA, OBQA) and raw
  `acc` for the fixed-lexicon tasks (Winogrande, CommonsenseQA) and MMLU.
  This matches the conventions on the EleutherAI and HELM leaderboards.
- MMLU uses `--num_fewshot 5` even though the YAML defaults to using the full
  `dev` split. Pinning 5 matches the published SmolLM2 number.
- For HellaSwag, the harness loads `Rowan/hellaswag` (a maintained mirror
  of the original). The original `hellaswag` dataset on HF Hub is fine too
  but the Rowan mirror is what v0.4.12 ships against.

## 3. CLI commands

### 3.1 Primary 0-shot suite (the win-condition tasks)

```bash
lm_eval --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM2-135M,trust_remote_code=False,dtype=bfloat16 \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,commonsense_qa \
    --num_fewshot 0 \
    --batch_size auto:4 \
    --output_path results/baseline-smollm2-135m/ \
    --log_samples \
    --seed 42
```

- `--batch_size auto:4` lets the harness grow from batch 1 up to 4 retries
  on OOM. For 135M at seq≤2048 a batch of 32 fits on a 24 GB GPU; auto
  will find it.
- `--log_samples` is required for `--output_path` to write a JSON; it also
  gives us per-example dumps for sanity-checking later.
- `--seed 42` matches the autoresearch seed pin (one seed only, per
  `feedback-one-seed-only`).
- `trust_remote_code=False` is the safe default; SmolLM2 ships native
  transformers code, no remote code needed.

### 3.2 MMLU 5-shot (reported, not a win condition)

```bash
lm_eval --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM2-135M,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path results/baseline-smollm2-135m/ \
    --log_samples \
    --seed 42
```

The `mmlu` group expands to 57 subject sub-tasks internally. Output JSON
includes the weighted-mean `acc` (group-aggregated) and per-subject
breakdowns.

### 3.3 Continuous metric — FineWeb-Edu held-out BPB

Run separately from lm-eval; uses a tiny HF model wrapper. See §5 for the
script outline. Pin to a fixed sample slice; results are not directly
comparable across `sample-10BT` / `sample-100BT` / `sample-350BT`.

```bash
python scripts/bpb_fineweb_edu.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --split train \
    --max_bytes 5242880 \
    --output results/baseline-smollm2-135m/bpb.json
```

## 4. CPU Mac vs Vast GPU box — concrete notes

The harness reads `--device cuda:0` / `--device mps` / `--device cpu` and
the `dtype` model arg. We are not trying to be portable across all three;
we pick per phase.

### 4.1 Phase 0 baseline (this script)

Run on the **rental Vast box** (H100 80 GB or RTX 4090 24 GB). The CPU Mac
is only used for:
- Smoke-testing harness install + model load (5-sample limit; the script
  `scripts/eval_baseline.sh` does this).
- Editing configs and reading results.

Vast box:
```bash
# Already-provisioned per the vast-runner-harness memory:
# - python at /venv/main/bin/python
# - repo at /root/universe-lm
# - GPU visible only after `export LD_LIBRARY_PATH=/usr/local/nvidia/lib64`
cd /root/universe-lm
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
/venv/main/bin/python -m lm_eval \
    --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM2-135M,dtype=bfloat16 \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,commonsense_qa \
    --num_fewshot 0 --batch_size auto:4 \
    --device cuda:0 \
    --output_path results/baseline-smollm2-135m/ \
    --log_samples --seed 42 2>&1 | tee logs/eval-baseline.log
```

CPU Mac (smoke only — `--limit 5`):
```bash
# No GPU; force fp32 to keep memory sane; limit to 5 examples per task
lm_eval --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM2-135M,dtype=float32 \
    --tasks hellaswag \
    --num_fewshot 0 --batch_size 1 \
    --device cpu \
    --limit 5 \
    --output_path results/smoke/
```

Expected smoke time: ~2–5 min on M-series Mac (5 examples is the bottleneck
of import/load, not eval). Full suite on the Vast box: ~30–60 min for the
7-task 0-shot list + ~60 min for MMLU (57 subjects × 14k examples).

### 4.2 Phase 1+ A/B evals

Same harness, same pin, same tasks. Wrapper in `tools/eval_lm_eval.sh`
takes a model path / HF id + run name and writes to
`results/<run_name>/lm_eval/`. Diff vs the baseline = the win condition.

## 5. FineWeb-Edu held-out BPB — script outline

Goal: a tokenizer-independent scalar that's comparable across recipe
candidates and across the SmolLM2 cosmo2 baseline.

### 5.1 Held-out slice

Use the FIRST ~5 MB of UTF-8 text from `HuggingFaceFW/fineweb-edu`,
config `sample-10BT`, `train` split. (FineWeb-Edu has no `validation`
split; reserving a deterministic prefix of `sample-10BT` is the standard
research trick and matches what modded-nanogpt does with
`kjj0/finewebedu10B-gpt2`'s pre-allocated `finewebedu_val_000000.bin`.)

We pin the slice by **byte offset** of the underlying parquet, not by row
count, so re-runs always get the same bytes. 5 MB is small enough to
eval in <1 min on the Vast box and large enough that BPB is noise-stable
(~3rd-decimal) for 135M-class models.

### 5.2 Script outline (`scripts/bpb_fineweb_edu.py`)

```
import argparse, json, math, os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)        # HF id or local path
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset",  default="sample-10BT")
    ap.add_argument("--split",   default="train")
    ap.add_argument("--max_bytes", type=int, default=5*1024*1024)
    ap.add_argument("--seq_len",   type=int, default=2048)
    ap.add_argument("--stride",    type=int, default=1024)  # sliding window
    ap.add_argument("--output",    required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # 1. Load enough rows to cover max_bytes of UTF-8 text.
    #    sample-10BT is row-shuffled but the first 5MB of UTF-8 is stable
    #    per the dataset's deterministic layout.
    ds = load_dataset(args.dataset, name=args.subset, split=args.split,
                      streaming=True)
    buf, n_bytes = [], 0
    for ex in ds:
        buf.append(ex["text"])
        n_bytes += len(ex["text"].encode("utf-8"))
        if n_bytes >= args.max_bytes:
            break
    text = "\n\n".join(buf)
    total_bytes = len(text.encode("utf-8"))  # re-measure exact boundary

    # 2. Sliding-window NLL over the text (no overlap double-counted).
    enc = tok(text, return_tensors="pt", truncation=False).input_ids[0]
    n = enc.size(0)
    nll_nats, counted = 0.0, 0
    L = args.seq_len
    for begin in range(0, n - 1, args.stride):
        end = min(begin + L, n)
        ids = enc[begin:end].unsqueeze(0).to(model.device)
        # Predict token t+1 from token t → only count end - 1 - begin targets
        with torch.no_grad():
            out = model(ids).logits
        # shift: predict ids[1:] from logits[:-1]
        shift_logits = out[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nll_nats += loss.item()
        counted += (end - begin - 1)
        if end == n: break

    bpb = nll_nats / math.log(2) / total_bytes   # tokenizer-independent
    mean_nll_per_token = nll_nats / counted
    ppl = math.exp(mean_nll_per_token)

    out = {
        "model": args.model,
        "dataset": args.dataset, "subset": args.subset, "split": args.split,
        "max_bytes": args.max_bytes, "actual_bytes": total_bytes,
        "tokens_evaluated": counted,
        "nll_nats": nll_nats,
        "bpb": bpb,
        "ppl": ppl,
        "harness": "lm-eval-harness@6d64254",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

### 5.3 How to read the result

- `bpb` is the **decision metric** (tokenizer-independent, comparable across
  all our 135M runs and against the SmolLM2 number we'd compute
  identically).
- `ppl` is provided for sanity (matches modded-nanogpt convention when
  applied to GPT-2 tokenized val). Don't make decisions on it.
- `nll_nats` is the raw sum; `bpb = nll_nats / ln(2) / actual_bytes`.

For 135M-class models we expect:
- SmolLM2-135M (cosmo2 tokenizer) on this 5 MB slice: BPB ≈ 0.72–0.78.
- Our 135M after Phase-1 stack: must clear the bracket AND be lower.
- Numbers will move ±0.005 across runs even at fixed seed because the
  streaming loader pulls a slightly different first-5MB under network
  cache misses; pin the actual byte count in the output JSON.

### 5.4 Why not use lm-eval-harness for this

lm-eval-harness has a `paloma` task family that does BPB, but it's geared
to GPT-2-style byte-pair accounting and doesn't cleanly expose "evaluate
an arbitrary HF causal LM on a held-out slice of FineWeb-Edu using this
specific tokenizer." The script above is 60 lines and reuses `transformers`
+ `datasets` directly. Reuse this script for all 135M A/B checkpoints;
its output JSON diffs are the BPB decision artifact.

## 6. Reproducibility checklist (for BASELINE.md)

- [ ] `git rev-parse HEAD` of this repo at run time
- [ ] Harness commit: `6d642546f4688648fced259eb3302efd36ece5af`
- [ ] Model id + SHA: `HuggingFaceTB/SmolLM2-135M` (record from
      `huggingface_hub.hf_hub_download` resolve)
- [ ] GPU: model + driver + CUDA version
- [ ] `--num_fewshot` and `--seed` echoed by the harness
- [ ] Output: `results/baseline-smollm2-135m/results_*.json` (8 files, one
      per task; or 2 files for grouped mmlu). Save the raw `--log_samples`
      jsonl as well — it's the audit trail.
- [ ] BPB result from §5 alongside
