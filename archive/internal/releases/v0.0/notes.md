# Universe v0.0

**Date:** 2026-05-28
**Commit:** see `COMMIT` in this folder
**HuggingFace:** https://huggingface.co/vukrosic/universe-7m-v0.0

> ⚠️ Raw-checkpoint release. Not in HuggingFace `transformers` format.
> To use, clone [universe-lm](https://github.com/vukrosic/universe-lm) (or this
> repo) and run `python -m scripts.generate --checkpoint model.pt --prompt "..."`.
>
> **This is a pipeline smoke test, not a usable model.** v0.0 trained for 13
> steps on 100K tokens — output is gibberish, by design. The goal was to wire up
> the full train → eval → sample → publish loop end-to-end. Real models start at
> v0.1.

## Config
- Params: 6.65M (Muon: 360K, AdamW: 6.29M)
- Layers / d_model / heads: 2 / 128 / 2 (1 KV head, GQA)
- d_ff: 512, seq_len: 2048
- Tokens trained: 100K (≈13 optimizer steps)
- Tokenizer: HuggingFaceTB/SmolLM2-135M (49,152 vocab)
- Dataset: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
- Optimizer: Muon (lr 0.024) for hidden matrices + AdamW (lr 0.006) for embeddings
- Train time: 2m 34s on Apple M-series GPU (MPS)

## Eval
```json
{
  "dataset": "wikitext/wikitext-2-raw-v1",
  "split": "test",
  "stride": 512,
  "perplexity": 8994.82
}
```

In-distribution val: loss 7.74, acc 7.9%, ppl 2,295. Both numbers are
essentially random for a model this undertrained — listed for the record.

## Generation samples
See `samples.txt`. Representative output (prompt + completion):

```
=== prompt: Once upon a time ===
Once upon a time for--
,. 2. and and and their the. 0 This by at and like to to to and about how
they their of and for and as and a the in. of the in and or to and into.
```

It's nonsense. That's expected. The model has barely moved off random
initialization.

## What changed vs last release
First release. Pipeline scaffold:
- `release.sh` orchestrator (smoke / train / eval-ppl / sample / publish)
- `scripts/generate.py` raw-checkpoint text generation
- `scripts/eval_ppl.py` wikitext-2 sliding-window perplexity
- `scripts/upload_to_hf.py` HuggingFace Hub uploader
- `configs/llm_config.py`: `FiveMillionConfig` preset (~7M params)
- `optimizers/muon.py`: gated `torch.compile` so Muon runs on MPS

## Next week
- v0.1: same architecture or slightly larger (~15M), trained on 50M–100M tokens
  so completions are at least grammatically coherent.
- Set up a tiny generation eval harness so we catch regressions across releases.
- Decide on a fixed weekday for the weekly cut.
