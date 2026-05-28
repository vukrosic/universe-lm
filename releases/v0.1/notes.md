# Universe v0.1

**Date:** 2026-05-28
**Commit:** see `COMMIT` in this folder
**HuggingFace:** https://huggingface.co/vukrosic/universe-7m-v0.1

> ⚠️ Raw-checkpoint release. Not in HuggingFace `transformers` format.
> Clone [universe-lm](https://github.com/vukrosic/universe-lm) and run
> `python -m scripts.generate --checkpoint model.pt --prompt "..."`.
>
> **Still a pipeline shakeout.** 150K tokens / 18 steps — model is undertrained
> and output is gibberish. Focus this week was hardening the release loop, not
> model quality.

## Config
- Params: 6.65M (same arch as v0.0)
- Layers / d_model / heads: 2 / 128 / 2 (1 KV head, GQA)
- Tokens trained: 150K (+50% vs v0.0)
- Optimizer: Muon (lr 0.024) + AdamW (lr 0.006)
- Train time: 1m 39s on Apple M-series GPU (MPS)

## Eval
```json
{
  "dataset": "wikitext/wikitext-2-raw-v1",
  "split": "test",
  "stride": 512,
  "perplexity": 7025.83
}
```

In-distribution val: loss 7.10 (v0.0: 7.74), acc 11.9% (v0.0: 7.9%).

## Generation samples
See `samples.txt`. Sample:

```
Once upon a time for of of of of, and,,..,.,.,,,. ,, and and and,,..,,,,
their of,.,, and, and in... in and or to,,...
```

Still gibberish. Slightly more punctuation, fewer raw vocab tokens.

## What changed vs v0.0
- Trained 1.5× longer (150K vs 100K tokens)
- Wikitext PPL: 8994 → 7026 (-22%)
- Val loss: 7.74 → 7.10
- Added `docs/release-pipeline.md` standardizing the weekly procedure

## Next week
- v0.2: bump to 1M+ tokens so the model is actually past the noise floor
- Decide on the weekly cut day
- Add a tiny sanity check that catches the kind of silent failure that hit v0.0's samples
