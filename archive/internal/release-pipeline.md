# Universe release pipeline

One page. Run this every release. Single entry point: `release.sh`.

## Cadence
Cut a release every week. Skipping a week is fine. Fake-shipping is not.

## Versioning
- `v0.X` — pipeline shakeout. Models may be undertrained / nonsense output.
- `v1.0` — first model that produces coherent English on the smoke prompts.
- `vN.X` after that — N is a major arch/scale jump, X is incremental tokens / fixes.

Tags are immutable. If something is wrong, ship `vN.(X+1)`.

## Required artifacts (in `releases/<version>/`)
- `COMMIT` — git SHA the model was trained from
- `train_log.txt` — full stdout of training
- `ppl.json` — wikitext-2 perplexity
- `samples.txt` — generation on the 4 standard prompts
- `notes.md` — human-written release card (becomes the HF README)

Checkpoint lives at `checkpoints/<version>/model.pt`, copied into the upload
stage at publish time.

## Steps

```bash
# 1. clean tree
git status              # must be clean — release.sh enforces this

# 2. train + eval + sample + scaffold notes
bash release.sh smoke v0.X --config <preset> --train_tokens <N> \
    --dataset_path HuggingFaceTB/smollm-corpus --device mps --compile false

# 3. edit the release card
$EDITOR releases/v0.X/notes.md

# 4. publish raw checkpoint to HuggingFace
bash release.sh publish v0.X vukrosic/universe-<size>-v0.X

# 5. push the tag
git push origin v0.X
```

Each subcommand (`train`, `eval-ppl`, `sample`, `publish`) can also be run
independently — useful when only one step failed.

## What "done" means
- HF repo loads at https://huggingface.co/vukrosic/universe-\<size\>-v0.X
- GitHub tag `v0.X` pushed
- `notes.md` says what changed vs the last release and what's coming next

## What to put in `notes.md`
Use `releases/TEMPLATE/notes.md`. Be honest about quality — if it's gibberish,
say so. The point of building in public is that the bad releases are visible too.
