# Issue 30: 10m warmup_decay_w002

Status: candidate leaderboard result, pending clean source commit/tag.

## Result

- Config: `10m`
- Seed: `42`
- Schedule: `warmup_decay_to_zero`
- Warmup ratio: `0.02`
- Dataset: `processed_data/pretrain_1B` from `vukrosic/blueberry-1B-pretrain`
- Train tokens: `200000000`
- Tokens seen: `200003584`
- Actual steps: `48829`
- Total time: `62.86522298256556` minutes

Final metrics:

```json
{
  "val_loss": 4.54859375,
  "val_accuracy": 0.2702808988764045,
  "val_perplexity": 94.4994250165279,
  "train_loss": 4.803034782409668
}
```

Baseline comparison:

- Leaderboard baseline: `5.015`
- Stored baseline file: `baselines/10m_baseline.json`
- Improvement over leaderboard baseline: `0.46640625`

## Command

```bash
/venv/main/bin/python3 train_llm.py \
  --config 10m \
  --schedule_type warmup_decay_to_zero \
  --warmup_ratio 0.02 \
  --seed 42 \
  --dataset_path processed_data/pretrain_1B \
  --output_dir runs/issue30/10m_warmup_decay_w002
```

## Environment

- Remote host: `b31837861509`
- GPU: `NVIDIA A16`
- Python: `/venv/main/bin/python3`
- BF16: supported

## Artifacts

Pulled locally:

- `runs/issue30/10m_warmup_decay_w002/metrics.json`
- `runs/issue30/10m_warmup_decay_w002/model.pt`
- `logs/issue30/issue30_full_queue_status.log`
- `logs/issue30/10m_warmup_decay_w002.log`

Hashes:

```text
0820d687d47fc4fb4c17e2841527a17246cd6f3e6a54329cd73f63f02dc41f45  runs/issue30/10m_warmup_decay_w002/model.pt
f75539cac387de229390d2fc7070b8dcb61641f9d3f43a7103b2d15006bbb2ae  runs/issue30/10m_warmup_decay_w002/metrics.json
6e7208d2bc839a14bef8aa9121398610cc23ec239243ea8b7a4c085f12cc7eb7  logs/issue30/issue30_full_queue_status.log
4644116dcd3ef80bcedc04fcb642a439b651c41bfbfe86dba5337b22c637d549  logs/issue30/10m_warmup_decay_w002.log
```

Checkpoint contents:

- `checkpoint_version`
- `config`
- `git_metadata`
- `metrics`
- `metrics_history`
- `model_state_dict`
- `optimizer_state_dicts`
- `rng_state`
- `scheduler_state_dicts`
- `step`
- `tokens_seen`

## Source State Caveat

The run artifact recorded:

```text
git_commit: 7dcf2192dea70896e8aea920249a98e7def21fc4
git_branch: main
git_dirty: true
```

This result should not be treated as leaderboard-clean until the exact dirty source changes are committed on an experiment branch and tagged.

Remote dirty source files at run time included:

- `configs/llm_config.py`
- `train_llm.py`
- `training/trainer.py`

## Generation Sample

Prompt:

```text
In a quiet laboratory, the researcher noticed
```

Generated with:

```bash
PYTHONPATH=. /venv/main/bin/python3 scripts/generate.py \
  --checkpoint runs/issue30/10m_warmup_decay_w002/model.pt \
  --prompt "In a quiet laboratory, the researcher noticed" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50 \
  --seed 1
```

Output:

```text
In a quiet laboratory, the researcher noticed how the test was used in the old hospital in 1876.
A student in the hospital was a first test for the researchers, and the researchers found them.
The findings were a few years later. The research was presented as a study of the most important problem.
He said the findings have been published in the 1800’s. The students said “
```
