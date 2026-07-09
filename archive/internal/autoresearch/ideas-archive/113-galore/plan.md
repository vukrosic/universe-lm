# Plan — 113 galore

## Flag
`use_galore: bool = False` in [configs/llm_config.py](/Users/vukrosic/my-life/llm-research-kit-scaling/configs/llm_config.py); `galore_rank`, `galore_proj_every`, `galore_lr`, `galore_beta1`, `galore_beta2`, and `galore_eps` are only live when `use_galore=True`.

## Change
- `optimizers/galore.py`: 2-D low-rank projected AdamW with QR/SVD basis refresh in fp32 and a plain AdamW fallback for 1-D params.
- `training/trainer.py`: route the 2-D non-embedding, non-norm bucket to `GaLoreAdamW` when `use_galore=True`; keep the default Muon path when off.
- `train_llm.py`: CLI overrides for the GaLore flags so the same config can be toggled outside the `_arq` stub.
- `configs/llm_config.py`: add `Tiny1M3MGaLoreConfig(Tiny1M3MConfig)` with the GaLore defaults enabled.
- `_arq_113-galore.py` and `autoresearch/ideas/113-galore/run.json`: seed-42 queue handoff; `C` wraps `Tiny1M3MGaLoreConfig` for the daemon build-smoke.

Flag off leaves the baseline model graph and the default Muon routing unchanged.

## Control
Control: `Tiny1M3MConfig`, seed 42, tiny1m3m, baseline val 6.4306. Treatment: `Tiny1M3MGaLoreConfig`, seed 42, tiny1m3m. The only delta is the optimizer on the 2-D non-embedding, non-norm slot.

## Cost
Model params delta: 0. Optimizer-state delta: lower on the 2-D bucket because AdamW moments live in rank-r space. FLOPs delta: extra projection and refresh work only on the GaLore bucket. Memory delta: lower optimizer state on 2-D tensors; model weights and activations are unchanged.

## Run
`cd /root/universe-lm && /venv/main/bin/python _arq_113-galore.py` or equivalent `train_llm.py --config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`. Tier: tiny1m3m, seed 42. Pass/fail bar from `idea.md`: PASS <= ctrl - 0.005; NULL band |delta| < 0.005; DRIFT > +0.005.
