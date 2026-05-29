# Full Baseline Run

Current source of truth for the real baseline pass on the remote RTX 3090.

## Remote

- GPU: NVIDIA GeForce RTX 3090
- VRAM: 24 GiB
- Runtime env: `/venv/main`
- Long runs stay in `tmux`

## Dataset

- Source: `vukrosic/blueberry-1B-pretrain`
- Local cache: `processed_data/pretrain_1B`
- Reused for every preset

## Run Order

Largest to smallest so the cached dataset stays warm:

1. `50m`
2. `25m`
3. `5m`
4. `default`

## Commands

```bash
ssh -p 35010 root@136.34.244.176
tmux attach -t llm50mseq
source /venv/main/bin/activate
python train_llm.py --config 50m --device auto --dataset_path processed_data/pretrain_1B --compile true --output_dir runs/full-baseline/50m-seq
python train_llm.py --config 25m --device auto --dataset_path processed_data/pretrain_1B --compile true --output_dir runs/full-baseline/25m
python train_llm.py --config 5m --device auto --dataset_path processed_data/pretrain_1B --compile true --output_dir runs/full-baseline/5m
python train_llm.py --config default --device auto --dataset_path processed_data/pretrain_1B --compile true --output_dir runs/full-baseline/default
```

## Completed

| Preset | Params | Train tokens | Steps | Train loss | Val loss | Val accuracy | Val perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5m` | `6,652,800` | `8,000,000` | `489` | `5.2120` | `5.4153` | `0.1997` | `224.82` |
| `25m` | `25,366,272` | `25,000,000` | `1,526` | `4.2603` | `4.3581` | `0.2897` | `78.11` |

## Active Run

- `50m` sequential compiled run is live in tmux session `llm50mseq`
- Current GPU state on the remote host: 100% util, about 15.1 GiB used
