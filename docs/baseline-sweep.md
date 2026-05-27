# Baseline Sweep

This is the first dense baseline pass on the remote Vast AI box.

## Remote GPU

- NVIDIA GeForce RTX 3090
- 24 GiB VRAM
- PyTorch `2.11.0+cu130`
- Device selection: `auto` resolved to CUDA

## Run Setup

- Dataset: `HuggingFaceTB/smollm-corpus` with `cosmopedia-v2`
- Token budget: `100,000` tokens per model
- Batch size: `1`
- Compile: `false`
- Sequence length: `2048`
- Seed: `42`

## Results

| Preset | Params | Final train loss | Final val loss | Final val accuracy | Wall time |
|---|---:|---:|---:|---:|---:|
| `5m` | `6,652,800` | `6.8429` | `7.1161` | `0.1298` | `4.68s` |
| `25m` | `25,366,272` | `6.8315` | `7.0268` | `0.1388` | `12.79s` |
| `50m` | `48,244,224` | `6.9167` | `7.1607` | `0.1380` | `15.46s` |
| `default` | `88,630,528` | `6.9672` | `7.2321` | `0.1345` | `15.26s` |

## Notes

- The `100m` preset was intentionally not run.
- All runs used the same data source, seed, token budget, and batch size.
- The sweep is short on purpose. It is a baseline pass, not the final full-budget training run.
