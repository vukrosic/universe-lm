# Tutorials Map

This folder holds the teachable writeups from the small-compute LLM ablation work.

The rough split is:

- `README.md` files are the main tutorial entrypoints.
- companion PDFs, translations, and figures live inside the same tutorial folder.
- compact machine-readable evidence lives in `../../results/`.
- raw run folders and checkpoints stay out of git.

| tutorial | path | status |
|---|---|---|
| Value embeddings | [`value_embeddings/README.md`](value_embeddings/README.md) | polished tutorial |
| QKV embeddings | [`qkv_embeddings/README.md`](qkv_embeddings/README.md) | polished tutorial with X/PDF companion assets |
| Embedding factorization depth | [`embedding_factorization_depth/README.md`](embedding_factorization_depth/README.md) | polished tutorial |
| QK gain | [`qk_gain/README.md`](qk_gain/README.md) | polished tutorial, figures, English PDF, Chinese version |
| Normalization | [`normalization/README.md`](normalization/README.md) | single tutorial with figures and supporting research notes |
| Sliding-window attention | [`swa_record/README.md`](swa_record/README.md) | polished tutorial, waterfall figure |
| RoPE base tuning | [`rope_base/README.md`](rope_base/README.md) | polished tutorial |
| LR warmup-decay schedule | [`lr_schedules/README.md`](lr_schedules/README.md) | polished tutorial |
| KV-head sharing (MHA/GQA/MLA) | [`kv_sharing/README.md`](kv_sharing/README.md) | polished tutorial (negative result) |
| Tied QK (PaLM) | [`tied_qk/README.md`](tied_qk/README.md) | polished tutorial (wins small, fades up) |
| Post-norm collapse | [`postnorm_collapse/README.md`](postnorm_collapse/README.md) | polished tutorial (closed axis) |
| FFN activations (ReLU²/GELU/SwiGLU) | [`ffn_activations/README.md`](ffn_activations/README.md) | polished tutorial (conditional lever) |
| Linear attention (Performer) | [`linear_attention/README.md`](linear_attention/README.md) | polished tutorial (closed at small scale) |

## Tutorial Folders

### QK Gain

All QK gain material stays together:

- [`qk_gain/README.md`](qk_gain/README.md) - English tutorial
- [`qk_gain/README.cn.md`](qk_gain/README.cn.md) - Chinese tutorial
- [`qk_gain/qk_gain.pdf`](qk_gain/qk_gain.pdf) - English PDF
- [`qk_gain/qk_gain_cn.pdf`](qk_gain/qk_gain_cn.pdf) - Chinese PDF
- [`qk_gain/images/`](qk_gain/images/) - source figures used by the tutorial

### Normalization

The normalization work is now in one folder so it can become one tutorial:

- [`normalization/README.md`](normalization/README.md) - main teaching draft
- [`normalization/ablations.md`](normalization/ablations.md) - short result table
- [`normalization/findings.md`](normalization/findings.md) - full findings log
- [`normalization/images/`](normalization/images/) - figures used by the tutorial
