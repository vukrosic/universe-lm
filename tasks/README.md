# Experiment tasks — paper implementations

Open, claimable research tasks. Each file in `papers/` is one self-contained task from a recent
paper: what to read, what to implement, what to train, and what an accepted result looks like.
Accepted results get named credit on the published report.

**Rules (all tasks):**
1. **Baseline first.** Your first contribution is reproducing the pinned baseline
   (`--config_class configs.llm_config.Ladder23M469MConfig`, seed 42) and reporting loss curve +
   final val loss + wall-clock + GPU. Every task compares against it.
2. Read the actual paper PDF before running anything.
3. Equal token budgets between arms, control run present, config diff + logs + **at least one
   figure** in the PR — no figure, not accepted.
4. All data-axis arms are scored on the **shared FineWeb-Edu held-out set**
   (`scripts/bpb_fineweb_edu.py`), never on each corpus's own split. Different tokenizers or
   corpora → report **bits-per-byte**, never per-token loss.

| # | Task | Paper venue | Config | ~GPU cost |
|---|---|---|---|---|
| [P1](papers/P1-proxy-lr-ranking.md) | Do small-model verdicts survive? (proxy-LR check) | ICLR 2026 | 23M ×4–6 | $5–8 |
| [P2](papers/P2-mixture-scaling-law.md) | Fit a mixture scaling law, extrapolate the mix | ICML 2026 ×2 | 23M grid + 52M | $15–25 |
| [P3](papers/P3-quality-filter-threshold.md) | What does the quality filter actually do? | ICML 2026 | 23M ×2 | $4 |
| [P4](papers/P4-repeat-vs-mix.md) | Repeat the good data, or add more kinds? | ICML 2026 | 52M ×2 | $8–12 |
| [P5](papers/P5-decay-phase-data.md) | Put the best data where the model can still learn | ICML 2026 spotlight | 52M ×3 | $12–18 |
| [P6](papers/P6-vocab-sweep.md) | Is a 49k vocab wrong for a small model? | ICML 2026 spotlight | 23M ×4–5 | $10 |
| [P7](papers/P7-late-to-early.md) | Late-to-Early Training (read-and-gate) | preprint | 23M ×2 | $4 |
| [P8](papers/P8-synthetic-swap.md) | Swap in free public synthetic data | preprint + public corpus | 23M ×1 | $2 |

Configs: `Ladder23M469MConfig` (23M params / 469M tokens) and `Ladder52M1042MConfig` (52M / 1.04B)
in `configs/llm_config.py`, loaded via `--config_class`. Winners must survive both scales before
entering the 135M flagship recipe.

More papers to browse and propose from: [PAPER-SUGGESTIONS.md](PAPER-SUGGESTIONS.md).
