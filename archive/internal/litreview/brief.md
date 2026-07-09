# Literature-review brief

> What papers should we read, and why? Scout agents read this before searching.
> Synced with [`../autoresearch/brief.md`](../autoresearch/brief.md) — lit review
> feeds the ablation pipeline, not a separate research program.

## Topic

Mechanisms for cheap LLM pretraining improvements: attention variants, optimizers
(Muon/Lion/Shampoo family), positional encodings, loss functions — anything
implementable in < 200 LoC with identity-init at `tiny1m3m`.

## Research questions

1. What papers propose **structural levers** (not HP sweeps) that plausibly lower
   val loss at small scale?
2. Which ideas are **already falsified** or duplicated in our repo / `closed.md`?
3. What **gaps** remain in our coverage (themes under-explored in autoresearch)?

## Search priorities (rotate weekly)

- arXiv `cs.LG`, `cs.CL` — 2025–2026: linear attention, delta rule, forget gates,
  Muon, cautious optimizers, RoPE alternatives, CoPE, poly-loss family
- Papers With Code, HuggingFace daily papers
- Su Jianlin / kexue.fm (RoPE author — mechanism-heavy posts)
- Follow-ups on **009-fire-pe** (position bias) and **011-cautious-lion** (WIN levers)

## Out of scope

- Inference-only tricks, quantization, tokenizer changes
- Multi-seed protocols (we run seed 42 only — note but don't adopt)
- Ideas requiring > 200 LoC or new data pipelines
- Pure scaling-law papers with no transferable mechanism

## Themes (for `theme:` frontmatter)

`attention` · `optimizer` · `position` · `loss` · `norm` · `moe` · `ssm`

## Success

- Screen queue rarely empty while autoresearch upstream is hungry for ideas
- Every `done` digest ends with a clear **Suggested action**
- `synthesis.md` updated when ≥5 new digests land in one theme
