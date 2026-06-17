# The Map — every decision to train an LLM, in order

Read this first. It is the spine of the manual: the decisions you actually make when you
sit down to train a model, in roughly the order you make them, each pointing to the section
that holds the rules. A `★` marks a decision where getting it wrong wastes most of your
compute.

```
                         ┌─────────────────────────────┐
                         │ 0. What am I building?       │
                         │   target use, compute budget │
                         │   C (FLOPs), data ceiling    │
                         └──────────────┬──────────────┘
                                        │
        ★ 1. ALLOCATE COMPUTE ──────────┼─────────────────────────────  §01 scaling-laws
           how big a model, how many tokens, for budget C?
           → Chinchilla: N and D scale together, ~20 tokens/param   (FM-01.1)
           → data-limited? repeat up to ~4 epochs ≈ free            (FM-01.2)
           → pick vocab size to match model size                    (FM-01.3)
                                        │
        ★ 2. GET THE DATA ──────────────┼─────────────────────────────  §02 data
           → quality > quantity; filter hard                        (FM-02.1)
           → deduplicate before everything else                     (FM-02.2)
           → set the domain mixture (code/math oversampled)         (FM-02.3)
                                        │
          3. CHOOSE THE ARCHITECTURE ───┼─────────────────────────────  §03 architecture
           → the convergent recipe: pre-RMSNorm + RoPE + SwiGLU + GQA (FM-03.1)
             decoder-only transformer, next-token prediction
                                        │
        ★ 4. SET HYPERPARAMETERS ───────┼─────────────────────────────  §04 optimization
           → AdamW β=(0.9,0.95), wd 0.1, clip 1.0                   (FM-04.1)
           → warmup + decay-to-(near-)zero schedule                 (FM-04.2)
           → batch size / weight decay scale with DATA              (FM-04.3)
           → tune small, transfer with μP                           (FM-04.4)
                                        │
          5. KEEP IT FROM DIVERGING ────┼─────────────────────────────  §05 stability
           → z-loss + qk-norm; watch logit growth                   (FM-05.1)
                                        │
          6. (UNDERSTAND WHAT IT LEARNS)┼─────────────────────────────  §06 knowledge+reasoning
           → capacity ceiling ≈ 2 bits/param                        (FM-06.1)
           → facts need varied phrasings to become *extractable*    (FM-06.2)
           → reasoning is a learnable process, measurable           (FM-06.3)
                                        │
          7. MAKE IT AN ASSISTANT ──────┼─────────────────────────────  §07 post-training
           → SFT → preference optimization (DPO/RLHF)               (FM-07.1)
                                        │
          8. (SCALE / SHIP CHEAPER) ────┼─────────────────────────────  §08 efficiency+scale
           → MoE: capacity ≫ active compute                         (FM-08.1)
           → BF16 default, FP8 frontier                             (FM-08.2)
           → extend context: PI → NTK → YaRN                        (FM-08.3)
                                        ▼
                                   ship + evaluate
```

> Steps 1–4 are the from-scratch core (and where most compute is won or lost). Step 8 is
> optional for a *first* model but is how a *competitive* one is built. **Not yet in the
> manual:** inference/serving and a real evaluation section — see §08's honest-gaps note.

## The few rules that dominate everything

If a reader remembers nothing else:

1. **Compute allocation is the master decision.** Most historically "big" models were
   *undertrained*: too many parameters, too few tokens. Fix the params↔tokens split first
   (§01). Everything downstream is a smaller lever.
2. **Data quality is a multiplier, not an additive term.** A clean, deduplicated,
   well-mixed corpus beats a larger dirty one at equal compute (§02).
3. **The architecture is nearly solved for this generation.** Five independent labs
   converged on the same recipe (§03). Spend your novelty budget elsewhere unless you have
   a specific reason.
4. **Tune cheap, transfer up.** You cannot afford to tune hyperparameters at full scale;
   μP and scaling-law fits exist so you don't have to (§04).
5. **Scope is everything.** Every number below was measured at *some* scale. Quoting it
   outside that scale is how the field misleads itself — and how we'll mislead ourselves if
   we're not careful. This is exactly why the **measured-rule** half of this manual
   ([PIPELINE.md](PIPELINE.md) + [drafts/](drafts/)) exists: to find out which of these
   field rules actually hold *at our scale*.

## How this map relates to our measured ledger

Our experiments (the `L###/D###/C###` entries + [drafts/](drafts/)) live mostly at **tiny
scale (1–3M params, 3M
tokens)** and focus on **structural** levers (attention, positional, norm, FFN). That means:

- We are well-positioned to interrogate **§03 architecture** and **§05 stability** rules
  directly.
- We are *not* positioned to confirm **§01 scaling laws** ourselves — those need scale we
  don't have. We cite them; we don't own them.
- The single most valuable thing we can add to the field: **does a structural rule that the
  literature established at 7B–70B still hold its sign at 1–3M, and does ours transfer up?**
  That is the open question the ledger's G3 agenda exists to answer.
