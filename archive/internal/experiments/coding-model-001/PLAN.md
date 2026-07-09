# Plan: Open Coding-Model Lab (experiment coding-model-001)

> **Status:** Phase 0 — Gate. Evals built, baseline not yet run. Plan written
> down so we don't re-debate it in a week.

---

## What we are doing

Build an **open-weights coding model lab** as a one-person operation, funded
by coaching revenue, shipping one concrete open release every ~6 weeks.

**In scope:** small (1-3B param) LLM, coding-specific, continued pretraining
+ SFT + DPO, open weights, training logs, public evals.

**Out of scope:** frontier-scale pretraining, chatbot product, agents,
infra-as-a-service, full company entity, "OpenAI competitor" framing in
any public copy.

## Why

- Coding is the only LLM submarket with clean public evals you can win on a budget.
- 1-3B is the size that fits on rented 4090s/A100s and iterates in days, not months.
- Crowded but not closed: Qwen 2.5 Coder, DeepSeek Coder, StarCoder already proved the path.
- Edge we actually have: training rig exists (`llm-research-kit-scaling`), shipped a real result
  (U-Net skips), YouTube channel for distribution.

## Funding ladder

Coaching funds the lab. Lab does not raise.

| Coaching MRR | Lab capacity |
|---|---|
| $139 (today) | 1 run per quarter, 1 GPU |
| $1,000 | 1 run per month, 8-GPU bursts |
| $3,000 | Multiple parallel runs, A100 80GB |
| $5,000+ | Layer grants + sponsorships |

Compute budget per release: **$500-1000** for a 1-3B continued-pretraining + SFT + DPO + eval cycle.

If coaching dies, the lab pauses. It does not raise to survive.

## Phases

### Phase 0 — Gate (this week)
- [x] Eval suite scaffolded: `evals/humaneval.py`, `evals/mbpp.py`, `evals/run_baseline.py`
- [x] Leaderboard scaffolded: `coding-model-leaderboard.md`
- [ ] Pick base model: **Qwen 2.5 Coder 1.5B** (Apache 2.0) or **DeepSeek Coder 1.3B** (MIT)
- [ ] Pick eval suite: HumanEval, HumanEval+, MBPP, LiveCodeBench-Easy
- [ ] Pick training stack: extend `llm-research-kit-scaling` trainer
- [ ] Pick dataset: focused subset of The Stack v2 (Python + TypeScript + Rust) + evol-instruct
- [ ] Write `experiments/coding-model-001/DECISIONS.md` locking the above

### Phase 1 — Continued pretraining (week 2-3, ~150 GPU-h)
- Continued pretraining of base model on 5-10B-token focused code corpus
- Track val loss + HumanEval every 1B tokens
- **First checkpoint to publish when HumanEval+ beats the base model by ≥3 points**

### Phase 2 — SFT + DPO (week 4-5, ~50 GPU-h)
- SFT on Magicoder-Evol-Instruct-110K + OSS-Instruct
- DPO on code preferences (UltraFeedback-Coder or self-generated)
- Re-eval, write results vs Qwen / DeepSeek of same size

### Phase 3 — Public release (week 6)
- Open weights (Apache 2.0 or MIT)
- Training data card
- Eval report (blog + paper-style writeup)
- YouTube deep-dive
- PostHog: track downloads and inferences if served

### Phase 4 — Iterate (monthly cadence)
- Scale to 3B
- Add languages / function-calling
- Keep coaching as floor, add grants/sponsors later

## Operating rules (commitments)

1. **One open release per 6 weeks.** Not "ongoing research" — a concrete artifact with a number on HumanEval+.
2. **No new products during a 6-week cycle.** Coaching + content + lab only.
3. **Public training logs.** No secret sauce. Brand = "I trained this on a budget, you can too."
4. **YouTube is the distribution.** Every milestone = video. Lab grows with the channel.
5. **No re-pivoting inside a 6-week cycle.** If we start, we finish.

## Open risks

1. This is **pivot #3 in ~1 week.** The "OpenAI competitor" framing is parked.
2. **Crowded space** — we will not "beat Qwen." Play is "smallest, cheapest, most reproducible."
3. **Coaching fragility** — $139/mo won't fund 2 failed runs. First run must be lean.
4. **No user yet** — 10 visitors, 0 inference users. Lab has no pull, only push.
5. **Time** — lab will eat content and coaching time. Be honest about the trade.

## The current next goal (Phase 0)

> Lock the four decisions in `experiments/coding-model-001/DECISIONS.md`:
> 1. Base model
> 2. Eval suite
> 3. Training stack
> 4. Dataset

After that is committed, the next goal is **run the eval baseline** to get a number on the
leaderboard and a starting pass@1 to beat.

The goal *after that* is **launch Phase 1** — first continued-pretraining run on rented GPU.

---

## Reference reading

- [Qwen 2.5 Coder technical report](https://arxiv.org/abs/2409.12186)
- [DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder)
- [Magicoder paper](https://arxiv.org/abs/2312.02120)
- [The Stack v2 dataset](https://huggingface.co/datasets/bigcode/the-stack-v2)
- [LiveCodeBench](https://livecodebench.github.io/)

## Companion files

- `evals/` — HumanEval + MBPP runner
- `coding-model-leaderboard.md` — pass@1 table
- `coaching/revenue.md` — the funding source (don't break this)
- `analytics/social-link-ledger.md` — tracking the lab's announcement posts
