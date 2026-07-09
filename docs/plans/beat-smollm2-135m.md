# Program plan: beat SmolLM2-135M

Status: draft v1 (2026-06-10)
Assumption: revenue funds compute as needed (~$3k for the flagship run).
This is the umbrella program; phases 1–2 run as autoresearch campaigns.

## Claim we are building toward

> At 135M parameters and 2T tokens, our open recipe (architecture + optimizer +
> data mix), developed end-to-end by the autoresearch pipeline, outperforms
> SmolLM2-135M on the pinned eval suite — with every decision logged and
> reproducible.

Intermediate (claimable early): at matched params and matched *partial* token
budget, our recipe beats the SmolLM2 recipe trained identically. This is the
honest token-matched A/B and is publishable/postable on its own.

## Target spec (verify in Phase 0 — do not trust from memory)

| | SmolLM2-135M |
|---|---|
| Params | ~135M |
| Layers / hidden / heads | 30 / 576 / 9 (3 KV, GQA) |
| Tokens | ~2T |
| Tokenizer | 49152 vocab (cosmo2) |
| Context | 2048 (base) |
| Data | FineWeb-Edu + DCLM + Stack mix |

## Benchmarks (pinned, all comparisons use OUR reruns, never paper numbers)

Primary suite — lm-eval-harness, pinned commit, pinned task versions, 0-shot
unless noted:

1. HellaSwag
2. ARC-easy + ARC-challenge
3. PIQA
4. Winogrande
5. OpenBookQA
6. CommonsenseQA
7. MMLU (5-shot) — reported but NOT in win condition (135M is near chance)

Continuous metric: FineWeb-Edu held-out BPB (bits per byte) — tokenizer-
independent, tracked during training, drives all A/B decisions before evals
have signal.

**Win condition (flagship):** ≥ SmolLM2-135M on ≥5 of 6 win-eligible tasks
AND lower FineWeb-Edu BPB.
**Win condition (token-matched A/B):** Δ BPB beyond the two-ctrl noise
bracket + no task regression beyond bracket.

## Phases

### Phase 0 — Pin the target (week of 2026-06-15, ~$0, CPU + small GPU)
- [ ] Run lm-eval-harness on released SmolLM2-135M → `BASELINE.md` (scores,
      harness commit, task versions, exact commands)
- [x] Harness pinned + smoke eval done (see `docs/plans/benchmark-protocol.md`,
      `scripts/eval_baseline.sh`, `results/baseline-smollm2-135m/`) — full
      suite still needs a GPU box
- DECIDED 2026-06-10: we take NOTHING from SmolLM2 — no arch config, no
  recipe. Their checkpoint is a row of benchmark scores; only constraint we
  inherit is params ≤ 135M. Our architecture is decided by our own pipeline.
  (Tokenizer for data shards: cosmo2 as practical default — benchmarks and
  BPB are tokenizer-independent, so this is convenience, not borrowing.)
- [ ] Eval harness wired into autoresearch evidence.md (benchmarks become
      first-class results, not just val_loss)

### Phase 1 — Recipe development at proxy scale (weeks 2–6, current Vast budget)
- Autoresearch campaign at 10M and ~30M tiers, seed 42, two-ctrl bracket
- Candidates: every done-idea WIN that cleared noise (qk-norm, value-residual,
  gated attention, muon variants, …) + data-mix ratios + schedule
- Stacking rule: a lever enters the recipe only if it clears the bracket at
  BOTH tiers; re-test the stack (interactions are real)
- Output: `recipe-v1.md` — full config diff vs ctrl-smollm2
- Doubles as: course module "how a recipe gets built", weekly posts

### Phase 2 — Scaling validation (weeks 6–10, ~$200–500)
- Scale ladder (DECIDED 2026-06-10): 1M = screen only (not in trend fits);
  trend tiers **10M → 37M → 135M**, log-even (~3.7× per step), ≥3 points so
  Δ(scale) curvature is visible per lever/stack
- Token-matched A/B: recipe-v1 vs our own base config at 135M/20B
- Gate: recipe Δ must persist at 135M or levers get dropped scale-by-scale
- Doubles as: held-out-scale validation for the Muon LR-law paper (question #4)
- Infra hardening here: multi-GPU (FSDP or DDP), resumable checkpoints,
  spot-instance death tolerance, eval-during-training

### Phase 3 — Data engineering (parallel with Phase 2, ~$100 + 4–5TB storage)
- 2T-token corpus: FineWeb-Edu + DCLM (+ Stack-Edu if code stays in scope)
- Pre-tokenize to uint16 shards (~4TB), streaming loader, shard checksums
- Mix ratios are Phase-1 experiments, not guesses
- This is the highest-risk workstream — start it early, it's the usual boss
  fight, not architecture

### Phase 4 — Flagship run (when funded, ~$2.5–3.5k)
- 135M × 2T tokens ≈ 1.6e21 FLOPs ≈ 1.1–1.3k H100-hours
  (≈6 days on 8×H100 at 35–40% MFU; Vast ~$2/GPU-hr)
- Mid-run evals every ~100B tokens vs SmolLM2 trajectory; abort/fix gates at
  10B and 100B tokens (compare BPB vs ctrl extrapolation)
- Two attempts budgeted; assume the first one finds a bug

### Phase 5 — Release & proof (1 week after run)
- HF release: weights, configs, data list, training logs, eval table vs our
  SmolLM2 reruns, full autoresearch decision log
- Writeup: release post + course capstone + paper draft material
- Repo quickstart so a stranger reproduces eval table in <1 day (client bar)

## Decision points (resolve in Phase 0/1, then frozen)

1. Tokenizer: reuse cosmo2 (default) or train own
2. Context: 2048 to match, or 4096 if cheap (decide from Phase-1 data)
3. Code in mix or not (affects Stack-Edu + eval expectations)
4. Exact param count: match 135M ±2% — never exceed, "smaller and better"
   beats "bigger and better"

## Risks

| Risk | Mitigation |
|---|---|
| Tiny-scale wins don't transfer | scale ladder 10M→30M→60M→135M, drop levers at the tier where they die |
| Data pipeline eats the schedule | Phase 3 starts during Phase 2, not after |
| Eval noise at 135M | BPB is the decision metric; tasks are the headline |
| Flagship run dies mid-way | resumable ckpts every 30 min, spot-tolerant launcher, 2-attempt budget |
| Solo-hours, not compute, becomes the constraint | every phase output is simultaneously course content + post + open-source doc; nothing written twice |
