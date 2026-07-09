---
id: 028-deep-thin-config
status: needs-run
round: 1
updated: 2026-06-10T12:43:18Z
transfer-risk: low
---

# 028 — Deep-and-Thin Config (depth/width ratio swap at fixed param budget)

## Source
Ma et al., "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases" (ICML 2024, arXiv:2402.14905). Reported +2.7% average gain at 125M and +4.3% at 350M from depth/width swap vs prior sub-billion SOTA, holding total parameters and training tokens fixed.

## Mechanism
At a fixed ~0.94M param budget, use a deeper-thinner architecture (more transformer layers, smaller d_model and d_ff, same total params). Hypothesis: each layer contributes a nonlinear transformation; more layers → more transformation steps → better representational depth per parameter. A/B is a new config class at ~same param count: n_layers↑, d_model/d_ff↓ proportionally to preserve the budget. Per-head `d_head = d_model / n_heads` is held at 16 (matches baseline); n_heads scales with d_model.

## Scale evidence
MobileLLM paper (Ma et al., ICML 2024): explicit ablations at 125M and 350M in Table 2 showing depth-prioritized architectures beat width-prioritized architectures at same parameter count by 2.7–4.3% on zero-shot benchmarks. transfer-risk: low — gains demonstrated at exactly the target model class (sub-400M, from-scratch training), though ≥133× our 0.94M budget so the open question is whether the lever still fires this small.

## Why it's worth a slot
The litrev (`plans/litrev-sub200m.md` §3) flags this as the highest-impact structural lever in winning sub-400M recipes, ahead of any optimizer or attention variant. Winning here compounds with every other stack lever (deeper model + FIRE + QKNorm). A null at tiny1m3m would indicate the depth benefit only materialises at longer training horizons (>3M tokens), which is informative for the ladder decision at 10M+ tier.

## Definition (gate 2)

### Ctrl vs trt
- **Ctrl**: `Tiny1M3MConfig` (`configs/llm_config.py:665`) — plain tiny1m3m baseline, val 6.4306 (LEADERBOARD.md row 14). Clean isolation of the depth/width lever; no FIRE/SWA/RoPE-tuned stack confound. (Alternative: stacking on the FIRE-equipped recipe like 020-025 would test compounding with the best attention stack but conflate two axes. For a structural config swap it's cleaner to A/B against the plain baseline first.)
- **Trt**: new `Tiny1M3MDeepThinConfig(Tiny1M3MConfig)` with the tuple below; all non-architectural fields inherited unchanged.
- **Config class**: `Tiny1M3MDeepThinConfig(Tiny1M3MConfig)`.

### Pinned tuple (B1 — closes the 0.94M budget cleanly)

| field         | baseline (`Tiny1M3MConfig`) | trt (`Tiny1M3MDeepThinConfig`) |
|---------------|-----------------------------|--------------------------------|
| `d_model`     | 64                          | **48**                         |
| `n_heads`     | 4                           | **3**                          |
| `n_kv_heads`  | 2                           | **3** (MHA-tied — see confound) |
| `n_layers`    | 12                          | **20**                         |
| `d_ff`        | 256                         | **192** (= 4·d_model)          |
| `d_head`      | 16 (=64/4)                  | 16 (=48/3) — preserved         |
| `emb_rank`    | 8                           | 8 — **preserved**              |
| `ffn_variant` | `squared_relu`              | `squared_relu` — **preserved** |
| `vocab_size`  | (inherited, ≈49152)         | (inherited) — preserved        |

Depth swap: 12→20 (1.67×). Width: 64→48 (0.75×). FFN expansion ratio held at 4× (192=4·48). Largest clean swap that closes the 0.94M budget without breaking the d_ff=4·d_model convention.

### Param-budget arithmetic (lands within ±5% of baseline 939k)
- **Baseline (`Tiny1M3MConfig`)**: per-block ≈ 12.3k (attn) + 32.8k (FFN: 2·64·256) + 0.3k (norms) ≈ 45.4k; ×12 = 545k. Embedding factorisation: 49152·8 + 8·64 ≈ 393.7k. **Total ≈ 939k ≈ 0.94M** ✓.
- **Trt B1**: per-block ≈ 9.2k (attn, n_heads=n_kv_heads=3, d_head=16) + 18.4k (FFN: 2·48·192) + 0.24k (norms) ≈ 27.9k; ×20 = 558k. Embedding factorisation: 49152·8 + 8·48 ≈ 393.6k. **Total ≈ 951k (+1.3%)** ✓.

### Known confound (call out explicitly, do not silently absorb)
B1's n_heads=3, n_kv_heads=3 is MHA (no kv-head sharing). Baseline is GQA 2:1 (n_heads=4, n_kv_heads=2). The depth/width swap therefore *also* collapses kv-sharing → MHA. Tied-QK / full-MHA is a known WIN signature at tiny1m3m (LEADERBOARD.md row 0 = vq-gain+rope250k+swa384+tiedqk, val 6.3041) — the trt's measured Δ partly reflects the n_kv_heads change, not pure depth/width.

This is the *least bad* clean tuple at the 0.94M budget: alternatives are (a) B1' = n_heads=3, n_kv_heads=1 (MQA, preserves "share kv across many query heads" ratio direction but compresses 3:1 instead of 2:1), or (b) B2 = d_model=32, n_heads=2, n_kv_heads=2, n_layers=24, d_ff≈272 (preserves GQA 2:1 only by forcing n_kv_heads=2 and bumping d_ff off the 4·d_model rule to absorb freed params — breaks the "pure depth/width swap" framing because FFN expansion ratio also changes from 4× to ~8.5×).

We pick **B1** because the d_ff=4·d_model convention is more load-bearing for "pure depth/width swap" than the GQA ratio, and call out the confound here so the runner reports it alongside the val-loss delta. A clean follow-up would be a 2-AB (B1 vs an n_kv_heads=2 control at the same depth/width) but that doubles the slot cost; deferred.

### Pass bar (tiny1m3m box noise ≈ ±0.01)
Run-to-run val-loss variance at this tier is ≈ ±0.01 (`closed.md:31-52` ctrls span e.g. 6.3875–6.4050 = 0.018 spread). Three non-overlapping bands tile the real line (mirrors 023-canon-conv `idea.md` r2 and 020-forgetting-attn):
- **WIN**: `trt_val < ctrl_val − 0.01` (strict; clears the cited noise floor)
- **NULL**: `|trt_val − ctrl_val| ≤ 0.01` (inclusive; sub-noise = inconclusive)
- **FAIL**: `trt_val > ctrl_val + 0.01` (strict; depth/width swap actively hurts at 0.94M)

ctrl_val baseline = 6.4306 (LEADERBOARD.md row 14). Result is interpreted against the **in-session** ctrl run, not the leaderboard number, to avoid cross-session drift confounds.

### Seed
**Seed 42 only.** Single fixed seed, no multi-seed sweep, no per-seed mean. A sub-noise delta is *inconclusive, not real*; never add "run more seeds to confirm" — log null and move on.

### Frozen (non-architectural) fields
All non-architectural fields inherited unchanged from `Tiny1M3MConfig`:
- `max_seq_len=2048`, `batch_size=2`, `train_tokens=3_000_000`, `compile_model=False`
- `warmup_ratio=0.02`, `schedule_type='warmup_decay_to_zero'`
- `eval_milestones=(0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700)` (the baseline tuple)
- all optimizer / Muon settings unchanged (no `muon_lr` bump, no schedule edit, no batch_size change)

The dataclass inheriting `Tiny1M3MConfig` already provides these by default; this is stated explicitly so the code-implementer does not bump LR or batch size to "rescue" a deeper model.

### LoC budget (≤30 LoC, well under the 200 ceiling)
- (a) new `Tiny1M3MDeepThinConfig(Tiny1M3MConfig)` dataclass overriding 5 fields (`d_model, n_heads, n_kv_heads, n_layers, d_ff`) ~10 LoC
- (b) docstring with the pass bar + Δ-vs-`Tiny1M3MConfig` plan + confound note ~10 LoC
- (c) param-count sanity assert (one-shot test) that `MinimalLLM(cfg)` has ≤ 0.99M params at init ~8 LoC

Total ≈ 28 LoC ≤ 30 cap. **No `models/layers.py`, `models/llm.py`, or any other shared-file edit needed.** Diff surface is `configs/llm_config.py` (one new dataclass, no existing-config edit) plus one test file — config-only, so the parallel-AI coordination memo (`MEMORY.md` `project-parallel-ai`) is a non-issue here.
