# Next Ladder Arms: Long-Context Mechanisms (post-DeepNet)

**Status:** DeepNet study mechanistically closed (Muon-redundant, confirmed 8M/13M). Awaiting 23M completion + ablations to finalize. Preparing the immediate next research direction.

## Research Thesis

**Muon + RMSNorm already own the optimization-stability regime.** Per-layer balancing (deepnet/rezero/layerscale), residual growth bounding, and gradient uniformity are all erased or provided by the optimizer + norm at ≤30 layers. The scaling lever, if one exists, must come from mechanisms Muon does NOT substitute for: **attention and long-context mechanisms**.

The release ladder's next arms are ranked in `LONG-CONTEXT-IDEAS.md`. Wiring order:

1. **RoPE base scaling** (`rope_base=100k+`) — ranked #1, lowest risk, de-aliases position at range
2. **QK-norm post-RoPE** (`use_qk_norm_post_rope=True`) — ranked #3, stabilizing, entropy-collapse guard
3. **Differential attention** (`use_diff_attn=True`) — ranked #2, strongest long-context signal, new operator
4. **Intra-doc mask** (`use_intra_doc_mask`) — ranked #5, highest capability upside, heavier wiring (defer)

## Implementation Readiness

All four are **already wired** in `models/layers.py` and `train_llm.py` (flag names noted). The work is:
- Create `_arq_ladder*_ropebase.py`, `_arq_ladder*_qknorm.py`, etc. (one-line config subclasses)
- Register them in `run_rung.py` arms list (alongside baseline/deepnet/deepnet_ab/rezero/layerscale)
- Run them at 8M/13M/23M/52M/135M (same ladder infrastructure as deepnet)
- Evaluate long-context capability (needle-in-haystack, long-doc QA, long-file code) in addition to loss

## Expected Outcomes

### RoPE base (arms 1)
- **Mechanism:** larger base de-aliases low-frequency RoPE dimensions over long contexts
- **Prediction:** steeper exponent (loss gradient grows with depth as aliasing intensifies)
- **Long-context eval:** needle-in-haystack at 2×–4× train length should improve

### QK-norm (arm 3)
- **Mechanism:** RMSNorm Q and K before dot-product, caps attention logit scale
- **Prediction:** constant intercept shift (stabilizing, not exponent-bending) OR steeper if logit-collapse is a scale-dependent failure mode
- **Long-context eval:** entropy-collapse rescue; should maintain diversity at long range

### Diff-attn (arm 2)
- **Mechanism:** softmax₁ − λ·softmax₂, cancels attention noise
- **Prediction:** steeper exponent (noise floor that scales with sequence length, benefit grows at scale)
- **Long-context eval:** "lost-in-the-middle" retrieval; needle at deep positions should improve

### Intra-doc mask (arm 5)
- **Mechanism:** forbid cross-document attention in packed training
- **Prediction:** steeper exponent on long-context eval ONLY (loss may stay flat if eval is on-distribution)
- **Long-context eval:** needle-in-haystack vs. distractor documents; multi-doc reasoning

## Timeline

1. **Immediate (post-23M):** wire arms 1 & 3 (RoPE-base, QK-norm) — trivial, no new params, step-0 active
2. **Parallel:** run arm 2 (diff-attn) at 8M/13M, watch tiny screen for convergence (new operator)
3. **After 1+2 confirm:** design and wire arm 5 (intra-doc mask collate+kernel) — heavier, highest upside
4. **Selection:** the ladder's fitted L(N) at 135M target N picks the winner (steeper α preferred)

## Success Criteria (D002 + Release Ladder)

A long-context arm earns the 135M run if:
- Its L(N) curve sits **below baseline at 135M**, ideally via **steeper exponent** (not flat intercept shift)
- It does **not regress the long-context eval** vs. full-attention baseline (D002 reopen clause)
- The ladder verdict is **robust** across the 5 rungs (not a lucky-seed artifact at 8M)

## Related Research

- `LONG-CONTEXT-IDEAS.md` — detailed lever descriptions, ranking rationale, implementation status
- `LADDER.md` — scaling-law search philosophy, decision rule, fit procedure
- `DECISIONS.jsonl` — D001/D002 gates (long-context non-negotiable, no distance-punishing attention)
- `autoresearch/bin/run_rung.py` — parametrized ladder runner (arms registered here)
