## r2 — 2026-06-15 — verdict: approve
- Wiring verified end-to-end: `use_vocab_bias: bool = False` at `configs/llm_config.py:545` (with the OH5 docstring block on lines 541-544), `vocab_bias` allocation at `models/llm.py:1311-1313`, forward hook `logits = logits + self.vocab_bias` at `models/llm.py:1883-1884`, and the canonical spec at `docs/research/output_head/plan.md:86` (OH5 VocabBias). Plan row matches the idea verbatim.
- Precedent stack validated: `Tiny1M3MAlibiConfig` at `configs/llm_config.py:2357` (current champion, val 6.2403), `Tiny1M3MLogitScaleConfig(Tiny1M3MAlibiConfig)` at line 2416 (idea 184, stacking pattern), `Tiny1M3MPreLMHeadRMSNormConfig(Tiny1M3MAlibiConfig)` at line 2449 (idea 183, stacking pattern). A `Tiny1M3MAlibiLMHeadBiasConfig(Tiny1M3MAlibiConfig)` subclass with `use_vocab_bias: bool = True` is the smallest-possible wiring change.
- `baseline-cache.json` confirms the cited numbers: box `5b8a7fea8963` shows `val_mean = 6.2403`, `val_std = 0.0088`, `noise_band = 0.04`, `n_measurements = 3`, measured `2026-06-15T07:04:48Z`. WIN bar `trt ≤ 6.2353` is correctly anchored to the post-175-ALiBi champion, not the pre-ALiBi 6.3988.
- `vocab_size = 49152` confirmed at `configs/llm_config.py:26` — +5.23% param cost (49,152 / 0.94M) is correctly characterized as a *budget-matched* lever, not a "small" injection. The bar-vs-cost framing is honest.
- Not a duplicate of any closed axis. `closed.md` and `LEADERBOARD.md` contain no per-vocab additive LM-head-bias entries (grepped for `vocab_bias`, `VocabBias`, `lm_head_bias`, `lm-head-bias`). The OH5 plan row is the *spec*, not a prior *run*. Closest family members (152-attn-logit-bias, 155-per-head-temp, 160-rms-gain-per-head, 166-t5-rpe) are all *per-head attention-shape* levers that nulled at 0.94M — 187 is *output-side per-vocab*, mechanistically orthogonal.
- Step-0 byte-identity claim is sound: `vocab_bias = zeros(V)` is a literal zero tensor; `logits + 0 = logits` in fp32 exact; backward introduces only the new bias gradient. Implementer must verify `max_abs_diff(trt_step0_loss, ctrl_step0_loss) == 0.0` against the ALiBi champion stack (not plain `Tiny1M3MConfig`) — the r1 finding explicitly pinned this and the reviser acknowledged it.
- Mechanism vs hyperparameter: per-vocab additive bias is a structural lever (one scalar per vocab token), not an LR/schedule/init-constant knob. Pass.
- Transfer-risk: low, justified. T5 220M-11B is genuine direct validation; the 2023+ frontier-decoder abandonment (LLaMA, Mistral, Qwen, Gemma, OLMo) is a *capacity* argument, not a mechanism invalidation. The 187 bet is that at 0.94M the per-token output knob has more value than at 7B+ where the model can adjust the embedding directly — that bet is specific, falsifiable, and the tier is correct (single seed 42, tiny1m3m).
- Falsifiable bar: WIN ≤ 6.2353 (champion − 0.005) clears two-ctrl rule; NULL is sub-noise `|Δ| < 0.01`; DRIFT > 6.2503. Expected Δ ∈ [−0.005, −0.02] per OH5 framing. Two-sided informative: a win unlocks a per-vocab bias lever (gated on param cost); a null closes the LM-head-bias axis at 0.94M and tells us the 2023+ frontier-decoder trend holds at this tier. Sub-noise is correctly tagged inconclusive per the one-seed-only rule.
- Diff to check on implement: confirm `models/llm.py:1311-1313` is unchanged and that the new subclass flips `use_vocab_bias = True` without touching the ALiBi config or any other lever.

## r1 — 2026-06-15 — verdict: revise

- **vocab_size is wrong (6× off) → param-count claim breaks.** idea.md:45 says
  `vocab_size = 8192 (typical for tiny1m3m; verify from the config)` and uses
  that to compute `8192 params (+0.87% of 0.94M)`. The actual `LLMConfig.vocab_size`
  is **49152** (configs/llm_config.py). True overhead is **49,152 params = ~+5.2%
  of 0.94M** — a six-fold correction and a *sizeable* param injection, not the
  "0.87%" figure. This changes the portfolio-fit argument (a +5% lever is not
  "small but not negligible" — it's larger than the entire budget for a typical
  OutputHead batch lever). The plan.md (OH5 VocabBias) explicitly tags it
  "many params but trivial compute"; please update the intution + portfolio
  framing to match.
- **baseline-cache reference is stale.** idea.md:58 cites
  `autoresearch/baseline-cache.json box 5b8a7fea8963 (RTX 3060), val_mean = 6.3988,
  noise_band = 0.04, n_measurements = 3`. The current pinned cache
  (autoresearch/baseline-cache.json, measured 2026-06-15T07:04:48Z) shows
  **`val_mean = 6.2403, n_measurements = 3` for that box_key**. The 6.3988
  number is the *pre-175-alibi* baseline; 175 alibi-slopes is a closed WIN
  (Δ-0.1585) and the champion stack is now `Tiny1M3MAlibiConfig` (per
  configs/llm_config.py, which is what `Tiny1M3MLogitScaleConfig` etc. subclass).
  The pass/fail bar must be against the **current champion**, not the pre-ALiBi
  baseline. Either (a) restate the bar as `Tiny1M3MAlibiConfig + use_vocab_bias`
  vs `Tiny1M3MAlibiConfig` baseline, or (b) keep the cache reference but pull a
  fresh number on run day and clearly mark this as the ABLATIVE Δ (i.e.
  vocab-bias-only vs the champ stack).
- **Lever is already implemented as OH5 VocabBias — say so in the plan, don't re-derive.**
  The exact lever `logits += b_v` with `b_v = zeros(vocab_size)` is already wired:
  - config flag `use_vocab_bias: bool = False` (configs/llm_config.py, line ~N
    in the OH5 VocabBias comment block)
  - parameter allocation `self.vocab_bias = nn.Parameter(torch.zeros(config.vocab_size))`
    gated on `use_vocab_bias` (models/llm.py)
  - forward hook `if self.use_vocab_bias: logits = logits + self.vocab_bias`
    after softcap (models/llm.py)
  The plan.md (docs/research/output_head/plan.md, Batch 2 row OH5) is this
  exact mechanism. The idea.md source/mechanism section is a clean re-statement
  but doesn't acknowledge either the config flag or the plan.md entry. The
  reviser must: (a) add an `## Existing wiring` section naming `use_vocab_bias`
  + the line in models/llm.py, (b) update the Design sketch to add a
  `Tiny1M3MAlibiLMHeadBiasConfig(Tiny1M3MAlibiConfig)` subclass (stacking on
  the current champion, matching the 184-logit-scale precedent) rather than a
  raw `Tiny1M3MConfig` subclass, (c) drop the redundant mechanism prose and
  instead cite the OH5 row in plan.md as the canonical spec.
- **Defensive comparison needs to cite OH5/plan.md, not just closed.md.** The
  idea.md "Distinct from closed axes" section claims "no prior lever in the
  repo tests a per-vocab LM head bias." That statement is true vs `closed.md`
  but false vs `docs/research/output_head/plan.md` (OH5 row). The plan row
  frames the lever as "mostly re-learns token frequency, a known small CE
  win"; the 187 pitch frames it as "output/input decoupling via per-vocab
  scalar." Both framings are correct, but the planner needs to pick one (or
  unify them) so the run spec is unambiguous about the *expected mechanism*
  vs the *expected magnitude*. Pre-binding on "expected Δ ~ -0.005 to -0.02"
  (OH5's framing) vs "expected Δ bound by the per-token decoupling capacity
  at 0.94M" (187's framing) leads to different pass bars.
- **Step-0 byte-identity claim is sound, but verify against champion.** The
  `lm_head_bias = zeros(V)` ⇒ `logits + 0 = logits` argument is correct for
  any tied-or-untied head. The implementer must verify
  `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0` AND
  `max_abs_diff(trt_step0_loss, ctrl_step0_loss) == 0.0` where the ctrl is the
  **champion stack (Tiny1M3MAlibiConfig)**, not plain Tiny1M3MConfig.
- **Transfer-risk: low holds.** T5 220M-11B validation is genuine; the
  frontier-decoder abandonment is a capacity argument not a mechanism
  invalidation. Fine as-is.
- **Falsifiable bar exists, but the threshold needs re-anchoring.** Current bar
  is `trt_val ≤ ctrl_val_mean - 0.005`. With `ctrl_val_mean` set to the
  pre-ALiBi 6.3988 baseline, that bar is *too loose* (we already know 175
  cleared 6.24). With it set to 6.2403 + the expected OH5 lever magnitude
  (-0.005..-0.02), the bar becomes meaningful: a vocab-bias win on top of ALiBi
  would have to clear `≤ 6.2353` (champion - 0.005) to count, and the noise
  band is ±0.04. Tighten or accept borderline-null, but make the threshold
  match the *actual* champion stack you're stacking on.

**Next step:** Reviser — apply the four corrections above (vocab_size
49152, baseline-cache fresh pull, existing-wiring acknowledgement, subclass
on Tiny1M3MAlibiConfig). Then re-flip to `needs-review` for another pass.
