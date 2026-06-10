# Code-review log — 025 Scalable-Softmax

## r1 — 2026-06-10 — verdict: accept

- **Mechanism faithful** (`models/layers.py:1522-1529, 1652-1659`): the
  multiplier is `s_h.view(1,H,1,1) * log_n.view(1,1,T,1)` where
  `log_n = log(arange(1, T+1))`, applied to `scores` after the additive
  positional biases (FIRE/CoPE/ALiBi) and before the causal mask. For
  the primary A/B (`Tiny1M3MSSMaxConfig` = baseline + ssmax, no
  FIRE/CoPE/etc.) the only score-side op before SSMax is
  `QKᵀ·scale`, so the effective forward is
  `softmax( (s_h · log n) · QKᵀ/√d )` — bit-equal to `idea.md:14`
  by scalar-multiplication associativity. log_n[0] = log(1) = 0 is
  exercised at query 0 and harmless (single unmasked key → softmax
  is temperature-invariant), per the plan's numerical note.
- **OFF path is bit-identical** (smoke verified): with
  `use_ssmax=False`, `self.ssmax_s` is never instantiated
  (`models/layers.py:695-696`), `hasattr(attn,'ssmax_s')` is False,
  and the elif chain at `models/layers.py:1560-1572` only pulls
  `or self.use_ssmax` into the manual path — so a build with all
  other manual-branch flags also off stays on the SDPA fast path.
  Smoke: `Tiny1M3MConfig` → 0 ssmax params, forward OK.
- **Step-0 non-bit-identical at flag-on is explicitly justified.** The
  paper's `s_h·log n` scaling IS the mechanism; `idea.md:14` and the
  reviewer's r2 verdict in `review.md` flag this as a permitted
  deviation. The plan's `plan.md:108-117` re-states the justification.
- **No silent HP drift.** Smoke field-parity check between
  `Tiny1M3MConfig` and `Tiny1M3MSSMaxConfig`: only `use_ssmax`
  differs across all `vars()` keys. No LR/schedule/init/seed change
  smuggled in.
- **Param routing correct.** `ssmax_s` is `nn.Parameter(torch.ones(H))`
  → ndim=1 → routes to AdamW in `training/trainer.py:109-115`
  (`param.ndim == 2` Muon predicate fails). Per-head scalar on AdamW
  is the standard treatment for the family (matches `cosine_tau`,
  `alibi_slope`, `q_gain`, etc.).
- **Cost matches plan.** Smoke: 48 ssmax params total
  (4 heads · 12 unique blocks), bit-equal to plan.md:77-79.
- **Shape correctness under GQA.** `n_heads=4, n_kv_heads=2` → K,V are
  `repeat_interleave`'d to 4 heads (`models/layers.py:1412-1414`)
  before the transpose at line 1500. The multiplier
  `[1,H,T,1]` broadcasts cleanly with `scores [B,H,T,T]`. Per-head
  scalar is on the expanded head count, which is correct (this is
  the head count seen by `scores`).
- **Routing through wiring layers verified.** Flag flows from
  `LLMConfig.use_ssmax` (`configs/llm_config.py:205`) →
  `MinimalLLM.use_ssmax` (`models/llm.py:241`) → `TransformerBlock`
  (`models/llm.py:363`, `models/layers.py:1976, 2026`) →
  `MultiHeadAttention.use_ssmax` (`models/layers.py:521, 694`).
  No missing pass-through.
- **Plan ↔ idea consistency.** Primary A/B in `plan.md:55-74` is
  `Tiny1M3MConfig` vs `Tiny1M3MSSMaxConfig`, seed 42, tiny1m3m,
  WIN bar ≤ −0.01, NULL band, anti-cheat ±0.0053 fence — all
  match `idea.md:19-30`. Follow-ups (FIRE-stack, qk-norm-stack)
  are correctly gated behind the primary clearing and NOT wired
  as runner targets — appropriate.
- **LoC budget.** ~42 SSMax-tagged diff lines across the three
  files (comments + 5 wiring points + ssmax_s init + multiplier
  in 2 branches + 1 elif `or` clause). Well under the 200-LoC
  cap for mined ideas.
- **Coordination.** Edits are anchored on the `use_softpick`
  neighbourhood (param init, elif chain, block kwargs); no
  overlap with the parallel agent's softpick/canon/gated-attn
  changes in the same files. No revert of unstaged work.
- **Soft note (non-blocking, fixed on accept):** `Tiny1M3MSSMaxConfig`
  is not in `configs/__init__.py`'s re-export list, but the arq
  runner pattern imports `from configs.llm_config import ...`
  directly (see `_arq_020.py:3`, and note that
  `Tiny1M3MFOXOnFireConfig` is similarly unexported and runs fine).
  Not a blocker.
- **Soft note (non-blocking):** `autoresearch/queue.md` "Ideas
  board" table stops at 020 — rows for 021–025 are missing. Not
  introduced by this diff (021–024 already missing). Out of scope
  for this gate.

**flip.sh release:** `accept: faithful, OFF bit-identical, ~42 LoC, smoke OK`
