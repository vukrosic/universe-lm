# Review log — 028 deep-thin-config

## r2 — 2026-06-10 — verdict: approve

**All 8 r1 findings applied. Spec now closes — pin tuple, arithmetic, pass bar,
freezes, confound disclosure all present. Approve to `needs-plan`, reset round
to 1 for the code gate.**

**Re-check sweep (r1 findings → r2 status):**
- **Definition (gate 2) block present** with `### Ctrl vs trt`, `### Pinned
  tuple`, `### Param-budget arithmetic`, `### Pass bar`, `### Seed`,
  `### Frozen (non-architectural) fields`, `### LoC budget`. ✓
- **B1 tuple pinned**: `d_model=48, n_heads=3, n_kv_heads=3, n_layers=20,
  d_ff=192`. Arithmetic re-verified against `configs/llm_config.py:665-680`:
  - Baseline (d_model=64, n_heads=4, n_kv_heads=2, n_layers=12, d_ff=256,
    emb_rank=8, vocab=49152, ffn=squared_relu 2-matrix):
    per-block attn 4·(64·64 mean) for q+o + 2·(64·32) for k,v = 4096+4096+
    2048+2048 = 12.3k; FFN 2·64·256 = 32.8k; norms ≈0.3k → 45.4k ×12 = 545k;
    emb 49152·8 + 8·64 = 393.7k; total ≈ 938.6k ≈ **0.94M** ✓.
  - Trt B1 (d_model=48, n_heads=n_kv_heads=3, MHA): per-block attn 4·(48·48)
    = 9.2k; FFN 2·48·192 = 18.4k; norms ≈0.24k → 27.9k ×20 = 558k; emb
    393.6k; total ≈ **951.6k (+1.3%)** — within the ±5% ceiling. ✓
  - Convention preserved: `d_ff = 4·d_model` (192 = 4·48); `d_head = 16`
    (48/3). ✓
- **emb_rank=8, ffn_variant=squared_relu, vocab_size=49152 preserved**
  explicitly in the tuple table. ✓
- **ctrl pinned = `Tiny1M3MConfig` (val 6.4306)** — clean isolation of the
  depth/width lever, no FIRE/SWA/RoPE stack confound; the alternative
  (stacking on the vq-gain+rope250k+swa384 recipe) is correctly noted as
  conflating two axes and deferred. In-session ctrl run is the comparison
  point, not the leaderboard number — correct anti-drift framing. ✓
- **Pass bar tiled at ±0.01**: WIN `< ctrl−0.01`, NULL `|Δ|≤0.01` inclusive,
  FAIL `> ctrl+0.01`. Tiles the real line without overlap. Noise floor cited
  against `closed.md:31-52` (ctrl spread 6.3875–6.4050 = 0.018). ✓
- **Seed 42 only**, explicit "no multi-seed sweep, no per-seed mean,
  sub-noise = inconclusive" — matches the prompt's hard rule. ✓
- **Frozen (non-arch) fields**: max_seq_len=2048, batch_size=2,
  train_tokens=3_000_000, compile_model=False, warmup_ratio=0.02,
  schedule_type, eval_milestones tuple, all Muon settings — stated
  explicitly so the code-implementer can't "rescue" with an LR bump. ✓
- **Known confound disclosed honestly** (`idea.md:50-55`): B1 collapses
  baseline GQA 2:1 → MHA (n_heads=n_kv_heads=3). The idea cites
  `LEADERBOARD.md` row 0 (vq-gain+rope250k+swa384+tiedqk @ 6.3041) as
  evidence MHA-tied is itself a known WIN signature, so the trt Δ
  partially reflects the kv-sharing axis change. The B1' (MQA) and B2
  (d_model=32 + d_ff off-convention) alternatives are spelled out and
  the d_ff=4·d_model invariant correctly prioritized over the GQA
  invariant. Calling for the runner to report the confound alongside the
  raw Δ is the right disposition; the cleaner 2-AB (B1 vs n_kv_heads=2
  depth-matched control) is correctly deferred as out-of-budget. ✓
- **LoC ~28 ≤ 30** with (a)/(b)/(c) breakdown, config-only diff surface.
  Coordination note that no `models/layers.py` / `models/llm.py` edit is
  needed → parallel-AI memo (memory `project-parallel-ai`) is a non-issue
  here. ✓
- **Source unchanged & still real**: MobileLLM (Ma et al., ICML 2024,
  arXiv:2402.14905). Citation matches the `transfer-risk: low` tag — gains
  demonstrated at sub-billion target class (125M/350M), 133–373× our
  budget, which is exactly what the open question on 0.94M tests. ✓
- **Not in `closed.md`**: no depth/width / MobileLLM entry; no n_kv_heads
  collapse closure that subsumes this. ✓

**Hand-off to code-implementer.** Spec is buildable as written:
`Tiny1M3MDeepThinConfig(Tiny1M3MConfig)` overriding the 5 fields, plus a
≤0.99M param-count assert. No model code change. Round resets to 1 for
the code gate's own 3-round budget.

---

## r1 — 2026-06-10 — verdict: revise

**Sound structural lever, citation real, but the spec is one-sentence
mechanism + handwave — pin the concrete (n_layers, d_model, n_kv_heads,
d_ff) tuple and a numeric pass bar before the code gate.**

**5-check sweep**
- **Source real:** MobileLLM (Ma et al., ICML 2024, arXiv:2402.14905) —
  resolves; depth/width swap at fixed param count is the paper's headline
  ablation; +2.7%/+4.3% at 125M/350M zero-shot. `plans/litrev-sub200m.md:98-101`
  lists this explicitly as the top sub-400M structural lever. ✓
- **Mechanism is structural, not HP:** config reallocation across
  (n_layers, d_model, d_ff, n_heads) at fixed param budget — architectural,
  not an LR/init/schedule lever. ✓
- **Not already closed:** no depth/width-swap entry in `closed.md`; the
  closed "Tied QK (on best baseline)" axis is kv-head sharing, not depth/width.
  Distinct from active 020-025 / 026-027 / 029-030 (all attention-side or
  norm tweaks). ✓
- **tiny1m3m + seed 42 only:** spec lives at tiny1m3m; no multi-tier or
  multi-seed language. ✓
- **transfer-risk: low + Scale-evidence section present and consistent
  with the source:** MobileLLM ablations at 125M/350M (≥133× our budget),
  same model class (sub-billion, from-scratch). The "low" tag is the
  *citation*'s claim; whether transfer down to 0.94M actually fires is the
  open question the A/B answers, which is the right framing. ✓

**Findings — must be addressed before `needs-plan` (all are spec-completeness,
no mechanism rewrite required):**

- **No `## Definition (gate 2)` block — blocks the code gate.** Mirror
  023-canon-conv/idea.md (see review.md r1): add `### Ctrl vs trt`,
  `### Pass bar`, `### Seed`, `### LoC budget`, `### Param-budget arithmetic`.
  Without it, the code-implementer has to guess every number below.

- **Pin the concrete (n_layers, d_model, n_heads, n_kv_heads, d_ff) tuple
  — currently a range, not a spec.** Idea says "n_layers=24, d_model=32-48,
  d_head=16 preserved, n_heads scales with d_model." The arithmetic doesn't
  close at the 0.94M budget for either endpoint:
  - Baseline `Tiny1M3MConfig`: d_model=64, n_heads=4, n_kv_heads=2,
    n_layers=12, d_ff=256, emb_rank=8, ffn=`squared_relu` (2-matrix),
    vocab≈49152. Per-block ≈ 12.3k (attn) + 32.8k (FFN) + 0.3k (norms) ≈
    45.4k; ×12 = 545k. Embedding factorisation ≈ 49152·8 + 8·64 = 393.7k.
    Total ≈ 939k ≈ 0.94M ✓.
  - Endpoint A (d_model=32, n_heads=2, n_kv_heads=2, n_layers=24,
    d_ff=4·d_model=128): per-block ≈ 4.1k (attn) + 8.2k (FFN) + 0.16k ≈
    12.5k; ×24 = 300k; +emb 393k = **693k ≈ 0.69M** — **26% UNDER budget**.
  - Endpoint B (d_model=48, n_layers=24, d_head=16 → n_heads=3, d_ff=192):
    n_kv_heads must divide n_heads=3 → only valid choices are 1 or 3. n_kv_heads=3
    gives per-block ≈ 9.2k (attn) + 18.4k (FFN) + 0.24k ≈ 27.9k; ×24 = 669k;
    +emb 393k = **1062k ≈ 1.06M** — **13% OVER budget**.
  - **Fix:** pin exactly one tuple and show the arithmetic lands within ±5% of
    the baseline 939k. Two clean candidates that close the budget:
    - **B1 (recommended):** d_model=48, n_heads=3, n_kv_heads=3 (MHA),
      n_layers=20, d_ff=192, emb_rank=8 → per-block 27.9k ×20 = 558k +
      emb 393.7k ≈ **951k (+1.3%)**. Depth swap 12→20 (1.67×), width 64→48
      (0.75×). Largest clean swap that closes the budget.
    - **B2:** d_model=32, n_heads=2, n_kv_heads=2, n_layers=24, emb_rank=8,
      with d_ff bumped to **~272** (not 4·d_model) to absorb the freed
      params: per-block (attn 4.1k + FFN 2·32·272 = 17.4k + 0.16k) ≈ 21.7k
      ×24 = 521k + emb 393.5k ≈ **915k (−2.6%)**. Wider depth swap (2×) but
      breaks the "d_ff = 4·d_model" convention from baseline (the swap
      isn't pure depth/width then; FFN expansion ratio also changes from
      4× to ~8.5×).
  - Pick B1 unless there's a reason to prefer the 2× depth swap; commit
    `(n_layers, d_model, n_heads, n_kv_heads, d_ff)` in the spec and show
    the per-block × n_layers + emb arithmetic.

- **GQA / n_kv_heads constraint silently dropped.** Baseline is GQA 2:1
  (n_heads=4, n_kv_heads=2). The idea's "d_head=16, n_heads scales with
  d_model" doesn't specify n_kv_heads. At d_model=32, d_head=16 → n_heads=2,
  so n_kv_heads ∈ {1, 2} — either MHA-tied (kv-sharing axis collapses) or
  MQA (8:1 → 2:1). At d_model=48 → n_heads=3, n_kv_heads ∈ {1, 3} (must
  divide n_heads). Pin n_kv_heads explicitly in the new config; **note in
  the spec** which kv-head pattern the swap collapses to, because depth/width
  + tied QK is a known WIN signature at tiny1m3m (LEADERBOARD.md tiny1m
  arch row 0, val 6.3041) — the A/B then partly measures the tied-QK lever,
  not pure depth/width. With B1 (n_heads=3, n_kv_heads=3) the swap is also
  MHA-tied; call this out explicitly as a known confound, or pick B1' =
  n_heads=3, n_kv_heads=1 (MQA) to preserve the kv-sharing ratio.

- **emb_rank=8 must be preserved — taste flagged this and spec must say so
  in one line.** Baseline emb_rank=8; the new config dataclass must
  inherit/state `emb_rank: int = 8` (unchanged). Otherwise the 393.7k
  embedding factorisation shifts and the A/B is contaminated by an
  embedding-side change. State explicitly that `vocab_size` is also
  unchanged (49152).

- **Pin numerical pass bar — currently absent.** Spec has no WIN/NULL/FAIL
  band. Use the now-standard tiling (mirror 023-canon-conv after r2 fix):
  - **WIN:** `trt_val < ctrl_val − 0.01` (strict; clears the ±0.01 box-noise
    floor cited in LEADERBOARD.md tiny1m3m section)
  - **NULL:** `|trt_val − ctrl_val| ≤ 0.01` (inclusive)
  - **FAIL:** `trt_val > ctrl_val + 0.01` (strict)
  These tile the real line without overlap. Cite `LEADERBOARD.md` tiny1m3m
  for the noise floor and `closed.md:31-52` for the gap-band convention.

- **Pin ctrl vs trt explicitly.** Two reasonable baselines; pick one and
  declare it. Recommendation: **ctrl = `Tiny1M3MConfig`** (plain baseline,
  val 6.4306 on LEADERBOARD.md row 14) — clean isolation of the depth/width
  lever, no FIRE/SWA/RoPE-tuned stack confound. The alternative (stacking
  on `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` mirroring
  020-025) tests compounding with the best attention stack but conflates
  two axes — for a structural config swap it is cleaner to A/B against
  the plain baseline first. Trt = new `Tiny1M3MDeepThinConfig` (or
  similar) with the tuple from the finding above.

- **Optimizer / LR / schedule / batch / tokens / seq-len must be frozen
  in the new config — declare it.** Add one line: "All non-architectural
  fields inherited unchanged from `Tiny1M3MConfig`: `max_seq_len=2048`,
  `batch_size=2`, `train_tokens=3_000_000`, `compile_model=False`,
  `warmup_ratio=0.02`, `schedule_type='warmup_decay_to_zero'`,
  `eval_milestones=…` (the baseline tuple), all optimizer/Muon settings."
  Otherwise the swap could be (mis)read as a joint depth/width × LR-tuning
  experiment. The dataclass inheriting `Tiny1M3MConfig` already gives this
  by default — but the spec should state it explicitly so the
  code-implementer doesn't bump muon_lr or batch_size.

- **FFN variant lock-in.** Baseline uses `ffn_variant="squared_relu"`
  (`configs/llm_config.py:582`; 2-matrix, `models/layers.py:5`,
  `SquaredReLUFeedForward`). Pin in the new config: `ffn_variant` unchanged
  ("squared_relu"). Otherwise switching to swiglu (3-matrix) shifts the
  FFN parameter accounting and breaks the budget math.

- **LoC budget — confirm the 30-LoC estimate.** Mirror 023's (a)/(b)
  breakdown:
  - (a) new `Tiny1M3MDeepThinConfig(Tiny1M3MConfig)` dataclass overriding
    `d_model, n_heads, n_kv_heads, n_layers, d_ff` (5 fields) ~10 LoC.
  - (b) docstring with the pass bar + Δ-vs-LEADERBOARD-row-14 plan ~10 LoC.
  - (c) param-count sanity assert (or one-shot test) that `MinimalLLM(cfg)`
    has ≤ 0.99M params at init ~8 LoC.
  - Total ~28 LoC ≤ 30 cap. No model/forward-pass code change. ✓

- **Coordination — no `models/layers.py` or other shared file edit needed
  for this idea.** The pipeline's parallel-AI coordination memo
  (`MEMORY.md`: parallel AI) is a non-issue here because the code-implementer
  only touches `configs/llm_config.py` (adds a new dataclass; doesn't edit
  any existing config). State this in the spec under LoC budget so the
  code gate knows the diff surface is config-only.

**Hand-off to reviser.** All findings are pure spec-completeness — pick the
B1 tuple, write the Definition block, add the pass bar and ctrl/trt pin,
declare emb_rank/ffn_variant/optimizer freeze. ~40 min of editing, no
mechanism change. Round will bump to 2 on the reviser's flip; round-3 cap
still leaves one reviewer pass after the next revise.
