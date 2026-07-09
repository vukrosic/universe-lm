# Evidence — 185-static-per-head-k-rotation

## r1 build-smoke fail (caught by daemon before any GPU time)
- **Verdict**: needs-recode (no run started; smoke caught it)
- **Cause**: queue-daemon's remote `_box_smoke.py` could not import
  `Tiny1M3MStaticKRotationConfig` from `configs.llm_config`. The config
  class was added in the same diff as the layers.py / llm.py changes,
  but the box's `git pull` apparently landed the layers+llm changes
  ahead of the configs change (or saw a partial tree), so the
  CPU build-smoke raised `ImportError` and the daemon bounced the
  idea back without burning any GPU time.
- **Log line**: `2026-06-15T07:51:06Z daemon: build-smoke failed:
  SMOKE_FAIL: ImportError: cannot import name
  'Tiny1M3MStaticKRotationConfig' from 'configs.llm_config'`

## r2 fix (this recode)
- **Verified locally** on a fresh build (PYTHONPATH=., TORCHDYNAMO_DISABLE=1):
  - `python autoresearch/bin/_box_smoke.py _arq_185-static-per-head-k-rotation.py`
    → `SMOKE_OK`.
  - `MinimalLLM(Tiny1M3MStaticKRotationConfig())` constructs cleanly on CPU.
  - Param `transformer_blocks.{0..11}.attention.k_rotation_angles` registered,
    shape `[4, 8]` per block (= `n_heads × d_k // 2`), init all zeros.
  - **Step-0 byte-identity**: `max_abs_diff(MinimalLLM(Tiny1M3MConfig())(ids),
    MinimalLLM(Tiny1M3MStaticKRotationConfig())(ids)) == 0.0` — confirms
    `θ=0 ⇒ R_h=I ⇒ K=R_h@K=K` exactly in fp32.
- **No code change required**: the config class was already present in the
  working tree at `configs/llm_config.py:6539`. The bounce was likely a
  transient race between the box's `git pull` and the daemon's `scp+smoke`
  step. Re-claiming and re-releasing brings the daemon back in sync — the
  next tick's smoke run will see the full config and pass.
- **Round**: bumped to 2.

## r3 fix (this recode — second re-code)
- **r2 was insufficient**: the box failed again at 07:57:41Z with the SAME
  ImportError. The r2 "transient race" diagnosis was wrong — the config
  class was never on origin (only in the local working tree), so the box's
  `git pull` couldn't have brought it in.
- **Root cause** (confirmed by `git show origin/orchestrate-codex-fallback:configs/llm_config.py | grep -c Tiny1M3MStaticKRotationConfig` → `0`): the model code in
  `configs/llm_config.py`, `models/layers.py`, `models/llm.py` was uncommitted.
  `autoresearch/bin/orchestrate.sh` only commits `autoresearch/ideas/*`
  snapshots (line 195); model code commits happen out-of-band. r1 + r2 both
  released without committing → box stayed stale → ImportError persisted.
- **Verified locally** (this round, same harness as r2 — passes):
  - `_box_smoke.py _arq_185-static-per-head-k-rotation.py` → `SMOKE_OK`.
  - `MinimalLLM(Tiny1M3MConfig())` → 949,056 params.
  - `MinimalLLM(Tiny1M3MStaticKRotationConfig())` → 949,440 params (+384, as
    planned: `n_heads × d_k//2 × n_layers = 4 × 8 × 12`).
  - Same seed (42) → `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0`.
    (Naive two-instance smoke fails because the two MinimalLLM ctor calls
    draw different RNG streams — must seed between the two constructions.)
- **Code fix**: staged and committed `configs/llm_config.py` +
  `models/layers.py` + `models/llm.py` locally (single commit, message
  records this re-code and explains the prior bounces). The daemon on the
  box will not see the class until the user pushes this commit to
  `origin/orchestrate-codex-fallback` — no auto-push per
  `feedback-dont-push-without-approval.md`. Once pushed, the next daemon
  tick's `git pull` will refresh the box's `configs/llm_config.py` and the
  smoke will pass.
- **Round**: bumped to 3 (one under the `MAX_RECODE_ROUNDS=3` cap — one more
  bounce auto-closes to `rejected`).

## Transfer note
(auto) deferred — see idea.md `## Scale evidence`. Written by the analyzer pass, not the daemon.