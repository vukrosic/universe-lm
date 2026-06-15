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

## Transfer note
(auto) deferred — see idea.md `## Scale evidence`. Written by the analyzer pass, not the daemon.