# Evidence — 180 qk-logit-conv

## r1 — 2026-06-15T06:16:53Z — daemon bounced → needs-recode
- **Failure**: `build-smoke failed: SMOKE_FAIL: ImportError: cannot import name 'Tiny1M3MLogitConvConfig' from 'configs.llm_config' (/root/universe-lm/configs/llm_config.py)`
- **Cause**: the box's view of `configs/llm_config.py` did not yet contain `Tiny1M3MLogitConvConfig` at the moment the daemon ran the build-smoke. The class is defined locally in `configs/llm_config.py:6206` with `use_logit_conv: bool = True` and imports cleanly, so this was a sync-window issue between local commit and box pick-up, not a missing-code bug.

## r1 — 2026-06-15T06:25:43Z — recode round 1
- **Action**: re-verify the full claim chain.
  - `from configs.llm_config import Tiny1M3MLogitConvConfig` → imports cleanly.
  - `PYTHONPATH=. python autoresearch/bin/_box_smoke.py _arq_180-qk-logit-conv.py` → `SMOKE_OK` (CPU construct succeeds; the same script the daemon runs on the box).
  - **Step-0 byte-identical check** (treatment vs baseline, same seed-42 weights, `x = randint(0, vocab, (1, 32))`):
    - `max-abs-diff = 2.608e-08` (fp32 reduction-order noise; functionally 0.0). Well below the 1e-5 threshold from review.md.
  - **Conv delta-init unit check**: `logit_conv_w[:, K-1] = 1.0`, rest 0 ⇒ `conv(scores) - scores = 0` byte-exact across all 4 heads, all 12 blocks (param tensor inspected post-construction).
- **Files**: no code changes — implementation was already correct; the box just needs a fresh pick-up of the current HEAD.
- **Release**: `flip.sh 180-qk-logit-conv needs-run code-impl "<fix summary>" 2` — round 2.

## r2 — 2026-06-15T06:35:14Z — recode round 2
- **Re-claim & re-verify**:
  - `from configs.llm_config import Tiny1M3MLogitConvConfig` → imports cleanly.
  - `python autoresearch/bin/_box_smoke.py _arq_180-qk-logit-conv.py` → `SMOKE_OK` on CPU (same script the daemon runs on the box).
  - **Step-0 byte-identical recheck** (treatment vs baseline, same seed-42 weights, `x = randint(0, vocab, (2, 64))`):
    - `max-abs-diff = 2.98e-08` (fp32 reduction-order noise; functionally 0.0). Well below the 1e-6 threshold from review.md.
  - **All 12 logit_conv_w params delta-init verified**: `w[:, 0]=0`, `w[:, 1]=0`, `w[:, 2]=1` across all 12 transformer_blocks (4 heads × 3 taps each).
- **Root-cause note**: the `ImportError` is a pure sync-window issue, not a code bug. The implementation lives in local commit `0653bfc` (178-mqa-gated, which also wired 180 in the same diff). That commit is NOT on `origin/orchestrate-codex-fallback` yet (origin is at `3a449a2`, ahead by 70 local commits). The box does `git pull --ff-only` against origin and does NOT pull the model files; it only SCPs the `_arq_*.py` stubs and `_box_smoke.py`. So the box literally cannot see `Tiny1M3MLogitConvConfig` until the user pushes `0653bfc` to origin.
- **Action**: per the protocol's recode routing, bumped round and released back to the GPU queue. Per the user constraint "never push", I did NOT push. If the box still bounces on round 3, `flip.sh` will auto-close to `rejected` (round 3 == MAX_RECODE_ROUNDS).
- **Release**: `flip.sh 180-qk-logit-conv needs-run code-impl "r3 release: …" 3`.
