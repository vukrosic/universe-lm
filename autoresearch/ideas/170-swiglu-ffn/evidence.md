# Evidence — 170 swiglu-ffn

## Verdict: still running (MEASURE-pass partial — ctrl landed, ctrl2/ctrl3/170 in flight)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_8.6, driver 580.159.03)
- baseline: MEASURE path (commit changed 2368d6c → 91b9864) — 3 ctrls queued to refresh `autoresearch/baseline-cache.json` before judging 170
  - ctrl landed: val=6.4188, train=6.3953, val_acc=0.1437 (ts 2026-06-14T10:07:47Z)
  - ctrl2, ctrl3, 170-swiglu-ffn still running in tmux arq (detached)
- treatment val: pending
- pass/fail bar (from plan.md): PASS = trt ≤ 6.4344 (Δ ≤ −0.005 vs cached mean); NULL = |Δ| < 0.01; DRIFT = Δ > +0.01
- box check: ctrl 6.4188 sits inside cached `6.4394 ± 0.04` band ⇒ baseline in-range, no DRIFT signal from this single ctrl
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (ctrl entry appended; logs alongside)
- date: 2026-06-14

## Code-sync note
- Local had dirty uncommitted plumbing for 170 (`use_swiglu_ffn` field on `LLMConfig`, `Tiny1M3MSwigluFFNConfig` subclass, `SwiGLUZeroInitFeedForward` in `models/components.py`, threading through `TransformerBlock.__init__`, `models/llm.py`). Box had parallel dirty work from idea 166 (T5-RPE) and others stashed.
- Action: `git stash push -u` on box (preserved box-side work in `stash@{0}` for later review), generated `/tmp/arq-170.patch` from local diff, applied via `git apply --3way` to box. Resolved a conflict in `models/layers.py` with `--theirs` (preserved box's existing t5_rpe plumbing). Then applied box's stash `components.py` hunk to add `ReLU2FeedForward` (153) which my patch did not carry.
- Smoke check on box: `MinimalLLM(Tiny1M3MConfig())` → CTRL_OK; `python autoresearch/bin/_box_smoke.py _arq_170-swiglu-ffn.py` → SMOKE_OK.
- 171 bounced to `needs-recode`: no `_arq_171-dropconnect-wo.py` stub, no `Tiny1M3MDropConnectWOConfig` subclass in `configs/llm_config.py`, no `plan.md` (was routed needs-run by `implement-button` after taste r2 accept, bypassing the code-implementer gate). flip.sh logged the bounce at round=2.

## Transfer note (preview)
SwiGLU is the standard FFN choice in LLaMA 1/2/3, Mistral, Qwen, Gemma, OLMo, Falcon — direct validation at 7B-540B with the 2/3-trick param parity. Shazeer 2020 (arXiv:2002.05202) original paper validates at T5 1.1B/1.6B/3B. Transfer risk: low (≥100M direct). The d_model=64 / 12L / 4H 0.94M tier is structurally small — gating has 32,640 FFN params vs 32,768 baseline (≈0.4% smaller), so the gate has ~64×64 = 4096 maskable weights per block to learn from. 153-relu2-ffn (activation-curvature swap, same 2-matrix FFN) already nulled at this tier (Δ=-0.0053, inside null band); 170 is structurally distinct (3-matrix gated FFN) so the 153 null does not pre-close 170.
