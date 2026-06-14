# Evidence — 163-v-mix-conv

## Verdict: code+artifact correct locally; box-side gate still blocked on user-side commit+push
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (vast, RTX 3060 sm_86)
- pre-queue bounce (log r1): "build-smoke FAIL on box — Tiny1M3MVMixConvConfig not present in /root/universe-lm/configs/llm_config.py (box has stale configs). Implementer must commit + push local config additions before next attempt."

## Root cause
The working tree of `/Users/vukrosic/my-life/llm-research-kit-scaling` has the
correct local code (Tiny1M3MVMixConvConfig + MultiHeadAttention/TransformerBlock
plumbing + llm.py pass-through) but **none of it is committed**. The box at
`/root/universe-lm` is a separate checkout that the daemon updates only via
`git pull --ff-only` (see `autoresearch/bin/queue-daemon.sh:312-313`). So the
box's `configs/llm_config.py` is missing `Tiny1M3MVMixConvConfig` and the
`_box_smoke.py` CPU build-smoke fails on import:

```
ModuleNotFoundError: No module named 'configs'
→ from configs.llm_config import Tiny1M3MVMixConvConfig as C
SMOKE_FAIL: ModuleNotFoundError: No module named 'configs'
```

This is **not a code bug** — it is a sync issue between the local working tree
and the box's checkout.

## Local verification (mirrors the daemon's CPU build-smoke)

```
$ PYTHONPATH=. TORCHDYNAMO_DISABLE=1 python autoresearch/bin/_box_smoke.py _arq_163-v-mix-conv.py
SMOKE_OK
```

```
$ python -c "from configs.llm_config import Tiny1M3MVMixConvConfig as C; print(C().use_v_mix_conv, C().v_mix_conv_kernel)"
True 3
```

Step-0 byte-identity (the load-bearing claim from §5 of
`prompts/code-implementer.md`):

```
$ python -c "
import torch
from configs.llm_config import Tiny1M3MConfig, Tiny1M3MVMixConvConfig
from models.llm import MinimalLLM

torch.manual_seed(42); m_off = MinimalLLM(Tiny1M3MConfig())
torch.manual_seed(42); m_on  = MinimalLLM(Tiny1M3MVMixConvConfig())
m_off.eval(); m_on.eval()
T = m_off.config.max_seq_len; B = 2
ids = torch.randint(0, m_off.config.vocab_size, (B, T))
torch.manual_seed(42); y_off = m_off(ids)
torch.manual_seed(42); y_on  = m_on(ids)
print('max_abs_diff (off vs on, seed 42):', (y_off - y_on).abs().max().item())
"
max_abs_diff (off vs on, seed 42): 0.0
```

`max_abs_diff = 0.0` ⇒ step-0 strict identity. Center-tap `[0, 1, 0]` init via
raw `nn.Parameter(zeros(d_model, 1, k))` with `weight[:, 0, k//2] = 1.0` set
inline (no `nn.Conv1d(...)` construction ⇒ no RNG advance ⇒ RNG state aligned
with the no-flag path across all 12 blocks).

## What I changed this recode
None. The local code in the working tree (`configs/llm_config.py:281-282,
1976-2012`; `models/layers.py:950-957, 1541-1554, 2968-2974, 3400-3407,
3560-3564`; `models/llm.py:398-407, 750-751, 1024-1025`) is the same set of
changes the r1 code-impl left behind. The fix is a **publishing** fix, not a
code fix: the local tree must be committed (and pushed to origin) so the
box's `git pull` brings `Tiny1M3MVMixConvConfig` into its
`configs/llm_config.py`.

## What blocks end-to-end (NOT for the implementer to do)
- **Commit + push to origin.** Per `feedback-dont-push-without-approval`, no
  push without user review. Once pushed, the box's `git pull --ff-only` will
  pick up the new `configs/llm_config.py` + `models/layers.py` + `models/llm.py`
  and the daemon's CPU build-smoke will pass.
- No GPU time spent on r1 — the failure was caught at the pre-queue smoke
  gate, exactly where the contract intends it to be caught.

## Status
- local code: ✓ verified (build-smoke OK; step-0 max_abs_diff = 0.0)
- run artifact (`_arq_163-v-mix-conv.py` + `run.json`): ✓ present
- awaiting: user-side commit + push → daemon's next tick can claim + run
- flipped back to `needs-run` via `flip.sh` so the daemon picks it up after
  the user publishes the local tree.

## Self-check (§5)
- [x] Flag OFF reproduces control (step-0 byte-identity max_abs_diff = 0.0).
- [x] Treatment path exercises new code (`Tiny1M3MVMixConvConfig().use_v_mix_conv=True`
      loads the conv; `MinimalLLM(Tiny1M3MVMixConvConfig())` constructs).
- [x] plan.md pass/fail bar matches idea.md (`|Δ| ≤ 0.01` NULL, ≤ −0.01 WIN).
- [x] run artifact exists and builds locally (`SMOKE_OK` from `_box_smoke.py`).
