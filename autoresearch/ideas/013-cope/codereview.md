## r1 — 2026-06-09 — verdict: accept

- **Mechanism is faithful to idea.md and the paper.** `models/cope.py` implements `score = einsum("btd,hd->bth", x, probe) → sigmoid(score-τ) → cumsum → diff via the prepend-zero trick` exactly as the plan specifies. The cumsum-to-offset tensor math is right: `cum_g_pad = cat([0, cum_g])` of length T+1, then `right = cum_g_pad[:, 1:, :]` (cum_g[i]) and `left = cum_g_pad[:, :-1, :]` (cum_g[j-1] with j=0 falling out to 0), and the final `.permute(0,3,1,2)` gives `[B, H, T, T]` with `offset[b,h,i,j] = cum_g[b,i,h] - cum_g[b,j-1,h]`. The `g ≈ 0.5 ± 0.04` at init (probe N(0,0.02) and d_model=64) → `offset ≈ (i-j+1)·0.5` RoPE-like is in the docstring and matches the plan's identity analysis. No cousin or shortcut.

- **Identity-safety at OFF is bit-identical.** When `use_cope=False`:
  - `self.rotary = Rotary(...)` is built (baseline), `self.cope = None` (no allocation in the OFF path).
  - The `if self.use_nope or self.use_cope` guard at `models/layers.py:1249` evaluates False, so the rotary call runs unchanged.
  - The `if self.use_cope:` adds at lines 1388 and 1485 are False, no bias.
  - The `or self.use_cope` clause in the manual-routing condition at line 1406 is False, so the SDPA fast path is still selectable when no other flag forces manual.
  - The new `assert not (self.use_cope and self.use_qk_norm_post_rope)` at line 1157 is a no-op in the OFF path.
  Baseline path is bit-identical. ✓

- **Treatment path actually exercises the new code, not a dead branch.** When `use_cope=True`: `self.cope` is built in `MultiHeadAttention.__init__` (line 581) and called in both attention branches — the FIRE branch (line 1388, additive on top of `fire_bias`) and the manual tweaks branch (line 1485, after all Q3/Q4/Q1/Q10/Q9/Q27 tweaks, before the mask, exactly as the plan promised). The standalone-CoPE case (no FIRE, no tweaks) is forced to the manual path by the `or self.use_cope` clause and then gets the CoPE add at line 1485.

- **Single boolean flag, default OFF.** `LLMConfig.use_cope: bool = False` at `configs/llm_config.py:164` with a docstring pinning probe init and τ. The preset `Tiny1M3MCoPEOnFireConfig` at line 675 sets `use_fire_pe=True; use_cope=True` — the stacked treatment per the plan. No CLI flag, no second config knob, no auto-enable from other flags.

- **No silent HP drift.** Diff does not touch LR, schedule, init (other than the new probe, which is pinned to N(0, 0.02) mirroring `models/fire_pe.py:60`), or seed. The `llm_config.py` add is one field + one preset; no LR/schedule/default tweaks slipped in alongside.

- **One seed, seed 42.** Plan's "Run" section shows seed 42 only; no sweep language; τ=0 pinned with explicit "no τ sweep" comment. Pipeline rule satisfied.

- **LoC budget.** `models/cope.py` is 88 LoC; the wiring additions in `models/layers.py` and `models/llm.py` are in the dozens. Well under the 200 LoC cap. The plan's "~50 LoC" estimate was for the wiring (the integration touches), not the new module — both numbers are honest.

- **Plan ↔ idea consistency.** PASS ≤ −0.01, NULL band |Δ| < 0.01, DRIFT > +0.01 all match between idea.md and plan.md. Ctrl anchor is the FIRE-equipped baseline (6.3234, per `closed.md` 009 WIN), not the historical V+q+SWA+HighRoPE reference. Tier is tiny1m3m, seed 42.

- **RoPE call-site audit is fully covered.** All 7 sites from the audit are gated:
  - `models/layers.py:12` and `:20` — the imported `RotaryPositionalEmbeddings` is still imported but is no longer applied when CoPE is on (the local `Rotary` class is what's gated in the diff, and the import is for the `from torchtune.modules import RotaryPositionalEmbeddings` line that's still used by the local `Rotary` wrapper). The plan's audit referred to a `Rotary` construction that needed gating — `models/layers.py:548` (now line 578) — and that one is correctly gated by `if self.use_cope`.
  - `models/layers.py:548` — `self.rotary = Rotary(...)` replaced with `self.rotary = None` when use_cope=True.
  - `models/llm.py:207` — `use_qk_norm_post_rope` is now defensively asserted against use_cope.
  - `models/llm.py:217`, `:322,346,347` — `rope_base` etc. are unchanged but harmless: when use_cope=True the rotary call is bypassed, so these are inert (no double-application, no extra param).
  - `models/llm.py:210-213` — the NoPE-style bypass is mirrored by the use_cope bypass.

- **Coordination.** Diff in `models/layers.py` is 119 lines, but the 013-only lines are isolated: the import (line 7), the `use_cope` kwarg in MHA (line 436), the rotary gate (lines 578-584), the assert (lines 1157-1158), the rotary-skip guard (line 1249), the FIRE-branch CoPE add (lines 1388-1389), the manual-routing clause (line 1406), the tweaks-branch CoPE add (lines 1485-1486), the `use_cope` kwarg in `TransformerBlock` (lines 1757-1760), the pass-through (line 1797). The other diffs in those files (QK-Norm #16, Sub-LN #17, Cautious-Lion #11) are on distinct lines and don't stomp. No rebase, no push, working-tree only.

- **Mechanism is real, not HP tuning.** Content-conditional position (probe → count of "important" tokens per head) is a structural lever — the bias lives in attention-logit space, the probe is a learned per-head content projection, and τ=0 is the only knob. Not a learning-rate shift.

- **Smoke test done by the implementer.** Plan's "Pre-flight" section confirms `MinimalLLM(cfg)` builds for ctrl + treatment + CoPE+FIRE stacked; forward is finite; max diff OFF vs ON-standalone is 0.072 (CoPE takes effect from step 0, consistent with the init analysis). Cheap CPU verification, no GPU burn.

The code is ready to run. Handing to the runner as `needs-run`.
