# Review log — 023 canon-conv

## r3 — 2026-06-10 — verdict: approve
- **r2 finding landed; bands tile.** Pass bar at `idea.md:71-82` is now
  `WIN: trt < ctrl − 0.01` (strict), `NULL: |trt − ctrl| ≤ 0.01` (inclusive),
  `FAIL: trt > ctrl + 0.01` (strict). At the boundaries Δ = −0.01 → NULL,
  Δ = +0.01 → NULL, so WIN/NULL and NULL/FAIL do not overlap; the real line
  is fully tiled with no result satisfying two bands. The cited `closed.md`
  ctrl spread of 0.018 is the documented noise floor and the −0.01 WIN bar
  clears it. (Tighter than 020-forgetting-attn's −0.02, which is fine —
  020 had a multiplicative gate and 023's lever is cheaper.)
- **Source still real and current.** Griffin arXiv:2402.19427 (De/Smith/
  Fernando, 2024-02-29) is the canonical local-DWConv mixer, and Allen-Zhu
  "Physics of LMs" Canon is the named ancestor. Mechanism = structural
  residual-stream mixer, not a hyperparameter; zero-init scalar gate gives
  exact step-0 identity, so a clean null is informative ("local mixing is
  redundant with RoPE+attention at 6L") and a win is a transferable lever.
- **Not in closed list.** Distinct from `closed.md:21` SWA / dilated-attn
  (window reshape) and `closed.md:25-26` NSA / diff-attn / hybrid heads
  (inside-attention). This is an *orthogonal* residual-stream mixer.
  Active queue neighbors 020/021/022 are all attention-side or
  normalization tweaks; 023 is the local-mixing half of a hybrid, in a
  separate lever family.
- **Spec is implementable without guessing.** Source ✓, ctrl/trt pinned
  to `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` (FIRE-equipped
  009-WIN baseline so the A/B partitions the orthogonal-axis question),
  placement pinned to 1 conv per block pre-attn pre-LN, kernel=3, scalar
  gate init 0, DWConv shape pinned, left-pad-2 causality with explicit
  warning against `padding=2`, LoC budget ~46 ≤ 50 cap with (a)-(e)
  breakdown, seed 42 only, single tier tiny1m3m.
- **Ready for the code gate.** Code-implementer should: (1) add
  `CanonConv` module (DWConv + scalar gate), (2) wire the optional branch
  in `TransformerBlock.forward` (left-pad, conv, gate-mul, residual-add)
  on `use_canon_conv=True`, (3) add `use_canon_conv: bool = False` to
  `LLMConfig` and `TransformerBlock`, (4) add
  `Tiny1M3MCanonOnFireConfig(Tiny1M3MConfig)` with both flags on, (5)
  causality test (`+1` at `t` only affects `≥ t`), (6) step-0 identity
  test (`use_canon_conv=True, g=0` ≡ baseline within 1e-6). Approve.

## r2 — 2026-06-10 — verdict: revise

- **r1 sweep landed.** 9/9 findings applied substantively: source tightened
  to Griffin arXiv:2402.19427; Definition (gate 2) block added with all
  required sub-sections; ctrl pinned to `Tiny1M3MVQGainSWAHighRoPE250KConfig
  + use_fire_pe=True` (verified at `configs/llm_config.py:773`); placement
  pinned to one conv per block on the residual stream immediately before the
  attention sublayer's pre-LN; kernel = 3; scalar gate `nn.Parameter(
  torch.zeros(1))` init 0; DWConv shape `groups=d_model, bias=False`; causal
  left-pad via `F.pad(x.transpose(1,2), (2,0))` with explicit warning against
  `padding=2`; pre-LN read; LoC budget ≈46 LoC ≤ 50 cap with (a)-(e)
  breakdown. Spec is now implementable without a code-implementer guess on
  any of the previously-open axes.
- **Pass bar is internally inconsistent — fix before code gate.** The spec
  cites box noise `≈ ±0.01` and `closed.md` ctrl spread `0.018` as the floor,
  then sets `WIN: trt < ctrl − 0.005`. A `−0.005` win bar sits **inside** the
  cited noise band and **overlaps the NULL band** `|Δ| < 0.01` in the range
  `Δ ∈ (−0.01, −0.005)`: a trt at `ctrl − 0.008` satisfies BOTH WIN and NULL,
  which is malformed. Compare 020-forgetting-attn (`idea.md:92-96`), which
  uses non-overlapping bars: `WIN < ctrl − 0.02`, `NULL |Δ| < 0.02`,
  `FAIL > ctrl + 0.01`. Concrete fix — replace the three lines in
  `idea.md:71-77` with:
  - **WIN**: `trt_val < ctrl_val − 0.01` (clears the cited noise floor)
  - **NULL**: `|trt_val − ctrl_val| ≤ 0.01` (inclusive, no gap with WIN)
  - **FAIL**: `trt_val > ctrl_val + 0.01` (unchanged)
  These three bands tile the real line without overlap. No other change to
  the spec needed; everything else is ready for the code gate.
- **Reminder — round 3 is the cap.** On the next reviser pass the round
  counter bumps to 3; if anything still blocks, the next reviewer pass is
  forced to `approve` or `reject` (`PIPELINE.md:222`). This single-finding
  fix is the entire r3 task — keep it surgical.

## r1 — 2026-06-10 — verdict: revise

- **Source verified.** Griffin arXiv:2402.19427 resolves: "Griffin: Mixing Gated
  Linear Recurrences with Local Attention for Efficient Language Models",
  De/Smith/Fernando, 29 Feb 2024. Causal depthwise Conv1d as local mixer is
  Griffin's standard pre-mixing block. Allen-Zhu "Physics of LMs" Canon line is
  the named ancestor but Griffin alone validates the mechanism. Not a
  fabrication. Distinct from `closed.md` SWA / dilated-attn (window reshape)
  and NSA/diff-attn (inside-attention) — this is an orthogonal residual-stream
  mixer.
- **Definition section missing — blocks the code gate.** No
  `## Definition (gate 2)` block; see 020-forgetting-attn/idea.md:76-115 for
  the required shape. Add ### Ctrl vs trt, ### Pass bar, ### Seed, ### LoC
  budget, ### Placement (called out below), ### Padding & causality (called
  out below).
- **Pin ctrl/trt to the FIRE-equipped baseline** (mirrors 020/021/022). Ctrl =
  `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (`configs/llm_config.py:773`, FIRE = 009 WIN per `closed.md:43`). Trt =
  ctrl + `use_canon_conv=True`. Stacking on FIRE tests whether a separate
  local conv mixer adds anything *on top of* the best attention-side win.
- **Pin numerical pass bar.** "Expect a drop" is not falsifiable. Use the
  shared ±0.01 box-noise band: **WIN** `trt < ctrl − 0.005`, **NULL**
  `|trt − ctrl| < 0.01`, **FAIL** `trt > ctrl + 0.01`. A FAIL here means the
  conv is actively interfering with attention (not just neutral).
- **Pin placement to ONE site per block (kills a hidden A/B axis).** Taste r1
  flagged this explicitly. Idea says "just before each attention/FFN sublayer"
  — that's 2 convs per block, plus a placement question
  (residual-stream vs sublayer-input). Pin to: **one conv per block, on the
  residual stream, immediately before the attention sublayer's pre-LN** (i.e.
  `x = x + g · DWConv(x); h = attn(LN(x)); x = x + h; …`). Not before FFN, not
  both. This matches Griffin's canonical Conv1d block placement and isolates
  the lever to a single variable.
- **Pin kernel size to a single value.** Idea says "kernel 3-4" — taste r1
  flagged this. Use **kernel = 3** (smaller is more conservative, fewer
  params; the win/null gap between 3 and 4 is well inside box noise at this
  scale, so the sweep is unjustified). No `kernel ∈ {3,4}` sweep.
- **Pin gate shape.** "Per-layer output gate init 0" — spec must say: single
  scalar `nn.Parameter(torch.zeros(1))` per CanonConv module (one per block),
  **not** per-channel and **not** per-token. Step-0 ≡ baseline because `g·… = 0`
  exactly. The model has `n_layers` such gates total.
- **Pin DWConv shape and causality.** Spec must say: `nn.Conv1d(d_model,
  d_model, kernel_size=3, padding=0, groups=d_model, bias=False)` (depthwise =
  one filter per channel; bias off because we have the gate). Causality is
  enforced by **left-padding** the input with `kernel_size − 1 = 2` zeros
  along the time axis *before* the conv (not by `padding=2` which pads both
  sides). Add an assertion test that a `+1` perturbation at position `t`
  changes only positions `≥ t` in the output.
- **State the LayerNorm interaction.** Will the conv read from the pre-LN
  residual stream (current spec) or from a separately-LN'd version? Pin:
  **read pre-LN** (cheaper, matches Griffin). One sentence in spec to avoid
  the code-implementer guessing.
- **LoC budget breakdown.** Mirror 020's (a)/(b)/(c)/(d)/(e): (a) `CanonConv`
  module class (DWConv + scalar gate, init zeros) ~12 LoC; (b) integration in
  `TransformerBlock.forward` (one branch, left-pad, gate-mul, residual-add)
  ~8 LoC; (c) flag wiring `use_canon_conv: bool = False` in `LLMConfig` +
  `TransformerBlock` + new config class `Tiny1M3MCanonOnFireConfig` ~10 LoC;
  (d) causality test (`+1` perturbation at `t` affects only `≥ t`) ~8 LoC; (e)
  step-0 identity test (`use_canon_conv=True, gate=0` ≡ baseline within 1e-6)
  ~8 LoC. Total ~46 LoC ≤ 50 cap.
