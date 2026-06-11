# Plan — 027 Moonlight Muon RMS × QK-Norm

Composition of two proven WINs (015 + 016) that operate at
non-overlapping code paths. Both flags already wired and shipped;
this gate is a single recipe addition.

## Flag

- `LLMConfig.use_moonlight_muon: bool = False` — already declared in
  `configs/llm_config.py:546` (default OFF, plumbed through
  `training/trainer.py:209-218` into the Muon optimizer's
  `scale_mode="moonlight"` branch with `c=0.2`).
- `LLMConfig.use_qk_layernorm: bool = False` — already declared in
  `configs/llm_config.py:559` (default OFF, plumbed through
  `models/llm.py:274,396` → `models/layers.py:554,675-680,1965,2119`
  → swaps Q/K RMSNorm for `nn.LayerNorm(d_head)`).
- No new flags. The composition flips **both** existing flags ON in
  a single new derived recipe.
- When both OFF: bit-identical to `Tiny1M3MConfig` baseline (already
  proven by 015's plan §5 + 016's plan §5 self-checks: Moonlight
  reads `getattr(config, "use_moonlight_muon", False)` and falls
  through to `shape_aspect`; QK-Norm reads
  `getattr(config, "use_qk_layernorm", False)` and falls through to
  `RMSNorm(d_head)`). No new code path; the OFF→OFF identity is
  guaranteed by the existing parent gating.

## Change

- `configs/llm_config.py`:
  - Add **one** new recipe `Tiny1M3MMoonlightMuonQKNormConfig`
    that inherits `Tiny1M3MConfig` and flips both flags on.
    Inserted after `Tiny1M3MVNormOnQKNormConfig` (line ~765)
    to avoid colliding with the unstaged 029/030 inserts already
    in `git diff`. ~14 LoC including docstring.
  - No edits to any existing line; pure additive insert.
- `optimizers/muon.py`, `training/trainer.py`, `models/llm.py`,
  `models/layers.py`: **no edits**. Both flag paths already exist
  and were validated by the parent A/Bs (closed.md:32-33).
- Step-0 invariance: Moonlight is identity on the first step
  because NS5(0)=0 (zero gradient → zero update regardless of
  scale formula). QK-Norm is identity on the first step because
  `nn.LayerNorm` inits γ=1, β=0. Independent identity proofs,
  no interaction. Off→OFF combined path is bit-identical to
  `Tiny1M3MConfig`; ON→ON path simply composes the two
  validated subpaths.

## Control

- **Control** (`ctrl`): `Tiny1M3MConfig` — plain Muon
  (`scale_mode="shape_aspect"`, `muon_lr=0.024`) + RMSNorm Q/K.
  Seed 42.
- **Treatment** (`trt`): `Tiny1M3MMoonlightMuonQKNormConfig` —
  `use_moonlight_muon=True` (Muon `scale_mode="moonlight"`, c=0.2)
  + `use_qk_layernorm=True` (LayerNorm γ=1 β=0 on Q,K per-head).
  Same `muon_lr=0.024`, same depth, same width, same data. Seed 42.
- **Recommended same-session bracket** (review finding on
  orthogonality disambiguation): include
  `Tiny1M3MMoonlightMuonConfig` (015 alone) and
  `Tiny1M3MQKNormConfig` (016 alone) in the **same A/B session**
  with two `Tiny1M3MConfig` ctrl bookends. This is a 5-run
  bracket on one box — directly measures whether the stack Δ is
  additive (independent levers) or subadditive (shared stability
  mechanism). The 5-run pattern matches the 015/016/017 bracket
  that originally produced the parent evidence. If the runner
  can only do one trt vs one ctrl this round, run just
  `Tiny1M3MConfig` vs `Tiny1M3MMoonlightMuonQKNormConfig` and
  use the 015/016 evidence.md numbers from
  `autoresearch/ideas/015-moonlight-muon-rms/evidence.md` and
  `autoresearch/ideas/016-qk-norm/evidence.md` as the parents'
  reference.
- **Tier**: tiny1m3m (3M tokens, ~92 steps).
- **Seed**: 42 (one seed only — per project rule).

## Cost

- **Params Δ**: +2·n_layers·2·d_head LayerNorm γ/β = +768 params
  at tiny1m3m (n_layers=12, d_head=16) = ~+0.08% over the
  ~0.94M baseline. From 016. Moonlight is 0 params.
- **FLOPs Δ**: +2 LayerNorm reductions per (token, head, layer)
  in the attention pre-softmax path (from 016). Optimizer-side
  cost from Moonlight is two `size()` reads + one sqrt + one mul
  per Muon param per step (negligible). Total ≪0.01% of model
  FLOPs.
- **Memory Δ**: +2·d_head per head per layer (LayerNorm γ+β,
  from 016). Moonlight adds no state. Negligible.
- **Wall-clock Δ**: ~0% — the two parents each measured ~0%
  overhead; composition does not introduce new hot paths.

## Run

- Tier: tiny1m3m.
- Command: existing runner with
  `--config Tiny1M3MMoonlightMuonQKNormConfig` for the trt and
  `--config Tiny1M3MConfig` for the ctrl. (If running the
  5-bracket: also `Tiny1M3MMoonlightMuonConfig` and
  `Tiny1M3MQKNormConfig`.)
- Seed: 42 (single seed, hard-pinned per project rule).
- Expected wall-clock: ~10–15 min per side on a single H100
  (matches 015/016 timings exactly — same recipe family).
- **Pass/fail bar** (from `review.md` r1 finding on control
  selection, matching 015's bar):
  - **PASS** (real win): `trt ≤ ctrl − 0.01` on val_loss at the
    final eval milestone. Bar set to 015's threshold (stricter
    than 016's −0.005) because the upper-bound additive
    hypothesis is ~−0.028; the stack must clear at least
    ⅓ of that to be a real composition win and not just a
    re-measure of the more-detectable parent.
  - **NULL** (informative, on-noise): `|Δ| < 0.01`. A clean
    null is itself the *interesting* answer — both taste log
    and review explicitly call this out: it indicates 015 and
    016 are substitutes for the same stability mechanism
    (per-head attention-logit headroom), and the 135M ladder
    should carry only one of the two. Log and move on; do
    **not** add seeds to confirm.
  - **DRIFT** (regression): `trt > ctrl + 0.01` — flag and
    reconsider. Not expected (no shared state, both parents
    monotone-improvers) but possible if e.g. the bounded Q·K
    magnitude under QK-Norm interacts badly with Moonlight's
    larger Muon updates by reducing effective update headroom.

## Interpretation key

- Δ ≈ −0.028 ± 0.005 → **additive WIN**: 015 and 016 are
  orthogonal levers (optimizer-side update scale vs runtime
  logit magnitude). Carry both into the 10M→135M recipe.
- −0.02 ≤ Δ ≤ −0.01 → **subadditive WIN**: levers overlap
  partially; stack still nets a real win. Carry the cheaper
  of the two (QK-Norm, ~0 params/FLOPs delta) and reserve the
  other for ablation at 10M.
- |Δ| < 0.01 → **substitute null**: both levers stabilise the
  same property. Carry whichever has the stronger
  parent-evidence Δ (tie at tiny1m3m → carry QK-Norm, simpler
  scaling story per Dehghani 22B).
- Δ ≥ +0.01 → **destructive interaction**: do not stack; halt
  composition track and re-test parents in a new session.
