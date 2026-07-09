# Plan — 015 Moonlight Muon RMS

## Flag
- `LLMConfig.use_moonlight_muon: bool = False` — default OFF, already
  declared in `configs/llm_config.py:436` (alongside the other Muon
  family flags).
- `LLMConfig.moonlight_muon_c: float = 0.2` — single global knob, the
  paper's tuned constant, declared at `configs/llm_config.py:437`.
- When OFF: Muon path is **bit-identical** to the existing
  `scale_mode="shape_aspect"` baseline (the `trainer.py` falls through
  to `getattr(config, "muon_scale_mode", "shape_aspect")`).
- New `scale_mode="moonlight"` branch added to `Muon.step()` in
  `optimizers/muon.py` — math `c * sqrt(max(d_in, d_out))` is the
  same family as the existing `"spectral"` branch but exposed under
  a paper-named key with `c` carried on the param group.

## Change
- `optimizers/muon.py`:
  - Add a `"moonlight"` branch in the `scale_mode` if/elif chain in
    `step()`. The branch reads `group.get("moonlight_c", 0.2)` and
    computes `scale = c * (max(p.size(-2), p.size(-1)) ** 0.5)`.
    ~7 LoC. (`__init__` already wires `moonlight_c` into `defaults`;
    only the step() branch is new.)
- `training/trainer.py`:
  - In `setup_muon_optimizer`, when
    `getattr(config, "use_moonlight_muon", False)` is True, the
    `scale_mode=` arg to the `Muon` constructor becomes
    `"moonlight"` (overriding the user's `muon_scale_mode` setting);
    otherwise it falls through to the existing default. Pass
    `moonlight_c=getattr(config, "moonlight_muon_c", 0.2)` so the
    constant is plumbed end-to-end. ~10 LoC of comment + one line
    of logic change.
- `configs/llm_config.py`:
  - No new fields. The flags `use_moonlight_muon` /
    `moonlight_muon_c` already exist at lines 436–437 (added in a
    earlier commit) and the `Tiny1M3MMoonlightMuonConfig` recipe at
    lines 586–596 already inherits `Tiny1M3MConfig` and flips the
    flag on. No edits needed.
- Step-0 invariance: orthogonalization of a zero gradient yields a
  zero update, and `0 * anything = 0`, so the first step is
  bit-identical regardless of the scale formula. No identity-init
  drift.

## Control
- **Control** (`ctrl`): `Tiny1M3MConfig` — plain Muon with
  `scale_mode="shape_aspect"` (the project default), `muon_lr=0.024`,
  seed 42.
- **Treatment** (`trt`): `Tiny1M3MMoonlightMuonConfig` — same as ctrl
  but with `use_moonlight_muon=True` (the trainer routes to
  `scale_mode="moonlight"`, `c=0.2`, same `muon_lr=0.024`).
- **Tier**: tiny1m3m (3M tokens, ~92 steps).
- **Seed**: 42 (one seed only — per project rule).

## Cost
- **Params Δ**: 0 (no parameter shape change; the flag is
  config-only).
- **FLOPs Δ**: 0 in the optimizer — same NS5 / polar-express cost.
  The scale formula is two `size()` reads + one sqrt + one mul per
  Muon param per step. Negligible.
- **Memory Δ**: 0 (no new state).
- **Wall-clock Δ**: ~0% (negligible CPU overhead in the optimizer
  loop).

## Run
- Tier: tiny1m3m.
- Command: existing runner with `--config
  Tiny1M3MMoonlightMuonConfig` (and `Tiny1M3MConfig` for ctrl).
- Seed: 42 (single seed, hard-pinned per project rule).
- Expected wall-clock: ~10–15 min per side on a single H100, in
  line with the other tiny1m3m A/Bs (cautious-muon etc.).
- **Pass/fail bar** (from `idea.md` / `review.md`):
  - **PASS** (real win): `trt ≤ ctrl − 0.01` on val_loss at the
    final eval milestone (per the reviewer: "call it
    trt ≤ plain-Muon ctrl − 0.01 to be a real win, since family
    variance is ~±0.01").
  - **NULL** (inconclusive, on-noise): `|Δ| < 0.01` — log and
    move on.
  - **DRIFT** (regression): `trt > ctrl + 0.01` — flag and
    reconsider.
- Note: per the taste review, a clean NULL at tiny1m3m is itself
  informative — it partitions "the lever" between ortho-update and
  shape-rescale. The 015 bet specifically tests the shape-rescale
  component.
