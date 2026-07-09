# Plan — 172 per-head-rope-base

## Flag
- `use_per_head_rope_base: bool = False` (default off) on `LLMConfig`,
  `configs/llm_config.py:1036`.
- `rope_base: int = 10000` (default) on `LLMConfig`, `configs/llm_config.py:492`.
- New trt subclass `Tiny1M3MPerHeadRopeConfig(Tiny1M3MConfig)` at
  `configs/llm_config.py:2405` — sets `use_per_head_rope_base: bool = True`
  and `rope_base: int = 500000` (the closed-axes winner).

## Change
**No code changes to `models/layers.py` or `models/llm.py` — mechanism
is already in place.**

Mechanism (already wired):
- `MultiHeadAttention.use_per_head_rope_base: bool` flag at
  `models/layers.py:1779` (ctor) and `:923` (signature).
- `MultiHeadAttention.per_head_rope_log = nn.Parameter(torch.zeros(self.n_heads))`
  at `models/layers.py:1780-1781` when the flag is on.
- Forward-graph use at `models/layers.py:2057-2060`:
  ```
  if self.use_per_head_rope_base:
      head_scale = torch.exp(self.per_head_rope_log)  # [H], init 1.0
  else:
      head_scale = torch.ones(self.n_heads, device=device)
  freqs = t[None, None, :, None] * inv_freq[None, None, None, :]  # [1, 1, T, d_k/2]
  freqs = freqs * head_scale[None, :, None, None]  # [1, H, T, d_k/2]
  ```
- Flag threading through `MinimalLLM` → `TransformerBlock` →
  `MultiHeadAttention` already complete (the `self.use_per_head_rope_base`
  reads at `models/llm.py:508`/`:745`/`:1015` pass it through every
  block construction site).

Config wiring added in this PR:
- New `Tiny1M3MPerHeadRopeConfig(Tiny1M3MConfig)` subclass at
  `configs/llm_config.py:2405` — docstring captures the family-fit,
  pass/fail bar, and the bit-identical-step-0 argument.

Step-0 byte-identity: with `per_head_rope_log = 0` (the default init
for a new `nn.Parameter`), `head_scale = exp(0) = 1.0` for every head
⇒ `freqs *= 1.0` ⇒ the per-head frequency spectrum matches the
`rope_base=500000` baseline exactly ⇒ cos/sin tables unchanged ⇒ Q/K
rotation unchanged ⇒ forward output unchanged ⇒ **byte-identical to
the `rope_base=500000` baseline at step 0 (max-abs-diff = 0.0, no
tolerance needed)**.

## Control
- **ctrl**: `Tiny1M3MConfig` (default `rope_base=10000`, no per-head
  RoPE). Daemon-owned via `CTRL_CONFIG="configs.llm_config.Tiny1M3MConfig"`
  in `autoresearch/bin/queue-daemon.sh:36`.
- **trt**: `_arq_172-per-head-rope-base.py` → `Tiny1M3MPerHeadRopeConfig`
  (sets `rope_base=500000` AND `use_per_head_rope_base=True`).
- **Seed**: 42 (single seed per project convention; pinned via
  `train_llm.py --seed 42`).
- **Tier**: tiny1m3m (`Tiny1M3MConfig`).

## Cost
- Params Δ: +48 scalars (`n_heads=4` × `n_layers=12`) ⇒ +0.005% of
  the 0.94M-parameter baseline (negligible).
- FLOPs Δ: identical to baseline. The `head_scale[None, :, None, None]`
  multiply is `O(B·H·T·d_k/2)` flops per block (a single elementwise
  multiply on `freqs` before cos/sin), well below fp32 noise and
  amortized by the existing rotary path.
- Memory Δ: +48 floats per block (≈ 0.2 KB). Negligible.

## Run
- **Command**: `/venv/main/bin/python _arq_172-per-head-rope-base.py`
  (the daemon's CPU build-smoke calls this with `--warmup false` per
  `RUN-CONTRACT.md`).
- **Tier**: tiny1m3m, 3M tokens, seed 42.
- **Expected wall-clock**: ~10 min on the Vast V100 box
  (`job_timeout: "12m"` in `run.json`).
- **Pass/fail bar** (from `idea.md` and `review.md`):
  - **WIN** — trt vs ctrl Δval < −0.01.
  - **NULL** — |Δval| < 0.01 between trt and ctrl.
  - **DRIFT** — Δval > +0.01.
  - **Informative null sub-band**: report `head_scale[h]` per head at
    the end of training. A null with `head_scale` close to 1.0 across
    all heads means "stay near baseline"; a null with spread to
    e.g. [0.7, 1.4] means the lever learned useful specialization
    but val-loss didn't move (informative re-evaluate at larger tier).

## Notes
- The A/B is "trt (500k base + per-head learning) vs ctrl (10000
  base)" — this conflates the closed-winner global-base effect with
  the per-head learning effect. The review explicitly accepts this:
  "isolates the per-head learning effect" by `rope_base=500000` on
  the trt side. The closed-winner global-base effect is settled, so
  any further Δval against the 10000 ctrl is attributable to the
  combination; if WIN, a follow-up at `rope_base=10000 +
  use_per_head_rope_base=True` would isolate per-head alone (not in
  scope here).
- Family-fit: attention-positioning cluster — 154-rebased-attn (WIN),
  155-per-head-temp (null), 161-dyt-temp (null). 172 is on the
  *frequency* axis (different from temperature and rebasing) but in
  the same family.