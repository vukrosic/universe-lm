# Plan — 168 av-output-carry

## Flag
`use_av_output_carry: bool = False` — default OFF on `LLMConfig` (`configs/llm_config.py`); ON on `Tiny1M3MAVOutputCarryConfig` (a `@dataclass` subclass of `Tiny1M3MConfig`). When ON, each block's `MultiHeadAttention` builds an `alpha_av = nn.Parameter(torch.zeros(()))` (one per block, 12 scalars total, +0.001% of 0.94M).

## Change

### Wiring (3 files, ~30 LoC of new code, the rest is comments)

1. **`configs/llm_config.py`** — add `use_av_output_carry: bool = False` on `LLMConfig`; add a `@dataclass`-decorated subclass `Tiny1M3MAVOutputCarryConfig(Tiny1M3MConfig)` with `use_av_output_carry: bool = True`.

2. **`models/layers.py`**:
   - `MultiHeadAttention.__init__` — new kwarg `use_av_output_carry: bool = False`. Builds `self.alpha_av = nn.Parameter(torch.zeros(()))` and initializes `self._av_carry = None` when the flag is on. Stubs `alpha_av = None`, `_av_carry = None` when off so attribute lookups are always valid.
   - `MultiHeadAttention.forward(...)` — new kwarg `av_carry=None`. At the post-SDPA / post-merge-reshape (`[B, T, d_model]`) / pre-W_O site (after `_apply_output_op`, after the optional 163 V-mix conv, BEFORE the `F.linear(... self.qkvo_proj[self.qkv_size:])` W_O projection), one branch:
     - `if av_carry is None: self._av_carry = attn_output.detach()` (layer 0 stash)
     - `else: attn_output = attn_output + self.alpha_av * av_carry` (layer l ≥ 1 blend)
   - `TransformerBlock.__init__` — new kwarg `use_av_output_carry: bool = False` passed through to `MultiHeadAttention(...)`.
   - `TransformerBlock.forward(...)` — new kwarg `av_carry=None` passed through to all three `self.attention(...)` call sites (parallel, post-norm, pre-norm).

3. **`models/llm.py`**:
   - `MinimalLLM.__init__` — `self.use_av_output_carry = getattr(config, "use_av_output_carry", False)`.
   - Pass-through to `MultiHeadAttention(...)` in both the YOCO upper-half block construction (~line 651) and the standard transformer block construction (~line 891).
   - Forward loop: stash `av_carry = None` before the loop; pass `av_carry=av_carry` to every `block(...)` call alongside `v_residual`/`q_carry`; after the layer-0 block, capture `av_carry = block.attention._av_carry` (with the same `not self.use_gau` guard as 021/164 since `GAUBlock` has no `.attention` attribute).

### Mechanism

For each block l ≥ 1:
```
av_l = softmax(Q_l K_l^T / sqrt(d)) V_l            # standard attention
out_l_pre_W_O = reshape(av_l) + α_l · av_{l-1}    # post-merge-reshape, pre-W_O
out_l = W_O @ out_l_pre_W_O                        # standard W_O projection
```
where `α_l` is a per-block 0-dim scalar (init 0 ⇒ identity blend at step 0), and `av_{l-1}` is the previous block's post-SDPA / post-merge-reshape / pre-W_O attention output (`.detach()`-ed in the model loop to mirror 021's V-residual contract).

### `prev_av` definition (the exact tensor)

Stashed on `block.attention._av_carry` at the post-SDPA / post-merge-reshape (`[B, T, d_model]`) / pre-W_O site inside `MultiHeadAttention.forward`. Shape is `[B, T, d_model]` (post-merge, NOT the per-head `[B, H, T, d_k]`). Pass to every layer l ≥ 1's `MHA.forward` as `av_carry=...` kwarg. Layer 0 has no previous block → carry term is exactly 0 (stash is set, blend branch not taken). Mirrors 021's `v_residual is None` → stash branch pattern at `models/layers.py:2392` and 164's `q_carry is None` → stash branch at `models/layers.py:2002`.

### Detach policy

`.detach()` on the stash so the only gradient through the carry path is `α_l`'s gradient (mirrors 021's V-residual contract at `models/layers.py:2394`; documented reason: "each layer's W_V trains on its own attention path"). Keeps `isolate_effects` clean — no gradient coupling between blocks beyond the per-block α scalar.

### Step-0 identity claim

With `α_l=0` (zero-init), `α_l · av_{l-1} = 0` exactly in fp32 ⇒ the additive mix is multiplied by 0 ⇒ the W_O projection sees the same `attn_output` as the baseline path. Plan verification: `step0_diff ≤ 1e-6` (fp32 max-abs-diff) between `use_av_output_carry=True` and `use_av_output_carry=False` at the model output, across all 12 blocks. Mirrors 021's bit-identity claim at `models/layers.py:2253-2254` and 164's bit-identity claim in `plan.md`.

## Control

| | config | seed | tier |
|---|---|---|---|
| control | `Tiny1M3MConfig` (baseline, no flags) | 42 | tiny1m3m |
| treatment | `Tiny1M3MAVOutputCarryConfig` (`use_av_output_carry=True`) | 42 | tiny1m3m |

The daemon runs the ctrl(s) — `baseline.sh check` returns `CACHED` (use the cached mean) or `MEASURE` (prepend ≥3 fresh ctrls). One treatment only; no multi-seed.

## Pass bar (NUMERIC GAP)

Idea says "structural cross-block carry" and contrasts with 021-value-residual (Δ = -0.034 WIN). Reviewer-pinned bar (revised to be tight enough to resolve at 92-step tiny1m3m with ~±0.01 val loss noise):

- **WIN** if `Δ ≤ -0.01` (treatment_val < ctrl_mean - 0.01).
- **NULL** if `|Δ| < 0.01` (inside the cached noise band).
- **DRIFT/hostile** if `Δ > +0.01` (any wrong-sign > noise is a strong null since the AV-output axis is structurally distinct from V).

Against `autoresearch/baseline-cache.json` cached mean `6.4346` ± 0.0458 noise band at tiny1m3m seed 42.

## Cost

- **Params Δ**: 12 scalars × 1 (one `alpha_av` per block) = +12 params (+0.001% of 0.94M).
- **FLOPs Δ**: +1 elementwise add (α · av_{l-1}) per block per step, ~+0.01% per-step at tiny1m3m (negligible).
- **Memory Δ**: +1 forward-pass-local `[B, T, d_model]` tensor stashed on `block.attention._av_carry` (auto-released after the forward pass).
- **Wall-clock Δ**: ≤1% — dominated by the existing attention matmul, not the carry add.

## Run

```bash
# daemon-driven; on the box, with the canonical entry:
python _arq_168-av-output-carry.py
```

- **Tier**: `tiny1m3m` (only).
- **Seed**: 42 (only).
- **Dataset**: `processed_data/pretrain_1B`.
- **Warmup**: `false` (per RUN-CONTRACT — `--warmup false`).
- **Job timeout**: `12m` (default — no genuine reason to bump).
- **Expected wall-clock**: ~10m on the box (V100-class), well within `12m` cap.

### Self-check before release

1. `run.json` + `_arq_168-av-output-carry.py` exist; stub defines top-level `C`.
2. `python -c "import _arq_168_av_output_carry as m; from models.llm import MinimalLLM; ml = MinimalLLM(m.C()); print('OK', ml.config.n_layers)"` succeeds on CPU.
3. `flag OFF` (baseline config) reproduces control numerically within fp32 rounding.
4. `flag ON` constructs without error; `alpha_av` is `0` init; `_av_carry` initialized to `None`.

## Composition matrix

- `use_value_residual` (021 V-side): OFF in this A/B — the experiment is the AV-output axis alone. A future 3-way interaction test is out of scope.
- `use_q_carry` (164 Q-side): OFF in this A/B — same reason.
- `use_qk_layernorm` (016): OFF (baseline).
- `use_v_layernorm` (029): OFF (baseline).
- `use_q_only_norm` (162): OFF (baseline).
- `use_k_only_norm` (165): OFF (baseline).
- All other baseline-tier flags: OFF (baseline).

168 is the **third** axis of the V/Q/AV-output attribution test and runs solo for clean attribution.
