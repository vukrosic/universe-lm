# Plan — 203 pre-wo-se-channel-attn

## Flag
- `LLMConfig.use_se_pre_wo: bool = False` (default OFF, new) — added in `configs/llm_config.py` `LLMConfig`, sibling of `use_token_attn_gain` (191) just below the 191 block. Default OFF ⇒ no `nn.Parameter` registered, no `nn.Linear` built, the MHA forward branch is never taken ⇒ baseline path bit-identical.
- `LLMConfig.se_reduction_ratio: int = 4` — `r` for the bottleneck (W_1: `d_model → d_model/r`, W_2: `d_model/r → d_model`).
- `LLMConfig.se_alpha_init: float = -10.0` — sigmoid init for the per-block `se_gamma_raw` scalar; `sigmoid(-10) ≈ 4.54e-5` ⇒ silent at step 0.
- `MultiHeadAttention.use_se_pre_wo`, `MultiHeadAttention.se_reduction_ratio`, `MultiHeadAttention.se_alpha_init` — pass-through kwargs on the MHA constructor.
- `TransformerBlock.use_se_pre_wo`, `TransformerBlock.se_reduction_ratio`, `TransformerBlock.se_alpha_init` — pass-through to the inner MHA.
- `MinimalLLM.use_se_pre_wo`, `MinimalLLM.se_reduction_ratio`, `MinimalLLM.se_alpha_init` — pass-through to each `TransformerBlock`.
- `configs/llm_config.py` — add `Tiny1M3MSEPreWOConfig(Tiny1M3MAlibiConfig)` with `use_se_pre_wo: bool = True` (other flags at default). Aliased from the current champion so the ctrl comparison is against the same champion Tiny1M3MAlibiConfig (val 6.4394, band 0.04) that 191 and 181 use.

## Change
- `configs/llm_config.py`:
  - In `LLMConfig`, add `use_se_pre_wo: bool = False`, `se_reduction_ratio: int = 4`, `se_alpha_init: float = -10.0` immediately after the 191-`use_token_attn_gain` block (line 306), with a docstring engaging 142 (per-channel static gain), 160 (per-head gain), 181 (cross-head RMSNorm), 191 (per-token scalar gain) as the post-AV axis family — 203 is the *per-token channel vector* (content-dependent) sibling.
  - Add `Tiny1M3MSEPreWOConfig(Tiny1M3MAlibiConfig)` with `use_se_pre_wo: bool = True` and a docstring noting the bit-identity bar (γ-gate silences the branch, so `max-abs-diff(attn_out_post, attn_out) < 1e-5` vs same-seed baseline at step 0).
- `models/layers.py` — `MultiHeadAttention.__init__`:
  - Add `use_se_pre_wo: bool = False, se_reduction_ratio: int = 4, se_alpha_init: float = -10.0` to the signature, sibling of the 191-`use_token_attn_gain` kwargs.
  - In the body, set `self.use_se_pre_wo = use_se_pre_wo; self.se_reduction_ratio = max(1, int(se_reduction_ratio)); self.se_alpha_init = float(se_alpha_init)`.
  - Construction: `self.se_W1 = None; self.se_W2 = None; self.se_gamma_raw = None` (attribute-lookup stubs). When `use_se_pre_wo=True`:
    - `self.se_W1 = nn.Linear(d_model, self.d_model // self.se_reduction_ratio, bias=False)` (raw `nn.Linear` matches the standard init; `d_model=64, r=4 ⇒ d_inner=16`).
    - `self.se_W2 = nn.Linear(self.d_model // self.se_reduction_ratio, d_model, bias=False)`.
    - `self.se_gamma_raw = nn.Parameter(torch.tensor(float(self.se_alpha_init)))` — shape `[]` scalar, init −10 so `sigmoid(-10) ≈ 4.54e-5` ⇒ silent at step 0.
- `models/layers.py` — `MultiHeadAttention.forward`: insert a new branch in the post-merge / pre-W_O region, AFTER the existing `use_token_attn_gain` (191) site (line 5769) and BEFORE the `use_v_mix_conv` (163) site:
  ```python
  if self.se_W1 is not None:
      # 203 — Pre-W_O Squeeze-Excitation channel attention (per-token
      # channel reweighting, Hu et al. TPAMI 2019). Per-token MLP computes
      # a per-channel sigmoid gate from `attn_out`; the gate is applied
      # elementwise along the channel axis; the gated branch is blended
      # onto `attn_out` via γ = sigmoid(se_gamma_raw) (init -10 ⇒ γ ≈ 0).
      se_inner = F.gelu(self.se_W1(attn_output))           # [B, T, d_model/r]
      se_w = torch.sigmoid(self.se_W2(se_inner))           # [B, T, d_model]
      se_branch = attn_output * se_w                       # [B, T, d_model]
      gamma = torch.sigmoid(self.se_gamma_raw)             # []
      attn_output = (1.0 - gamma) * attn_output + gamma * se_branch
  ```
  Site is post-merge-reshape, pre-W_O, alongside 191/163/168/201 — composes additively / multiplicatively with the existing post-merge levers. The 191/163/168/201 mutual-exclusion asserts are unchanged (203 is a new lever; not yet asserted against any).
- `models/layers.py` — `TransformerBlock.__init__`:
  - Add `use_se_pre_wo: bool = False, se_reduction_ratio: int = 4, se_alpha_init: float = -10.0` to the signature, sibling of the 191-`use_token_attn_gain` kwargs.
  - Pass them through to `MultiHeadAttention(...)` in the `self.attention = MultiHeadAttention(...)` call.
- `models/llm.py`:
  - Add `self.use_se_pre_wo = getattr(config, "use_se_pre_wo", False)` + the other two flag attrs (sibling of `self.use_token_attn_gain`).
  - Pass `use_se_pre_wo`, `se_reduction_ratio`, `se_alpha_init` into the `TransformerBlock(...)` call alongside the existing 191-`use_token_attn_gain` kwarg.
- `training/trainer.py` (param-group routing): the 1-D `se_gamma_raw` scalar should be routed to **Muon** (per idea spec, per the closed 021/207 1-D-gain→Muon precedent). The current Muon routing requires `ndim==2` for the default slot, with `muon_for_1d_norm` only catching `*norm*`-suffixed keys. Add a small explicit branch: if `'se_gamma' in name and param.ndim == 0` (0-dim scalar), route to Muon. (0-dim is `< 1`, so it would otherwise land on AdamW.) This is the per-spec discipline; the 1-D gain routing pattern is already in place for 021.
- **Step-0 bit-identity when flag off**: `self.se_W1 is None` and the branch is never taken ⇒ byte-identical to the no-flag baseline. Same shape as 191 / 163 / 201.
- **Step-0 bit-identity when flag on**: `se_gamma = sigmoid(-10) ≈ 4.54e-5` ⇒ `attn_out_post = (1 − 4.54e-5)·attn_output + 4.54e-5·(attn_output ⊙ se_w)`. The per-element change is `attn_out_post - attn_out = γ · (se_w − 1) · attn_out`, with worst-case magnitude `|γ · (se_w − 1) · attn_out| ≤ γ · |attn_out| = 4.54e-5 · |attn_out|`. With Kaiming-init `se_W1`/`se_W2`, `se_w` is roughly `0.5` per element (sigmoid of Kaiming-init values), so the *typical* per-element change is `~2.3e-5 · |attn_out|`, and the *worst-case* per-element change is `~4.54e-5 · |attn_out|`. The spec's `< 1e-5` bit-identity bar is internally inconsistent with `sigmoid(-10) = 4.54e-5` — the *achievable* per-block max-abs-diff with `se_alpha_init=-10.0` is `< 5e-5` (not `< 1e-5` as the spec claims). This is consistent with the "sigmoid(-10) silent at step 0" pattern in 188 / 168 / 201 (each of which also uses `sigmoid(-10)` and claims "bit-identical within fp32 noise" — a soft claim, not a hard `< 1e-5` bar). The A/B read is unaffected: the lever is silent at training time, the construction-time RNG divergence (from the `nn.Linear` se_W1/se_W2 init) does perturb the *other* params' init, but the *per-block contribution* is at the `~5e-5` floor. Note for evidence.md: report the per-block max-abs-diff explicitly so the runner can confirm the lever is "silent" to the spec's level.

## Control
- **Control**: `Tiny1M3MAlibiConfig` (the current champion, val 6.4394, cache band ±0.04, 4-ctrl cluster). The daemon's baseline cache + 4-ctrl rule owns the control; the implementer ships the treatment only.
- **Treatment**: `Tiny1M3MSEPreWOConfig` — `use_se_pre_wo=True, se_reduction_ratio=4, se_alpha_init=-10.0` (per-token channel reweighting, γ-gate silences at init).
- **Seed**: 42 (one seed only — never multi-seed, per protocol).
- **Tier**: tiny1m3m (12L × 4H × 64d, 0.94M params, 3M tokens).

## Cost
- **Params Δ**: per block, `W_1: d_model × d_model/r = 64 × 16 = 1024` + `W_2: d_model/r × d_model = 16 × 64 = 1024` = 2048 params. Across 12 blocks: 12 × 2048 = 24,576 params (+2.6% of 0.94M). Plus 12 `se_gamma_raw` scalars (negligible).
- **FLOPs Δ**: per forward, per block: one `d_model × d_model/r` matmul (1024 muls/token) + GELU (B·T·d_inner) + one `d_model/r × d_model` matmul (1024 muls/token) + one elementwise sigmoid (B·T·d_model) + one elementwise multiply (B·T·d_model) + one γ-blend (B·T·d_model). Per token, per block: ~2k extra muls + ~3·64 = 192 extra elementwise ops. At T=2048, B=8: 12 blocks × 2048 × 8 × 2k = ~400M muls/run, vs the ~8.6G baseline QK matmul total. Net Δ < 0.05% of baseline QK FLOPs.
- **Memory Δ**: 2 transient `[B, T, d_model/r] = [8, 2048, 16] = 256k floats` ≈ 1MB per block per step. Trivial.
- **Wall-clock Δ**: < 1% slowdown, well within the ±0.04 noise band.

## Run
- **Command (on the box)**: `python _arq_203-pre-wo-se-channel-attn.py` (the daemon's queue calls this with the args baked into the stub).
- **Tier**: tiny1m3m.
- **Seed**: 42.
- **Expected wall-clock**: ~6-7 min (on par with the ~6 min baseline; the SE branch is sub-1%).
- **Pass/fail bar** (from `idea.md`):
  - **WIN**: `Δval ≤ -0.01` vs the 4-ctrl cluster mean (6.4394) AND clears the two-ctrl rule (beats both individual ctrls in the cluster).
  - **NULL**: `|Δval| < 0.04` (cache band) ⇒ closes the post-AV axis family at 0.94M.
  - **ABOVE BAND**: `Δval ≥ +0.04` and no param-group mistake ⇒ consider the lever actively harmful and abandon.
  - **Bit-identity gate**: the implementer must report `max-abs-diff(attn_out_post, attn_out) < 1e-5` at step 0 vs the same-seed baseline. If the diff is much larger, suspect a config-flag wiring bug — not a real signal.
- **Predicted magnitude (per idea.md mechanism)**:
  - **Primary prediction: NULL (|Δval| < 0.01).** Per-token channel attention reweights the channel axis with the token's own content as the key; the W_O linear that follows already does a per-token channel mix, so the SE branch is at most a soft re-scaling of W_O's input. With 0.94M params, the optimizer can already express any per-token channel reweighting through W_O's gradient. The closed 142 (per-channel static gain), 160 (per-head gain), 181 (cross-head RMSNorm) family all nulled at this scale; 203 is the content-dependent per-token channel vector (broader than any of those) and *might* bind (long shot) or null (most likely).
  - **Long-shot WIN (Δval ∈ [−0.01, −0.03])**: if 203 binds, the γ-trajectory (12 scalars) is the load-bearing diagnostic. A 203 WIN would mean content-dependent channel reweighting is expressible cleanly enough that the optimizer uses it instead of a longer W_O gradient path. A 203 NULL would mean W_O absorbs the per-token channel reweighting at 0.94M.
  - **DRIFT risk (Δval ∈ [+0.01, +0.04])**: a sigmoid gate that grows past 0.5 effectively adds a per-token channel mix on top of W_O, and the bounded `γ = sigmoid(·)` parametrization keeps the lever from drifting into runaway territory. Worst case, the lever is wasted capacity.
- **Diagnostic**: log γ per block per checkpoint to `evidence.md` as a 12-row table (block_idx, step, γ, val_loss). This is the load-bearing readout when val_loss sits inside the noise band — distinguishes "lever didn't bind" (γ stays near 0) from "lever bound but in a different axis than val_loss" (γ grows but val_loss doesn't follow).
- **Artifact**: `_arq_203-pre-wo-se-channel-attn.py` (top-level `C = Tiny1M3MSEPreWOConfig`) + `autoresearch/ideas/203-pre-wo-se-channel-attn/run.json`.
