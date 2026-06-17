from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LLMConfig:
    """Default legacy tuned large preset: 88,630,528 parameters."""

    # Model architecture (88M Params)
    d_model: int = 512       
    n_heads: int = 8         
    n_layers: int = 22
    d_ff: int = 2048         
    
    # GQA parameters
    n_kv_heads: int = 4      
    
    # Data params
    # Use the pre-built dataset downloaded as described in the README
    # (`python data/download_hf_data.py`). The repo author recommends NOT
    # changing the data or max_seq_len. If you think you need to, ASK THE
    # REPO AUTHOR FIRST — it is not recommended. The downloaded data is
    # chunked at seq_len 2048, which the RoPE cache depends on; a mismatch
    # causes runtime errors.
    max_seq_len: int = 2048  # do not change; matches the downloaded data
    vocab_size: int = 49152

    # Low-rank embedding factorization (ALBERT-style). None = full vocab x d_model
    # embedding (default, current behavior). An int r factorizes it into
    # (vocab x r) @ (r x d_model), freeing params to reallocate into transformer
    # depth/width at a fixed total budget. lm_head stays tied to the factorization.
    emb_rank: Optional[int] = None
    # Optional additive low-rank output adapter. This keeps the cheap factorized
    # input embedding but gives the tied softmax a small independent correction.
    # None = baseline tied head only.
    output_adapter_rank: Optional[int] = None

    # Residual-stream levers (both default off; identity/baseline-initialized, so
    # "off" reproduces the current model bit-for-bit).
    # #20 embed residual: re-inject the token embedding x0 each block via a learnable
    #   per-dim mix x = m0*x + m1*x0 (m0 init 1, m1 init 0). Fights representation
    #   drift as depth grows. Costs 2*d_model params/block.
    use_embed_residual: bool = False
    # #22 zero-init resid: zero-init the attention O-projection + FFN down-projection
    #   so each block is an exact identity at step 0 (clean signal propagation through
    #   the deep stack). Zero extra params — purely an init change.
    zero_init_resid: bool = False
    # #27 SmearGate: add a learned per-channel amount of the previous token's
    # embedding before the transformer stack. Causal, zero-init, costs d_model.
    use_smear_gate: bool = False
    # 159 — Embedding pre-LayerNorm (LLaMA 3 / Gemma 2 / Mistral /
    # Qwen 2.5 pattern). Apply a single `nn.LayerNorm(d_model)` on the
    # scaled token embedding right before the first transformer block.
    # Default off → baseline path bit-identical (the LN module is
    # never built, the forward branch is never taken). When on, the
    # LN params are init to `weight = std(x_post)`, `bias = mean(x_post)`
    # (empirical, computed at construction) so `LN(x_post) ≈ x_post`
    # at step 0 within fp32 rounding noise — the model starts as
    # exactly the baseline residual stream and the LN earns its
    # normalisation during training. Cost: 2·d_model params (128 at
    # tiny1m3m — negligible). See
    # `autoresearch/ideas/159-emb-layernorm/idea.md`.
    use_emb_layernorm: bool = False
    # #23 U-Net skips: add zero-init learned bridges from early block outputs to
    # mirrored late blocks. Helps deep narrow stacks preserve early lexical detail.
    use_unet_skips: bool = False
    # Number of early-to-late bridge pairs. None = n_layers // 2 (full U).
    # Must be <= n_layers // 2 (bridges read from un-saved early activations
    # otherwise). Only active when use_unet_skips=True.
    unet_skip_count: Optional[int] = None
    # Gate parameterization for U-Net skips. "raw" applies the gate directly
    # (`x = x + gate * skip`); "sigmoid" wraps as `x = x + sigmoid(gate) * skip`
    # (modded-nanoGPT speedrun style). Default "raw" + init 0.0 reproduces the
    # current behaviour bit-for-bit.
    unet_gate_type: str = "raw"
    # Initial gate parameter value (broadcast to (skip_count, d_model)).
    # With unet_gate_type="raw", 0.0 means no contribution at step 0 (default,
    # current behaviour). With unet_gate_type="sigmoid", -1.5 matches the
    # speedrun: sigmoid(-1.5) ~= 0.18 of the early activation flows in at step 0.
    unet_gate_init: float = 0.0
    # RMSNorm each early-layer skip before the gated add. Aims to keep the bridge
    # contribution scale-stable as the residual stream grows over training. Off
    # reproduces the current behaviour. Only active when use_unet_skips=True.
    unet_bridge_norm: bool = False
    # #28 Attention output gate: zero-init per-head multiplier on attention output.
    # Starts as exact baseline via output *= (1 + gate).
    use_attn_output_gate: bool = False
    # Value-channel gate: zero-init per-head, per-channel multiplier on V.
    # Starts as exact baseline via V *= (1 + gate). Distinct from the
    # scalar per-head output gate above because it acts before the
    # weighted sum.
    use_value_channel_gate: bool = False
    # Attention-output channel gate: zero-init per-head, per-channel
    # multiplier on the post-AV head outputs. Starts as exact baseline via
    # output *= (1 + gate). This is the channelwise sibling of
    # use_attn_output_gate.
    use_attn_output_channel_gate: bool = False
    # 152 — Per-head attention logit bias (PaLM 2 §arch, OLMo 2).
    # Learnable `b_h ∈ R^H` added to attention logits pre-softmax.
    # Init 0 → step-0 byte-identical to baseline. NB: per-head
    # *scalar* bias cancels in softmax over the key axis for all
    # subsequent steps too (per-(b,h,t) `e^{b_h}` factor cancels in
    # the per-row normalizer); the experiment is a recorded null.
    # Default off → baseline path bit-identical. See
    # `autoresearch/ideas/152-attn-logit-bias/idea.md`.
    use_attn_logit_bias: bool = False
    # 205 — Per-Head Post-Softmax Convex Interpolation Toward
    # Uniform. Per-head `m_h = σ(raw_h)` (init `raw_h = -4` ⇒
    # `m_h ≈ 0.018` at step 0). After softmax, blend the per-head
    # attention distribution toward the per-row uniform 1/(t+1)
    # over the active (causal) positions: `attn_h_post = (1 − m_h)·
    # attn_h + m_h · uniform`. Bounded in [0, 1] via sigmoid ⇒ the
    # optimizer can only soften any head toward uniform, never
    # sharpen it. At init `m_h ≈ 0.018` ⇒ per-cell deviation in
    # `attn_w` is bounded by ~0.018, well within fp32 noise. The
    # forward branch is gated on `self.use_per_head_post_softmax_mix`,
    # parameter is not registered when off, so the no-flag baseline
    # path is bit-identical. Forces the manual attention path. Cost:
    # H × L = 48 params (+0.005% of 0.94M). Distinct from the
    # closed per-head attention-shape nulls 152/155/160 (pre-softmax
    # bias/temp, post-AV gain) — this is the bounded *post-softmax*
    # axis. See
    # `autoresearch/ideas/205-per-head-mult-logit-scale/idea.md`.
    use_per_head_post_softmax_mix: bool = False
    # 205 — `raw_h` init for the post-softmax-mix per-head scalar.
    # Default -4.0 ⇒ `m_h = σ(-4) ≈ 0.018` at init (close to
    # identity, lever-bound but cheap). Pinned to -4.0 per the
    # committed parameterization in the idea spec. See
    # `autoresearch/ideas/205-per-head-mult-logit-scale/idea.md`.
    per_head_post_softmax_mix_init_raw: float = -4.0
    # 166 — T5-style bucketed relative position bias
    # (Raffel et al. JMLR 2020, arXiv:1910.10683; re-used in
    # BigBird, REALM, LongT5). Per-head learnable logit bias
    # `rpe_bias ∈ R^{H × B}` added pre-softmax, indexed by
    # `bucket(|i-j|) = floor(log2(|i-j|+1)).clamp_max(B-1)`.
    # B=32 (T5-XXL's spec default; clamped to ≥1 at construction).
    # `rpe_bias = 0` init ⇒ `scores + 0` is bit-identical to the
    # no-RPE baseline at step 0. Composes additively with RoPE /
    # FIRE / CoPE / per-head-temp / per-head-logit-bias (all live
    # on the score side). Forces the manual attention path so the
    # bucket-indexed bias can't go through SDPA's flash kernel.
    # Cost: H × B = 4 × 32 = 128 params/block at tiny1m3m
    # (~+0.014% — negligible). Default off → baseline path bit-
    # identical (no Parameter registered, no branch taken). See
    # `autoresearch/ideas/166-t5-rpe/idea.md`.
    use_t5_rpe: bool = False
    t5_rpe_buckets: int = 32
    # 155 — Per-head learnable attention temperature
    # (PaLM 2 §arch, OLMo 2, Gemma 2). Replace the standard
    # `1/sqrt(d_k)` attention scale with a per-head learnable
    # scalar `τ_h ∈ R^H` so the per-head logit scale becomes
    # `Q_h K_h^T * τ_h`. Init `τ_h = 1/sqrt(d_k)` exactly ⇒
    # `Q_h K_h^T * (1/sqrt(d_k))` ≡ `Q_h K_h^T / sqrt(d_k)`
    # (bit-identical to the standard pre-softmax scale) at step 0.
    # Each head can then adjust its own temperature during
    # training — heads wanting sharper focus can lower `τ_h`,
    # heads wanting broader context can raise it. Cost: H
    # scalars/layer (4 at tiny1m3m, total 48 — negligible). Default
    # off → baseline path bit-identical (no Parameter registered,
    # no branch taken). See
    # `autoresearch/ideas/155-per-head-temp/idea.md`.
    use_per_head_temp: bool = False
    # 161 — Per-layer learnable attention temperature. Replace the
    # standard `1/sqrt(d_k)` attention scale with a per-layer
    # learnable scalar `τ_l ∈ R^{n_layers}` so the per-layer logit
    # scale becomes `Q_h K_h^T * τ_l` (the same scale factor across
    # all heads in a layer, but different across layers). Init
    # `τ_l = 1/sqrt(d_k)` exactly ⇒ `Q_h K_h^T * (1/sqrt(d_k))`
    # ≡ `Q_h K_h^T / sqrt(d_k)` (bit-identical to the standard
    # pre-softmax scale) at step 0. Each layer can then adjust
    # its own temperature — early layers can broaden attention,
    # late layers can sharpen it. Cost: 1 scalar/layer
    # (12 at tiny1m3m, total 12 — negligible). Forces the manual
    # attention path so SDPA's flash/efficient backends don't
    # perturb step-0 numerics. Distinct from `use_per_head_temp`
    # (155): per-head varies WITHIN a layer (H scalars/layer),
    # per-layer varies ACROSS layers (1 scalar/layer). The two
    # are orthogonal axes. Default off → baseline path bit-
    # identical (no Parameter registered, no branch taken). See
    # `autoresearch/ideas/161-dyt-temp/idea.md`.
    use_per_layer_temp: bool = False
    # 188 — Per-block QK-rms scaling (a.k.a. per-block attention
    # temperature, learned). One scalar `s_l ∈ R^1` per block,
    # parameterized as `s_l = exp(s_param_l)` with `s_param_l`
    # init 0 ⇒ `s_l = exp(0) = 1.0` exactly ⇒ `scores * s_l =
    # scores` byte-identical to the baseline at step 0. The
    # forward multiplies the pre-softmax `Q·K^T / sqrt(d_k)`
    # scores by `s_l` BEFORE the causal mask + softmax so a
    # learned `s_l` can sharpen (`> 1`) or flatten (`< 1`) the
    # attention distribution for that block. Different from
    # `use_per_layer_temp` (161 — not implemented in forward) and
    # `use_per_head_temp` (155 — per-head within a block); 188
    # is per-block (one scalar per MHA, shared across heads).
    # Forces the manual attention path so SDPA's flash/efficient
    # backends don't perturb step-0 numerics. Default off → no
    # Parameter registered, no branch taken, baseline path
    # bit-identical. See `autoresearch/ideas/188-qk-rms-scaling/
    # idea.md`.
    use_qk_rms_scaling: bool = False
    # 195 — Tight hard QK logit clamp (min/max bound on pre-softmax
    # QK^T). Apply `torch.clamp(scores, min=-c, max=+c)` after the
    # standard `Q·K^T / sqrt(d_k)` so no single logit can dominate
    # softmax. Default off → baseline path bit-identical (the
    # `if self.use_qk_clamp:` branch is never taken). When on, the
    # lever is intentionally NOT bit-identical at step 0 — at
    # Kaiming init, QK^T entries are O(1) Gaussian and a tight
    # c (default 2.0) actively clips ~5% of the 2-sigma tail at
    # step 0, so the regularizer effect is exercised immediately
    # and the gradient at the boundary is discontinuous (exactly
    # 0 outside the clamp). Forces the manual attention path so
    # SDPA's flash kernel doesn't fuse QK^T+softmax+AV (the
    # pre-softmax logit must be exposed for clamping). Distinct
    # from the closed `logit softcap` axis (smooth tanh at c=8
    # — inactive at step 0 at tiny1m3m; here it's a *hard* clip
    # at c=2.0 — *active* at step 0). See
    # `autoresearch/ideas/195-qk-clamp-min-max/idea.md`.
    use_qk_clamp: bool = False
    qk_clamp_c: float = 2.0
    # 193 — Blockwise attention temperature schedule (fixed cosine-
    # depth, no learned params). One multiplicative scalar `τ_b ∈ R^1`
    # per block `b ∈ [0, L-1]` (`L = n_layers`, 12 at tiny1m3m),
    # shape `τ_b = 1 + α · cos(π · b / L)`, applied to the pre-softmax
    # attention scores as `scores_b = Q_b K_b^T / (τ_b · √d_k)`. The
    # committed scalar is `α = -0.3` ⇒ at `b=0`, `τ_0 = 0.7` (sharper
    # softmax on early layers, the "early layers do local pattern-
    # matching" prior — consistent with 175-alibi's locality-rewarding
    # additive WIN); at `b=L-1=11`, `τ_11 ≈ 1.29` (softer softmax on
    # late layers, "late layers mix broad context"). The schedule is
    # hard-coded: zero new parameters. `α = 0` ⇒ `τ_b = 1` for all `b`
    # ⇒ `scores / (1·√d_k) = scores / √d_k` byte-identical to the
    # standard pre-softmax scale at step 0 (and at every step where
    # `α = 0` is held). The buffer of `τ_b` values is computed once at
    # model construction (registered as a non-Parameter `Buffer` on the
    # MHA) so the per-forward cost is one elementwise divide on
    # `[B, H, T, T]` per block. Distinct from 188 (per-block *learned*
    # `exp(s_param_l)` on the same axis — 193 is the *fixed-shape*
    # control), 155 (per-head *learned* scalar), 161 (per-layer
    # *learned* scalar — closed DRIFT). Forces the manual attention
    # path so SDPA's flash kernel doesn't fuse QK^T+softmax+AV (the
    # pre-softmax score must be exposed for the `τ_b` divide). Default
    # off → no Buffer registered, no branch taken, baseline path
    # bit-identical. See `autoresearch/ideas/193-blockwise-attn-temp-
    # schedule/idea.md` and the conditional framing in review.md
    # (193 is informative whether 188 wins or nulls — at worst, it
    # closes the fixed-shape depth-conditional scale axis).
    use_block_temp_schedule: bool = False
    block_temp_alpha: float = 0.0
    # 160 — Per-head RMS gain on the attention output (Gemma 2 /
    # Qwen 2.5). After the AV product and softmax aggregation, multiply
    # each head's output `o_h = (A·V)_h ∈ R^{T×d_k}` by a learnable
    # scalar `g_h ∈ R^H` so each head controls the magnitude of its
    # contribution to the residual stream without changing direction.
    # Init `g_h = 1.0` exactly ⇒ `o_h *= 1 = o_h` byte-identical to
    # baseline at step 0. Distinct from `use_attn_output_gate` (reparam
    # `(1+g_h)` with g_h=0 init): that one starts at 1.0 but its
    # magnitude reparam has the gradient concentrated in `g_h`; this
    # one is a direct `g_h` multiplier so the magnitude *and*
    # gradient are both `g_h`. Distinct from `use_layerscale`/
    # `use_layer_scale`: those operate on the residual add after the
    # O projection; this lever fires on the *head* dimension before
    # the O projection, normalizing per-head output magnitudes
    # independently of the residual stream. Cost: H scalars/layer
    # (4 at tiny1m3m, total 48 — negligible). Default off → baseline
    # path bit-identical (no Parameter registered, no branch taken).
    # See `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
    use_head_gain: bool = False
    # 181 — Cross-Head Channel RMSNorm (normalize attention output
    # across the H head dim within each d_k slice, so all H heads
    # land on the same per-(t, k) scale before the W_O projection).
    # Per-head scalar gate `α_h = relu(α_raw_h)` (init `−1e-3` ⇒
    # `relu(−1e-3) = 0` exactly ⇒ identity blend at step 0) and a
    # per-(h, k) post-normalization gain
    # `γ_h[k] = 1 + tanh(γ_raw_h[k])` (init 0 ⇒ γ=1 exactly at
    # step 0). Mixed output:
    # `out = (1 − α_h)·out + α_h·(out / rms)·γ_h` where
    # `rms[b, t, k] = sqrt(mean_h(out[b, h, t, k]²) + ε)`. At init
    # α=0, γ=1 ⇒ step-0 forward is byte-identical to baseline
    # (max-abs-diff = 0.0). Distinct from 160 (per-head scalar
    # gain, no cross-head coupling), 176 (V-pre-AV per-head
    # RMSNorm, normalizes each head independently along d_k),
    # 162/165 (pre-softmax Q/K RMSNorm). Cross-head coupling is a
    # qualitatively different lever axis from any closed lever.
    # Default off → no Parameter registered, no branch taken,
    # baseline path bit-identical. See
    # `autoresearch/ideas/181-cross-head-rmsnorm/idea.md`.
    use_cross_head_rmsnorm: bool = False
    # 191 — Per-token attention output gain (Shleifer et al. 2021
    # "NormFormer" / Touvron et al. 2021 "CaiT" class-attention
    # gain, arXiv:2110.09423). After the AV product + softmax
    # aggregation + merge-reshape, multiply the per-position
    # attention output by a learnable per-position scalar
    # `(1 + γ_t)` where `γ_t ∈ R^{T_max}` is shared across
    # batch and the d_model axis. Init γ=0 ⇒ (1 + 0) = 1
    # exactly ⇒ `attn * 1 = attn` byte-identical to baseline
    # at step 0 (max-abs-diff = 0.0, algebraic identity).
    # Sits BEFORE the W_O projection, alongside the post-merge
    # `use_v_mix_conv` (163) and `use_av_carry` (168) sites.
    # Per-token granularity (T_max=2048 scalars/block) is a
    # different axis from the closed per-head (160: H=4
    # scalars), per-channel (142: d_model=64 scalars), and
    # per-(h, k) (181: H·d_k=64 scalars) levers. Cost: T_max
    # scalars/block × 12 blocks = 24,576 params (+2.6% of
    # 0.94M). Default off → no Parameter registered, no branch
    # taken, baseline path bit-identical. See
    # `autoresearch/ideas/191-token-attn-gain/idea.md`.
    use_token_attn_gain: bool = False
    # 203 — Pre-W_O Squeeze-Excitation channel attention (Hu et al.
    # "Squeeze-and-Excitation Networks", TPAMI 2019, arXiv:1709.01507).
    # Per-token channel reweighting on the post-merge attention output
    # via a tiny bottleneck MLP: `se_w = sigmoid(W_2 · gelu(W_1 · x_t))`,
    # `W_1 ∈ R^{d_model × d_model/r}`, `W_2 ∈ R^{d_model/r × d_model}`
    # (default `r=4` ⇒ 64×16 + 16×64 = 2048 params/block). Same W_1,
    # W_2 applied to every token/position (no T-axis pooling — the
    # original SE-Net CNN pattern pools over the spatial axis, but
    # here the lever is the per-token content-dependent cell, not
    # the original CNN cell). Output
    # `attn_out_post = (1 − γ) · attn_out + γ · (attn_out ⊙ se_w)`
    # with `γ = sigmoid(se_gamma_raw)`, init `se_alpha_init=-10.0`
    # ⇒ `sigmoid(-10) ≈ 4.54e-5` ⇒ silent at step 0 (bit-identical
    # to no-flag baseline within fp32 noise of one extra multiply-
    # add — the spec's bit-identity bar is `max-abs-diff(attn_out_post,
    # attn_out) < 1e-5`). The internal `se_w` at step 0 is ~0.5 per
    # channel (sigmoid of Kaiming-init), but the γ-gate silences
    # the whole branch regardless. Applied at the same post-merge /
    # pre-W_O site as 191 (per-token scalar gain), 163 (v_mix_conv),
    # 168 (av_carry), 201 (gmlp_sgu) — composes additively with
    # the residual stream after the γ-blend. The 1-D `se_gamma_raw`
    # scalar is routed to Muon per the spec (1-D gain → Muon,
    # mirrors 021/207 reviewer precedent). Default off → no Parameter
    # registered, no `nn.Linear` built, no branch taken, baseline
    # path bit-identical. Cost: 2 × d_model × d_model/r per block ×
    # 12 blocks = 24,576 params (+2.6% of 0.94M) plus 12 γ scalars
    # (negligible). Distinct from 142 (per-channel static gain),
    # 160 (per-head gain), 181 (cross-head RMSNorm), 191 (per-token
    # scalar gain) — 203 is the *per-token channel vector* (content-
    # dependent channel reweighting). See
    # `autoresearch/ideas/203-pre-wo-se-channel-attn/idea.md`.
    use_se_pre_wo: bool = False
    se_reduction_ratio: int = 4
    se_alpha_init: float = -10.0
    # #107 Exclusive self-attn: subtract the component of the attention
    # output that lies along the current token's value vector. Zero-init
    # per-head coefficient → step-0 is baseline; default off keeps the
    # existing attention path bit-identical.
    use_exclusive_self_attn: bool = False
    # #21 LayerScale: zero-init per-channel scales on attention and MLP residual
    # outputs. Starts as exact baseline via branch *= (1 + gate).
    use_layerscale: bool = False
    # 142 — LayerScale (Touvron et al. 2021, arXiv:2103.17239). Per-channel
    # learnable diagonal scale `gamma ∈ R^{d_model}` on each sublayer's
    # residual branch. Direct form `x = x + gamma * sub_block(x)` (NOT the
    # reparam `(1+g)` form used by the existing `use_layerscale` flag above).
    # Init `gamma = layer_scale_init * ones(d_model)` (default 1e-4) → at
    # step 0 the residual contribution is `1e-4 × sub_block(x)`, four
    # orders of magnitude smaller than the residual stream magnitude, so
    # the val loss at step 0 is within fp32 noise of the baseline (the
    # "soft warmup" the paper specifies). Distinct from the existing
    # `use_layerscale` lever: that one is reparam `(1+g)` with g=0 init
    # (identity at step 0); this one is direct `g·sub_block` with g=ε init
    # (soft-warmup). Per-channel vs scalar (ReZero) is the headline
    # architectural novelty — see `autoresearch/ideas/142-layerscale/`.
    use_layer_scale: bool = False
    layer_scale_init: float = 1e-4
    # 130 — ReZero (Bachlechner et al. 2020, arXiv:2003.04887): per-sublayer
    # learnable scalar α on the residual add (one for attention, one for
    # FFN), init 0 ⇒ the entire stack is the identity at step 0 and each
    # layer earns its residual contribution during training. Replaces
    # the baseline add `x = x + f(x)` with `x = x + α·f(x)`. Off by
    # default → baseline path bit-identical. Cost: 2 scalars/block
    # (n_layers × 2 = 24 scalars at tiny1m3m; negligible). See
    # `autoresearch/ideas/130-rezero/idea.md`.
    use_re_zero: bool = False
    # 197 — DeepNet α residual init (Wang et al. 2022,
    # arXiv:2203.00555). Fixed (NOT learned) depth-conditional scalar
    # `α = (2·n_layers)^(-1/2)` applied to every block's sublayer
    # output (attention AND FFN) before the residual add. The fixed
    # form bounds the residual stream's magnitude growth to `O(1)`
    # throughout the network — at L=12 the per-block contribution
    # scales by 1/√24 ≈ 0.204 so the residual stream's expected
    # magnitude at the LM head is `(1-α^2·L) ≈ 0.71` of one block's
    # contribution (vs `√L ≈ 3.5×` un-scaled). 0 new params
    # (`α` is a Python float computed once at block construction
    # from `n_layers`). Distinct from the closed learned per-block
    # depth-conditional forms — 017 Sub-LN-sandwich (per-block
    # LN), 111 drop-path (regularizer), 116 hyper-connections
    # (multi-stream), 130 ReZero (per-block learned α, init 0),
    # 142 LayerScale (per-channel learned γ, init 1e-4). 197 is
    # the first *fixed* *global* scalar in the family. Step-0
    # forward is NOT byte-identical to baseline by construction
    # (the lever's purpose is the bounded regime from step 0) —
    # this is the explicit trade documented in idea.md. Default
    # off → baseline path bit-identical. See
    # `autoresearch/ideas/197-output-residual-sqrt-2l/idea.md`.
    use_deepnet_alpha: bool = False
    # 288/289 — DeepNet β init-downscaling: the init-side companion to the
    # forward `use_deepnet_alpha` branch-scale. Canonical DeepNet (Wang 2022,
    # arXiv:2203.00555) couples the forward residual scale α with an *init*
    # down-scaling β of the value + output + FFN projection weights so the
    # model UPDATE is bounded at step 0 (Theorem 1). The champion implements
    # ONLY α (the forward 0.204 branch-scale) — β is its untested other half.
    # When on, AFTER `_init_weights` the V-slice + O-slice of every block's
    # fused `qkvo_proj` and both FFN projections (`up_proj`/`down_proj`) are
    # multiplied by β. `deepnet_beta <= 0` ⇒ canonical decoder gain
    # β = (8·n_layers)^(-1/4) (≈0.319 at L=12); set it explicitly to bracket
    # the strength (289 uses (2L)^(-1/2) ≈ 0.204, matched to the forward α).
    # 0 new params (a pure init rescale). NOT step-0 byte-identical — it is an
    # init-CONDITIONING lever (the family that binds at 0.94M/92 steps), not an
    # identity-at-step-0 add. Default off → baseline init path untouched. See
    # `autoresearch/ideas/288-deepnet-beta-init/idea.md`.
    use_deepnet_beta_init: bool = False
    deepnet_beta: float = 0.0
    # #29 value embeddings (speedrun records 55/63): inject the (factorized) token
    # embedding into attention V at every layer via a tiny per-layer projection,
    # zero-inited so step 0 == baseline. Reuses the existing rank-r table as the
    # source, so cost is only ~r*kv_size/layer (~56k total) — stays in budget.
    use_value_embed: bool = False
    # #30 query embeddings: same trick on Q. Tests whether V's win was
    # V-specific or generalizes to "token identity straight into attention."
    use_query_embed: bool = False
    # #31 key embeddings: same trick on K. K goes through RoPE downstream,
    # so the projection's term is positionally rotated — different operating
    # point from V (no RoPE) or Q (also RoPE'd).
    use_key_embed: bool = False
    # #33 output embeddings: same trick, but applied AFTER the O projection
    # (output side of attention, not input side). The token's raw embedding
    # bypasses the attention computation entirely and lands straight in
    # the residual stream. This is the modded-nanogpt speedrun "value
    # embeddings" position — tests whether V-embed's win is V-specific or
    # is "any token-signal-to-residual" wins.
    use_output_embed: bool = False
    # #37 per-head Q-gain: a learnable per-head scalar that multiplies
    # the Q vector after norm+RoPE. Zero-init = baseline at step 0.
    # Equivalent to a per-head temperature on attention scores.
    # Non-embed lever: changes the attention math, not the inputs.
    use_q_gain: bool = False
    # #42 per-head K-gain: symmetric to Q-gain. Multiplies K after
    # norm+RoPE. Tests if K side also benefits from per-head scaling
    # and if V+q+k_gain beats V+q_gain.
    use_k_gain: bool = False
    # #45 deep value embeddings: 2-layer non-linear V projection.
    # V += GELU(ve @ W1) @ W2. Tests if the linear V-embed (#29) has
    # saturated at a single projection, or whether a non-linear
    # "bottleneck" V-embed has more capacity. Mutually exclusive
    # with use_value_embed. Hidden dim is deep_value_embed_hidden
    # (default 96 = 2× emb_rank for Screen10M20M).
    use_deep_value_embed: bool = False
    deep_value_embed_hidden: Optional[int] = None
    # #47 FFN embeddings: add a learned projection of the factorized
    # token embedding to the FFN input. Different position from
    # V-embed (in attention) and O-embed (post-O residual). Tests
    # whether the V-embed win is about attention content or about
    # residual content. The FFN now has direct access to token
    # identity without going through attention. Cost: 24 × (d_model
    # 144 × emb_rank 48) = 165,888 extra params (~2.1%).
    use_ffn_embed: bool = False
    # 153 — Squared-ReLU FFN activation (So et al. "Primer", arXiv:2109.08668,
    # 2021). When True, swap the FFN's activation for `relu2(x) = x *
    # F.relu(x)` (equivalently `(max(0, x))^2`) regardless of `ffn_variant`.
    # Two-projection shape (up_proj, down_proj, dropout) — same param count
    # as `SquaredReLUFeedForward` so the lever is purely the activation
    # change. At init with normal-distributed pre-activations, both GELU
    # and `ReLU²` produce zero-mean, similar-variance outputs — the first
    # forward differs by < 1e-3 in fp32 max-abs-diff (well inside the
    # harness tolerance for non-bit-identical flags). Default off → no
    # `ReLU2FeedForward` is constructed and the standard `ffn_variant`
    # path runs bit-identical to baseline at step 0. See
    # `autoresearch/ideas/153-relu2-ffn/idea.md`.
    use_relu2_ffn: bool = False
    # 170 — SwiGLU FFN (Shazeer 2020, arXiv:2002.05202; LLaMA-family
    # FFN). Drop-in replacement for the standard 2-projection FFN with
    # a 3-projection gated linear unit: `y = down_proj(silu(W_gate·x)
    # ⊙ (W_up·x))`. Built lazily as `SwiGLUZeroInitFeedForward` with
    # `gate_proj.weight = 0` so at step 0 `silu(0) = 0` ⇒ FFN output is
    # exactly 0 ⇒ the residual stream carries only the attention
    # sub-block (a clean ReZero-style step 0). d_ff is scaled by the
    # Shazeer 2/3 trick (`(2 * d_ff) // 3`) so total FFN param count
    # matches the baseline to within ~0.4%. Default off → the new FFN
    # class is never constructed and the standard `ffn_variant` cascade
    # runs bit-identical to baseline. See
    # `autoresearch/ideas/170-swiglu-ffn/idea.md`.
    use_swiglu_ffn: bool = False
    # 196 — MishGLU FFN (Misra 2019 + Shazeer 2020 composition).
    # Inner-activation axis orthogonal to 170's outer-GLU axis:
    # `y = down_proj(mish(W_gate·x) ⊙ (W_up·x))` — structurally identical
    # to SwiGLU (170) except the gate activation is `mish` (`x·tanh(
    # softplus(x))`) instead of `silu`. `mish(0)=0` gives the step-0
    # silence automatically (no explicit zero-init needed — and one
    # would actually mask the gradient signal the lever depends on,
    # since `dMish/dx|_{x=0} ≈ 0.6` vs `dSiLU/dx|_{x=0} = 0.5` is the
    # lever). d_ff is scaled by the Shazeer 2/3 trick (`(2 * d_ff) //
    # 3`) so total FFN param count matches SwiGLU to within ~0.4%.
    # Default off → the new FFN class is never constructed, the
    # standard `ffn_variant` cascade runs bit-identical to baseline.
    # Mutually exclusive with `use_swiglu_ffn` (170) — they target the
    # same FFN slot, different inner activations; the dispatch in
    # `models/layers.py` puts MishGLU ahead so the new lever isn't
    # silently shadowed. See `autoresearch/ideas/196-ffn-glu-mish/idea.md`.
    use_mish_glu: bool = False
    # 198 — Pre-FFN Attention Mixing (FiLM-style cross-stream
    # conditioning; Perez et al. 2018, arXiv:1709.07871). The
    # standard FFN reads `ffn_in = norm2(x)` (pre-norm) where `x`
    # already includes the attention add. 198 instead mixes the
    # *raw* attention output (post-`self.attention(...)`, before any
    # layerscale / sub_ln / rezero / dropout wrapping) into the
    # FFN input as a learned residual:
    #     ffn_in = norm2(x + sigmoid(γ_raw) · attn_out_raw.detach())
    # The `.detach()` keeps γ's gradient cleanly tied to FFN-side
    # loss only (no gradient through the attention path's
    # Q/K/V/O projections at step 0 — the same discipline as 021's
    # value-residual V.detach). Init `pre_ffn_attn_mix_init=-10`
    # ⇒ `sigmoid(-10) ≈ 4.5400e-5` ⇒ the mix contribution is
    # `~4.5e-5 · O(1) ≈ 4.5e-5` in fp32 at step 0 ⇒ baseline path
    # is fp32-noise bit-identical at step 0. Placement: pre-norm2
    # path only (the parallel-block / post-norm paths are
    # alternative architectures off by default; the lever is
    # silently shadowed if the user combines 198 with those
    # flags — documented in `models/layers.py`). Default off ⇒
    # no Parameter registered, no forward branch taken, baseline
    # path bit-identical. Cost: 1 scalar × 12 blocks = 12 scalars
    # (+0.0013% of 0.94M). See
    # `autoresearch/ideas/198-pre-ffn-attnmix/idea.md`.
    use_pre_ffn_attn_mix: bool = False
    pre_ffn_attn_mix_init: float = -10.0
    # 217 — Per-Block RMSNorm/LayerNorm Mixture. When True, each
    # `TransformerBlock` builds a fresh `nn.LayerNorm(d_model)` per
    # pre-norm site (norm1_ln, norm2_ln) and a single scalar
    # `mix_norm_alpha` initialized to `mix_norm_init` (default 4.6,
    # so `sigmoid(4.6) ≈ 0.99` ⇒ output is `0.99·RMSNorm(x) +
    # 0.01·LayerNorm(x)` at step 0, ~1% deviation well within the
    # step-0 noise band). Each block's `mix_norm_alpha` is a
    # per-block learnable scalar; the optimizer can move each one
    # independently over training. Default off ⇒ no Parameter
    # registered, no LayerNorm modules built, no forward branch
    # taken ⇒ baseline path bit-identical. Cost: 1 scalar × 12
    # blocks = 12 scalars (+0.0013% of 0.94M) + 12 × 2 =
    # 24 LayerNorm modules (d_model=64 affine, ~3K params total,
    # +0.32% of 0.94M). See
    # `autoresearch/ideas/217-mix-norm/idea.md`.
    use_mix_norm: bool = False
    mix_norm_init: float = 4.6
    # 157 — Depthwise Conv inside FFN (ConvBERT/ConvNeXt-style, Jiang
    # et al. 2020 arXiv:2008.02496; Woo et al. 2020). When True, each
    # block builds a `ConvFFN(d_model, kernel=k)` that applies a
    # symmetric depthwise Conv1d to the FFN output (post-FFN, pre-
    # residual-add). Conv weights are identity-initialized (center tap
    # = 1, rest = 0) so the conv is a strict identity at step 0 ⇒
    # baseline path bit-identical when the flag is off (the `ConvFFN`
    # module is never built, the forward branch is never taken).
    # `conv_ffn_kernel` defaults to 3 (spec pin); valid range is odd
    # integers ≥ 3. Differs from 143-shortconv by placement (post-FFN
    # vs pre-attention) and causality (symmetric vs causal). Cost:
    # n_layers × (kernel × d_model) extra params (~2.3K at tiny1m3m
    # with k=3, +0.25%). See `autoresearch/ideas/157-conv-ffn/idea.md`.
    use_conv_ffn: bool = False
    conv_ffn_kernel: int = 3
    # 163 — Post-Attention V-Mix Depthwise Convolution (Poli et al.
    # "Hyena", 2023, arXiv:2302.10866). After the attention output
    # is computed (post-SDPA, post-reshape [B,T,H,D]→[B,T,d_model],
    # pre-W_O projection), apply a symmetric depthwise Conv1d on the
    # time axis over the post-attention tensor. Conv weights are
    # built as raw `nn.Parameter(zeros(d_model, 1, k))` with center
    # tap = 1.0 set inline ⇒ the conv is a strict identity at step 0.
    # Padding = k//2 symmetric (causal+future) — the attention
    # sublayer has already integrated the full causal context, so the
    # conv may look at both neighbors. `v_mix_conv_kernel` defaults to
    # 3 (spec pin); valid range is odd integers ≥ 3. Third axis of
    # the 3-axis locality test (143-shortconv pre-attn, 157-conv-ffn
    # post-FFN, 163-v-mix-conv post-attention on V). Default off →
    # baseline path bit-identical (no Parameter registered, no
    # forward branch taken). Cost: n_layers × k × d_model extra
    # params (12 × 3 × 64 = 2,304 at tiny1m3m, +0.25%). See
    # `autoresearch/ideas/163-v-mix-conv/idea.md`.
    use_v_mix_conv: bool = False
    v_mix_conv_kernel: int = 3
    # 201 — Degenerate gMLP Spatial Gating Unit on Attention Output
    # (Liu et al. "Pay Attention to MLPs", NeurIPS 2021,
    # arXiv:2105.08050, §3.1). The committed shape is
    # `z = attn_out.mean(dim=T) → gelu(z) → z @ W_g → broadcast(T)`,
    # with `attn_out_post = attn_out + α · z` and `α = σ(sgu_alpha)`,
    # sgu_alpha init -10 ⇒ α ≈ 4.5e-5 ⇒ silent at step 0 (bit-
    # identical to no-flag baseline). Note: the original gMLP SGU
    # applies a T×T spatial mix along the token axis; the committed
    # shape reduces the cross-token axis to a parameter-free mean
    # (so the lever is a *per-channel gate broadcast* with channel
    # mixing of the global summary, not a true T×T spatial mix —
    # the "degenerate" prefix in the docstring captures this).
    # Applied to the attention output pre-W_O alongside attention
    # (not as a replacement, unlike gMLP proper). Sits on the same
    # post-merge / pre-W_O site as 163 (local conv, depthwise) and
    # 175 (alibi, post-attn pre-O bias) — three axes (local /
    # global / bias) on the same architectural site.
    # `gmlp_sgu_block_stride=3` applies the SGU to block_idx ∈
    # {0, 3, 6, 9} ⇒ 4 of 12 blocks at tiny1m3m (per-block-
    # stochastic to stay in the fair-by-param regime at 0.94M).
    # Default off → baseline path bit-identical (no Parameter
    # registered, no forward branch taken). Cost: 4 blocks ×
    # d_model² extra params (~16K at tiny1m3m, +1.74% of 0.94M)
    # plus 4 α scalars (negligible). See
    # `autoresearch/ideas/201-mlp-token-mixer/idea.md`.
    use_gmlp_sgu: bool = False
    gmlp_sgu_block_stride: int = 3
    gmlp_sgu_alpha_init: float = -10.0
    # #49 QK-norm-post-RoPE: apply RMSNorm to Q,K AFTER RoPE (modded-
    # nanogpt variant) instead of the default BEFORE RoPE. Flag-only,
    # no extra params. The post-RoPE norm constrains post-RoPE Q,K
    # magnitudes per head, which can help with attention score
    # stability. Mathematically different from pre-RoPE norm.
    use_qk_norm_post_rope: bool = False
    # #51 sliding-window attention: replace the full causal mask with
    # a local causal window of width `sliding_window_size`. Flag-only,
    # no extra params. Tests whether the attention *pattern* (not just
    # the inputs) has headroom at this scale — i.e. whether most of the
    # useful long-range signal can be replaced by a short window. If
    # window matches our 2k seq_len's natural coherence, this is a
    # capacity-shaping lever. If not, it caps the model's context.
    use_sliding_window: bool = False
    sliding_window_size: int = 512
    # #53 NoPE: skip RoPE entirely. Flag-only, no extra params. The
    # purest test of whether positional information is load-bearing
    # at this scale. If NoPE ≈ baseline, position is mostly conveyed
    # by the causal mask + token identity (and our token identity
    # injection via V-embed may be partially substituting for RoPE).
    # If NoPE << baseline, RoPE is critical and there's no slack.
    use_nope: bool = False
    # FIRE positional encoding (Li et al., NeurIPS 2023, arXiv:2306.02613):
    # content-aware additive logit bias `bias(t,s) = γ(|t-s|) · f([φ(x_t); φ(x_s)])`
    # with fixed γ (Lp-norm kernel) and small per-head learnable φ/f. Drop-in
    # for RoPE when ON. Default off → baseline path bit-identical. f is
    # zero-init so step-0 bias = 0 even with the flag on.
    # See autoresearch/ideas/009-fire-pe/plan.md.
    use_fire_pe: bool = False
    fire_pe_d_phi: int = 4
    # 013 — CoPE (Golovneva et al. 2024, arXiv:2405.18719, Meta):
    # content-aware positional bias replacing RoPE. Position offset
    # between query i and key j is the count of "important" tokens
    # (those with high content dot-product to a per-head learned probe)
    # in [j, i] — not the literal index distance. Replaces RoPE when
    # ON: the Rotary construction is gated off, the Q/K RMSNorm still
    # runs (magnitude-stabilizer role), and the CoPE bias is added to
    # attention scores in the manual branch. Default off → baseline
    # path bit-identical. Probe init N(0, 0.02) (mirrors FIRE's per-
    # head content projection init at `models/fire_pe.py:60`); τ=0
    # pinned (one-seed-only rule forbids the τ sweep). A/B is
    # "FIRE + CoPE" vs "FIRE" — stacked lever, not a replacement.
    use_cope: bool = False
    # 020 — Forgetting Transformer (FoX, Lin et al. 2025, arXiv:2503.02130):
    # per-head, per-token learnable forget gate that multiplies the
    # attention matrix element-wise after softmax, then row-renormalizes.
    # Conservative extension of softmax attention (the softmax stays; the
    # projection stays; the V path is unchanged). Strictly orthogonal to
    # FIRE (which is additive on logits) — FIRE changes *which* key wins,
    # FoX changes *how much mass* even the winners keep. Identity-init:
    # W_f^h = 0, b_f^h = +10 (pinned at `models/fox.py:FOX_BF_INIT`) →
    # f ≈ 0.99995 at init, D[i,j] within 9% of 1 over the full T=2048
    # context, so the post-softmax decay barely fires at step 0 and the
    # model has to *learn* to forget from scratch. Default off → baseline
    # path bit-identical. Forces the manual attention path (the
    # post-softmax multiply can't go through SDPA's flash kernel). See
    # `autoresearch/ideas/020-forgetting-attn/plan.md`.
    use_fox: bool = False
    # 022 — Softpick (Zuhri/Fuadi/Aji 2025, arXiv:2504.20966):
    # rectified-softmax attention `softpick(x) = relu(exp(x)−1) /
    # (Σ|exp(x)−1| + ε)`. Drop-in for `torch.softmax` in the FIRE
    # manual-path branch. Permits zero total attention mass → kills
    # the attention-sink pathology without adding a learnable sink
    # token. ε=1e-6 pinned (paper default). `exp−1` is computed in
    # fp32 then cast back, otherwise large positive scores overflow
    # in fp16/bf16. No new params, no schedule, no init tuning.
    # Default off → softmax baseline path bit-identical. See
    # `autoresearch/ideas/022-softpick-attention/plan.md`.
    use_softpick: bool = False
    # 173 — Entmax-1.5 sparse attention (Peters/Niculae/Martins,
    # ACL 2019, arXiv:1905.09018). Replace `torch.softmax` in the
    # manual attention path with the Tsallis α-entmax projection at
    # α=1.5 (`p_i = max(0, 0.5·(s_i − λ))^2 / Σp`). Per-head
    # learnable α_h is parameterized as
    # `α_h = 1 + 0.5·(1 + tanh(α_raw_h))`, init `α_raw_h = 0` ⇒
    # `α_h = 1` ⇒ the helper short-circuits to `torch.softmax` for
    # byte-identity at step 0. As training proceeds the optimizer can
    # push `α_raw_h` positive to make the attention sparser
    # (approaching sparsemax at α=2). Default off → baseline path
    # bit-identical (no Parameter registered, no branch taken, no
    # bisection in the forward). Distinct from 022-softpick (no
    # params, fixed operator), 025-SSMax (per-head scaling on
    # softmax, not a replacement), 020-FoX (post-softmax forget gate,
    # softmax stays). See `autoresearch/ideas/173-entmax-15/idea.md`.
    use_entmax: bool = False
    # 192 — Pre-softmax per-row hard top-k sparse attention
    # (Touvron et al. 2021, "Going Deeper with Image Transformers" /
    # DeiT III, arXiv:2103.17239). Keep only the k largest pre-softmax
    # scores per row, scatter -inf to the rest, then softmax-
    # renormalize over the surviving k positions. `k` is a config
    # int (`topk_k`, default 512 = T/4 at the tiny1m3m `max_seq_len
    # = 2048`, i.e. 75% sparsity). 0 new params, no learnable
    # scalar — `k` is fixed. Step-0 is NOT bit-identical to baseline
    # when flag-on (topk of random Gaussians is a different operator
    # than full softmax) — same structural-lever category as 173 /
    # 022 / 154. Forces the manual attention path (the scatter write
    # can't go through SDPA's flash kernel). Causal-mask
    # interaction: topk runs on the already-masked scores, so
    # -inf future positions are below the topk budget and never
    # selected. `k = min(topk_k, scores.size(-1))` is the defensive
    # bound (handles shorter eval contexts). Default off → baseline
    # path bit-identical. See
    # `autoresearch/ideas/192-topk-attn/idea.md`.
    use_topk_attn: bool = False
    topk_k: int = 512
    # 025 — Scalable-Softmax (SSMax, Nakanishi 2025, arXiv:2501.19399):
    # per-head learnable scalar s_h that scales the attention logits
    # by `s_h · log(n)` pre-softmax, where n is the per-query causal
    # key count. Restores per-position sharpness at long range so the
    # softmax distribution does not flatten toward uniform as context
    # grows. Drop-in to the manual attention branch; forces the
    # manual path (score-side tweak, can't go through SDPA's flash
    # kernel). Init s_h = 1.0 — the paper's natural starting point
    # and an effective identity on the operator. NOT bit-identical to
    # vanilla softmax at flag-on, n > 1: the log(n) scaling IS the
    # mechanism (paper §3.1), so the step-0 numerical drift is
    # explicitly justified, not a bug. Default off → baseline path
    # bit-identical. See
    # `autoresearch/ideas/025-scalable-softmax/plan.md`.
    use_ssmax: bool = False
    # 191 — ReLU Attention (Primer / Mercury-Style; So et al. 2021,
    # arXiv:2109.08668). Replace the `torch.softmax` in the manual
    # attention path with `F.relu(scores) / (scores.sum(-1, keepdim=True)
    # + 1e-6)` — drop-in non-softmax operator that zeros negative
    # logits and L1-normalizes the remaining positive scores to a
    # convex combination. No parameters (ReLU+renormalize is purely
    # functional). Default off → softmax baseline path bit-identical
    # (the manual-path branch is not entered, the swap site is
    # bypassed, no branch is taken). Forces the manual attention path
    # (the post-normalize AV can't go through SDPA's flash kernel).
    # When on at step 0 the distribution is *sparser* than softmax
    # (half-zeros; the idea.md design sketch acknowledges this is a
    # documented step-0 drift, not a bug — Primer validates the lever
    # on language modeling at 100M-1.5B). See
    # `autoresearch/ideas/191-relu-attn/idea.md`.
    use_relu_attn: bool = False
    # 023 — Canon conv (Griffin / Mamba local-mixing, De/Smith/Fernando
    # 2024, arXiv:2402.19427; Allen-Zhu et al. Canon-layer line):
    # one causal depthwise Conv1d (kernel=3, left-pad 2) on the
    # residual stream per block, immediately before the attention
    # sublayer's pre-LN. Single scalar output gate `g` per block
    # init 0 → step-0 ≡ no-conv baseline. Pre-LN read (no extra
    # norm on the conv path). Strictly orthogonal to the attention-
    # side levers (FIRE/CoPE/FoX/Softpick all live inside the
    # attention computation; canon conv is on the residual stream
    # itself). Default off → baseline path bit-identical. See
    # `autoresearch/ideas/023-canon-conv/plan.md`.
    use_canon_conv: bool = False
    # 143 — ShortConv (Hyena ShortConv variant, Poli/Massaroli et al.
    # 2023, arXiv:2302.10866): one identity-init depthwise causal
    # Conv1d per block on the residual stream, immediately before the
    # attention sublayer's pre-LN (same placement as CanonConv 023).
    # Weights are identity-initialized (center tap = 1, rest = 0) and
    # a per-block scalar output gate `g` is init 0 → step-0 ≡
    # no-conv baseline (the conv has identity init but the gate
    # scales the contribution to 0, so `x = x + 0·x = x` at step 0).
    # The conv is a *pre-attention* local aggregator: cheap k-neighbor
    # context before the global attention pass. Differs from
    # CanonConv by (a) the identity-init weights (vs Kaiming-uniform)
    # and (b) the parameterizable kernel `short_conv_kernel` (3 or 4).
    # Pre-LN read. Default off → baseline path bit-identical (the
    # `ShortConv1D` module is never built, the forward branch is
    # never taken). See `autoresearch/ideas/143-shortconv/idea.md`.
    use_short_conv: bool = False
    short_conv_kernel: int = 3
    # 021 — Value Residual Learning (Zhou/Wu/Jiang 2024,
    # arXiv:2410.17897). Cross-layer V shortcut: stash the projected V
    # from layer 0 (post-W_V, post-GQA repeat_interleave, post-transpose,
    # shape `[B, n_heads, T, d_k]`); in every later layer l > 0, blend
    # `V_l ← (1 - λ_l)·V_l + λ_l·V_1` BEFORE `attn_weights @ V`, with
    # `λ_l = nn.Parameter(torch.zeros(()))` per-block on MHA. λ=0 init ⇒
    # `V_l = V_l` bit-identical to baseline at step 0; the model has to
    # *learn* to mix in the cross-layer shortcut. `.detach()` on the
    # layer-0 stash so the layer-l blend's gradient does not flow back
    # into layer-0 W_V (each layer's W_V trains on its own attention
    # path). Distinct from the closed V/Q/K/O *embedding* axis (input-
    # side projection scaling) and from every active attention-side lever
    # (020-FoX is post-softmax A·D, 022-softpick is the softmax swap,
    # 024-gated-attn is post-AV o_h gate, 025-SSMax is logit temperature).
    # 021 is the only lever on the *projected V stream*, and the cross-
    # layer formulation is what makes it orthogonal to the closed
    # value-embed axis. Default off → baseline path bit-identical
    # (no `nn.Parameter` created, no stash, no blend). See
    # `autoresearch/ideas/021-value-residual/plan.md`.
    use_value_residual: bool = False
    # 167 — Output logit z-loss (PaLM-style `log(Z)²` penalty,
    # Chowdhery et al. 2022, arXiv:2204.02311, §3.3). Auxiliary
    # training loss term `λ · mean(log(Z)²)` where `Z =
    # sum_v exp(logits_v)`, computed via
    # `logits.logsumexp(dim=-1).pow(2).mean()`. Penalises logit
    # *magnitude* so the largest logit cannot grow without bound
    # and the softmax cannot collapse to a near-delta. Used in
    # PaLM 540B / LLaMA 2 / OLMo 2 / Gemma with λ=1e-4. Train-only;
    # eval stays plain CE (the term is added inside the train
    # branch in `training/trainer.py:1599, 1702`). `use_z_loss=False`
    # OR `z_loss_lambda=0.0` ⇒ term is exactly 0 ⇒ baseline forward
    # + backward is bit-identical to no-z-loss. The trainer reads
    # these via `getattr(config, "use_z_loss", False)` /
    # `getattr(config, "z_loss_lambda", 0.0)` so adding them as
    # proper LLMConfig fields with defaults `False` / `0.0` is
    # drop-in compatible with existing configs that don't set
    # them. The `Tiny1M3MZLossConfig` subclass sets them to
    # `True` / `1e-4`. See `autoresearch/ideas/167-logit-zloss/`.
    use_z_loss: bool = False
    z_loss_lambda: float = 0.0
    # 198 — Residual-stream L2-norm z-loss (PaLM-style magnitude
    # regularizer on the per-token residual, complementary to 167's
    # logit-side z-loss). Auxiliary training loss term
    # `c · mean(log(1 + ||r||²))` where `r ∈ R^{d_model}` is the
    # per-token final residual stream (after the final norm + output
    # dropouts, right before the LM head). Penalises *residual-stream*
    # magnitude so the L2 norm cannot grow without bound — a soft
    # upper bound that is quadratic for small `||r||` and logarithmic
    # for large. Computed via
    # `torch.log1p((x ** 2).sum(dim=-1)).mean()`. The
    # `Tiny1M3MResidualZLossConfig` subclass sets them to `True` /
    # `1e-4`. The trainer reads these via
    # `getattr(config, "use_residual_zloss", False)` /
    # `getattr(config, "zloss_coef", 0.0)` so adding them as proper
    # LLMConfig fields with defaults `False` / `1e-4` is drop-in
    # compatible with existing configs that don't set them.
    # `use_residual_zloss=False` OR `zloss_coef=0.0` ⇒ term is
    # exactly 0 ⇒ baseline forward + backward is bit-identical to
    # no-z-loss. See `autoresearch/ideas/198-z-loss-on-residual/`.
    use_residual_zloss: bool = False
    zloss_coef: float = 1e-4
    # 168 — AV-Output Carry (post-AV cross-block residual). For
    # each block l ≥ 1, augment the post-SDPA/post-reshape/pre-W_O
    # attention output with a learnable α_l-scaled carry from the
    # previous block's same-stage tensor:
    # `out_l = W_O @ (av_l + α_l · av_{l-1})`. `α_l` is a per-block
    # 0-dim scalar (init 0 ⇒ identity blend at step 0). The carry
    # is `.detach()`-ed (mirroring 021's V-residual contract). Site
    # is post-merge-reshape (`[B, T, d_model]`), pre-W_O. Default
    # off → baseline path bit-identical (no Parameter created, no
    # stash, no blend). The third axis of the cross-block carry
    # family (021 = V-side pre-AV, 164 = Q-side pre-AV, 168 = post-
    # AV). See `autoresearch/ideas/168-av-output-carry/plan.md`.
    use_av_output_carry: bool = False
    # 186 — Within-Block V-Carry (per-head learnable V
    # recurrence). Per-head scalar `α_h = tanh(v_carry_alphas_h)`
    # (init 0 ⇒ `α_h = 0` exactly) drives a left-to-right
    # recurrence on V: `V_new[0] = V[0];  V_new[t] = V[t] + α_h ·
    # V_new[t-1]` for `t ≥ 1`. Closed form: `V_new[t] = Σ_{k=0}^{t}
    # α_h^k V[t-k]`, a 1-pole IIR low-pass on V per head (the
    # Katharopoulos 2020 linear-attention recurrence restricted to
    # the V side). Implemented via depthwise `F.conv1d` along T.
    # Local to each block (no cross-block stash). Default off ⇒
    # baseline path bit-identical (α_h=0 ⇒ conv kernel is `[1, 0,
    # …, 0]` ⇒ V is unchanged). Cost: H × n_layers = 48 scalars
    # (+0.005% of 0.94M). See
    # `autoresearch/ideas/186-v-carry-block/plan.md`.
    use_v_carry_block: bool = False
    # 188 — Cross-Block K/V Projection Sharing (Universal
    # Transformers-style learnable parameter sharing across depth,
    # Dehghani et al. ICLR 2019, arXiv:1807.03819). Each block's
    # effective K, V projection is a learnable convex blend of its
    # own (new) projection and the previous block's projection:
    #   `W_K_eff = (1 − σ(α_K_raw)) · W_K_self + σ(α_K_raw) · W_K_prev`
    # (same for V). Init `α_K_raw = α_V_raw = -10.0` ⇒
    # `σ(-10) ≈ 4.5e-5` ⇒ the blend is numerically dominated by
    # `W_K_self` at step 0, so the K, V projection is bit-
    # identical (within fp32 noise) to the no-flag baseline.
    # `prev_W_K` / `prev_W_V` are detached at the call site so
    # the cross-block gradient is bounded to the 2 scalar α
    # params per block. Default off → baseline path bit-
    # identical. Cost: 2 scalars/block × 12 blocks = 24
    # (+0.003% of 0.94M). See
    # `autoresearch/ideas/188-cross-block-kv-share/idea.md`.
    use_cross_block_kv_share: bool = False
    # 206 — Cross-Block W_up / W_down Projection Sharing (the
    # FFN-side analog of 188, narrowed to the two largest FFN
    # matrices only). Each block's `W_up_eff` and `W_down_eff`
    # are learnable convex blends of the block's own
    # (per-block) projection and the previous block's
    # (detached) projection:
    #   `W_up_eff   = (1 - σ(α_up_raw))   · W_up_self   + σ(α_up_raw)   · W_up_prev`
    #   `W_down_eff = (1 - σ(α_down_raw)) · W_down_self + σ(α_down_raw) · W_down_prev`
    # W_gate is LEFT PER-BLOCK (the gating decision is a per-block
    # axis; only the FFN's expansion / compression subspace is
    # shared across depth). Init `α_up_raw = α_down_raw = -10.0`
    # ⇒ `σ(-10) ≈ 4.5e-5` ⇒ the blend is numerically dominated by
    # `W_up_self` / `W_down_self` at step 0, so the FFN is bit-
    # identical (within fp32 noise) to the no-flag baseline.
    # `prev_W_up` / `prev_W_down` are `.detach()`-ed at the call
    # site so the cross-block gradient is bounded to the 2 scalar
    # α params per block. Default off → baseline path bit-
    # identical (no Parameter registered, no stash, no blend).
    # Cost: 2 scalars/block × 12 blocks = 24 (+0.003% of 0.94M).
    # See `autoresearch/ideas/206-cross-block-ffn-share/idea.md` /
    # `plan.md`.
    use_cross_block_ffn_share: bool = False
    ffn_share_alpha_init: float = -10.0
    # 204 — Cross-Block Attention Score Sharing (Sukhbaatar et al.
    # Memorizing Transformers ICLR 2022, arXiv:2203.08913 — within-
    # model cross-block score-reuse lever). Each block's attention
    # scores are blended with the previous block's pre-softmax
    # scores via a learnable per-block scalar α:
    #   `scores_b_eff = (1 − σ(α_raw)) · scores_b_self + σ(α_raw) · scores_{b-1}.detach()`
    # `prev_block_scores` is the PRE-SOFTMAX logit
    # `Q_{b-1} · K_{b-1}^T / √d_k` (NOT the post-softmax attention
    # distribution — a different lever, see review.md finding B),
    # `.detach()`-ed so gradients flow only through `α_raw` and
    # the current block's Q, K (never through the previous
    # block's QK computation, mirroring the 021 / 164 / 168
    # cross-block detach contract). Init `α_raw = -10.0` (via
    # `score_share_alpha_init`) ⇒ `σ(-10) ≈ 4.5e-5` ⇒
    # `scores_eff ≈ scores_self` at step 0 (max-abs-diff across
    # all 12 blocks < 1e-4; identity at fp32 noise of one extra
    # multiply-add). Default off ⇒ baseline path bit-identical
    # (forward branch gated on `use_cross_block_score_share`,
    # `score_share_alpha_raw` not registered, `_prev_block_scores`
    # attribute never written). Forces the manual attention path
    # (the score-blend can't go through SDPA's flash kernel).
    # Cost: 1 scalar/block × 12 blocks = 12 α scalars (+0.001% of
    # 0.94M). See
    # `autoresearch/ideas/204-cross-block-attn-score-share/idea.md`.
    use_cross_block_score_share: bool = False
    score_share_alpha_init: float = -10.0
    # #55 layer tying (ALBERT-style): when tie_layer_groups=N, every
    # group of N consecutive blocks shares weights. The model creates
    # n_layers // N unique blocks and the forward pass cycles through
    # them. group_size=1 (default) is the standard non-tied baseline.
    # group_size=2 means 12 unique blocks for 24 layers — half the
    # unique depth params, with each block seeing two distinct
    # positions. Incompatible with U-Net skips.
    tie_layer_groups: int = 1
    # #63 RoPE base: control the wavelength of the rotary. The default
    # base=10000 is GPT-Neo style; Llama uses 500000 which extends the
    # useful positional range. Tests whether the default decay is
    # hurting at our seq_len=2048 (e.g. a 500k base keeps more
    # headroom for distant positions).
    rope_base: int = 10000
    # #71 logit softcap (Gemma-style): cap logits at ±softcap via
    # `logits = softcap * tanh(logits / softcap)`. A 0.0 value
    # disables it (default). Gemma uses 30.0; we test smaller caps
    # (15.0, 20.0) since our model is smaller. The cap is applied
    # right before the LM head loss, so the gradient is backpropped
    # through the tanh. Real arch change — known stabilizer that
    # can change the loss landscape and unlock better minima.
    logit_softcap: float = 0.0
    # OH4 OutputTemp (OutputHead Batch 2 — see docs/research/output_head/plan.md):
    # divides logits by a learnable scalar τ. τ=1 init = no-op at step 0.
    # 1-D parameter, routes to AdamW. Logit op — flows into eval CE legitimately.
    use_output_temp: bool = False
    # OH5 VocabBias (OutputHead Batch 2 — see docs/research/output_head/plan.md):
    # adds a learnable per-vocab bias b_v to the logits (logits += b_v). b=0
    # init = no-op at step 0. 1-D parameter of size vocab_size, routes to AdamW.
    # Logit op — flows into eval CE legitimately. Re-learns token frequency.
    use_vocab_bias: bool = False
    # 184 — Learned Logit Scale (CLIP-style, Radford et al. 2021,
    # arXiv:2103.00020). Single learned scalar `logit_scale_param`
    # applied as `logits = logits * exp(logit_scale_param)`. Init 0 ⇒
    # `exp(0) = 1.0` exactly in IEEE 754 ⇒ logits unchanged ⇒ step-0
    # byte-identical to baseline. The exp-parameterization guarantees
    # positivity without an explicit clamp. 1 scalar param, routes to
    # AdamW. Default off → baseline path bit-identical. See
    # `autoresearch/ideas/184-logit-scale/idea.md`.
    use_logit_scale: bool = False
    # 193 — μP Joint Init (Tensor Programs V, Yang et al. 2022,
    # arXiv:2203.03466). When on, the embedding is re-initialized to
    # `std = 1.0` (vs the GPT-2 baseline `std = 0.02`; i.e. 50× larger
    # magnitude) AND a learned scalar `logit_scale_param` (init
    # `-log(50)` so `exp(...) = 1/50` at step 0) is applied to the
    # output logits. The two changes compose to a step-0
    # byte-identical output: `(50 · W_emb_baseline @ x) · (1/50) =
    # W_emb_baseline @ x` exactly in fp32. The optimizer then sees a
    # 50×-larger gradient on the embedding (because
    # `∂L/∂W_emb = 50 · ∂L/∂W_emb_baseline`) and a normal-scale gradient
    # on the logit scale, matching the spirit of μP's
    # `lr_emb = lr_base · d_model` rule (the 50× is `1/0.02 = 50`).
    # Default off → baseline path byte-identical (no flag, no
    # param, no forward-graph branch). See
    # `autoresearch/ideas/193-mup-init/idea.md`.
    use_mup_joint_init: bool = False
    # 183 — Pre-LM-Head RMSNorm (Gemma 2 §2, LLaMA 3 §3.1, Qwen 2.5
    # §2.3, OLMo 2 §2.2: a final RMSNorm right before the tied LM
    # head). When on, the model builds an `nn.RMSNorm(d_model)` plus
    # a scalar gate `pre_head_scale = nn.Parameter(torch.zeros(()))`
    # and applies `x = (1 − scale) · x + scale · RMSNorm(x)` between
    # `output_dropout` and `lm_head`. Init `scale = 0` ⇒ the mix is
    # exactly `x` at step 0 (byte-identical to the no-flag baseline,
    # including the dropout interaction — the dropout divides by
    # `(1−p)` in expectation, and the gate is a strict no-op on
    # both the forward and the backward graph). The optimizer grows
    # `scale` toward `1` to engage the RMSNorm during training.
    # Cost: 1 scalar (AdamW) + `d_model` gain weights (Muon) — 65
    # extra params at tiny1m3m (~+0.007%, negligible). Default off
    # ⇒ no module built, no parameter registered, baseline forward
    # path byte-identical to the champion. See
    # `autoresearch/ideas/183-pre-lm-head-rmsnorm/idea.md`.
    use_pre_lm_head_rmsnorm: bool = False
    # 144 — Mixture of Softmaxes (Yang, Chen, et al. 2017,
    # arXiv:1711.03953, "Breaking the Softmax Bottleneck"). When on,
    # replace the single output softmax with `n_mos_components` parallel
    # vocab-sized heads mixed by per-token π = softmax(W_π · h). The
    # mixture output is `P(v) = Σ_k π_k · softmax(W_k · h)[v]`, computed
    # in log space via `logsumexp_k (log π_k + log_softmax(W_k · h))`.
    # The structural lever is the *rank* of the output distribution: a
    # single `softmax(W·h)` has rank ≤ d_model, but a K-mixture has
    # effective rank ≤ K·d_model. Identity at step 0: W_π.weight = 0 and
    # W_π.bias = [+1e4, -1e4, ..., -1e4] ⇒ `softmax(W_π·h) = [1, 0, ...,
    # 0]` exactly in fp32 (the `exp(-2e4)` terms underflow to 0), so
    # `logsumexp` reduces to `log_softmax(W_0 · h)` — bit-identical to
    # the standard tied head. The K fresh heads cost `K·vocab·d_model`
    # params (12.6M at tiny1m3m with K=4 — a sizeable param injection,
    # acknowledged as a confound). Default off → baseline path
    # bit-identical (no MoS module built, no forward-graph branches).
    # See `autoresearch/ideas/144-mos/idea.md`.
    use_mos: bool = False
    n_mos_components: int = 4
    # Forward chunk size (along B*T) for the MoS head. Default 128
    # tokens keeps peak memory well under 1 GiB at tiny1m3m (with
    # K=4, V=49152, fp32). The runner report on round 1 showed K=4
    # with chunk=256 still OOM'd on the RTX 3060 12GB because the
    # downstream `F.cross_entropy(logits.view(-1, V), …)` internally
    # materializes a full (N, V) tensor at fp32 (≈ 3.0 GiB at
    # tiny1m3m). Halving the chunk size to 128 keeps the per-chunk
    # peak ~2× smaller; this combined with K=2 in the
    # `Tiny1M3MMoSConfig` subclass keeps MoS training inside the
    # 12GB envelope. Increase for fewer kernel launches at the cost
    # of more peak memory; decrease further if you hit OOM on
    # smaller GPUs. Only consulted when `use_mos=True`.
    mos_chunk_size: int = 128
    # #72 Tied QK (PaLM-style): Q and K share the same projection
    # matrix. The merged QK is shape [q_size + kv_size, d_model],
    # output is split into Q (q_size) and K (kv_size). Real arch
    # change — PaLM's signature attention choice. The Q and K
    # weights are no longer independent; this is a structural
    # constraint, not a hyperparam.
    use_tied_qk: bool = False
    # #73 Multi-head Latent Attention (MLA, DeepSeek-V2): compress
    # K, V into a low-rank latent of dim `mla_latent_dim`, then
    # up-project per head. Different attention design from the
    # standard projection-per-head. Real arch change.
    use_mla: bool = False
    mla_latent_dim: Optional[int] = None
    # #74 Dilated attention: like SWA but the window consists of
    # every `dilation`-th position (instead of contiguous). Tests
    # whether strided/sparse patterns beat contiguous locality.
    # dilation=1 (default) means contiguous (SWA). dilation=2 means
    # every other position. The window still covers ~`window_size`
    # positions by token count, but spread across a longer range.
    attention_dilation: int = 1
    # #75 Post-norm: instead of pre-norm (norm before attn/ffn),
    # place the norm AFTER the residual addition. The original
    # Transformer used post-norm; pre-norm is the modern default.
    # Tests whether post-norm is a real lever at our depth=24.
    use_post_norm: bool = False
    # #76 embedding scale: the standard code multiplies the
    # token embedding by sqrt(d_model). When set to a value
    # other than -1.0, that value is used instead. -1.0 = use
    # the standard sqrt(d_model). Tests whether the standard
    # scaling is a hidden knob.
    embedding_scale: float = -1.0
    # 194 — Embedding 1/sqrt(d_model) scaling (Primer-style,
    # So et al. 2021, arXiv:2109.08668). When `True`, the
    # `_embed_input` forward substitutes `1/sqrt(d_model)` for
    # the standard `sqrt(d_model)` emb_scale (i.e. the residual
    # stream input is scaled DOWN by 1/d_model relative to the
    # baseline). The lever is forward-time / init-time only —
    # 0 new parameters, the magnitude rescaling is computed
    # deterministically from `d_model`. Step-0 loss is
    # approximately the same as the baseline (the initial
    # Kaiming-init LM-head produces near-uniform logits
    # regardless of the input scale, so cross-entropy on
    # random labels is ~log(vocab_size) in both cases), but
    # the gradient signal is different (a smaller-magnitude
    # residual stream ⇒ flatter softmax ⇒ more uniform
    # gradient across vocab tokens). `False` ⇒ the existing
    # emb_scale path is bit-identical to the baseline. See
    # `autoresearch/ideas/194-embed-sqrt-d/idea.md`.
    use_embed_sqrt_d_scaling: bool = False
    # #77 Q/K dim ratio: by default Q and K have the same dim
    # (d_k). When set != 1.0, K is widened to d_k * qk_k_ratio.
    # Tests whether asymmetric Q/K dims change dynamics.
    qk_k_ratio: float = 1.0
    # #79 LayerNorm vs RMSNorm: the base code uses RMSNorm. Set
    # this flag to use LayerNorm instead (with learned affine).
    # Flag-only — drops in via nn.LayerNorm in place of
    # nn.RMSNorm. Tests whether the choice of norm is a real
    # architecture lever on the best baseline.
    use_layernorm: bool = False
    # #80 Linear attention (Performer-style): replace
    # softmax(QK^T / sqrt(d_k)) V with phi(Q) (phi(K)^T V)
    # where phi(x) = elu(x) + 1 (the standard positive
    # random-feature kernel). Flag-only — different attention
    # math, can be O(n) instead of O(n^2) in the windowed
    # case. Tests whether linear-attention math unlocks a new
    # operating point on the best baseline.
    use_linear_attn: bool = False
    # 189 — CosFormer-Style Linear Attention (Qin et al. NeurIPS 2022,
    # arXiv:2202.08791). Replace softmax(QK^T/√d) V with the kernel-
    # replacement form `out = (Q'·(K'^T·V)) / (Q'·K'^T)` where
    # `Q' = cos(Q)` and `K' = exp(γ·K)·cos(K)` (γ is a learnable
    # per-block scalar, init 0 ⇒ `K' = cos(K)`, the cosFormer cosine
    # feature map). Linear in sequence length, with the prefix-sum
    # cumsum trick at `models/layers.py` for causal masking. The
    # denominator `Q'·K'^T` is MANDATORY (no skip-flag) — it is the
    # softmax replacement, not a global mean-pool. γ lives on the
    # MODEL (`MinimalLLM.cosformer_gammas`), one Parameter of size
    # `n_layers` (the 161-`layer_temperature` pattern), so the
    # optimizer sees one entry not 12. Mutually exclusive with
    # `use_linear_attn` / `use_diff_attn` / `use_nsa_global` /
    # `use_hybrid_heads` / `use_multiscale_heads` (the cosFormer
    # branch IS the attention path; combining with another is
    # double-attention and a structural lever change). Default off
    # → baseline path bit-identical (the branch is gated on
    # `self.use_cosformer`, no Parameter registered, the flag is
    # a strict no-op). See
    # `autoresearch/ideas/189-cosformer-linear-attn/idea.md`.
    use_cosformer: bool = False
    cosformer_gamma_init: float = 0.0
    # #86 Interleaved global attention (DeepSeek-V4 hybrid-attention
    # analog): the model is otherwise all-SWA (cheap local context
    # everywhere, like V4's compressed/sparse path). When
    # global_attn_every_k > 0, every k-th layer (1-indexed) drops the
    # sliding window and runs full causal attention instead — a cheap
    # periodic "global" layer (V4's HCA-style global context). 0 = off
    # (every layer uses whatever use_sliding_window says). Requires
    # use_sliding_window=True to have any effect; on a full-attention
    # baseline every layer is already global. Flag-only, no extra
    # params — it only changes which layers see the window mask.
    global_attn_every_k: int = 0
    # #87 Differential Attention (Microsoft DIFF Transformer, adapted for
    # small heads): split each head's d_k in half, compute two softmax
    # attention maps, output map1 - lambda*map2 (learnable per-head
    # lambda). Cancels common-mode attention noise. Needs even d_k.
    use_diff_attn: bool = False
    # #88 NSA-style compressed-global attention (DeepSeek Native Sparse
    # Attention, adapted): local sliding window PLUS a global branch over
    # block-mean-pooled K/V summaries (block size nsa_block). Zero-init
    # per-head gate, so step 0 == the local-attention baseline.
    use_nsa_global: bool = False
    nsa_block: int = 64
    # #89 Hybrid heads (DeepSeek-V4 hybrid attention at head granularity):
    # first half of heads attend locally (sliding_window_size), second half
    # attend over full causal context, every layer. Zero extra params.
    use_hybrid_heads: bool = False
    # #90 Residual-stream normalization type. "rmsnorm" (default) / "layernorm"
    # / invented variants: "peak" (L-inf), "manhattan" (L1), "squash" (DyT-style
    # tanh, reduction-free), "center" (mean-only), "manifold" (fractional-power
    # RMS with learnable strength rho). See models/layers.make_norm.
    norm_type: str = "rmsnorm"
    # #91 Robust QK-norm: norm applied to Q,K before the attention dot product
    # (default "rmsnorm" == current behaviour; e.g. "pnorm1.5" for outlier-
    # robust attention logits). #92 Robust V-norm: norm applied to V before the
    # softmax-weighted sum ("" / "none" = off; e.g. "pnorm1.5").
    qk_norm_type: str = "rmsnorm"
    v_norm_type: str = ""
    # #97 Multi-scale heads: each head a different sliding-window size
    # (geometric spread around sliding_window_size). #98 Parallel block
    # (PaLM/GPT-J): attention + FFN read one shared norm and sum into the
    # residual, instead of running sequentially.
    use_multiscale_heads: bool = False
    use_parallel_block: bool = False
    # 162 — Q-Only RMSNorm (asymmetric QK pre-softmax normalization). Apply
    # RMSNorm to Q only, leave K raw. nn.RMSNorm weight=1, bias=0 init
    # ⇒ step-0 ≡ RMSNorm-rescaled Q (spec-allowed fp32 max-abs-diff < 1e-3
    # tolerance, same trade-off as 159-emb-layernorm). Default off ⇒ no
    # module built, baseline path bit-identical. See
    # autoresearch/ideas/162-q-only-norm/idea.md.
    use_q_only_norm: bool = False
    # 165 — K-Only RMSNorm (asymmetric QK pre-softmax normalization,
    # K-side). Apply RMSNorm to K only, leave Q raw. nn.RMSNorm
    # weight=1, bias=0 init ⇒ step-0 ≡ RMSNorm-rescaled K (spec-allowed
    # fp32 max-abs-diff < 1e-3 tolerance, same trade-off as 159-emb-
    # layernorm, 162-q-only-norm). Default off ⇒ no module built,
    # baseline path bit-identical. The K-mirror of 162 — together with
    # 016 (symmetric) and 162 (Q-only), the three levers form a clean
    # 3-way orthogonal attribution test for the 016 WIN. See
    # autoresearch/ideas/165-k-only-norm/idea.md.
    use_k_only_norm: bool = False
    # #99 Attention sink slot (softmax-off-by-one): append a zero K/V the query
    # can attend to, so it isn't forced to dump probability on a real token.
    use_attn_sink: bool = False
    # 017 — Sub-LN / Sandwich block (Wang et al. 2022, DeepNet §3.1; Shleifer
    # et al. 2021, NormFormer): wrap each sublayer output with a fresh
    # `nn.LayerNorm(d_model)` (γ=1, β=0 init → identity at step 0) so the
    # pre-norm baseline path stays bit-identical when the flag is off.
    # On the pre-norm path, `y = x + LN_post(Sublayer(LN_pre(x)))` for both
    # attention and FFN. The pre-LN remains whatever norm_type/use_layernorm
    # selected; the post-LN is always `nn.LayerNorm` (the residual-stream
    # re-bounding role, separate from the magnitude-stabilizing pre-LN).
    use_sub_ln: bool = False
    # 111 — DropPath / Stochastic Depth (Huang et al. 2016, arXiv:1603.09382).
    # Per-block Bernoulli gate during training: with probability `1 - p_l`
    # skip the whole block (residual update `x ← x`), with probability `p_l`
    # keep and rescale the block's contribution by `1/p_l` so the expected
    # residual magnitude is preserved. `p_l` is linearly scheduled from 1.0
    # at the first block to `1 - drop_path_max` at the last
    # (`p_l = 1 - drop_path_max * l / (n_layers - 1)`, l = 0-indexed layer
    # position). The coin is shared across the batch (one flip per block per
    # step) — matches the paper and avoids per-token noise that hurts causal
    # LM. Eval has no stochasticity: full block, no rescale. Default off →
    # baseline path bit-identical; flag on + drop_path_max=0.1 is the original
    # paper default (ViT-B/16 12L used 0.1; ConvNeXt 18-36L uses 0.1-0.4).
    # See `autoresearch/ideas/111-drop-path/idea.md`.
    use_drop_path: bool = False
    drop_path_max: float = 0.1
    # 131 — LayerDrop (Fan, Grave, Joulin 2019, arXiv:1904.09728, ICLR
    # 2020). Whole-layer stochastic depth: per-block Bernoulli gate
    # during training: with probability `1 - p_l` skip the entire block
    # (`x ← x`); with probability `p_l` keep and rescale by `1/p_l` so
    # the expected residual matches baseline. Coin is shared across
    # the batch (one flip per block per step) — different from
    # DropPath (111) which is per-batch coin AND per-sample (well, here
    # both are per-batch, but LayerDrop is BLOCK-level, not residual-
    # branch-level). `layerdrop_schedule`:
    #   "constant"          → p_l = layerdrop_p for all l (paper default)
    #   "linear"            → p_l = (l/(L-1)) · layerdrop_p (paper stable variant)
    #   "stochastic_depth"  → p_l = layerdrop_p · (l/(L-1)) (drops start at 0)
    # Eval has no stochasticity. With `use_layerdrop=False` (default)
    # the gate is never applied → baseline path bit-identical.
    # NOTE: with the flag ON, step-0 is NOT byte-identical to baseline
    # — the kept-block rescale `1/p_l` magnifies the residual by 1/p_l
    # (e.g. 5× at p_l=0.2). The lever is explicitly an own-control,
    # not an identity trick. See
    # `autoresearch/ideas/131-layer-drop/idea.md`.
    use_layerdrop: bool = False
    layerdrop_p: float = 0.2
    layerdrop_schedule: str = "constant"
    # 024 — Gated Attention (Qiu et al. 2025, arXiv:2505.06708): per-head
    # *scalar* sigmoid gate on the head output `o_h = A_h V_h`, applied
    # post-AV and pre-merge with the O projection: `o_h ← o_h · 2·σ(W_g·x+b)`.
    # `W_g : nn.Linear(d_model, n_heads)` (one scalar gate per head —
    # NOT the per-head vector form, which would blow the parameter budget
    # at this tier). Gate input is the **sublayer input residual** (pre-LN,
    # NOT `o_h` itself — that would be circular). Identity-init: W=0, b=0
    # → 2·σ(0) = 1.0 exactly at step 0, so the pre-norm baseline path
    # stays bit-identical when the flag is off AND when the flag is on
    # at step 0. Categorically distinct from the pre-existing
    # `use_attn_output_gate` (which is a per-head *learnable scalar gain*
    # `o_h *= (1 + g_h)`, not input-conditional, ReZero-style). Default
    # off → baseline path bit-identical. See
    # `autoresearch/ideas/024-gated-attention/plan.md`.
    use_gated_attn: bool = False

    # 147 — DropKey (Xu et al. 2022, arXiv:2207.01058). Per-head
    # Bernoulli gate on K during training: sample `M ~ Bernoulli(1-p)`
    # of shape `[B, n_heads, T, 1]` and apply `K ← K · M / (1-p)`
    # (inverted-dropout rescale so the expected magnitude matches the
    # un-dropped baseline). The mask is per-head, per-token, and
    # independent across batch — finer granularity than DropPath
    # (111, per-batch coin, per-block) and orthogonal to value-side
    # regularizers (use_value_channel_gate, use_kda_channel_gate) and
    # to score-side regularizers (use_fox, use_ssmax). At inference
    # (training mode off) the mask is identity ⇒ forward graph is
    # bit-identical to the no-DropKey baseline. With
    # `use_drop_key=False` (default) the K tensor is never modified,
    # so baseline path is bit-identical at any rate. Default
    # `drop_key_rate=0.1` matches the ViT-B/16 paper default;
    # `drop_key_rate=0.0` collapses to no masking regardless of the
    # flag. See `autoresearch/ideas/147-dropkey/idea.md`.
    use_drop_key: bool = False
    drop_key_rate: float = 0.1

    # 171 — DropConnect on W_O (Wan et al. 2013, arXiv:1304.3174).
    # Per-weight Bernoulli mask on the attention output projection
    # matrix during training. Sample `M ∈ {0,1}^{d_model × d_model}`
    # with `M_ij ~ Bernoulli(1-p)` and use `W_O_masked = W_O ⊙ M / (1-p)`
    # (inverted-dropout rescale) for the forward pass. Distinct from
    # DropKey (147) which masks K activations at the per-token level
    # and from DropPath (111) which drops whole residual branches:
    # 171 zeroes individual entries of W_O (per-weight, weight-level
    # noise). The mask is sampled per forward pass and shared across
    # all batch elements and positions. At eval (`self.training ==
    # False`) and with `dropconnect_wo_rate=0.0` the branch is never
    # taken ⇒ forward graph bit-identical to the no-DropConnect
    # baseline. Default off ⇒ no Parameter, no branch, no RNG consumed,
    # baseline path bit-identical. Default `dropconnect_wo_rate=0.1`
    # matches Wan et al.'s vision-CIFAR/ImageNet sweet spot. See
    # `autoresearch/ideas/171-dropconnect-wo/idea.md`.
    use_dropconnect_wo: bool = False
    dropconnect_wo_rate: float = 0.05
    # 171 — DropConnect warmup schedule (steps). Linearly ramps the
    # effective DropConnect rate from 0.0 to `dropconnect_wo_rate` over
    # the first `dropconnect_wo_warmup_steps` training forwards, then
    # holds at `dropconnect_wo_rate`. Step-0 effective rate = 0.0 ⇒ the
    # mask branch is short-circuited before any RNG is consumed ⇒ the
    # trt forward is bit-identical to baseline at step 0 (max-abs-diff
    # = 0.0 across the full forward). The ramp gives the optimizer time
    # to find a stable path through the noise before it reaches its full
    # magnitude. Locked at 100 in `Tiny1M3MDropConnectWOConfig` (matches
    # the `~92 remaining` step count after ramp at the 192-step
    # tiny1m3m schedule). Default 100 to match the locked treatment
    # (the trt subclass doesn't override). See
    # `autoresearch/ideas/171-dropconnect-wo/idea.md`.
    dropconnect_wo_warmup_steps: int = 100

    # 207 — W_O Low-Rank Bottleneck (learnable rank-r residual
    # correction on the W_O projection, Arora et al. "Linear
    # Algebraic Structure of Word Senses" + LoRA-style trained-
    # from-scratch low-rank factorization, Hu et al. 2021,
    # arXiv:2106.09685). Replace W_O with
    #   `W_O_eff = W_O + σ(α) · (W_O_A @ W_O_B)`
    # where `W_O_A ∈ R^{d_model × r}`, `W_O_B ∈ R^{r × d_model}`
    # (`r = wo_rank`, default 16), and `α` is a 0-dim learnable
    # scalar (init `wo_lowrank_alpha_init`, default −10 ⇒
    # `σ(α) ≈ 4.5e-5` at step 0). `W_O_A` is normal-init std=0.02
    # (matches the existing `out_proj` / `qkvo_proj` init at
    # line 6043); `W_O_B` is **zero-init** so the rank-r
    # correction is exactly 0 at step 0 ⇒ `W_O_eff == W_O`
    # bit-identical at step 0. As training proceeds, the
    # optimizer can grow `W_O_B` and `α`, activating a learnable
    # rank-r correction that soft-bottlenecks what each
    # attention block can write to the residual stream.
    # Composes with 171-DropConnect (the 171 mask runs first on
    # `w_o`, the 207 correction adds after — both are
    # joint-by-default and individually silent at step 0).
    # Distinct from 197-tied-wo (sharing axis), 199-spectral-norm
    # (Lipschitz axis), 203-pre-wo-se (pre-projection axis) —
    # 207 is the **rank** axis on W_O. Trains A, B, and base
    # jointly from scratch (NOT the LoRA-style frozen-base
    # adaptation; see `autoresearch/ideas/207-wo-lowrank-bottleneck/
    # idea.md` for the joint-training caveat). Default off → no
    # Parameter registered, no branch taken, baseline path
    # bit-identical. See
    # `autoresearch/ideas/207-wo-lowrank-bottleneck/idea.md` /
    # `plan.md`.
    use_lowrank_wo: bool = False
    wo_rank: int = 16
    wo_lowrank_alpha_init: float = -10.0
    # 194 — W_V Low-Rank Residual Correction (LoRA-style trained-
    # from-scratch low-rank factorization on the value projection,
    # Hu et al. 2021 arXiv:2106.09685 + Arora et al. on linear-
    # algebraic word-sense structure). Replace W_V with
    #   `W_V_eff = W_V + σ(α) · (W_V_A @ W_V_B)`
    # where `W_V_A ∈ R^{d_model × r}`, `W_V_B ∈ R^{r × d_model}`
    # (`r = wv_rank`, default 8), and `α` is a 0-dim learnable
    # scalar (init `wv_lowrank_alpha_init`, default −10 ⇒
    # `σ(α) ≈ 4.5e-5` at step 0). `W_V_A` is normal-init std=0.02
    # (matches the existing `qkvo_proj` init); `W_V_B` is
    # **zero-init** so the rank-r correction is exactly 0 at
    # step 0 ⇒ `W_V_eff == W_V` bit-identical at step 0. As
    # training proceeds, the optimizer can grow `W_V_B` and `α`,
    # activating a learnable low-rank correction on the V
    # projection — V is the only single-side attention projection
    # that *positively* binds at 0.94M (021-value-residual WIN),
    # so this is the highest-prior untested rank axis.
    # Complementary to 207-W_O-LowRank (same mechanism, different
    # sub-block); a null at 0.94M closes the entire low-rank-
    # residual sub-block family (FFN tested in r1, W_O tested in
    # 207, W_V tested here). Default off → no Parameter
    # registered, no branch taken, baseline path bit-identical.
    # See `autoresearch/ideas/194-lowrank-ffn/idea.md` / `plan.md`.
    use_lowrank_wv: bool = False
    wv_rank: int = 8
    wv_lowrank_alpha_init: float = -10.0
    # 199 — W_Q Low-Rank Residual Correction (LoRA-style trained-
    # from-scratch low-rank factorization on the query projection,
    # Hu et al. 2021 arXiv:2106.09685). Replace W_Q with
    #   `W_Q_eff = W_Q + σ(α) · (W_Q_A @ W_Q_B)`
    # where `W_Q_A ∈ R^{d_model × r}`, `W_Q_B ∈ R^{r × d_model}`
    # (`r = wq_rank`, default 16), and `α` is a 0-dim learnable
    # scalar (init `wq_lowrank_alpha_init`, default −10 ⇒
    # `σ(α) ≈ 4.5e-5` at step 0). `W_Q_A` is normal-init std=0.02
    # (matches the existing `qkvo_proj` init); `W_Q_B` is
    # **zero-init** so the rank-r correction is exactly 0 at
    # step 0 ⇒ `W_Q_eff == W_Q` bit-identical at step 0. As
    # training proceeds, the optimizer can grow `W_Q_B` and `α`,
    # activating a learnable low-rank correction on the Q
    # projection. Completes the rank-residual sub-block family
    # with 207 (W_O) and 194-r2 (W_V). W_Q is the only d_model ×
    # d_model attention sub-block unowned in the active queue;
    # the 162 (Q-only norm, null) and 165 (K-only norm, null)
    # priors do NOT bind this lever because **norm ≠ rank** —
    # norm and rank are orthogonal in mechanism space (a
    # normalized W_Q can still be low-rank, and vice versa).
    # Default off → no Parameter registered, no branch taken,
    # baseline path bit-identical. See
    # `autoresearch/ideas/199-attn-output-lowrank/idea.md` /
    # `plan.md`.
    use_lowrank_wq: bool = False
    wq_rank: int = 16
    wq_lowrank_alpha_init: float = -10.0
    # 199 — Spectral-Norm-Bounded W_O Projection (per-block
    # learnable Lipschitz cap on the attention output projection,
    # Miyato et al. 2018 "Spectral Normalization for GANs" ICLR
    # 2018, arXiv:1802.05957 + Gouk et al. 2021 arXiv:1804.04368).
    # Per-block *l*, apply an *asymmetric* (clip-only) Lipschitz
    # cap on W_O's spectral norm σ_max(W_O^[l]):
    #   cap_l       = σ_max(W_O_init^[l]) · exp(γ_l)
    #   W_O_eff^[l] = W_O^[l] · min(1, cap_l / σ_max(W_O^[l]))
    #                = W_O^[l] · min(1, σ_max_init·exp(γ_l) / σ_max_current)
    # `γ_l` is a per-block learnable 0-dim scalar (init 0 ⇒
    # `exp(γ_l)=1`). `σ_max_init` is the spectral norm of W_O
    # captured on the FIRST forward (then frozen — never recomputed
    # from a perturbed W_O; this is the byte-identity guarantee,
    # per the review's implementation note). `σ_max_current` is
    # tracked via power iteration (1 step per block per forward, a
    # single `[d_model]`-sided vector `u` per block updated as
    # `u ← W_O · u / ||·||₂`, then `σ_max ≈ u^T · W_O · u /
    # (u^T · u)`). At step 0 `γ_l = 0` and `σ_max_current = σ_max_init`
    # ⇒ the factor is exactly 1 ⇒ `W_O_eff == W_O` byte-identical
    # to baseline (the lever is dormant). As training proceeds,
    # σ_max(W_O) typically grows under SGD; the optimizer can push
    # `γ_l < 0` to tighten the cap (and bind the Lipschitz
    # constant), or `γ_l > 0` to loosen it (a no-op since the
    # factor is already 1). The asymmetry (clip-only) is the bet:
    # the informative direction is `γ_l < 0`. Power-iteration
    # state `u` is a Buffer (`.buffers()` not `.parameters()`) so
    # it survives optimizer state serialization but does not
    # consume an optimizer slot. `wo_spectral_cap_pi_iters` is the
    # number of power-iteration steps per forward (default 1, the
    # minimum that tracks the σ_max drift under standard SGD at
    # 0.94M/12L). Distinct from 128-spectral-decoupling (gradient-
    # space orthogonalization) and 160-rms-gain-per-head (post-AV,
    # post-W_O magnitude gain): 199 operates *intra-W_O* on the
    # projection's forward Lipschitz constant. Default off → no
    # Parameter registered, no Buffer registered, no branch taken,
    # baseline path bit-identical. See
    # `autoresearch/ideas/199-spectral-attn-output/idea.md` /
    # `plan.md`.
    use_wo_spectral_cap: bool = False
    wo_spectral_cap_pi_iters: int = 1

    # 151 — RoV (Rotary Value Embeddings, gated). Apply the same rotary
    # position embedding already used on Q,K to the value vector V as
    # well, mixed in via a learnable per-block scalar gate
    # `rov_gate = nn.Parameter(torch.zeros(1))`. Init 0 ⇒ V_combined =
    # V + 0·V_rot = V ⇒ step-0 forward graph bit-identical to baseline.
    # The base rotary is reused from Q,K (no extra buffer). When
    # `use_nope`/`use_cope` is on, RoPE is bypassed and RoV is a no-op
    # (the geometric lever is unavailable). Default off → baseline
    # path bit-identical. See `autoresearch/ideas/151-rov-gated/idea.md`.
    use_rov: bool = False
    # 174 — xPos exponential decay on RoPE-magnitude (Sun et al. 2022,
    # arXiv:2212.10554). One learnable per-layer scalar `xpos_gamma`
    # (init 0) applied as a per-position decay on K after RoPE:
    # `K = K · exp(-xpos_gamma · t)` (the paper's `g_t = (1 − γ)^t`,
    # in `exp` form for numerical stability; at γ=0 both equal 1).
    # With γ=0 (init) `K = K * 1 = K` exactly ⇒ attention scores are
    # unchanged ⇒ forward is bit-identical to the 500k-base RoPE
    # baseline at step 0 (max-abs-diff = 0.0). γ > 0 biases attention
    # toward recent tokens; γ < 0 extends context. Decay applied to
    # K only (not Q) so the score factor is `g_s = (1-γ)^s` on K's
    # position. Default off → no Parameter created, no branch taken,
    # baseline path bit-identical. See
    # `autoresearch/ideas/174-xpos-decay/idea.md`.
    use_xpos: bool = False

    # 156 — Mixture-of-Attentions (MoA). Run `E` parallel attention
    # computations per layer with separate K_e, V_e projections (Q
    # is shared across experts). Mix the E attention outputs by a
    # per-token router `g_e = softmax(W_g x)_e`. At init the
    # (E-1) extra K/V projections are zero (extra experts produce
    # 0 attention) and the router bias is one-hot on expert 0
    # (g_0 = 1.0) ⇒ step-0 output is bit-identical to a single
    # standard attention. Distinct from MoS (144, closed) which
    # mixes softmax variants within one attention — MoA mixes full
    # attention computations. Default off → baseline path bit-
    # identical (no MoA parameters built, no MoA branch taken).
    # `moa_num_experts=E` is the expert count; default E=2 (one
    # extra expert). Cost when on: (E-1) × (2·kv_size × d_model)
    # extra K/V + d_model × E router params per layer ≈ 4-5K
    # params/layer at tiny1m3m (~5% of the 0.94M model). See
    # `autoresearch/ideas/156-moa/idea.md`.
    use_moa: bool = False
    moa_num_experts: int = 2

    # 174 — xPos exponential decay on RoPE (Sun et al. 2022,
    # arXiv:2212.10554). One learnable per-layer scalar
    # `xpos_gamma ∈ R` applied as `K *= exp(-xpos_gamma · t)` to
    # the rotated K, so attention scores pick up a factor
    # `g_s = (1-γ)^s` that shrinks with distance — biases
    # attention toward recent tokens without altering Q.
    # Init `xpos_gamma = 0` ⇒ `g_t = 1` for all t ⇒ forward is
    # bit-identical to the 500k-base RoPE baseline at step 0
    # (max-abs-diff = 0.0). The optimizer can dial γ positive
    # (decay far-away attention) or negative (extend context).
    # Default off → no parameter created, no branch taken,
    # baseline path bit-identical. Cost when on: +1 scalar per
    # MHA × n_layers = 12 scalars at tiny1m3m (~+0.001% of
    # 0.94M). See `autoresearch/ideas/174-xpos-decay/idea.md`.
    use_xpos: bool = False

    # 154 — Rebased Attention (Shi et al. 2024, arXiv:2407.06641):
    # pool K and V along the time axis with a fixed stride-R average
    # *before* the softmax, so attention reads from a learned set of R
    # summary positions instead of T raw ones. `rebase_stride=R` (default
    # 8) gives R = ceil(T/R) ≈ 256 rebasins at tiny1m3m's T=2048.
    # Implementation: `K' = avg_pool(K, R)`, `V' = avg_pool(V, R)`,
    # then `softmax(Q @ K'^T) @ V'` with a causal mask at the
    # rebased-time level (query t can only attend to rebasin r when
    # t >= r·R). When `rebase_stride >= T` the pool collapses to a
    # single block per token, equivalent to the standard full attention
    # → bit-identical to baseline. When `use_rebased_attn=False`
    # (default) the rebase branch is never built and the standard
    # softmax path runs unchanged. Forces the manual attention path
    # (the rebased causal mask can't go through SDPA's flash kernel).
    # See `autoresearch/ideas/154-rebased-attn/idea.md`.
    use_rebased_attn: bool = False
    rebase_stride: int = 8
    # 185 — Static per-head learned K-rotation (learned orthogonal
    # rebase of K only, position-independent). Each head has its own
    # `R_h ∈ R^{d_k × d_k}` orthogonal matrix applied as `K_h =
    # R_h @ K_h` pre-RoPE / pre-qk_norm, parametrized as a product of
    # `d_k/2 = 8` 2D rotations on disjoint `(2i, 2i+1)` planes — one
    # angle `θ_{h,i} ∈ R` per plane. Init `θ_{h,i} = 0` ⇒
    # `R_h = I_{d_k}` exactly in fp32 ⇒ `K = R_h @ K = K` exactly
    # ⇒ step-0 forward is byte-identical to the no-flag baseline.
    # `R_h` orthogonal preserves norms and dot products, so QK^T
    # magnitudes are unchanged (no softmax temperature shift) — same
    # "preserve the dot product" property RoPE has for its position
    # rotation and 154's fixed orthogonal rebase has. Default off ⇒
    # no Parameter registered, no branch taken, baseline path bit-
    # identical. See `autoresearch/ideas/185-static-per-head-k-rotation/idea.md`.
    use_static_k_rotation: bool = False

    # 200 — Static per-layer × per-pair learned K-rotation
    # (depth-axis twin of 185, shared across heads, K-only). Each
    # block has its OWN orthogonal rebase matrix `R_l ∈ R^{d_k ×
    # d_k}` applied to K ONLY (Q untouched) — parameterized as a
    # product of `d_k/2 = 8` 2D rotations on disjoint `(2i, 2i+1)`
    # planes. One learnable angle `φ_{l,i} ∈ R` per (layer, plane),
    # SHARED across heads (depth-axis: 185 varies angles across
    # heads, 200 varies angles across layers). The K-only
    # application breaks QK^T inner-product preservation (Q is
    # in baseline basis, K is in R_l-rotated basis) — gives the
    # lever a real axis to bind on, unlike a QK-symmetric
    # application which would be a provable no-op. `R_l`
    # block-diagonal-orthogonal preserves K's norm and the
    # softmax temperature. Init `φ_{l,i} = 0` ⇒ `cos(0)=1`,
    # `sin(0)=0` in fp32 ⇒ `R_l = I_{d_k}` exactly ⇒
    # `K = R_l @ K = K` exactly ⇒ step-0 forward is bit-
    # identical to the no-flag baseline. Default off ⇒ no
    # Parameter registered, no branch taken, baseline path bit-
    # identical. See
    # `autoresearch/ideas/200-rope-phase-offset-per-layer/idea.md`.
    use_per_layer_k_rotation: bool = False

    # 202 — V-Only Soft-Blend Probe (Isolate V-Sharing From
    # K-Sharing). Per head h, soft-blend per-head V with a group-
    # shared V via per-head `sigmoid(α_h) ∈ R^H`:
    #   `V_h_eff = (1 − σ(α_h)) · V_h_local + σ(α_h) · V_group_g(x)`
    # where `g = h // v_group_size` is the head's group and
    # `V_group_g(x) ∈ R^{d_k}` is the output of a fresh group-shared
    # projection `W_V_group_g ∈ R^{d_k × d_model}`. K is **never
    # touched** — every head keeps its own W_K_h, so the K-axis is
    # the held-out implicit control. Group V projs (G = n_heads //
    # v_group_size, default G=2 at tiny1m3m with v_group_size=2 and
    # H=4) are allocated and init to the elementwise mean of the
    # in-group per-head W_V_h weights; α_h init `-25.0` ⇒
    # `σ(α_h) ≈ 1.4e-11` (well below fp32 precision) ⇒
    # `V_h_eff ≈ V_h_local` exactly at step 0 ⇒ forward is bit-
    # identical to the no-flag baseline. K remains untouched, so the
    # K-axis is the held-out implicit control (the family-dead or
    # family-keep attribution is read off the σ(α) trajectory, not
    # val loss). Default off ⇒ no Parameter registered, no branch
    # taken, baseline path bit-identical. See
    # `autoresearch/ideas/202-grouped-value-projection/idea.md`.
    use_grouped_v: bool = False
    v_group_size: int = 2

    # 192 — Pre-RoPE per-head × per-pair learned Q+K rotation
    # (Su et al. 2024 RoFormer / RoPE, arXiv:2104.09864, position-
    # dependent rotation context). Per head h and per pair i
    # (d_k/2 = 8 planes), one learnable scalar angle `φ_{h,i} ∈ R`
    # applied to BOTH Q and K as a static (position-independent)
    # 2D rotation on disjoint `(2i, 2i+1)` planes BEFORE RoPE's
    # position-dependent rotation. The block-diagonal `R_h` (product
    # of d_k/2 2D rotations) is orthogonal; applied to both Q and
    # K the inner product `<Rq, Rk> = <q, k>` would absorb the
    # lever under identity RoPE — but RoPE is NOT identity: the
    # rotation layers cleanly BEFORE RoPE, so the (Q, K) stream
    # entering RoPE is in a *learned static basis*, and RoPE's
    # position-dependent per-pair rotation then mixes that basis
    # with position. The pre-RoPE placement is the fresh axis
    # (185 rotates K post-RoPE, 200 rotates K post-RoPE with
    # shared angles, 154 uses a fixed rebase on K,V pre-softmax
    # — 192 is the only learned QK rotation that lives *before*
    # the position mix). Init `φ_{h,i} = 0` ⇒ `cos(0)=1,
    # sin(0)=0` in fp32 ⇒ `R_h = I_{d_k}` exactly ⇒ `Q = R_h @
    # Q = Q` and `K = R_h @ K = K` exactly ⇒ step-0 forward is
    # bit-identical to the no-flag baseline. Default off ⇒ no
    # Parameter registered, no branch taken, baseline path bit-
    # identical. See
    # `autoresearch/ideas/192-pre-rope-qk-rotation/idea.md`.
    use_pre_rope_rotation: bool = False
    pre_rope_rotation_init: float = 0.0

    # 134 — Mega: Moving Average Equipped Gated Attention
    # (Ma et al. 2022, arXiv:2209.10655, ICLR 2023). Replaces the
    # standard V projection with `V_mega = concat(V, V_ema)` where
    # `V_ema = β·V_ema_{t-1} + (1-β)·V_raw_t` is a learned per-channel
    # exponential moving average over the V projection input. The
    # attention weights then softmax over the doubled key dim and
    # the AV product sums over both halves. `mega_beta` is a
    # per-channel learnable scalar in `[0, 1]` parametrized as
    # `β = σ(raw)` so it stays bounded during training; raw is
    # zero-init ⇒ β = 0.5 at step 0 (the natural "half-smoothed"
    # midpoint between the paper's β=0 "no smoothing" and β=1
    # "constant EMA" extremes). At step 0 β=0.5 ⇒ V_mega is NOT
    # identical to V; the lever is explicitly NOT a baseline-
    # identity trick. At β=0 the EMA collapses to the current
    # token's V (concat → gated attention, the closed 024 lever);
    # at β=1 it collapses to a constant u_t (no signal). The Mega
    # paper's default β=0.9 is achieved when `raw = log(9) ≈ 2.2`;
    # the optimizer finds the right operating point during
    # training. `mega_use_input=True` (default) feeds the residual
    # stream `x` into the EMA; `False` would feed the projected
    # V (but `x` carries the most recent context and is what the
    # paper actually smooths). Default off → baseline path
    # bit-identical (no Parameter created, no concat applied).
    # See `autoresearch/ideas/134-mega-ema/idea.md`.
    use_mega: bool = False
    mega_beta: float = 0.9  # paper default; raw scalar parametrized as sigmoid
    mega_use_input: bool = True  # EMA on pre-projection residual x (paper form)

    # 129 — YOCO: You Only Cache Once (Sun et al. 2024, arXiv:2405.05254,
    # ICLR 2024 workshop). Decoder-decoder cross-layer KV reuse: the
    # model is split into a lower half (standard sliding-window self-
    # attention, default `yoco_lower_window=512`) and an upper half
    # where each layer's attention reads a SHARED `(K_g, V_g)` cache
    # projected from the lower half's final residual stream — instead
    # of computing per-layer K, V from the input. Saves ~50% of the
    # upper-half K/V projection params (the W_K, W_V slices of
    # `qkvo_proj` are unused on the upper half) and at inference
    # collapses the KV cache from `O(L·d·T)` to `O(d·T)`. The lever
    # is the cross-layer information flow itself, not the cache
    # saving (which doesn't affect the tiny1m3m val-loss A/B).
    # `yoco_split` is the 0-indexed layer where the split happens (the
    # LAST lower-half layer is `yoco_split - 1`; the FIRST upper-half
    # layer is `yoco_split`); with default 6 and n_layers=12 the lower
    # half has 6 layers, the upper half has 6 layers. The lower half
    # runs standard sliding-window self-attention (turning
    # `use_sliding_window=True` and `sliding_window_size=yoco_lower_window`
    # on those blocks only). The upper half uses `YOCOLlamaBlock` whose
    # MHA has `use_shared_kv=True` — the K, V projections are skipped
    # and replaced with a single shared `GlobalKVHead` projection that
    # runs ONCE on the lower-half output. Identity at step 0: the
    # `GlobalKVHead` projections have normal init std=0.02 (matching
    # the rest of the model), so K_g, V_g are `O(0.02)` at step 0 →
    # upper-half attention output is small but non-zero. NOT
    # byte-identical to the standard self-attention baseline at
    # step 0 (the standard path uses per-layer K, V projections
    # with std=0.02 init → same magnitude order), but the deviation
    # is bounded by `O(0.02²)` which is within the NULL band. With
    # `use_yoco=False` (default) the YOCO path is never built and
    # the baseline forward graph is bit-identical.
    # See `autoresearch/ideas/129-yoco/idea.md`.
    use_yoco: bool = False
    yoco_split: int = 6
    yoco_lower_window: int = 512

    # 117 — Soft MoE (Puigcerver, Riquelme, Mustafa, Houlsby 2024,
    # arXiv:2406.06589, ICLR 2025). Drop-in FFN replacement: E parallel
    # narrower FFNs + softmax-based dispatch/combine so gradients flow to
    # all experts (no top-k, no balancing loss, no straight-through).
    # Each expert has width `d_ff / soft_moe_n_experts` so total FFN
    # params stay at the budget. Dispatch and combine are derived from
    # small per-token linear projections (`W_d, W_c` of shape
    # `[soft_moe_n_experts * soft_moe_n_slots, d_model]`) — zero-init
    # ⇒ uniform softmaxes at step 0 ⇒ every expert sees roughly the
    # same weighted average of all input tokens ⇒ layer collapses to
    # ~a single FFN applied to `mean(X)`. NOT byte-identical to the
    # single-FFN baseline when flag is ON (the mean-over-tokens
    # aggregation changes the per-token output), but with
    # `use_soft_moe=False` (default) the `SoftMoEFFN` module is never
    # built and the baseline path is bit-identical. See
    # `models/soft_moe.py` for the full mechanism +
    # `autoresearch/ideas/117-soft-moe/idea.md`.
    use_soft_moe: bool = False
    soft_moe_n_experts: int = 4
    soft_moe_n_slots: int = 4

    # 118 — Mixture-of-Depths (Raposo et al. 2024, arXiv:2404.02258):
    # per-token router at each transformer block decides whether the
    # block fires for the token. Top-k tokens (k = mod_capacity·T) get the
    # block's residual update, the rest are passed through unchanged. The
    # kept tokens' residual update is rescaled by `c = k/T` so the
    # expected per-token contribution matches the dense baseline.
    # `mod_capacity=0.5` is the paper's default. `mod_router_hidden=64`
    # gives a 2-layer MLP `W_1 ∈ R^{d×h}`, `W_2 ∈ R^{h×1}` per block;
    # zero-init both ⇒ `σ(0) = 0.5` uniform scores ⇒ top-k is an
    # arbitrary subset at step 0. With `use_mod=False` (default) the
    # `MoDRouter` is never built and the baseline forward graph is
    # bit-identical. See `models/mod_router.py` +
    # `autoresearch/ideas/118-mixture-of-depths/idea.md`.
    use_mod: bool = False
    mod_capacity: float = 0.5
    mod_router_hidden: int = 64

    # 148 — Focal Modulation Networks (Yang et al. 2022,
    # arXiv:2203.11926, NeurIPS 2022). Replaces the attention sub-block
    # with a three-stage focal modulator: (1) hierarchical context
    # aggregation via a stack of depthwise causal Conv1d at multiple
    # kernel sizes (default 3, 5, 7); (2) gather linear that projects
    # the multi-scale context into modulation space; (3) modulate via
    # `output = x + σ(W_g x + b_g) * (W_q x ⊙ W_h · context)`. Different
    # inductive bias from softmax attention: no QKᵀ, no softmax, no
    # O(T²) memory. Step-0 identity: `gather` and `h_proj` are both
    # zero-init, so the modulation signal is exactly `0` at step 0
    # and `output = x` — bit-identical to baseline when flag is off.
    # With `use_focal_mod=False` (default) the `FocalModulationBlock`
    # is never built and the MHA path is bit-identical. Cost when on:
    # 3 × d_model × K depthwise conv params + 3 × d_model² linear
    # params ≈ 3 × 64 × 5 + 3 × 64² ≈ 960 + 12,288 ≈ 13.2K extra
    # params per block × 12 blocks ≈ 159K (~17% of the tiny1m3m
    # budget). See `autoresearch/ideas/148-focal-mod/idea.md`.
    use_focal_mod: bool = False
    focal_mod_kernels: tuple = (3, 5, 7)

    # 146 — Switch FFN (Fedus, Zoph, Shazeer 2022, arXiv:2101.03961):
    # replace the dense FFN with N parallel FFN "experts" and a
    # top-1 learned router per token. The simplest form of sparse
    # mixture-of-experts in the FFN position. Distinct from
    # 117-soft-moe (slot assignment, all experts always used) and
    # 118-MoD (skip-routing) — Switch uses *top-1 hard routing*.
    # When `use_switch_ffn=True`, swap the standard dense FFN for
    # `SwitchFFN` (E parallel full-width FFNs + top-1 router).
    # Each expert is full-width (no narrowing), so the FFN-param cost
    # multiplies by `n_ffn_experts` (default 4×) — a real param
    # injection. `expert_capacity_factor` controls the per-expert
    # token cap = `ceil(N/E) * capacity_factor`; tokens beyond
    # capacity pass through unchanged (residual identity, paper §2.2).
    # Identity at step 0: `W_router` is zero-init ⇒ argmax over
    # uniform-zero returns index 0 for every token ⇒ all tokens
    # route to expert 0 ⇒ output = expert_0(x) = a standard dense
    # FFN (with the same squared_relu/swiglu/etc. variant the
    # baseline would have used). With `use_switch_ffn=False`
    # (default) the `SwitchFFN` module is never built and the
    # baseline FFN path is bit-identical. See `models/switch_ffn.py`
    # and `autoresearch/ideas/146-sparse-ffn/idea.md`.
    use_switch_ffn: bool = False
    n_ffn_experts: int = 4
    expert_capacity_factor: float = 1.25

    # 145 — Expert-Choice MoE (Zhou, Lei, et al. 2022,
    # arXiv:2202.09368). Inverted routing direction vs Switch FFN:
    # each expert picks its own top-k tokens (k = ceil(N/E)) instead
    # of each token picking its top-1 expert. Load balance is by
    # construction — every expert processes exactly k tokens — so
    # NO auxiliary load-balancing loss is required. When
    # `use_expert_choice_moe=True`, swap the standard dense FFN for
    # `ExpertChoiceMoE` (E parallel full-width FFNs + a
    # `nn.Linear(d_model, n_experts)` zero-init router). Each expert
    # is full-width so the FFN-param cost multiplies by `n_moe_experts`
    # (default 4×). At step 0 the router is zero-init ⇒ all
    # expert-token scores are 0 ⇒ every expert processes the same
    # set of k tokens with uniform softmax weights ⇒ output ≈
    # uniform mean of E identically-init'd FFNs (close to a single
    # FFN but NOT byte-identical — same caveat as 117-soft-moe).
    # With `use_expert_choice_moe=False` (default) the
    # `ExpertChoiceMoE` module is never built and the baseline FFN
    # path is bit-identical. See `models/expert_choice_moe.py` and
    # `autoresearch/ideas/145-expert-choice/idea.md`.
    use_expert_choice_moe: bool = False
    n_moe_experts: int = 4

    # 149 — TTT-Linear (Sun, Yang, et al. 2024, arXiv:2407.04620,
    # §3.2). Drop-in FFN replacement: the FFN's up-projection is
    # swapped for `TTTLinear` — a per-input closed-form fast-weight
    # linear that updates its own weight from the input on the fly
    # (one Newton-style gradient step on the auto-encoding loss
    # `||W·x − x||²`). The down-projection stays a standard
    # `nn.Linear` so the FFN output side is unchanged. Per-input
    # fast weights act as a capacity multiplier: a 0.94M model with
    # per-input W_f behaves like a much larger static model in
    # expectation. The fast path costs O(B·T·out·in) extra FLOPs per
    # layer. `ttt_lr_init=0.0` (default) zero-inits the per-layer TTT
    # learning rate so `lr=0` at step 0 ⇒ `TTTLinear` short-circuits
    # to `F.linear(x, weight, b)` with the same `kaiming_uniform_`
    # weight as `nn.Linear` ⇒ the FFN is bit-identical to a vanilla
    # `SquaredReLUFeedForward` at step 0. With `use_ttt_ffn=False`
    # (default) the `TTTFeedForward` module is never built and the
    # baseline FFN path is bit-identical. See
    # `models/ttt_linear.py` and
    # `autoresearch/ideas/149-ttt-linear/idea.md`.
    use_ttt_ffn: bool = False
    ttt_lr_init: float = 0.0

    # 109 — KDA channel gate (Kimi Linear, arXiv:2510.26692): per-channel
    # *bounded* diagonal gate on the V stream of each head. KDA replaces
    # the single scalar forget/decay gate in delta-rule attention with a
    # per-channel diagonal `Γ = diag(γ_1, …, γ_d)`. In this repo's softmax
    # attention, the closest analog is a per-(head, channel) gate on V
    # before the AV product. Parametrized as a *bounded* `2·σ(g)` (not the
    # unbounded `1+g` of the closed `use_value_channel_gate`) so each
    # channel can independently amplify or dampen its own value stream
    # within `(0, 2)`. `g ∈ R^{n_heads × d_k}` zero-init ⇒ `2·σ(0) = 1.0`
    # exactly at step 0 ⇒ baseline graph bit-identical when the flag is
    # off AND when the flag is on at step 0. Categorically distinct from
    # the closed `use_value_channel_gate` (unbounded, can drift to
    # extremes) and from every active attention-side lever (021-V-residual
    # is cross-layer V, 022-softpick is the softmax swap, 024-gated-attn
    # is post-AV o_h gate, 020-FoX is post-softmax A·D). The lever is the
    # *diagonal* and *bounded* per-channel V gain. Default off → baseline
    # path bit-identical (no Parameter created, no application site taken).
    # See `autoresearch/ideas/109-kda-channel-gate/idea.md`.
    use_kda_channel_gate: bool = False

    # ============================================================================
    # Query-tweaks plan (29 experiments, 6 batches). All defaults are
    # identity/zero-init so step-0 == baseline unless the flag is on.
    # See docs/research-plans/query-tweaks/plan.md for the spec.
    # ============================================================================

    # q-side normalization (Q-only). Defaults to qk_norm_type at
    # construction (see __post_init__) so existing configs are
    # bit-identical. Sweep in Batch 4 (Q11-Q16) sets this directly.
    q_norm_type: str = ""
    # ---- Batch 1: high-signal levers ----
    # Q1 ALiBi-style per-head distance bias `scores += -m_h·(i-j)`.
    use_alibi_bias: bool = False
    # 230 Polynomial-distance ALiBi `scores -= (m_h·d + c_h·d²/L)` — a
    # per-head superset of Q1 (c_h=0 recovers linear alibi). See
    # Tiny1M3MPolyAlibiConfig and models/layers.py poly-alibi branch.
    use_poly_alibi: bool = False
    # 231 Kerple log-distance ALiBi `scores -= m_h·log(1 + r_h·d)` — a concave
    # per-head distance kernel (Chi et al. 2022). See Tiny1M3MKerpleLogConfig.
    use_kerple_log: bool = False
    # Q2 Token-conditioned per-head temperature `Q *= (1+tanh(x·w_h))`.
    use_q_temp_token: bool = False
    # Q3 Cosine attention (L2-normalize Q,K; learnable per-head τ).
    use_cosine_attn: bool = False
    # Q4 Per-channel relevance `score = Q^T diag(d_h) K` (d_h init 1).
    use_qk_bilinear: bool = False
    # ---- Batch 2: flagship + positional ----
    # Q5 Talking-heads on Q: logit-mix via learned n_h × n_h M (M=I init).
    use_talking_heads_q: bool = False
    # Q6 Per-head learnable RoPE base (Q and K share head h's θ).
    use_per_head_rope_base: bool = False
    # Q7 Partial rotary: rotate only fraction p of Q/K dims (default 1.0).
    partial_rotary_p: float = 1.0
    # ---- Batch 3: exotic ----
    # Q8 Multi-query expansion: project Q to 2·q_size, 2 attention reads, mean.
    use_q_expansion: bool = False
    # Q9 Decoupled content/position attention (DeBERTa-style).
    use_decoupled_content_pos: bool = False
    # Q10 Antisymmetric Q·K coupling via learnable skew S (init 0).
    use_antisym_qk: bool = False
    # ---- Batch 5: learnable-param zoo ----
    # Q17 Per-head bias `Q += b_h` after q_norm and RoPE (constant prior).
    use_q_per_head_bias: bool = False
    # Q18 Per-channel gain `Q *= g_d` (d_k) after RoPE.
    use_q_per_channel_gain: bool = False
    # Q19 Head×channel gain `Q *= g_hd` (n_h × d_k) after RoPE.
    use_q_hd_gain: bool = False
    # Q20 Norm-gate: per-head scalar `g_h = σ(a_h·‖x‖+b_h)` on Q.
    use_q_norm_gate: bool = False
    # Q21 Low-rank refine: `Q ← Q + (W1·x) @ W2` (zero-init, default r=8).
    use_q_lowrank_refine: bool = False
    q_lowrank_refine_rank: int = 8
    # Q22 LayerScale on Q: `Q *= (1 + ls_d)` per-channel after RoPE.
    use_q_layerscale: bool = False
    # Q23 Softplus gain: `Q *= softplus(g_h)` per-head — always ≥ 0.
    use_q_softplus_gain: bool = False
    # ---- Batch 6: architecture / mixing ----
    # Q24 Head-mix: `Q ← Q + Q @ M` (M=I init) pre-attention.
    use_q_head_mix: bool = False
    # Q25 Time-conv: `Q += conv1d(Q, k=3)` zero-init along position axis.
    use_q_time_conv: bool = False
    # Q26 EMA-smooth over position: `Q ← α·Q + (1−α)·Q_prev` (α=1 init).
    use_q_ema_smooth: bool = False
    q_ema_alpha: float = 0.0  # sigmoid'd; 0 → α=0.5 at init
    # Q27 Feature-map attention: phi(Q) @ phi(K)^T with learnable phi.
    # NOT identity-init — see plan.md note. Needs own control.
    use_q_feature_map: bool = False
    q_feature_map_hidden: int = 64
    # Q28 Per-token RoPE: each token's θ via small MLP (default hidden=32).
    use_q_per_token_rope: bool = False
    q_per_token_rope_hidden: int = 32
    # Q29 Noise reg: `Q += N(0, σ²)` in training only (learnable σ).
    use_q_noise_reg: bool = False

    # Base Training Defaults
    seed: int = 42  # seeds model init AND data order; override via --seed
    device: str = "auto"  # auto, cuda, mps, or cpu
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 8000000
    
    # Learning Rate (Aggressive for pre-training)
    muon_lr: float = 0.024
    muon_momentum: float = 0.95
    adamw_lr: float = 0.006
    # SWAN (Ma et al. 2024/2025, arXiv:2412.13148): stateless whitening
    # on matrix gradients. Reuses Muon's lr for the matrix slot; the
    # algorithm has no momentum buffer.
    use_swan: bool = False
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"
    # 112 — Lookahead Optimizer Wrapper (Zhang et al. 2019, arXiv:1907.08610).
    # Wraps the *list* of inner optimizers (Muon, AdamW, ...): every k inner
    # steps, pull slow weights halfway toward fast weights and reset fast to
    # slow. Also clears the inner optimizer's momentum buffers so the next
    # inner step doesn't see stale gradients from before the slow reset.
    # k=5, alpha=0.5 are the paper's defaults. With use_lookahead=False
    # (default) the wrapper is fully inert → baseline path bit-identical.
    # Identity at step 0: slow = theta_init, first inner step uses the
    # baseline Muon/AdamW path unchanged.
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    # 116 — Hyper-Connections (mHC, Xie et al. 2024, arXiv:2409.19606):
    # multi-stream residual that splits `d_model` into `hc_n_resid`
    # parallel streams of width `d_l = d_model // hc_n_resid`, each mixed
    # via per-position `(A_l, B_l, C_l) ∈ R^{n_resid × n_resid}`. Identity
    # init (A=B=C=I) ⇒ streams don't mix at step 0 ⇒ baseline forward
    # graph is bit-identical to the pre-norm residual path (B=A=C=I
    # reduces to `block(x)`). Default off → baseline path bit-identical.
    # Cost: 3·n_resid² × n_layers scalars (288 at tiny1m3m with n_resid=4,
    # negligible). See `autoresearch/ideas/116-hyper-connections/idea.md`.
    use_hyper_connections: bool = False
    hc_n_resid: int = 4
    # 150 — Cross-Layer Feedback Attention (Holtzman et al. 2020,
    # Feedback Transformer, arXiv:2002.09402; lean "previous K=2 layers"
    # variant). Each block reads from a small cache of the previous K
    # blocks' pre-FFN residual states via a `XLayerCrossAttn` head, and
    # adds the result as a gated residual branch. Per-block scalar
    # `xlayer_gate = nn.Parameter(torch.zeros(1))` ⇒ contribution is
    # exactly 0 at step 0 ⇒ baseline forward is bit-identical. K=2 by
    # default (the spec pin — the spec also lets K=4 / 8 be tested).
    # The cross-attn head is single-head with `head_dim=16` to keep
    # params compact at 0.94M: per-block param overhead is
    # 2·d_model·16 + 2·d_model² = 8.2K at d_model=64, ≈10% of the
    # 0.94M budget across 12 blocks. With `use_xlayer_feedback=False`
    # (default) the cross-attn module is never built, the per-block
    # gate is not allocated, and the baseline path is bit-identical.
    # See `models/xlayer_attn.py` and
    # `autoresearch/ideas/150-xlayer-feedback/idea.md`.
    use_xlayer_feedback: bool = False
    xlayer_k: int = 2
    # 115 — R-Drop: Regularized Dropout for Neural Networks
    # (Liang et al. 2021, arXiv:2106.14448, NeurIPS 2021). Run the model
    # forward twice per step with different dropout masks, average the two
    # CE losses, and add `rdrop_alpha · 0.5·(KL(p_1‖p_2)+KL(p_2‖p_1))`
    # to pull the model's logits toward dropout-invariance. `rdrop_alpha`
    # is linearly warmed from 0 → target over `rdrop_warmup_steps` so at
    # step 0 the loss is the (mean of two) CE only — bit-identical to the
    # single-CE baseline modulo the doubled forward (which is runtime,
    # not math). With `use_rdrop=False` (default) the trainer takes the
    # single-forward path → byte-identical to baseline. See
    # `autoresearch/ideas/115-rdrop/idea.md`.
    use_rdrop: bool = False
    rdrop_alpha: float = 1.0   # target alpha; paper sweeps 1.0–5.0
    rdrop_warmup_steps: int = 1000  # step-0 invariance: alpha=0 here
    # 110 — Model-Weight EMA (Polyak-Ruppert averaging, Polyak 1990;
    # used in RoBERTa, MAE, MoCo v3, modded-nanogpt speedrun SWA).
    # Maintain a shadow copy `θ_ema ← μ·θ_ema + (1−μ)·θ` updated each
    # step. `μ` ramps linearly from 0 to `ema_decay` over the first
    # `ema_warmup_steps` ⇒ step-0 EMA = live θ ⇒ step-0 val byte-
    # identical to baseline. `ema_eval_only=True` (default) means the
    # live `θ` is the saved/resumed model and the EMA is *only* swapped
    # in for the val pass; training and checkpointing stay on the live
    # trajectory. With `use_ema_eval=False` (default) the trainer does
    # no shadow copy and the baseline path is bit-identical. See
    # `autoresearch/ideas/110-weight-ema/idea.md`.
    use_ema_eval: bool = False
    ema_decay: float = 0.999
    ema_warmup_steps: int = 100
    ema_eval_only: bool = True
    # 119 — SAM: Sharpness-Aware Minimization
    # (Foret et al. 2020, arXiv:2010.01412, ICLR 2021). Wraps the
    # 1-D / embedding / norm AdamW path with an adversarial ascent
    # step `w ← w + ρ · ∇L(w) / ‖∇L(w)‖` followed by descent at the
    # perturbed point. The Muon 2-D path is unchanged — SAM only
    # applies to the AdamW bucket (per-paper default for Adam-SAM).
    # At step 0 the perturbation is non-zero (O(ρ) along the
    # gradient direction), so the first-step gradient differs from
    # AdamW by O(ρ). With `rho = 0.0` SAM collapses to AdamW (the
    # first_step is a no-op, the second_step is parent's step on
    # the same grad) — the flag-off path stays bit-identical. With
    # `use_sam=False` (default) the trainer uses plain
    # `torch.optim.AdamW` unchanged. See `optimizers/sam.py` for
    # the mechanism and `autoresearch/ideas/119-sam/idea.md` for
    # the bet.
    use_sam: bool = False
    sam_rho: float = 0.05
    # 138 — LookSAM: Periodic Sharpness-Aware Minimization (Du et al.
    # 2022, ICLR 2023, arXiv:2205.13539). Compute-efficient variant of
    # SAM (119): the SAM-style 2-backward ascent-descent step fires
    # only every K steps; the K-1 steps in between are plain AdamW.
    # With paper default K=5, effective compute is ~1.2x (vs. SAM's
    # 2x) at ~80% of the flatness benefit. Mutex with `use_sam`: if
    # both are on, `use_sam` wins (full SAM is the more aggressive
    # variant). With `use_looksam=False` (default) the trainer uses
    # plain `torch.optim.AdamW` unchanged — the LookSAM class is
    # never instantiated, baseline path bit-identical. Identity at
    # step 0: with K=5 the first 4 steps are plain AdamW
    # (`step_count=0..3`, `next_is_sam=False`); the first SAM step
    # fires at `step_count=4`. So LookSAM is *more* bit-identical
    # at step 0 than full SAM (119), which always runs the SAM
    # ascent on the first step. See `optimizers/looksam.py` and
    # `autoresearch/ideas/138-looksam/idea.md`.
    use_looksam: bool = False
    looksam_k: int = 5
    looksam_rho: float = 0.05
    # 121 — Prodigy: An Expeditiously Adaptive Parameter-Free Optimizer
    # (Mishchenko & Defazio 2023, arXiv:2306.06101, NeurIPS 2023 L4DC /
    # COLT 2024). Successor to D-Adaptation (120): smooth *continuous*
    # Adam-style gradient similarity `s_t = ⟨sign(g_t/√v_t), sign(g_{t-k}/√v_{t-k})⟩`
    # feeds `D ← D · exp(β3·s_t)` — eliminating D-Adaptation's noisy
    # binary ramp-up. Plus a *displacement-based* warm-start: the first
    # `prodigy_warmup_steps` (default 10) steps are unit-LR AdamW and
    # `D_0` is set to `‖w_0 − w_k‖ / k` — the natural step size for the
    # measured trajectory, no hand-tuned guess. `prodigy_d0` is the
    # warm-start D scalar (paper default 1.0; *not* the production LR —
    # the production LR is `D_t`, which Prodigy discovers). `beta3` is
    # the D-update coefficient η (paper default 0.01; bounded per-step
    # multiplicative change in [exp(-0.01), exp(0.01)] ≈ [0.99, 1.01]).
    # Identity at step 0: the first `warmup_steps` calls are unit-LR
    # AdamW (i.e. `D_0 = d0` is the multiplier on the AdamW update),
    # so they are NOT bit-identical to AdamW with `adamw_lr` — this is
    # the lever. After warmup, D jumps to the measured displacement
    # and the LR-discovery loop engages. With `use_prodigy=False`
    # (default) the trainer uses `torch.optim.AdamW` unchanged — the
    # Prodigy class is never instantiated. See `optimizers/prodigy.py`
    # and `autoresearch/ideas/121-prodigy/idea.md`.
    use_prodigy: bool = False
    prodigy_d0: float = 0.01
    prodigy_warmup_steps: int = 10
    prodigy_beta3: float = 0.01
    prodigy_d_max: float = 1.0     # paper §3.1 default; upper clamp on D.
                                    # Without this, D grows as e^t per step
                                    # (~1e40 by step 92) and explodes on
                                    # the first small-gradient plateau. The
                                    # re-code uses d0=0.01 *and* d_max=1.0
                                    # for defense in depth (the previous
                                    # d0=1.0 caused a 12.01 → 10348 blowup
                                    # at step 25 of the 2026-06-13 GPU run).
    prodigy_min_d: float = 1e-6    # lower clamp on D (prevents collapse on
                                    # sign-disagreement spike).
    prodigy_update_clip: float = 1.0  # per-param max-norm on
                                       # delta = eff_lr · adam_update. Final
                                       # safety net against a too-large
                                       # eff_lr from a discovery-loop spike.
    # 113 — GaLore: Gradient Low-Rank Projection
    # (Zhao et al. 2024, arXiv:2403.03507, NeurIPS 2024). For each 2-D
    # weight matrix, project the gradient into a rank-`galore_rank`
    # subspace via orthonormal P, Q, run AdamW in the r×r projected
    # space, then project the update back. AdamW state is r×r instead
    # of n×m (memory win, moot at 0.94M). Every `galore_proj_every`
    # steps, P, Q are refreshed from the SVD of a running gradient
    # EMA. Routes ONLY the 2-D non-embed, non-norm slot; 1-D / embed
    # / norm stay on plain AdamW. The forward graph is unchanged, so
    # val_loss at step 0 (computed before any optimizer step) is
    # bit-identical to baseline. The first optimizer step itself
    # differs from AdamW's first step (it operates on a rank-r
    # projection), which is the inherent behavior of GaLore. With
    # `use_galore=False` (default) the trainer's existing Muon path
    # is unchanged. See `autoresearch/ideas/113-galore/idea.md`.
    use_galore: bool = False
    galore_rank: int = 4           # projection rank r (paper sweet spot 4-256)
    galore_proj_every: int = 200   # SVD basis refresh cadence (paper default)
    galore_lr: float = 0.006       # matches adamw_lr; tune in tandem if at all
    galore_beta1: float = 0.9
    galore_beta2: float = 0.999
    galore_eps: float = 1e-8
    # Cautious Muon (Liang et al. 2024, arXiv 2411.16085): one-line sign-mask
    # on the orthogonalized update — zero out components whose sign disagrees
    # with the current gradient. Suppresses stale-momentum artifacts. Bit-
    # identical to baseline when False (default). When True, the masked
    # components reduce effective step norm ~10-20% on average; pair with
    # a small muon_lr bump (e.g. 0.024 → 0.025) to compensate. Applies only
    # to the Muon path; AdamW is unchanged (separate flag `use_cautious_adamw`
    # if/when we add it). See docs/research/muon/cautious-muon/plan.md.
    use_cautious_muon: bool = False
    # Cautious AdamW (Liang et al. 2024, arXiv 2411.16085): same sign-mask
    # as Cautious Muon, applied to the AdamW path (1D / embedding / head).
    # Selects WHICH AdamW bucket(s) the mask fires on (the AdamW path is
    # independent of Muon — `use_cautious_muon` does NOT affect it).
    # Allowed values: "none" (default — bit-identical baseline AdamW),
    # "embedding" (mask on `token_embedding` + `emb_proj` only),
    # "gain" (mask on `*.norm.weight` + 1D scalars), "all" (mask every
    # AdamW param). See autoresearch/ideas/002-cautious-adamw/plan.md.
    use_cautious_adamw: str = "none"
    # SOAP (Vyas et al. 2024, arXiv 2409.11321): Adam in the eigenbasis
    # of the Shampoo preconditioner, with periodic basis refresh. Routes
    # ONLY the 2D non-Muon AdamW params (`token_embedding.weight`,
    # `emb_proj.weight`, `out_proj.weight`) to SOAP; 1D scalars and
    # `*.norm.weight` stay on plain AdamW (eigendecomp is meaningless on
    # 1D). Default off → bit-identical to baseline. Pair with the
    # bf16 pre-flight in `autoresearch/ideas/003-soap/plan.md` before
    # the full screen20m run. See autoresearch/ideas/003-soap/idea.md.
    use_soap: bool = False
    use_soap_precondition_freq: int = 10
    # Schedule-Free AdamW (Defazio et al. 2024, arXiv:2405.15682): eliminates
    # the LR schedule by maintaining a Polyak-Ruppert average alongside the
    # gradient-following iterate. Drop-in replacement for the AdamW path only;
    # Muon path is unchanged. Default off → bit-identical to baseline AdamW.
    # When True, the AdamW optimizer's LR scheduler is set to constant (the
    # averaging handles late-training stabilization). See
    # autoresearch/ideas/006-schedule-free-adamw/plan.md.
    use_schedule_free_adamw: bool = False
    # 120 — D-Adaptation (Defazio 2023, arXiv:2301.11933 / arXiv:2201.11941,
    # ICML 2023). Eliminates the learning-rate knob by maintaining a log-scale
    # running lower bound `D` on the distance from `w_init` to `w_optimal` and
    # deriving the effective LR as `lr_t = D_t / ‖g_t‖`. The 1st/2nd moments
    # of AdamW are retained intact — only the outer LR scaling is replaced.
    # Routes ONLY the 1-D / embedding / norm / head path to `DAdaptAdamW`; the
    # Muon 2-D path is unchanged (D-Adapt is ortho to Muon, lives only on the
    # AdamW bucket). At step 0 `D = 1e-6` warm-start ⇒ `lr_0 ≈ 1e-6 / ‖g_0‖`
    # (essentially zero); after ~10–20 steps `D` reaches a typical AdamW-
    # equivalent value. This first-step ramp-up is the lever's signature, not
    # a bug. Default off → trainer uses plain `torch.optim.AdamW` unchanged,
    # baseline path bit-identical. See `optimizers/dadaptation.py` for the
    # mechanism and `autoresearch/ideas/120-dadaptation/idea.md` for the bet.
    use_dadapt: bool = False
    dadapt_d0_lr: float = 1.0     # η, log-LR update constant (paper default 1.0)
    dadapt_min_lr: float = 0.0    # lower clamp on D (paper default 0.0)
    dadapt_d_max: float = 1.0     # upper clamp on D (paper §3.1, default 1.0).
                                  # Also caps the derived lr_t = D/‖g_t‖.
                                  # Required for stability at tiny1m3m — without
                                  # this `D` grows as e^t per step (~1e40 by
                                  # step 92) and explodes on the first small-
                                  # gradient plateau (val 10.81 → 36.89 → 7e15).
    dadapt_eps: float = 1e-8      # floor for lr_t = D/‖g_t‖ (also Adam eps)
    # 114 — MARS: Variance-Reduced AdamW (Yuan et al. 2024,
    # arXiv:2401.03855). Subclass of AdamW that adds a lag-based
    # variance-reduction correction `g̃_t = g_t + mix_coef *
    # (m_{t-lag} − m_{t-2*lag})` to the *gradient* passed to AdamW.
    # Per-parameter `v` is untouched; only the gradient input is
    # modified. Ring buffer of past `exp_avg` snapshots of length
    # `2*lag` is maintained per param. Identity at step 0: the
    # buffer is empty for the first `2*lag` steps ⇒ correction
    # undefined ⇒ g̃_t = g_t ⇒ bit-identical to plain AdamW. Paper
    # default `lag=10`, `mix_coef=0.5`; `lr_scale=1.0` (paper does
    # not require LR re-tuning). With `use_mars=False` (default)
    # the trainer uses `torch.optim.AdamW` unchanged — the
    # MARSAdamW class is never instantiated. See
    # `autoresearch/ideas/114-mars/idea.md`.
    use_mars: bool = False
    mars_lag: int = 10
    mars_mix_coef: float = 0.5
    mars_lr_scale: float = 1.0
    # RetNet retention kernel (Sun et al. 2023, arXiv 2307.08621):
    # per-head learnable decay γ_h replaces softmax attention with a
    # linear-recurrence kernel. v1 = kernel + synthetic probe only
    # (`models/retention.py` + `tests/test_retention.py`); v2 will
    # wire it into `MultiHeadAttention.forward` as a separate PR.
    # Default off → baseline path bit-identical. See
    # autoresearch/ideas/004-retnet-retention/plan.md.
    use_retention: bool = False
    # Lion optimizer (Chen et al. 2023, arXiv:2302.06675): sign-based
    # optimizer that replaces Muon on the 2-D non-embedding, non-norm
    # routing slot when use_lion=True. Default off → Muon path is
    # bit-identical. `lion_lr=3e-4` matches Chen et al.'s default at
    # much larger scale — do not change without sweeping. `use_lion` and
    # the Muon path are mutually exclusive: enabling Lion routes the
    # Muon 2-D slot to Lion (no parallel Muon instance is created).
    # 1-D / embedding / head stay on AdamW — Lion's fixed-LR sign update
    # is known to diverge on the embedding (Chen et al. 2023 §5). See
    # autoresearch/ideas/011-cautious-lion/plan.md.
    use_lion: bool = False
    lion_lr: float = 3e-4
    lion_beta1: float = 0.9
    lion_beta2: float = 0.98
    # Cautious-Lion (Liang et al. 2024, arXiv:2411.16085) — the Cautious
    # sign-mask generalized to Lion. After computing `update = sign(c)`,
    # zero out components whose sign disagrees with the current gradient
    # and rescale by `1 / mask.mean().clamp(min=0.1)` to keep the effective
    # LR constant. Default off → bare Lion, bit-identical to the
    # use_lion=True baseline. Only takes effect when use_lion=True; the
    # `use_cautious_lion` flag is gated by the trainer so it cannot fire
    # on the AdamW path. See autoresearch/ideas/011-cautious-lion/plan.md.
    use_cautious_lion: bool = False
    # Tiger optimizer (Chen et al. 2024, arXiv:2401.16691): sign-based
    # optimizer with per-parameter magnitude EMA — `update = m / (√v + ε)`
    # where `m` is the gradient EMA (β1=0.9) and `v` is the EMA of |g|
    # (β2=0.999). Distinct from Lion (which has unit-magnitude sign
    # updates); Tiger's per-parameter magnitude EMA gives a tighter LR
    # sensitivity and matches AdamW at ~5-10x lower LR (paper).
    # Replaces Muon on the 2-D non-embedding, non-norm routing slot
    # when `use_tiger=True`. Default off → Muon path is bit-identical.
    # 1-D / embedding / head stay on AdamW — Tiger's sign-stable but
    # magnitude-scaled update can be aggressive on the embedding
    # (paper §5.2 recommends AdamW for embedding). Cold-start with
    # `m_0 = 0`, `v_0 = 0` ⇒ first step is `update = 0/ε = 0` ⇒ no
    # parameter change at step 0 ⇒ byte-identical to baseline at
    # step 0 (no paper warmstart `v_0 = |g_0|`, which would shift
    # the first step to a unit sign step). `tiger_lr=1e-3` matches
    # `adamw_lr / 6` (paper-recommended for tiny models). See
    # `autoresearch/ideas/122-tiger/idea.md`.
    use_tiger: bool = False
    tiger_lr: float = 1e-3
    tiger_beta1: float = 0.9
    tiger_beta2: float = 0.999
    tiger_eps: float = 1e-8
    # 123 — CAME: Confidence-guided Adaptive Memory Efficient
    # Optimization (Luo et al. 2023, arXiv:2307.02085, NeurIPS 2023).
    # AdamW replacement for the 1-D / embedding / norm / head path
    # when `use_came=True`. The update is
    #     m_t = β1·m_{t-1} + (1−β1)·g_t
    #     v_t = β2·v_{t-1} + (1−β2)·g_t²
    #     res_t = (m_t − g_t) / (√v_t + ε)
    #     conf_t = max(res_t, 0) + ε
    #     update = m_t / (√v_t + ε) · conf_t / (|m_t| + ε)
    # — i.e. a confidence-rescaled AdamW where the rescaling
    # down-weights updates when the gradient agrees with the
    # running momentum (residual small) and applies a residual-
    # shaped step when they disagree. Cold-start `m_0 = 0`,
    # `v_0 = 0` ⇒ first-step residual is negative, clipped to
    # 0, confidence = ε, update ≈ 0 ⇒ byte-identical to baseline
    # at step 0. Default off → AdamW path unchanged, baseline
    # bit-identical. See `optimizers/came.py` for the mechanism
    # and `autoresearch/ideas/123-came/idea.md` for the bet.
    use_came: bool = False
    came_lr: float = 0.006
    came_beta1: float = 0.9
    came_beta2: float = 0.999
    came_eps: float = 1e-8
    # Per-element magnitude clip on the raw `update` before the LR
    # scaling in `optimizers/came.py`. Bounds any single step's
    # per-element displacement to `±came_update_clip · lr`. Protects
    # against the `m̂ / ε² ≈ 1e16` blowup when `v̂ ≈ 0` and `m̂` is
    # non-trivial (the 2026-06-13 GPU divergence — val loss 10.81 →
    # 6.79e7 at step 25). Default `10.0` is well above the natural
    # ~1.0 per-element magnitude on a healthy trajectory, so it is
    # effectively inactive on a normal Adam-like regime and only
    # triggers in the runaway-`v̂`-zero blowup case. See
    # `autoresearch/ideas/123-came/idea.md` for the post-mortem.
    came_update_clip: float = 10.0
    # 124 — RAdam: Rectified Adam (Liu et al. 2019, arXiv:1908.03265,
    # ICLR 2020). Replaces the AdamW 1-D / embedding / norm / head
    # path with `RAdam` when `use_radam=True`. The 2-D Muon path is
    # unchanged (RAdam is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME). The update applies
    # a variance-bounded correction `ρ_t` to Adam's bias-corrected
    # step: when the variance of `1/(1−β2^t)` is high (early steps),
    # RAdam falls back to an SGD-only `m̂_t` step (no `v̂_t`); once
    # `ρ_t > 4` (≈ `t > 4/(1−β2)`) it switches to the full Adam-
    # normalized update with the variance-aware `√ρ_t` rescale. This
    # *removes the manual warmup knob* — RAdam auto-detects when the
    # effective LR is safe. At step 0 (t=1) `ρ_1 ≪ 4` ⇒ SGD-fallback
    # path ⇒ `update = (1−β1)·g_0`. NOT bit-identical to AdamW's first
    # step (which uses the full Adam-normalized update), but the
    # magnitude is comparable (O(β1) smaller). This first-step
    # divergence is the lever, not a bug. With `use_radam=False`
    # (default) plain `torch.optim.AdamW` is used — baseline
    # bit-identical. `radam_lr=0.006` matches `adamw_lr` (paper does
    # not require re-tuning). See `optimizers/radam.py` for the
    # mechanism and `autoresearch/ideas/124-radam/idea.md` for the bet.
    use_radam: bool = False
    radam_lr: float = 0.006
    radam_beta1: float = 0.9
    radam_beta2: float = 0.999
    radam_eps: float = 1e-8
    # 126 — AdaShift: Decorrelated Adam via Delayed Gradients
    # (Zhou et al. 2019, arXiv:1810.00143, NeurIPS 2019 workshop).
    # Replaces the AdamW 1-D / embedding / norm / head path with
    # `AdaShift` when `use_adashift=True`. The 2-D Muon path is
    # unchanged (AdaShift is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam). The
    # update uses a *delayed* gradient `g_{t-n}²` for the 2nd
    # moment, decorrelating `v_t` from `m_t` (which both use `g_t`):
    #     m_t = β1·m_{t-1} + (1-β1)·g_t
    #     v_t = β2·v_{t-1} + (1-β2)·g_{t-n}²
    #     update = m̂_t / (√v̂_t + ε)
    # Per-parameter state keeps a queue of past `n` gradients
    # (clones, fp32, length bounded by n). The paper's
    # warm-start `v_0 = g_0²` is used on the first step so
    # `v_1 = β2·g_0²` — NOT bit-identical to AdamW's first step
    # (`v_1 = (1-β2)·g_0²`) but same magnitude order (O(β2)
    # different). The first-step displacement is the lever, not a
    # bug. With `n = 0` AdaShift collapses to AdamW; the
    # `adashift_n = 3` default is the paper's recommended delay.
    # With `use_adashift=False` (default) plain `torch.optim.AdamW`
    # is used — baseline path bit-identical. See
    # `optimizers/adashift.py` for the mechanism and
    # `autoresearch/ideas/126-adashift/idea.md` for the bet.
    use_adashift: bool = False
    adashift_lr: float = 0.006
    adashift_beta1: float = 0.9
    adashift_beta2: float = 0.999
    adashift_eps: float = 1e-8
    adashift_n: int = 3
    # 135 — Adan: Adaptive Nesterov Momentum with N-Step Lookback
    # (Xie et al. 2022, arXiv:2208.06677, TPAMI 2022 / ICLR 2023
    # workshop). Replaces the AdamW 1-D / embedding / norm / head
    # path with `Adan` when `use_adan=True`. The 2-D Muon path is
    # unchanged (Adan is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam,
    # 126-AdaShift, 127-GC, 128-SD). The mechanism (paper Algorithm
    # 1) combines (1) a 1-step first moment, (2) an N-step lookback
    # variance estimate, and (3) a Nesterov-style extrapolated
    # gradient:
    #     g_la = g_t + β_la · (g_t − g_{t−1})
    #     m_t = β1·m + (1−β1)·g_la
    #     v_t = β2·v + (1−β2)·mean(g_{t..t-N+1}²)
    #     update = m_t / (√v_t + ε)         (no bias correction)
    # `adan_n_lookback=4` is the paper's default N. `adan_lookahead_beta=0.5`
    # is the paper's default Nesterov coefficient. At step 0
    # `prev_grad=None` ⇒ lookahead term falls back to `g_0` ⇒
    # `m_1 = (1−β1)·g_0`, `v_1 = (1−β2)·g_0²` (queue length 1) ⇒
    # `update_0 = g_0 / (|g_0| + ε) ≈ sign(g_0)`. NOT bit-identical
    # to AdamW's first step (which uses bias-corrected
    # `m̂/√v̂`), but the magnitudes are similar — the first-step
    # displacement is the lever's signature. The N=4 lookback ramps
    # in over the first 4 steps. With `use_adan=False` (default) the
    # `Adan` class is never instantiated and the trainer uses
    # `torch.optim.AdamW` unchanged. See `optimizers/adan.py` and
    # `autoresearch/ideas/135-adan/idea.md`.
    use_adan: bool = False
    adan_lr: float = 0.006
    adan_beta1: float = 0.9
    adan_beta2: float = 0.999
    adan_eps: float = 1e-8
    adan_lookahead_beta: float = 0.5
    adan_n_lookback: int = 4
    # 140 — Sophia: Scalable Stochastic Second-order Optimizer
    # (Liu, Wang, et al. 2023, arXiv:2305.14342, ICML 2023). Replaces
    # the AdamW 1-D / embedding / norm / head path with `Sophia` when
    # `use_sophia=True`. The 2-D Muon path is unchanged (Sophia is an
    # AdamW replacement, like 114-MARS, 119-SAM, 121-Prodigy, 135-Adan).
    # The mechanism is the diagonal-Hessian-aware update
    #     m_t  = β1·m + (1−β1)·g_t
    #     h_t  = β2·h + (1−β2)·h_hat_t         (h_hat sampled every k)
    #     update = clip(g, ±ρ) / max(h, ε)
    #     θ_t  = θ_{t−1} − lr·(update + λ·θ_{t−1})   (decoupled WD)
    # The diagonal Hessian is sampled via Hutchinson's trace
    # estimator: `u ~ Rademacher(±1)` per parameter, then
    # `h_hat = u · ∇(g·u)`, computed by an extra backward on the
    # scalar `g·u` (one extra backward every `sophia_hessian_freq`
    # steps — paper default 10, so ~1.1× amortized backward cost at
    # 92 update steps). The trainer handles the extra backward; see
    # `training/trainer.py` for the wiring. Defaults match the
    # paper's 125M model (lr=6e-3, β1=0.965, β2=0.99, ρ=0.04) with
    # a per-parameter `update_clip=1.0` safety guard that bounds
    # the cold-start `h_t≈0` amplification to a single AdamW-
    # magnitude step. With `use_sophia=False` (default) the
    # `Sophia` class is never instantiated and the trainer uses
    # plain `torch.optim.AdamW` unchanged — baseline path
    # bit-identical. See `optimizers/sophia.py` for the mechanism
    # and `autoresearch/ideas/140-sophia/idea.md` for the bet.
    use_sophia: bool = False
    sophia_lr: float = 6e-3
    sophia_beta1: float = 0.965
    sophia_beta2: float = 0.99
    sophia_eps: float = 1e-8
    sophia_rho: float = 0.04
    sophia_hessian_freq: int = 10
    sophia_update_clip: float = 1.0
    # 136 — AdaPNM: Adaptive Positive-Negative Momentum
    # (Ding, Zhou, Zhu, Ye, Jiao 2019, arXiv:1906.01520, NeurIPS 2019).
    # Replaces the AdamW 1-D / embedding / norm / head path with
    # `AdaPNM` when `use_adapnm=True`. The 2-D Muon path is
    # unchanged (AdaPNM is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam,
    # 126-AdaShift, 135-Adan, 127-GC, 128-SD). The mechanism
    # maintains TWO parallel momentum buffers — one for the
    # positive part of the gradient `m+_t = β1·m+_{t-1} +
    # (1−β1)·max(g_t, 0)` and one for the negative part
    # `m-_t = β1·m-_{t-1} + (1−β1)·max(-g_t, 0)`. The combined
    # direction `m_t = m+_t − m-_t` is algebraically equal to
    # the standard EMA `β1·m_{t-1} + (1−β1)·g_t` because
    # `max(g, 0) − max(-g, 0) = g` element-wise — the lever's
    # factored-state trick preserves AdamW's update direction
    # while opening the door to future per-side processing.
    # The 2nd moment `v_t = β2·v_{t-1} + (1−β2)·g_t²` is
    # standard Adam-style. Cold-start `m+_0 = m-_0 = v_0 = 0`
    # ⇒ first-step update = `(1−β1)·g_0 / (√((1−β2)·g_0²) + ε)`,
    # approximately equal to AdamW's first step (within an
    # `O(β1)` factor — AdamW applies bias correction `m̂_1 =
    # m_1 / (1−β1)`, AdaPNM does not). With `use_adapnm=False`
    # (default) plain `torch.optim.AdamW` is used — baseline
    # path bit-identical. See `optimizers/adapnm.py` and
    # `autoresearch/ideas/136-adapnm/idea.md`.
    use_adapnm: bool = False
    adapnm_lr: float = 0.006
    adapnm_beta1: float = 0.9
    adapnm_beta2: float = 0.999
    adapnm_eps: float = 1e-8
    # 127 — Gradient Centralization (Yong et al. 2020, arXiv:2004.01461,
    # ICONIP 2020). Pre-step hook that subtracts the mean from each
    # gradient matrix before the AdamW update runs. For 2-D weight
    # `W ∈ R^{n×m}` the mean is taken along `gc_axis` (default 1, the
    # output axis), giving each output neuron zero-mean input
    # gradient. For 4-D conv weights, the mean is taken per-filter
    # over the spatial axes. The transform is `g ← g − mean(g,
    # dim=axis)` — a single linear operator that removes the
    # constant component without changing the variance.
    # Compositional: when `use_gc=True` and no specific AdamW
    # replacement is active, the trainer routes AdamW-eligible params
    # through `GCAdamW` (subclass of `torch.optim.AdamW`). The per-
    # parameter `(m, v)` state is untouched — only the gradient
    # input is centered. The forward graph is unchanged, so `val_loss`
    # at step 0 (computed before any optimizer step) is bit-identical
    # to baseline. The first optimizer step itself differs from
    # AdamW's first step (the centered gradient has zero mean per
    # output neuron, removing the constant component that AdamW's
    # first step otherwise sees) — this is the lever's signature, not
    # a bug. With `use_gc=False` (default) plain `torch.optim.AdamW`
    # is used — baseline path bit-identical. See
    # `optimizers/grad_centralization.py` for the mechanism and
    # `autoresearch/ideas/127-grad-centralization/idea.md` for the bet.
    use_gc: bool = False
    gc_axis: int = 1
    # 125 — PSGD: Preconditioned Stochastic Gradient Descent
    # (Li, Chen, Milenkovic, Giannakis 2024, arXiv:2405.13856,
    # NeurIPS 2024). The most recent (NeurIPS 2024) high-quality
    # optimizer paper with explicit ≥100M-scale LM wins (GPT-2
    # small/medium/large match or beat AdamW at same compute).
    # Replaces Muon on the 2-D non-embedding, non-norm routing slot
    # when `use_psgd=True`. PSGD learns an online preconditioner
    # that whitens the gradient per axis. For 2-D W ∈ R^{n×m}:
    #     P ← P + α · (g g^T / m − I)        (n×n)
    #     Q ← Q + α · (W W^T / n − I)        (m×m)
    #     update = P · g · Q                  (whitened step)
    #     w ← w − lr · (β·m_prev + (1−β)·update)
    # For 1-D params (norms, biases, embeddings): diagonal `D` with
    # `D ← D + α · (g² − 1)` and `update = D · g`. The 1-D / embedding
    # / norm slot stays on AdamW per the paper's default (we keep
    # the same routing as Muon / Lion / Tiger / GaLore). `psgd_alpha`
    # is the preconditioner EMA rate (paper default 1e-3). `psgd_beta`
    # is the momentum coefficient (paper default 0.9). At α=0 PSGD
    # collapses to SGD-with-momentum. At step 0 (P=I, Q=I, m=0) the
    # first update is `I · g · I = g` and the first step is
    # `w ← w − lr · g` (SGD, not AdamW — the lever's first-step
    # signature). With `use_psgd=False` (default) the `PSGD` class
    # is never instantiated and the trainer uses the existing Muon
    # path bit-identically. See `optimizers/psgd.py` for the
    # mechanism and `autoresearch/ideas/125-psgd/idea.md` for the bet.
    use_psgd: bool = False
    psgd_lr: float = 0.01
    psgd_alpha: float = 1e-3
    psgd_beta: float = 0.9
    # 128 — Spectral Decoupling (Yong, Pehlivan, Morariu, Tsang 2022,
    # arXiv:2202.05380, NeurIPS 2022). Replaces the AdamW 1-D /
    # embedding / norm / head path with `SDAdamW` — a thin subclass
    # of `torch.optim.AdamW` that projects each per-param gradient
    # off the weight direction (`g ← g − (⟨g,w⟩/‖w‖²)·w`) before
    # delegating to AdamW's `.step()`. Decoupled WD `λ·w` is
    # unchanged (it acts along w — magnitude shrinkage is preserved).
    # The 2-D Muon path is unchanged (SD is an AdamW replacement, like
    # 119-SAM, 120-DAdapt, 121-Prodigy, 114-MARS, 123-CAME, 124-RAdam,
    # 126-AdaShift, 127-GC). Identity at step 0: with symmetric inits
    # `⟨g_0, w_0⟩` is small but nonzero, so the projection removes
    # an `O(1/n)` component of `g_0`. NOT bit-identical to AdamW's
    # first step (small `O(1/n)` correction), but the deviation is
    # bounded and well within the NULL band. `sd_lambda=1.0` is the
    # paper's full projection. `sd_lambda=0.0` collapses SD to
    # plain AdamW (the projection is inert). With `use_sd=False`
    # (default) plain `torch.optim.AdamW` is used — baseline path
    # bit-identical. See `optimizers/spectral_decoupling.py` and
    # `autoresearch/ideas/128-spectral-decoupling/idea.md`.
    use_sd: bool = False
    sd_lambda: float = 1.0
    # 137 — AdamP: Adam with Projection-Based Update
    # (He, Liu, Mao, Chen, Zhang 2020, arXiv:2006.08217, NeurIPS
    # 2020). Replaces the AdamW 1-D / embedding / norm / head path
    # with `AdamP` when `use_adamp=True`. The 2-D Muon path is
    # unchanged (AdamP is an AdamW replacement, like 114-MARS,
    # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam,
    # 126-AdaShift, 127-GC, 128-SD). The mechanism projects the
    # Adam update `Δ = m̂/√v̂` onto the orthogonal complement of
    # `w` (removes the component of Δ along w, leaving only the
    # perpendicular component) so the update rotates direction
    # without changing magnitude. The L2 reg is applied as the
    # paper's `λ · ‖w‖ · ŵ` (pure magnitude shrinkage, no
    # rotation). Identity at step 0: for symmetric inits
    # `‖Δ_0 · w_0 / ‖w_0‖²‖` is `O(1/√d)`, so the projection
    # removes a small component of Δ_0 and the first AdamP step
    # ≈ the first AdamW step modulo an `O(1/√d)` correction. With
    # `adamp_lambda=0.0` the projection is fully inert and
    # `AdamP` collapses to plain AdamW — bit-identical baseline.
    # With `use_adamp=False` (default) plain `torch.optim.AdamW`
    # is used — baseline bit-identical. See `optimizers/adamp.py`
    # and `autoresearch/ideas/137-adamp/idea.md`.
    use_adamp: bool = False
    adamp_lr: float = 0.006
    adamp_beta1: float = 0.9
    adamp_beta2: float = 0.999
    adamp_eps: float = 1e-8
    adamp_lambda: float = 1.0  # projection strength (0.0 = inert)
    # 141 — AdaBelief: Adapting Stepsizes by the Belief in Observed
    # Gradients (Zhuang, Liu, Tran, Hoang, Chang, et al. 2020,
    # arXiv:2010.07468, NeurIPS 2020). Replaces the AdamW 1-D /
    # embedding / norm / head path with `AdaBelief` when
    # `use_adabelief=True`. The 2-D Muon path is unchanged
    # (AdaBelief is an AdamW replacement, like 114-MARS, 119-SAM,
    # 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam, 126-AdaShift,
    # 127-GC, 128-SD, 135-Adan, 136-AdaPNM, 137-AdamP). The
    # mechanism replaces AdamW's 2nd moment `v_t = E[g²]` with
    # the *residual* variance `s_t = E[(g_t − m_t)²] + ε`, where
    # `m_t` is the running momentum. Step magnitude is large when
    # the current gradient agrees with the momentum (small
    # residual — we trust the direction) and small when they
    # disagree (large residual). AdamW does the *opposite* — a
    # large `g²` shrinks the step — which is wrong when a large
    # gradient is a *good* direction, not a noisy one. At step 0
    # `m_0 = 0`, `s_0 = ε`; first-step residual is `g_0 −
    # (1−β1)·g_0 = β1·g_0`, so `s_1 = (1−β2)·β1²·g_0² + ε` and
    # `update_0 ≈ g_0 / √(0.081·g_0² + ε) ≈ 3.5·sign(g_0)` — NOT
    # bit-identical to AdamW's first step (AdamW would use
    # `m̂/√v̂ = g_0/|g_0| = sign(g_0)`), but the magnitude is the
    # same order. The first-step displacement is the lever's
    # signature, not a bug. The forward graph is unchanged, so the
    # *pre-step-0 forward* output is bit-identical to baseline.
    # With `use_adabelief=False` (default) plain `torch.optim.AdamW`
    # is used — baseline path bit-identical. See
    # `optimizers/adabelief.py` and
    # `autoresearch/ideas/141-adabelief/idea.md`.
    use_adabelief: bool = False
    adabelief_lr: float = 0.006
    adabelief_beta1: float = 0.9
    adabelief_beta2: float = 0.999
    adabelief_eps: float = 1e-8
    # Moonlight Muon RMS rescale (Kimi / Moonshot AI, arXiv:2502.16982,
    # Feb 2025): replaces the default `shape_aspect` per-tensor scale
    # with `c·sqrt(max(d_in, d_out))` so every 2-D weight has an
    # approximately unit-RMS element-wise update. Geometric calibration
    # of step magnitude across matrix shapes (1:1 attention heads,
    # 1:4 FFN up). Default off → Muon path is bit-identical to the
    # `shape_aspect` baseline. `moonlight_muon_c=0.2` is the paper's
    # tuned single global knob. See
    # autoresearch/ideas/015-moonlight-muon-rms/plan.md.
    use_moonlight_muon: bool = False
    moonlight_muon_c: float = 0.2
    # #16 QK-Norm (Dehghani et al. 2023, ViT-22B, arXiv:2302.05442): apply
    # a `nn.LayerNorm(d_head)` to Q and K along the head-dim axis, before
    # the attention dot product. Bounds the per-head logit
    # `Q·K/√d_head` to `|·| ≤ √d_head` — prevents logit explosion that
    # softens the softmax at depth. Default off → Q/K stay on the existing
    # RMSNorm (qk_norm_type="rmsnorm", the current baseline), so step-0 is
    # bit-identical. Init γ=1, β=0 → identity at step 0. Affects ONLY the
    # Q/K norms (the residual stream norms stay on `norm_type`); the
    # global `use_layernorm` flag is a heavier hammer that flips every
    # norm in the block. See
    # autoresearch/ideas/016-qk-norm/plan.md.
    use_qk_layernorm: bool = False
    # 029 — V-Norm (Wortsman et al. 2023, arXiv:2309.14322): per-head
    # `nn.LayerNorm(d_head)` on V along `d_head` before the AV product,
    # symmetric partner of 016's QK-Norm. Bounds the per-head V vector
    # magnitude so outlier V entries do not dominate the AV aggregation.
    # Separate `nn.LayerNorm(d_head)` module (no weight sharing with
    # q_norm/k_norm from 016). When v_norm_type is also set, the
    # existing v_norm_type wins (explicit > implicit — the closed-#92
    # lever takes precedence). Default off → no v_norm module is built
    # and the baseline path stays bit-identical. See
    # autoresearch/ideas/029-v-norm/plan.md.
    use_v_layernorm: bool = False
    # 176 — Pre-AV V RMSNorm with per-head α-gate + per-head γ-gain
    # (Wortsman et al. 2023, arXiv:2309.14322 V-norm primitive +
    # per-head α-gating following the closed-016 family of
    # pre-interaction normalizations). Apply RMSNorm to V along the
    # `d_k` axis BEFORE the AV product, with a learnable per-head
    # scalar `α_h = relu(α_raw_h)` (init 0 ⇒ identity gate at
    # step 0) and a learnable per-head gain `γ_h ∈ R^{d_k}` (init
    # 1.0 ⇒ identity gain at step 0). Output
    # `V_out = (1 − α_h)·V + α_h·RMSNorm(V)·γ_h`. With α=0,γ=1 the
    # lever is exactly identity ⇒ forward is byte-identical to
    # baseline at step 0 (max-abs-diff = 0.0). Mutually exclusive
    # with use_v_layernorm (closed-029 LayerNorm V) and the
    # closed-#92 v_norm_type zoo (asserted at MHA forward).
    # Default off → no Parameter is registered, no branch is taken,
    # baseline path bit-identical. See
    # `autoresearch/ideas/176-v-pre-av-norm/idea.md`.
    use_v_rmsnorm: bool = False
    # 169 — Depth-Conditional QK-Norm (per-block learnable scale on
    # top of 016's WIN). Keep 016's per-head `q_norm`/`k_norm` and
    # add a single scalar `qk_norm_scale = nn.Parameter(torch.ones(()))`
    # per MHA, init 1.0, applied AFTER the per-head norm and BEFORE
    # the QK matmul: `Q ← Q · qk_norm_scale; K ← K · qk_norm_scale`
    # (the MoA `extra_K` branch mirrors the same multiply). α_l = 1.0
    # init ⇒ step-0 multiplicative gain is exactly the identity ⇒
    # forward is byte-identical to 016's step-0 (max-abs-diff = 0.0).
    # Mutually exclusive with use_q_only_norm / use_k_only_norm /
    # use_qk_norm_post_rope (asserted at MHA forward). Default off →
    # no `qk_norm_scale` Parameter is registered, no branch is taken,
    # baseline path bit-identical. Mirrors NormFormer's per-layer
    # attention-output gains (Shleifer et al. 2021) applied to the
    # QK-norm output. See
    # `autoresearch/ideas/169-qk-norm-depth/idea.md`.
    use_qk_norm_depth: bool = False
    # 190 — Per-Layer QK-Norm (scalar γ per block per side, replaces
    # 016's per-channel γ). Sits on top of 016's WIN shape: the per-head
    # `q_norm`/`k_norm` (RMSNorm/LayerNorm over d_head) is kept intact,
    # and a single scalar `qk_norm_scalar_{q,k} = nn.Parameter(
    # torch.ones(()))` per MHA per side is multiplied AFTER the per-head
    # norm and BEFORE the QK matmul. Default off ⇒ no Parameter
    # registered, no branch taken, baseline path bit-identical. When
    # ON with init 1.0, the multiply is exactly the identity in fp32 ⇒
    # step-0 forward is byte-identical to 016's step-0 (max-abs-diff =
    # 0.0). Distinct from 169 (`use_qk_norm_depth`, single SHARED scalar
    # across Q and K) — 190 keeps Q and K scalars separate by default
    # to preserve 016's QK symmetry; the shared variant is gated behind
    # `qk_norm_scalar_qk_shared` (collapses to 169's axis if both are
    # on). 190 default `qk_norm_scalar_qk_shared=False` ⇒ 12 × 2 × 1 =
    # 24 γ params total (vs 016's 384 per-channel); shared variant ⇒
    # 12 × 1 = 12. Mutually exclusive with `use_q_only_norm` /
    # `use_k_only_norm` / `use_qk_norm_post_rope` (asserted at MHA
    # forward) — those levers restructure the norm, not the gain, and
    # combining them confounds 190's axis. See
    # `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
    qk_norm_scalar_per_block: bool = False
    qk_norm_scalar_qk_shared: bool = False

    # 132 — Born-Again Networks: Self-Distillation with EMA Teacher
    # (Furlanello, Lipton, Tschiatschek, Prabhudesai, Urbach 2018,
    # arXiv:1805.04770). Maintain a shadow copy of the model
    # `θ_teacher ← (1−β)·θ_teacher + β·θ_student` updated each
    # optimizer step. Add a per-step distillation term
    # `L_distill = α · T² · KL(softmax(teacher/T) ‖ softmax(student/T))`
    # on top of CE. Identity at step 0: the shadow is a clone of the
    # live init, so the teacher forward produces identical logits to
    # the student ⇒ KL = 0 ⇒ loss = CE (byte-identical to baseline
    # at step 0). With `use_born_again=False` (default) the teacher
    # is never built and the loss term is zero ⇒ baseline path
    # bit-identical throughout. See `autoresearch/ideas/132-born-again/idea.md`.
    use_born_again: bool = False
    born_again_beta: float = 0.999  # EMA "speed" (higher = teacher tracks closer)
    born_again_alpha: float = 1.0   # KL weight on top of CE
    born_again_temp: float = 2.0    # distillation temperature; KL scaled by T²

    # 133 — SeqMix: Token-Level Mixup for Language Modeling
    # (Guo, Mao, Zhang 2019, arXiv:1908.02951, extended to LM).
    # When on, the trainer samples a paired sequence from the batch,
    # computes embeddings for both via the model's existing
    # token_embedding (and emb_proj if emb_rank is set), and mixes
    # them at the embedding level:
    #   emb_mixed = λ · emb_a + (1 − λ) · emb_b,   λ ~ Beta(α, α)
    # The residual stream is fed `emb_mixed * emb_scale`; the rest of
    # the model runs unchanged. The loss is the λ-weighted mix of the
    # two CEs against the unmixed targets:
    #   L_mixed = λ · CE(logits, y_a) + (1 − λ) · CE(logits, y_b)
    # α=0.4 is the paper default; λ is almost always in (0.05, 0.95),
    # so the mixed loss differs from the unmixed CE by a non-trivial
    # amount at step 0 — the lever's documented signature. With
    # `use_seqmix=False` (default) `model.seqmix_forward` is never
    # called and the trainer takes the standard `model(x)` +
    # `F.cross_entropy(...)` path — baseline path bit-identical.
    # See `autoresearch/ideas/133-seqmix/idea.md`.
    use_seqmix: bool = False
    seqmix_alpha: float = 0.4

    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None
    
    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True
    ffn_variant: str = "squared_relu"
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        # Query-tweaks Batch 4 prereq wire: q_norm_type defaults to
        # qk_norm_type so existing configs are bit-identical unless the
        # Q-side norm is explicitly set (see plan.md Batch 4 note).
        if not self.q_norm_type:
            self.q_norm_type = self.qk_norm_type

    def active_flags(self) -> dict:
        """Return {field_name: value} for every field whose value differs
        from the LLMConfig default. Used by the metrics writer to dump
        only the non-default knobs — the "what was on" summary of a
        run. Forward-only: new runs emit this; old metrics.json
        don't have it (DESC in runs/make_evidence_index.py is the
        fallback for those).
        """
        import dataclasses
        defaults = LLMConfig()
        out = {}
        for f in dataclasses.fields(self):
            cur = getattr(self, f.name)
            dflt = getattr(defaults, f.name, None)
            if cur != dflt and not f.name.startswith("_"):
                out[f.name] = cur
        return out


# ============================================================================
# SCREEN tier — undertrained (NOT 20x). Cheap, fast filters to find a mechanism's
# sign + basin and kick out bad ideas before paying for a Full run. Screen
# results never transfer-promote; the optimum drifts with training duration.
# ============================================================================


@dataclass
class Screen10M20MConfig(LLMConfig):
    """Screen — ~7.7M params · 20M tokens · ~4880 steps. Confirms sign survives more tokens.

    The 10M architecture: low-rank embedding (emb_rank=48) + depth (24 layers).
    Embedding factorized 49152x144 -> (49152x48)@(48x144), freeing ~4.7M params
    from the lookup table and spending them on transformer depth at a fixed budget.
    """
    d_model: int = 144
    n_heads: int = 6
    n_layers: int = 24
    d_ff: int = 576
    n_kv_heads: int = 2
    emb_rank: int = 48
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 20_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 4880, 200))


@dataclass
class Screen10M1MConfig(Screen10M20MConfig):
    """Ultra-fast screen — ~10M params · 1M tokens · ~250 steps.

    Kept for checkpoint compatibility and fast experiment screens.
    """
    train_tokens: int = 1_000_000
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 250, 25))


@dataclass
class Screen10M5MConfig(Screen10M20MConfig):
    """Short screen — ~10M params · 5M tokens.

    Kept for checkpoint compatibility and short transfer checks.
    """
    train_tokens: int = 5_000_000


@dataclass
class Tiny1M3MConfig(LLMConfig):
    """Tiny screen — ~0.94M params · 3M tokens.

    Fast idea filter. This is a separate tier from screen20m:
    use it to rank ideas cheaply, then re-test winners on screen20m
    before making stronger claims.
    """
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 256
    n_kv_heads: int = 2
    emb_rank: int = 8
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 3_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = (
        0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700
    )


@dataclass
class Tiny1M3MReZeroConfig(Tiny1M3MConfig):
    """Tiny1M3M with ReZero residual scaling (Bachlechner et al. 2020,
    arXiv:2003.04887).

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Each
    transformer block builds two learnable scalars `α_attn` and
    `α_ffn` (init 0) on the residual adds. At step 0 both gates are
    0, so the residual add becomes a no-op and the model is the
    identity function — bit-identical to baseline in the limit of
    fp32 (the `α·f` term is exactly 0). The optimizer then grows
    the α's during training; the lever is whether layer-specific
    residual scaling helps at 12L. Cost: 2 scalars/block × 12
    blocks = 24 scalars total (negligible).

    Transfer-risk: high. The paper's headline wins are at 100L
    (CIFAR-10 / T2T-ViT) and modest at 12L (GPT-2 125M). tiny1m3m
    is at 12L so the lever is least likely to fire. NULL band
    |Δ| < 0.01. DRIFT > +0.01. PASS ≤ −0.01. See
    `autoresearch/ideas/130-rezero/idea.md`.
    """
    use_re_zero: bool = True


@dataclass
class Tiny1M3MSWANConfig(Tiny1M3MConfig):
    """Tiny1M3M with SWAN on the matrix-weight slot."""
    use_swan: bool = True


@dataclass
class Tiny1M5MConfig(Tiny1M3MConfig):
    """Tiny screen — ~0.94M params · 5M tokens.

    Same architecture as Tiny1M3MConfig, longer only when a 3M result
    looks promising but too undertrained.
    """
    train_tokens: int = 5_000_000
    eval_milestones: Optional[Tuple[int, ...]] = (
        0, 50, 100, 150, 200, 300, 400, 500, 600, 750, 900, 1100, 1200
    )


@dataclass
class Tiny1M3MQGainConfig(Tiny1M3MConfig):
    """Tiny1M3M with per-head Q-gain."""
    use_q_gain: bool = True


@dataclass
class Tiny1M3MAlibiConfig(Tiny1M3MConfig):
    """Tiny1M3M with learnable per-head ALiBi-style linear-distance
    bias on attention scores (Press, Smith, Lewis, ICLR 2022,
    arXiv:2108.12409; validated at BLOOM-176B by BigScience 2022).

    Adds `score[b,h,t,s] -= m_h · (t − s)` pre-softmax for s ≤ t
    (causal positions only; the rest are masked by the −1e9 causal
    mask first so the bias has no effect on them). `m_h` is a
    per-head `nn.Parameter(torch.zeros(self.n_heads))` — *direct
    linear* parameterization init 0 ⇒ `m_h = 0` ⇒ bias is 0 ⇒
    scores unchanged ⇒ softmax unchanged ⇒ AV unchanged ⇒ output
    unchanged ⇒ **byte-identical to baseline at step 0**
    (max-abs-diff = 0.0).

    The published ALiBi paper uses *fixed* geometric-sequence slopes
    `1/2^(8k/H)` and writes the bias as `−s_h·(i−j)`. The repo
    implementation writes `+m_h·(i−j)` (sign-flipped; for causal
    positions `(i−j) ≥ 0`, the effective bias is `−m_h·(i−j)`).
    Since `m_h` is learnable and init 0, the optimizer can recover
    ALiBi's decay behavior by learning `m_h > 0` — the sign
    convention is irrelevant to the experiment's validity. The
    fixed-geometric-slope recipe is not used here (we don't constrain
    the per-head slope to a pre-set sequence); the lever is fully
    unstructured per-head so the optimizer can pick any decay shape.

    Forces the manual attention path (`models/layers.py:3096-3104`)
    so the score-side bias can't go through SDPA's flash kernel.
    Cost: 4 scalars/block × 12 blocks = 48 params (+0.005% — negligible).

    Distinct from the closed *content-free* per-head scalars at 0.94M:
    - 152-attn-logit-bias (NULL): free additive bias `b_h`; cancels in
      softmax's per-row normalizer → mathematical null.
    - 155-per-head-temp (NULL): scalar `τ_h` on Q·K/√d; identity at
      init `τ_h=1/sqrt(d_k)`; the lever absorbed into the Q/K scales.
    - 160-rms-gain-per-head (NULL): per-head RMS gain on V; identity
      at init `g_h=1`.
    - 166-t5-rpe (NULL): bucketed per-head bias indexed by
      `floor(log2(|i-j|+1))`; identity at init `b=0`. *Closest to
      175*, but bucketed-discrete vs 175's continuous-linear axis.
    175 is the *position-distance-structured* member of the
    per-head-attention-shape family — the bias is a function of
    `(i−j)`, not a free per-head offset. The structured axis gives
    the optimizer something to specialize (each head picks its own
    decay rate) that the free-scalar levers lack.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val
    mean 6.4447±0.0244 over the box's 14 cached measurements).
    Expected Δval ∈ [−0.005, −0.025] (modest, similar to other
    locality-prior wins: 009-FIRE, 154-rebased-attn, 023-canon-conv).
    NULL band |Δ| < 0.02 expected (sub-noise inconclusive).
    PASS ≤ ctrl − 0.02.

    See `autoresearch/ideas/175-alibi-slopes/idea.md` for the full
    mechanism, lever-mode pin, and zero-init rationale.
    """
    use_alibi_bias: bool = True


@dataclass
class Tiny1M3MQKNormConfig(Tiny1M3MConfig):
    """Tiny1M3M with QK-Norm (LayerNorm on Q,K before the attention dot product).

    A/B vs the tiny1m ctrl. Replaces the default RMSNorm on Q,K with
    `nn.LayerNorm(d_head)` (γ=1, β=0 init → identity at step 0).
    Residual-stream norms stay on RMSNorm — the lever is strictly the
    per-head logit bounding, not a residual-stream re-centering. PASS
    ≤ ctrl − 0.005 on val_loss (taste review puts leverage at the low
    end of the hypothesis range for 6 layers). NULL band |Δ| < 0.005.
    See autoresearch/ideas/016-qk-norm/plan.md.
    """
    use_qk_layernorm: bool = True


@dataclass
class Tiny1M3MScaleNormConfig(Tiny1M3MConfig):
    """Tiny1M3M with ScaleNorm (scalar-gain RMSNorm on the residual stream)."""
    norm_type: str = "scalenorm"


@dataclass
class Tiny1M3MVNormOnQKNormConfig(Tiny1M3MQKNormConfig):
    """Tiny1M3M with QK-Norm + V-Norm (per-head LayerNorm on V before AV).

    A/B vs the QK-Norm ctrl (`Tiny1M3MQKNormConfig`, the 016 WIN
    signature). Adds a per-head `nn.LayerNorm(d_head)` on V along
    `d_head` (γ=1, β=0 init → identity at step 0), the symmetric
    partner of 016's QK-Norm. Independent `v_norm` module (no weight
    sharing with q_norm/k_norm). Bounds per-head V vector magnitude
    so outlier V entries do not dominate the AV aggregation output.

    PASS ≤ ctrl − 0.005 (matches 016's bar — the symmetric-partner
    bet is at the low end of the hypothesis range per the taste
    caveat that Wortsman used V-norm as a diagnostic, not a primary
    lever). NULL band |Δ| < 0.005. DRIFT > +0.005. See
    autoresearch/ideas/029-v-norm/plan.md.
    """
    use_v_layernorm: bool = True


@dataclass
class Tiny1M3MVQGainConfig(Tiny1M3MConfig):
    """Tiny1M3M with V-embed + per-head Q-gain."""
    use_value_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Tiny1M3MSWAConfig(Tiny1M3MConfig):
    """Tiny1M3M with SWA(window=512) only."""
    use_sliding_window: bool = True
    sliding_window_size: int = 512


@dataclass
class Tiny1M3MVQGainSWAHighRoPEConfig(Tiny1M3MConfig):
    """Tiny1M3M with the current screen20m best recipe."""
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000


@dataclass
class Tiny1M3MVQGainHighRoPESWA384Config(Tiny1M3MConfig):
    """Tiny1M3M with V+q+HighRoPE + SWA(window=384)."""
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 384
    rope_base: int = 500000


@dataclass
class Tiny1M3MVQGainSWAHighRoPE250KConfig(Tiny1M3MConfig):
    """Tiny1M3M with V+q+SWA(window=512) + RoPE base 250k."""
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 250000


@dataclass
class Tiny1M3MSoftpickOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + Softpick (rectified-softmax normalization).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:40`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates the
    softpick swap (function-level normalization tweak) on top of the
    same FIRE-equipped foundation, not silent HP drift. The treatment
    replaces `torch.softmax` in the manual attention path with softpick
    `relu(exp(x)−1) / (Σ|exp(x)−1| + ε)`. Permits zero total
    attention mass → kills the attention-sink pathology without
    adding a learnable sink token (categorically distinct from
    the closed `attn-sink` lever, which *added* a sink token;
    distinct from 020-FoX, which multiplies post-softmax; distinct
    from 013-CoPE, which adds a content-aware bias on logits).
    ε=1e-6 pinned; `exp−1` computed in fp32 then cast back.

    PASS ≤ −0.005 vs the FIRE-equipped ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01. Step-0 smoke gate (see plan.md) — build trt,
    run one fwd+bwd, assert loss is finite and Q/K/V grads are
    non-zero (zero attn output at step 0 ⇒ zero grad ⇒ lever
    dead on arrival). See
    `autoresearch/ideas/022-softpick-attention/plan.md`.
    """
    use_fire_pe: bool = True
    use_softpick: bool = True


@dataclass
class Tiny1M3MCanonOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + Canon conv (gated depthwise causal Conv1d).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:40`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates
    the canon-conv swap (one depthwise causal Conv1d per block on
    the residual stream) on top of the same FIRE-equipped
    foundation, not silent HP drift. The treatment stacks
    `use_canon_conv=True` on top: one causal depthwise Conv1d
    (kernel=3) per block on the residual stream, immediately
    before the attention sublayer's pre-LN, with a single scalar
    output gate `g` (init 0 → step-0 ≡ no-conv baseline). Pre-LN
    read (no extra LN on the conv path). Strictly orthogonal to
    FIRE (additive on logits) and to CoPE/FoX/Softpick (all live
    inside the attention computation); this is an *outside-
    attention* local-mixing lever on the residual stream — the
    Griffin/Mamba local-mixing half. Default off → baseline path
    bit-identical. Cost: n_layers × (3·d_model + 1) extra params
    (~2.3K at tiny1m3m, +0.25%).

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| ≤ 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/023-canon-conv/plan.md`.
    """
    use_fire_pe: bool = True
    use_canon_conv: bool = True


@dataclass
class Tiny1M3MUNetSigmoidOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + U-Net sigmoid skips (modded-nanogpt fix).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates
    the U-Net swap (residual-stream architectural lever) on top of
    the same FIRE-equipped foundation, not silent HP drift.

    Adds learnable U-Net skip connections bridging early layer
    outputs into mirrored late layers. The gate parameter is
    initialised to -1.5 and wrapped in sigmoid (modded-nanogpt
    PR #125 fix:
    https://github.com/KellerJordan/modded-nanogpt/pull/125), so
    `sigmoid(-1.5) ≈ 0.18` of the early activation flows in at
    step 0 — small, bounded to (0, 1), non-zero starting point
    with non-zero gradient. Categorically distinct from our
    previous broken attempt
    (`docs/youtube-architecture-ablation-log.md §5`, val +0.0003
    worse) which used `unet_gate_type="raw"` +
    `unet_gate_init=0.0` — the dead-gate bug. The mechanism never
    actually ran in that test; it was a bug-experiment, not a
    mechanism A/B.

    At tiny1m3m's 6-layer depth the U-Net mirrors are 0↔5 / 1↔4 /
    2↔3 — only 3 short pairs, so the predicted effect is "small
    but non-zero", not big-if-true. A clean null after the fix
    definitively closes U-Net skips for this model class; a win
    plausibly amplifies at 135M where depth grows. Strictly
    orthogonal to FIRE (which is an attention-side lever);
    orthogonal to all closed levers (no residual-stream
    architectural change in closed.md). transfer-risk: low —
    modded-nanogpt's +1.25% speedup is at ≥100M parameter scale,
    directly comparable to tiny1m3m's model class.

    PASS ≤ −0.005 vs the FIRE-equipped ctrl (taste's "small but
    non-zero" prediction; not −0.01 because the 3-pair U at 6L
    is a smaller bet than the deeper-stack version). NULL band
    |Δ| < 0.005. DRIFT > +0.005. See
    `autoresearch/ideas/030-unet-skip-sigmoid/plan.md`.
    """
    use_fire_pe: bool = True
    use_unet_skips: bool = True
    unet_gate_type: str = "sigmoid"
    unet_gate_init: float = -1.5


@dataclass
class Tiny1M3MGatedAttnOnFireConfig(Tiny1M3MConfig):
    """Tiny1M3M with FIRE + Gated Attention (Qiu et al. 2025).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val 6.3234
    per `closed.md:40`). The treatment stacks `use_gated_attn=True` on
    top: a per-head *scalar* input-conditional sigmoid gate on the head
    output `o_h = A_h V_h`, applied post-AV and pre-merge with the O
    projection: `o_h ← o_h · 2·σ(W_g·x+b)`. `W_g : nn.Linear(d_model, H)`
    (per-head scalar, NOT the per-head vector form — vector would be
    42% of the 0.94M model). Gate input is the sublayer input residual
    `x` (pre-LN), NOT `o_h` itself (circularity). Identity-init: W=0,
    b=0 → 2·σ(0) = 1.0 exactly at step 0, so the gated forward graph
    is bit-identical to the no-gate forward graph at step 0; the new
    params start receiving gradient from step 1. Categorically
    distinct from every closed lever and every active attention-side
    lever (020-FoX → A-prob decay, 021-V-residual → cross-layer V,
    022-softpick → softmax swap, 023-canon-conv → pre-attn conv,
    025-SSMax → logit temperature) — 024 is the *only* lever on the
    post-AV head-output value site. 009's additive position bias is
    additive on logits; the head-output gate is multiplicative on
    `o_h`; the two compose cleanly when both are on.

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| < 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/024-gated-attention/plan.md`.
    """
    use_fire_pe: bool = True
    use_gated_attn: bool = True


@dataclass
class Tiny1M3MExclusiveSelfAttnConfig(Tiny1M3MConfig):
    """Tiny1M3M with exclusive self-attention correction.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). After standard
    attention, subtract the component of the head output that points along
    the current token's value vector. The per-head coefficient is zero-init,
    so step 0 is the baseline graph.
    """
    use_exclusive_self_attn: bool = True


@dataclass
class Tiny1M3MVResidualOnFireConfig(Tiny1M3MConfig):
    """Tiny1M3M with FIRE + Value Residual Learning (cross-layer V shortcut).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`; ctrl spread 6.3875–6.4050 per
    `closed.md:41-44`). The treatment stacks `use_value_residual=True`
    on top: stash the projected V at layer 0 (post-W_V, post-GQA
    repeat_interleave, post-transpose, shape `[B, n_heads, T, d_k]`);
    in every later layer l > 0, blend
    `V_l ← (1 - λ_l)·V_l + λ_l·V_1` BEFORE `attn_weights @ V`, with
    `λ_l = nn.Parameter(torch.zeros(()))` per-block on MHA (identity-
    init at step 0 ⇒ baseline-bit-identical at flag-on, step 0).
    `.detach()` on the V_1 stash ⇒ each layer's W_V trains on its own
    attention path; the cross-layer shortcut only learns the blend
    weight, not the layer-0 projection. Strictly orthogonal to FIRE
    (which is *additive* on logits): FIRE chooses *which* key wins;
    021 changes *which* value-stream the winners read from. The bet
    is that tiny1m3m's narrow heads (d_k=32 at H=8) suffer attention
    concentration, and a direct shortcut to the first-layer value
    representation gives later blocks a cleaner value signal.

    Categorically distinct from the closed V/Q/K/O *embedding* axis
    (input-side projection scaling, an added embedding to the value
    *source*; 021 is a cross-layer residual on the value *stream*)
    and from every active attention-side lever (020-FoX = post-softmax
    A·D, 022-softpick = softmax swap, 024-gated-attn = post-AV o_h
    gate, 025-SSMax = logit temperature) — 021 is the only lever on
    the projected V stream.

    PASS ≤ −0.005 vs the FIRE-equipped ctrl (low-to-moderate bar; the
    bet is at the small end of the paper's reported effect — the
    taste r1 reviewer asked for exactly this band). NULL band
    `|Δ| < 0.01` (sub-noise; the lever does not fire on top of FIRE
    at this scale). DRIFT > +0.01 (cross-layer mix hurts attention
    concentration rather than helping). See
    `autoresearch/ideas/021-value-residual/plan.md`.
    """
    use_fire_pe: bool = True
    use_value_residual: bool = True


@dataclass
class Tiny1M3MQKNormOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
    """Tiny1M3M with FIRE + QK-Norm (LayerNorm on Q,K head-dim).

    A/B vs the FIRE-equipped baseline (the 009 WIN signature, val
    6.3234 per `closed.md:44`). Parent is
    `Tiny1M3MVQGainSWAHighRoPE250KConfig` so VQ-gain + SWA(512) +
    RoPE 250K carry over from the ctrl recipe — the A/B isolates the
    QK-Norm swap (per-head LayerNorm on Q,K along `d_head`) on top
    of the same FIRE-equipped foundation, not silent HP drift. The
    treatment stacks `use_qk_layernorm=True` on top: bounds the
    per-head logit `Q·K/√d_head` to `|·| ≤ √d_head`. Categorically
    distinct from FIRE — FIRE is *additive* (bias added to logits
    post-dot-product); QK-Norm is *multiplicative-normalizing*
    (LayerNorm bounds the dot-product magnitude that the bias gets
    added to). The two operate at different points and on different
    mathematical axes.

    The 013-CoPE DRIFT (+0.143 vs FIRE-alone, `closed.md`) is the
    relevant prior, but 013 failed by stacking *two additive position
    bias levers* — QK-Norm does not compound additively with FIRE's
    bias; it bounds the magnitude of the Q·K product. This is the
    qualitative difference that makes 026 a different bet from 013.

    Expected: additive (~−0.078 vs FIRE-alone, computed as 009's
    −0.064 + 016's −0.014). Superadditive (~−0.09+) would mean the
    per-head logit bounding makes FIRE's learned position bias more
    consistent across heads. A null or regression would mean the
    013-CoPE precedent generalises — attention-domain headroom is
    exhausted by FIRE at this scale.

    PASS ≤ −0.01 vs the FIRE-equipped ctrl. NULL band |Δ| ≤ 0.01.
    DRIFT > +0.01. See
    `autoresearch/ideas/026-fire-x-qknorm/plan.md`.
    """
    use_fire_pe: bool = True
    use_qk_layernorm: bool = True


@dataclass
class Tiny1M3MDeepThinConfig(Tiny1M3MConfig):
    """Tiny1M3M deep-and-thin: depth/width swap at fixed ~0.94M budget.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306
    per `LEADERBOARD.md` row 14). The treatment reallocates the 0.94M
    budget across more, thinner layers: n_layers 12→20 (1.67×),
    d_model 64→48 (0.75×), d_ff 256→192 (= 4·d_model preserved),
    n_heads/n_kv_heads 4/2 → 3/3 (MHA-tied — see confound below).
    Per-head `d_head = 16` preserved (was 64/4, now 48/3); `emb_rank=8`,
    `ffn_variant="squared_relu"`, `vocab_size=49152` all inherited
    unchanged from `Tiny1M3MConfig`. Param budget arithmetic:
    per-block attn 9.2k + FFN 18.4k + norms 0.24k ≈ 27.9k; ×20 = 558k
    + embedding factorisation 393.6k ≈ 951k (+1.3% vs baseline 939k,
    inside the ±5% ceiling). MobileLLM (Ma et al., ICML 2024,
    arXiv:2402.14905) reports +2.7% / +4.3% on zero-shot benchmarks
    at 125M / 350M from this exact depth/width swap; the open question
    is whether the lever still fires at 0.94M (133× smaller than the
    paper's smallest ablation).

    Known confound (see `idea.md:50-55`). Baseline is GQA 2:1
    (n_heads=4, n_kv_heads=2). The depth/width swap also collapses
    kv-sharing → MHA (n_heads=n_kv_heads=3). Tied-QK / full-MHA is a
    known WIN signature at tiny1m3m (`LEADERBOARD.md` row 0 = vq-gain
    + rope250k + swa384 + tiedqk, val 6.3041) — the trt Δ partly
    reflects the kv-sharing collapse, not pure depth/width. Picked B1
    over B1' (MQA n_kv_heads=1) and B2 (d_model=32, d_ff off the
    4·d_model rule) because the `d_ff = 4·d_model` convention is more
    load-bearing for "pure depth/width swap" than the GQA ratio.
    Runner reports the confound alongside the raw val-loss Δ.

    PASS ≤ ctrl − 0.01 (clears the cited ±0.01 box-noise floor).
    NULL band |Δ| ≤ 0.01 (inclusive — sub-noise = inconclusive,
    no multi-seed rescue). DRIFT > ctrl + 0.01. ctrl_val baseline
    6.4306 (`LEADERBOARD.md` row 14) — interpreted against the
    in-session ctrl run to avoid cross-session drift. Seed 42 only.
    See `autoresearch/ideas/028-deep-thin-config/plan.md`.
    """
    d_model: int = 48
    n_heads: int = 3
    n_kv_heads: int = 3
    n_layers: int = 20
    d_ff: int = 192


@dataclass
class Screen10M20MSwiGLUConfig(Screen10M20MConfig):
    """Screen10M20M with SwiGLU feed-forward blocks."""
    ffn_variant: str = "swiglu"
    d_ff: int = 384  # Parameter-matched to squared-ReLU d_ff=576.


@dataclass
class Screen10M20MOutputAdapterConfig(Screen10M20MConfig):
    """Screen10M20M with a rank-32 additive output adapter.

    Tests whether the rank-48 tied factorized softmax is too narrow after the
    embedding/depth reallocation. Adds ~1.58M parameters while staying under
    the 10M class.
    """
    output_adapter_rank: int = 32


@dataclass
class Screen10M20MSmearGateConfig(Screen10M20MConfig):
    """Screen10M20M with SmearGate previous-token embedding blend."""
    use_smear_gate: bool = True


@dataclass
class Screen10M20MUNetSkipConfig(Screen10M20MConfig):
    """Screen10M20M with zero-init U-Net skip bridges across depth."""
    use_unet_skips: bool = True


@dataclass
class Screen10M20MAttnOutputGateConfig(Screen10M20MConfig):
    """Screen10M20M with per-head attention-output gates."""
    use_attn_output_gate: bool = True


@dataclass
class Screen10M20MLayerScaleConfig(Screen10M20MConfig):
    """Screen10M20M with per-channel attention/MLP LayerScale gates."""
    use_layerscale: bool = True


@dataclass
class Screen10M20MValueEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value embeddings injected into attention V."""
    use_value_embed: bool = True


@dataclass
class Screen10M20MQueryEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token query embeddings injected into attention Q."""
    use_query_embed: bool = True


@dataclass
class Screen10M20MKeyEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token key embeddings injected into attention K.

    The natural mirror of #29/#30. K goes through RoPE downstream, so the
    projection's term is positionally rotated — a different operating point
    from V (no RoPE) and Q (also RoPE'd). The cheapest probe of whether
    the token-identity-into-attention lever has more headroom in the K
    direction.
    """
    use_key_embed: bool = True


@dataclass
class Screen10M20MVQEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value + query embeddings injected into attention.

    #32 — combination probe. V is the end-of-training winner (#29, 4.7728),
    Q is the fast-warmup winner (#30, 4.8159). Tests whether the lever is
    additive (Q's warmup advantage + V's end-game edge) or whether V's
    V-specific position is the unique story. Cost = V-embed + Q-embed
    projections = ~166k extra params (~2% over baseline).
    """
    use_value_embed: bool = True
    use_query_embed: bool = True


@dataclass
class Screen10M20MOutputEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token embeddings injected into the attention OUTPUT.

    #33 — fundamentally different lever. Where #29-#32 inject the raw
    token embedding into the attention INPUTS (V/Q/K, inside the score
    computation), this one injects it into the attention OUTPUT (after
    the O projection, straight into the residual stream). The token
    identity bypasses attention entirely. Tests "is V-embed winning
    because V is a unique position, or because any token-signal-to-
    residual helps?" Most likely outcome: underperforms V-embed (since
    the signal bypasses attention) but a clean probe of an architectural
    question we haven't asked yet. Cost = 24 × d_model 144 × emb_rank
    48 = 165,888 extra params (~2.1%).
    """
    use_output_embed: bool = True


@dataclass
class Screen10M20MVQKEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value + query + key embeddings injected into attention.

    #34 — full combo probe. V-embed alone is the natural-end winner
    (4.7728), Q-embed adds 0.03 to V (V+Q = 4.7428). K-embed is
    essentially tied with Q at the natural end (4.8228 vs 4.8159,
    inside noise). Tests "is K redundant with Q in the V+Q combo?"
    If V+Q+K ≈ V+Q, K adds nothing. If V+Q+K < V+Q, K is hurting.
    If V+Q+K > V+Q, K is contributing beyond Q.

    Cost = 24 × (q_size 144 + 2 × kv_size 48) × emb_rank 48
        = 24 × 240 × 48 = 276,480 extra params (~3.6% over baseline).
    """
    use_value_embed: bool = True
    use_query_embed: bool = True
    use_key_embed: bool = True


@dataclass
class Screen10M20MVOEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token value (inside attention) + output (post-O) embeddings.

    #35 — across-boundary combo. V-embed wins inside attention (4.7728),
    O-embed is the worst of the family (4.8350). Tests whether the
    inside-attention and post-O positions are additive — i.e. whether
    adding the token signal to the residual stream helps when V is
    already injecting it into attention. If V+O < V, the residual
    signal interferes with V's inside-attention signal. If V+O > V,
    the residual signal adds value.

    Cost = 24 × (kv_size 48 + d_model 144) × emb_rank 48
        = 24 × 192 × 48 = 221,184 extra params (~2.9% over baseline).
    """
    use_value_embed: bool = True
    use_output_embed: bool = True


@dataclass
class Screen10M20MVOKEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with value + output + key embeddings.

    #36 — K's role across contexts. K is anti-additive in V+Q (4.8250 vs
    V+Q's 4.7428). Tests whether K is universally bad or just bad in
    the V+Q context. With O replacing Q (since V+O = best so far at
    4.7188), K has a different gradient environment. If V+O+K > V+O,
    K helps when paired with O. If V+O+K < V+O, K is universally
    bad in any embed combo.

    Cost = 24 × (kv_size 48 + kv_size 48 + d_model 144) × emb_rank 48
        = 24 × 240 × 48 = 276,480 extra params (~3.6% over baseline).
    """
    use_value_embed: bool = True
    use_output_embed: bool = True
    use_key_embed: bool = True


@dataclass
class Screen10M20MQGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head learnable Q-gain (post-RoPE).

    #37 — first non-embed lever. Each attention head has a learnable
    scalar that multiplies its Q vector after norm+RoPE. Zero-init so
    step 0 is exact baseline. Equivalent to per-head temperature on
    the attention scores. Known modded-nanogpt speedrun trick
    (q_gain). Cost: 24 × 6 = 144 extra params (negligible).

    If q_gain helps, the model benefits from per-head attention
    temperature — a way for different heads to specialize. If
    q_gain is in noise, the heads don't need to rescale their
    attention patterns.
    """
    use_q_gain: bool = True


@dataclass
class Screen10M20MKGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head learnable K-gain (post-RoPE).

    #42 — symmetric to q_gain but on K. Tests whether scaling K
    helps as much as scaling Q. If k_gain is similar to q_gain,
    the lever is "per-head temperature" in general. If only q_gain
    helps, it's specifically the Q side.
    """
    use_k_gain: bool = True


@dataclass
class Screen10M20MVQKGainConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + per-head Q-gain + per-head K-gain.

    #43 — V+q+k_gain. Tests whether q_gain and k_gain are
    additive. If V+q+k_gain < V+q_gain, k_gain is hurting or
    redundant. If V+q+k_gain > V+q_gain, k_gain adds value.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_k_gain: bool = True


@dataclass
class Screen10M20MQKGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head Q-gain + per-head K-gain (no embed).

    #44 — pure double-gain (no embed). Tests if the Q-gain and
    K-gain levers together beat either alone.
    """
    use_q_gain: bool = True
    use_k_gain: bool = True


@dataclass
class Screen10M20MDeepVEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with 2-layer non-linear V-embed (GELU bottleneck).

    #45 — deep V-embed probe. Tests whether the linear V-embed (#29,
    4.7728) has saturated at a single projection, or whether a
    non-linear "bottleneck" V-embed has more capacity. The
    architecture:

        V += GELU(ve @ W1) @ W2
        W1: [emb_rank=48, hidden=96]  zero-init
        W2: [hidden=96, kv_size=48]   zero-init

    Both zero-init so step 0 = exact baseline. GELU has a dead-zone
    at 0 so the first gradient step only flows through W2 (similar
    to standard deep ResNets).

    Cost = 24 × (48 × 96 + 96 × 48) = 24 × 9,216 = 221,184 extra
    params (+2.9% over baseline, +1.4% over V-embed).

    If deep V-embed > V-embed (linear), the V-embed win has more
    capacity to unlock. If deep V-embed ≈ V-embed, the linear
    projection is already sufficient. If deep V-embed < V-embed,
    the non-linearity hurts (likely due to overfitting or gradient
    issues with the dead-zone at init).
    """
    use_deep_value_embed: bool = True
    deep_value_embed_hidden: int = 96


@dataclass
class Screen10M20MDeepVQGainConfig(Screen10M20MConfig):
    """Screen10M20M with deep V-embed + per-head Q-gain.

    #46 — combines #45 (deep V-embed) with #37 (q_gain). Tests
    whether the q_gain lever is also additive with the deeper
    V-embed architecture. If V_deep+q_gain > V+q_gain (4.6815),
    deep V-embed is the new capacity ceiling.
    """
    use_deep_value_embed: bool = True
    deep_value_embed_hidden: int = 96
    use_q_gain: bool = True


@dataclass
class Screen10M20MFFNEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with token embeddings injected into FFN input.

    #47 — new position probe. The FFN-embed adds a learned projection
    of the factorized token embedding to the FFN input (post-attention,
    pre-FFN, after norm2). Different path from V-embed (in attention)
    and O-embed (post-O residual). The FFN now has direct access to
    token identity without going through attention.

    Cost = 24 × (d_model 144 × emb_rank 48) = 165,888 extra params
    (~2.1% over baseline).

    Tests:
    - If FFN-embed ≈ V-embed (4.7728), the lever is "token identity
      into residual content" regardless of position.
    - If FFN-embed > V-embed, the FFN is a more useful position than
      attention's V (because it's a more direct path).
    - If FFN-embed < V-embed, the position matters — V-embed's win is
      specifically about attention content, not residual content.
    """
    use_ffn_embed: bool = True


@dataclass
class Screen10M20MVQGFFNEmbedConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + FFN-embed.

    #48 — combines #29 (V-embed), #37 (q_gain), and #47 (FFN-embed).
    Tests whether the FFN-embed lever is also additive with V+q_gain.
    If V+q_gain+ffn_embed < V+q_gain (4.6815), FFN-embed conflicts
    with V+q_gain. If V+q_gain+ffn_embed > V+q_gain, FFN-embed adds
    a new dimension to the win.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_ffn_embed: bool = True


@dataclass
class Screen10M20MVQGainQKPostNormConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + QK-norm-post-RoPE.

    #49 — applies the modded-nanogpt QK-norm-post-RoPE variant on top
    of V+q_gain. Different mathematical operating point: the post-RoPE
    norm constrains post-RoPE Q,K magnitudes per head. Flag-only, no
    extra params. Tests whether the normalization story (where the
    norm is applied) breaks the V+q_gain plateau.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_qk_norm_post_rope: bool = True


@dataclass
class Screen10M20MVQGainSwiGLUConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + SwiGLU FFN.

    #50 — combines V+q_gain with SwiGLU FFN (instead of squared_relu).
    SwiGLU is a different FFN activation. Tests whether the FFN
    activation is part of the V+q_gain plateau. Uses d_ff=384 so the
    3-matrix SwiGLU FFN is parameter-matched to squared-ReLU d_ff=576:
    3 * d_model * 384 == 2 * d_model * 576.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    ffn_variant: str = "swiglu"
    d_ff: int = 384


@dataclass
class Screen10M20MVQGainSlidingWindowConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + sliding-window attention.

    #51 — first attention-pattern axis in the ladder. V+q_gain (the
    plateau) plus a local causal window of 512 tokens. Flag-only, no
    extra params. Tests whether the attention *pattern* (not just the
    inputs) has headroom at this scale. Window 512 = quarter of
    seq_len 2048, a clean first probe. If this beats V+q_gain
    (4.6815), the attention matrix itself was a hidden lever; if it
    ties, long-range is a wash; if it loses, long-range is load-
    bearing and SWA is closed.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512


@dataclass
class Screen10M20MSlidingWindowConfig(Screen10M20MConfig):
    """Screen10M20M with sliding-window attention ONLY (no embeds, no gains).

    #52 — clean ablation. Same window (512) as #51, but the V-embed and
    q_gain levers are off. Tests whether sliding-window attention is a
    standalone lever (in which case this should land near V+q+SWA's
    4.6700) or just a small add-on to V+q_gain (in which case this
    should land near control 4.7984, or even worse if long-range is
    load-bearing). The most informative single screen for deciding
    whether the architecture change is "use SWA" or "use V+q_gain".
    """
    use_sliding_window: bool = True
    sliding_window_size: int = 512


@dataclass
class Screen10M20MNoPEConfig(Screen10M20MConfig):
    """Screen10M20M with no positional encoding (NoPE).

    #53 — fresh axis: positional encoding. Skips the rotary call
    entirely while keeping the Q/K RMSNorm. Flag-only, no extra
    params. Tests whether RoPE is load-bearing at this scale. If
    NoPE ≈ control (4.7984), position is mostly conveyed by the
    causal mask + token identity injection (and our V-embed lever
    is partially substituting for RoPE). If NoPE << control, RoPE
    is critical and there's no slack there. If NoPE < control,
    position is hurting — surprising but worth measuring.
    """
    use_nope: bool = True


@dataclass
class Screen10M20MVQGainNoPEConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + NoPE.

    #54 — tests whether NoPE is additive with the V+q_gain plateau
    (3-seed mean 4.6815). If V+q+NoPE < 4.6815, NoPE is a real lever
    on the best baseline. If V+q+NoPE > 4.6815, RoPE is load-bearing
    for V+q and NoPE is closed. If V+q+NoPE ≈ 4.6815, position is a
    wash when paired with V+q_gain.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_nope: bool = True


@dataclass
class Screen10M20MLayerTied2Config(Screen10M20MConfig):
    """Screen10M20M with layer tying (ALBERT-style, group_size=2).

    #55 — fresh axis: weight sharing across depth. 24 layers, every
    group of 2 consecutive blocks shares weights, so 12 unique
    TransformerBlock modules are used twice each. Drops unique
    depth params by ~50% (qkvo+FFN per block). Tests whether depth
    uniqueness or depth *re-use* matters more at this scale. If
    layer_tied ≈ control (4.7984), unique depth is critical. If
    layer_tied < control, weight sharing acts as cheap
    regularization.
    """
    tie_layer_groups: int = 2


@dataclass
class Screen10M20MVQGainLayerTied2Config(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + layer tying (group_size=2).

    #56 — combines V+q_gain (the plateau, 3-seed mean 4.6815) with
    layer tying (group_size=2, 12 unique blocks). Tests whether the
    V+q lever still works when each block is used twice — i.e.
    whether V-embed projections and q_gain scalars can survive the
    weight-sharing constraint. If V+q+tied < V+q, layer tying is
    additive with V+q; otherwise it conflicts.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    tie_layer_groups: int = 2


@dataclass
class Screen10M20MGQA1Config(Screen10M20MConfig):
    """Screen10M20M with aggressive GQA (n_kv_heads=1).

    #57 — fresh axis: GQA ratio. The base config has n_kv_heads=2
    (each KV head shared across 3 Q heads). This drops it to 1 (6:1
    GQA — every Q head reads from the same K/V). Fewer KV params,
    more aggressive sharing. Tests whether the GQA ratio is a real
    architecture lever at this scale. If GQA1 < base, the model
    benefits from more attention-head sharing; if GQA1 > base, KV
    diversity is load-bearing.
    """
    n_kv_heads: int = 1


@dataclass
class Screen10M20MMHAConfig(Screen10M20MConfig):
    """Screen10M20M with full multi-head attention (n_kv_heads=n_heads).

    #58 — the other end of the GQA axis: no sharing at all.
    n_kv_heads=6 means each Q head has its own K and V projection.
    More KV params, no information sharing between heads. Tests
    whether the current 3:1 GQA is a wash (in which case MHA ≈ GQA)
    or whether more KV capacity helps. This is the "no GQA" point.
    """
    n_kv_heads: int = 6


@dataclass
class Screen10M20MVQGainMHAConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + full MHA (n_kv_heads=6).

    #59 — combines V+q_gain (the plateau, 3-seed mean 4.6815) with
    full MHA. Tests whether the GQA ratio is additive with V+q_gain
    on the best baseline. If V+q+MHA < V+q, full MHA is a real lever
    on top of the plateau; otherwise GQA is sufficient.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    n_kv_heads: int = 6


@dataclass
class Screen10M20MGELUConfig(Screen10M20MConfig):
    """Screen10M20M with GELU FFN activation (no gating, no squaring).

    #60 — fresh axis: MLP activation. The base config uses
    squared_relu (Primer-style); SwiGLU was tried and washed (#50).
    This is plain GELU, the most common transformer activation.
    Single up-projection, no gating, parameter-matched to
    squared_relu d_ff=576. Tests whether the activation is itself
    a real architecture lever — a cleaner test than SwiGLU, which
    differs in BOTH activation AND number of projections.
    """
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MVQGainGELUConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + GELU FFN.

    #61 — combines V+q_gain (the plateau, 3-seed mean 4.6815) with
    GELU FFN. Tests whether the activation swap is additive with
    V+q. If V+q+GELU < V+q, GELU is a real lever; otherwise the
    activation is a wash when V+q is in play.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MVQGainSWAGELUConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + Q-gain + sliding-window + GELU FFN.

    #62 — combines the current best screen20m levers (V+q+SWA at
    4.6700 single-seed) with the only untried MLP activation
    (GELU). Tests whether GELU is additive with the V+q+SWA
    plateau. If V+q+SWA+GELU < 4.6700, GELU is the new best add-on.
    If V+q+SWA+GELU > V+q+SWA, GELU conflicts.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MHighRoPEConfig(Screen10M20MConfig):
    """Screen10M20M with Llama-style RoPE base (500000).

    #63 — fresh axis: positional decay. The default base=10000 gives
    short wavelength (positional information blurs fast). Llama's
    500000 keeps positional information sharper over longer
    distances. Tests whether our seq_len=2048 is hitting the edge
    of the default RoPE's useful range.
    """
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPEConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + Llama-style RoPE base.

    #64 — combines the new best screen20m levers (V+q+SWA at
    4.6676 2-seed mean) with the high RoPE base. Tests whether
    Llama-style positional decay is additive with V+q+SWA.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPEGELUConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + High RoPE + GELU FFN.

    #65 — combines the new best screen20m levers (V+q+SWA+
    HighRoPE at 4.6364) with GELU FFN. Tests whether GELU is
    additive with the V+q+SWA+HighRoPE plateau.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    ffn_variant: str = "gelu"


@dataclass
class Screen10M20MVQGainSWAHighRoPETied2Config(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + High RoPE + layer tying (group=2).

    #66 — V+q+LayerTied2 was anti-additive (4.7419 vs V+q 4.6797),
    but on the new best baseline (V+q+SWA+HighRoPE 4.6364) the
    question is whether SWA+RoPE-base have changed the
    regularization story enough that tying now adds value.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    tie_layer_groups: int = 2


@dataclass
class Screen10M20MVQGainSWAHighRoPEMHAConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + High RoPE + full MHA (n_kv_heads=6).

    #67 — MHA alone was a wash on control (4.7981 vs 4.7984).
    On the new best baseline the question is whether the GQA
    ratio becomes a lever when paired with the other wins.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    n_kv_heads: int = 6


@dataclass
class Screen10M20MVQGainHighRoPESWA256Config(Screen10M20MConfig):
    """Screen10M20M with V+q + High RoPE + SWA (window=256).

    #68 — window size sweep on the V+q+HighRoPE plateau. The current
    best baseline uses window=512. Tests whether a smaller window
    (256 = 1/8 of seq_len) is better. Smaller window = more
    aggressive locality. Mask density: sum(min(256,i+1))/2048^2 =
    0.13.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 256
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainHighRoPESWA1024Config(Screen10M20MConfig):
    """Screen10M20M with V+q + High RoPE + SWA (window=1024).

    #69 — window size sweep (larger). Window=1024 = half of
    seq_len. Tests whether the default window=512 is sub-optimal.
    Mask density: sum(min(1024,i+1))/2048^2 = 0.378.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 1024
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainHighRoPENoSWAConfig(Screen10M20MConfig):
    """Screen10M20M with V+q + High RoPE + NO SWA.

    #70 — V+q+HighRoPE without SWA. Tests whether SWA is still
    load-bearing on the new RoPE-base=500000 baseline. If this
    lands at ~4.6364, SWA is redundant on top of HighRoPE. If
    it's much worse, SWA is still load-bearing.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = False
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPELogitSoftcapConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + logit softcap (Gemma-style).

    #71 — logit softcap=15.0 on the new best baseline. Real
    architecture change: logit = softcap * tanh(logit/softcap).
    Tests whether the cap is a real lever. Gemma uses 30.0; we
    test 15.0 because our model is smaller.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    logit_softcap: float = 15.0


@dataclass
class Screen10M20MVQGainSWATiedQKConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA + Tied QK (PaLM-style).

    #72 — Tied QK: Q and K share the same projection matrix.
    Real arch change. PaLM uses this as the default attention
    design. Tests whether tying QK weights is a real lever.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    use_tied_qk: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPETiedQKConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + Tied QK (PaLM-style).

    #72b — same as #72 but on the V+q+SWA+HighRoPE best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_tied_qk: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPEMLAConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + MLA (DeepSeek-V2-style).

    #73 — Multi-head Latent Attention: K,V are computed via a
    low-rank latent (d_c=mla_latent_dim, default d_model//4=36).
    Real arch change. DeepSeek-V2 uses this. Tests whether
    the latent bottleneck is a real lever on our small model.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_mla: bool = True
    mla_latent_dim: int = 36


@dataclass
class Screen10M20MVQGainSWAHighRoPEDilatedConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + Dilated Attention.

    #74 — dilation=2. Same window_size=512 by token count, but
    positions are spread (every other position in the window
    range). Tests whether strided patterns beat contiguous
    locality at this scale. Effective range: 2*512=1024 tokens.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    attention_dilation: int = 2


@dataclass
class Screen10M20MVQGainSWAHighRoPEPostNormConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + post-norm.

    #75 — fundamental arch change: norm goes AFTER the residual
    addition (original Transformer) instead of before (modern
    pre-norm). Tests whether post-norm is a real lever at
    our depth=24.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_post_norm: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPEGQA1Config(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + GQA=1 (max GQA).

    #76 — n_kv_heads=1 means every Q head reads from the same
    K,V. On the best baseline — does max GQA add or hurt?
    GQA1 standalone (#76) was bad; the question is whether
    the best baseline changes that.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    n_kv_heads: int = 1


@dataclass
class Screen10M20MVQGainSWAHighRoPENoEmbScaleConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + no embedding scale.

    #77 — the standard code multiplies the token embedding by
    sqrt(d_model). Set to 1.0 (no scaling). Tests whether the
    standard scaling is a hidden knob at this scale.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    embedding_scale: float = 1.0


@dataclass
class Screen10M20MVQGainSWAHighRoPESWAFullConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + SWA=seq_len (full).

    #78 — window=2048 = seq_len. Effectively full causal attention
    but with the SWA code path. The cleanest "is SWA helping at
    all" test on the best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 2048
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPELayerNormConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + LayerNorm (vs RMSNorm).

    #79 — RMSNorm is the default. LayerNorm is the older
    alternative with learned bias. Tests whether the choice
    of norm is a real lever on the best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_layernorm: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPELinearAttnConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + linear attention.

    #80 — Performer-style linear attention. Replaces the
    softmax(QK^T / sqrt(d_k))V with phi(Q) (phi(K)^T V) where
    phi(x) = elu(x) + 1. Different attention math (O(n) instead
    of O(n^2) in the full case, but windowed in our case).
    Tests whether linear attention unlocks a new operating point
    on the best baseline.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_linear_attn: bool = True


@dataclass
class Screen10M20MVQGainSWAHighRoPEQKPostNormConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA+HighRoPE + QK norm after RoPE.

    #81 — modded-nanogpt style Q/K normalization position, but on
    the current best baseline. V+q alone tied with this knob; the
    question is whether SWA+HighRoPE changes the operating point.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 500000
    use_qk_norm_post_rope: bool = True


@dataclass
class Screen10M20MVQGainHighRoPESWA384Config(Screen10M20MConfig):
    """Screen10M20M with V+q + HighRoPE + SWA(window=384).

    #82 — finer locality sweep between the losing 256 window and
    the current best 512 window.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 384
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainHighRoPESWA768Config(Screen10M20MConfig):
    """Screen10M20M with V+q + HighRoPE + SWA(window=768).

    #83 — finer locality sweep between the current best 512 window
    and the losing 1024 window.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 768
    rope_base: int = 500000


@dataclass
class Screen10M20MVQGainSWAHighRoPE250KConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA(window=512) + RoPE base 250k.

    #84 — finer RoPE-base sweep. Default 10k lost, 500k won; this
    checks whether the optimum sits below 500k.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 250000


@dataclass
class Screen10M20MVQGainSWAHighRoPE1MConfig(Screen10M20MConfig):
    """Screen10M20M with V+q+SWA(window=512) + RoPE base 1M.

    #85 — finer RoPE-base sweep above the current 500k winner.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 512
    rope_base: int = 1000000


@dataclass
class Screen10M20MVOQGainConfig(Screen10M20MConfig):
    """Screen10M20M with V+O + per-head Q-gain. Best embed (V+O 4.7188)
    + non-embed lever (q_gain). Tests whether q_gain is additive
    with V+O.
    """
    use_value_embed: bool = True
    use_output_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Screen10M20MVQGainConfig(Screen10M20MConfig):
    """Screen10M20M with V-embed + per-head Q-gain (no O-embed).

    #39 — partial-ablation probe. Best arch (V+O+q_gain = 4.6789 mean
    across 3 seeds) drops the O-embed. Tests whether O-embed is the
    necessary addition or whether q_gain alone is enough to push
    V-embed down. If V+q_gain ≈ V+O+q_gain, O is unnecessary. If
    V+q_gain >> V but V+q_gain << V+O+q_gain, O is the load-bearing
    piece.
    """
    use_value_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Screen10M20MVQQGainConfig(Screen10M20MConfig):
    """Screen10M20M with V+Q + per-head Q-gain (no O-embed).

    #40 — V+Q+q_gain is an alternative to V+O+q_gain. Tests
    whether V+O is the unique best embed combo, or whether V+Q
    with q_gain also helps. If V+Q+q_gain ≈ V+O+q_gain, O and Q
    are interchangeable (just a different place to put the embed).
    If V+Q+q_gain > V+O+q_gain, Q is the better add-on.
    """
    use_value_embed: bool = True
    use_query_embed: bool = True
    use_q_gain: bool = True


@dataclass
class Screen10M20MQGainConfig(Screen10M20MConfig):
    """Screen10M20M with per-head Q-gain (no embeds).

    #41 — pure non-embed lever. Tests whether q_gain is the
    load-bearing piece (if so, this should land near V+O+q_gain).
    If q_gain alone is in noise, q_gain is only additive WITH
    the embeds.
    """
    use_q_gain: bool = True


# ============================================================================
# FULL ladder — 20x tokens (compute-optimal / Chinchilla). Transfer-valid: this
# is where a mechanism's real optimum is locked. Ladder 10M→25M→50M→135M lets
# you fit optimum-vs-size and extrapolate to the 135M release target. Same
# architecture at every size (RoPE + GQA + RMSNorm + squared-ReLU + Muon);
# scaling is hyperparameters + engineering, not an architecture change.
# Param counts use tied embeddings (vocab 49,152).
# ============================================================================


@dataclass
class Full10M200MConfig(LLMConfig):
    """Ladder — ~7.7M params · 200M tokens (20x) · ~48,800 steps. The 10m record target.

    The 10M architecture: low-rank embedding (emb_rank=48) + depth (24 layers),
    same shape as Screen10M20MConfig but trained to the 20x regime — the cheapest
    transfer-valid point, runnable locally. First rung of the release ladder.
    """
    d_model: int = 144
    n_heads: int = 6
    n_layers: int = 24
    d_ff: int = 576
    n_kv_heads: int = 2
    emb_rank: int = 48
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 200_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 48800, 2000))


@dataclass
class Full10M200MOutputAdapterConfig(Full10M200MConfig):
    """Full10M200M with a rank-32 additive output adapter."""
    output_adapter_rank: int = 32


@dataclass
class Full10M200MSmearGateConfig(Full10M200MConfig):
    """Full10M200M with SmearGate previous-token embedding blend."""
    use_smear_gate: bool = True


@dataclass
class Full10M200MUNetSkipConfig(Full10M200MConfig):
    """Full10M200M with zero-init U-Net skip bridges across depth."""
    use_unet_skips: bool = True


@dataclass
class Full10M200MAttnOutputGateConfig(Full10M200MConfig):
    """Full10M200M with per-head attention-output gates."""
    use_attn_output_gate: bool = True


@dataclass
class Full10M200MLayerScaleConfig(Full10M200MConfig):
    """Full10M200M with per-channel attention/MLP LayerScale gates."""
    use_layerscale: bool = True


@dataclass
class Full10M200MValueEmbedConfig(Full10M200MConfig):
    """Full10M200M with token value embeddings injected into attention V."""
    use_value_embed: bool = True


@dataclass
class Full10M200MQueryEmbedConfig(Full10M200MConfig):
    """Full10M200M with token query embeddings injected into attention Q."""
    use_query_embed: bool = True


@dataclass
class Full10M200MKeyEmbedConfig(Full10M200MConfig):
    """Full10M200M with token key embeddings injected into attention K."""
    use_key_embed: bool = True


@dataclass
class Full135M2700MConfig(LLMConfig):
    """Release target — ~134.5M params · 2.7B tokens (20x). SmolLM2-135M class.

    The model we race to release: benchmark head-to-head vs SmolLM2-135M.
    """

    d_model: int = 576
    n_heads: int = 9          # head_dim 64
    n_layers: int = 30
    d_ff: int = 2304          # 4x d_model
    n_kv_heads: int = 3       # 3:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 2_700_000_000  # ~20x params (Chinchilla-optimal)


# ============================================================================
# RELEASE LADDER — scaling-law rungs (see autoresearch/LADDER.md)
# ----------------------------------------------------------------------------
# Rungs for fitting L(N) and extrapolating to the 135M release point
# (Full135M2700MConfig). Same architecture FAMILY as the target — full tied
# embeddings, head_dim 64, d_ff = 4·d_model, RoPE + RMSNorm + sq-ReLU + Muon,
# Chinchilla 20×-total-params token budget. SHAPE (depth, width, n_heads, GQA
# ratio) is deliberately VARIED to span ~1.9 decades of non-embedding N
# LOG-UNIFORMLY — the log(N) lever arm is what pins the exponent, and loss
# depends mainly on non-embed N and only weakly on shape (Kaplan et al. 2020),
# so spreading N across orders of magnitude matters far more than holding the
# shape fixed. Two nearly-redundant clustered rungs are NOT what you want.
#
# FIT ON NON-EMBEDDING N (the 49k-vocab embedding dominates total params at
# small N and does not follow the transformer's scaling — standard Kaplan/
# Chinchilla convention). Non-embed N per rung: 1.45M, 3.17M, 10.9M, 33.2M →
# extrapolate to 106.8M (the target). Train the same arch (champion + a carry-
# lever) at every rung; the rung whose curve bends BELOW the baseline's wins.
# Distinct from Full10M200MConfig (a low-rank-emb "10m record" model, not a rung).
# Param/token counts verified by build. log10(non_embed N): 6.16/6.50/7.04/7.52/8.03.
# ============================================================================


@dataclass
class Ladder8M155MConfig(LLMConfig):
    """Ladder rung 1 — 7.7M total / 1.45M non-embed · 155M tokens (20×). Local-runnable.

    Smallest rung — extends the log(N) lever arm cheaply (the high-ROI end).
    head_dim 64, GQA 2:1.
    """
    d_model: int = 128
    n_heads: int = 2          # head_dim 64
    n_layers: int = 8
    d_ff: int = 512           # 4× d_model
    n_kv_heads: int = 1       # 2:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 155_000_000


@dataclass
class Ladder13M252MConfig(LLMConfig):
    """Ladder rung 2 — 12.6M total / 3.17M non-embed · 252M tokens (20×). Local-runnable.

    head_dim 64, GQA 3:1.
    """
    d_model: int = 192
    n_heads: int = 3          # head_dim 64
    n_layers: int = 8
    d_ff: int = 768           # 4× d_model
    n_kv_heads: int = 1       # 3:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 252_000_000


@dataclass
class Ladder23M469MConfig(LLMConfig):
    """Ladder rung 3 — 23.5M total / 10.9M non-embed · 469M tokens (20×). Local-runnable.

    head_dim 64, GQA 2:1.
    """
    d_model: int = 256
    n_heads: int = 4          # head_dim 64
    n_layers: int = 15
    d_ff: int = 1024          # 4× d_model
    n_kv_heads: int = 2       # 2:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 469_000_000


@dataclass
class Ladder52M1042MConfig(LLMConfig):
    """Ladder rung 4 — 52.1M total / 33.2M non-embed · 1.04B tokens (20×). Needs a contributor GPU.

    Closest sub-target anchor. head_dim 64, GQA 3:1.
    """
    d_model: int = 384
    n_heads: int = 6          # head_dim 64
    n_layers: int = 21
    d_ff: int = 1536          # 4× d_model
    n_kv_heads: int = 2       # 3:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 1_042_000_000


# ============================================================================
# Query-tweaks plan — 29 Screen10M20M<Name>Config recipes (Batches 1-6).
# See docs/research-plans/query-tweaks/plan.md and manifest.md.
# ============================================================================

# ---- Batch 1: high-signal levers (Q1-Q4) ----

@dataclass
class Tiny1M3MBlockTempConfig(Tiny1M3MConfig):
    """Tiny1M3M with a fixed cosine-depth attention temperature
    schedule (193). Each block `b` multiplies its pre-softmax
    attention score by `τ_b = 1 + α · cos(π · b / L)`, where
    `L = n_layers` (12 at tiny1m3m) and `α = block_temp_alpha`
    (default `-0.3` here ⇒ early-layer sharpen / late-layer soften,
    consistent with the 175-alibi locality-rewarding prior on the
    multiplicative depth-varying side). The buffer of `τ_b` values
    is computed once at model construction and divided into
    `Q·K^T/√d_k` in the manual attention path.

    A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`,
    cache-mean 6.3988 ± 0.04, n=3, measured 2026-06-15). 0 new
    parameters (the schedule is hard-coded). At `α=0` the path
    is byte-identical to baseline (`τ_b = 1` for all `b`). Forces
    the manual attention path so SDPA's flash kernel doesn't fuse
    QK^T+softmax+AV. The committed single value is `α = -0.3`
    (per the r2 sign-convention resolution in `idea.md` — `α > 0`
    would be DRIFT-bound and is closed).

    PASS: `trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl rule.
    NULL: `|trt_val − ctrl_val| < 0.01` (closes the fixed-shape
    depth-conditional scale axis).
    DRIFT: `trt_val > ctrl_val + 0.01` (sharpen-early past the
    locality-rewarding optimum).
    CONDITIONAL: 188-qk-rms-scaling is the *learned* sibling on
    the same axis. If 188 has reported a WIN ≥ −0.005 before 193
    is committed to the queue, redirect 193 to a different axis
    per the conditional in `idea.md`. Otherwise run as planned —
    a 193 null closes the fixed-shape axis decisively; a 193 WIN
    opens a new fixed-prior lever on the depth-conditional
    multiplicative side.

    See `autoresearch/ideas/193-blockwise-attn-temp-schedule/
    {idea,plan}.md` for the full mechanism, the
    sharpen-early-soften-late sign convention, and the 188-
    conditional framing.
    """
    use_block_temp_schedule: bool = True
    block_temp_alpha: float = -0.3


