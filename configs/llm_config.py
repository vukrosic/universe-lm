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
    # #21 LayerScale: zero-init per-channel scales on attention and MLP residual
    # outputs. Starts as exact baseline via branch *= (1 + gate).
    use_layerscale: bool = False
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
    # #99 Attention sink slot (softmax-off-by-one): append a zero K/V the query
    # can attend to, so it isn't forced to dump probability on a real token.
    use_attn_sink: bool = False

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
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"
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
class Tiny1M3MCautiousMuonConfig(Tiny1M3MConfig):
    """Tiny1M3M with cautious-muon sign-mask + small LR bump.

    A/B vs the tiny1m ctrl (6.4306) — should land ≤ 6.4206 for a pass.
    """
    use_cautious_muon: bool = True
    muon_lr: float = 0.025  # +4% to compensate for masked components

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
# Query-tweaks plan — 29 Screen10M20M<Name>Config recipes (Batches 1-6).
# See docs/research-plans/query-tweaks/plan.md and manifest.md.
# ============================================================================

# ---- Batch 1: high-signal levers (Q1-Q4) ----

@dataclass
class Screen10M20MAlibiBiasConfig(Screen10M20MConfig):
    """Q1 — ALiBi-style per-head distance bias. scores += -m_h·(i-j)."""
    use_alibi_bias: bool = True

@dataclass
class Screen10M20MQTempTokenConfig(Screen10M20MConfig):
    """Q2 — Token-conditioned per-head Q temperature. Q *= (1 + tanh(x·w_h))."""
    use_q_temp_token: bool = True

@dataclass
class Screen10M20MCosineAttnConfig(Screen10M20MConfig):
    """Q3 — Cosine attention. L2-normalize Q,K; per-head learnable τ."""
    use_cosine_attn: bool = True

@dataclass
class Screen10M20MQKBilinearConfig(Screen10M20MConfig):
    """Q4 — Per-channel relevance. score = Q^T diag(d_h) K (d_h init 1)."""
    use_qk_bilinear: bool = True

# ---- Batch 2: flagship + positional (Q5-Q7) ----

@dataclass
class Screen10M20MTalkingHeadsQConfig(Screen10M20MConfig):
    """Q5 — Talking-heads on Q. learned n_h×n_h M on attention logits pre-softmax."""
    use_talking_heads_q: bool = True

@dataclass
class Screen10M20MPerHeadRopeBaseConfig(Screen10M20MConfig):
    """Q6 — Per-head learnable RoPE base. θ_h init = global base."""
    use_per_head_rope_base: bool = True

@dataclass
class Screen10M20MPartialRotaryConfig(Screen10M20MConfig):
    """Q7 — Partial rotary. Rotate only 50% of Q/K dims."""
    partial_rotary_p: float = 0.5

# ---- Batch 3: exotic (Q8-Q10) ----

@dataclass
class Screen10M20MQExpansionConfig(Screen10M20MConfig):
    """Q8 — Multi-query expansion. Q += W·x (zero-init W; step-0 baseline)."""
    use_q_expansion: bool = True

@dataclass
class Screen10M20MDecoupledContentPosConfig(Screen10M20MConfig):
    """Q9 — Decoupled content/position attention (DeBERTa-style)."""
    use_decoupled_content_pos: bool = True

@dataclass
class Screen10M20MAntisymQKConfig(Screen10M20MConfig):
    """Q10 — Antisymmetric Q·K coupling. +Q^T S K with skew S (init 0)."""
    use_antisym_qk: bool = True

# ---- Batch 4: query-norm zoo (Q11-Q16) ----

@dataclass
class Screen10M20MNormPNormConfig(Screen10M20MConfig):
    """Q11 — Q-side pnorm p=1.5 (Lp norm, outlier-robust)."""
    q_norm_type: str = "pnorm1.5"

@dataclass
class Screen10M20MNormClipConfig(Screen10M20MConfig):
    """Q12 — Q-side Winsorized RMSNorm (clip k=3)."""
    q_norm_type: str = "clipnorm3"

@dataclass
class Screen10M20MNormChannelScaleConfig(Screen10M20MConfig):
    """Q13 — Q-side ChannelScale (learnable pre-scale)."""
    q_norm_type: str = "channelscale"

@dataclass
class Screen10M20MNormManhattanConfig(Screen10M20MConfig):
    """Q14 — Q-side Manhattan (L1 MAD) norm."""
    q_norm_type: str = "manhattan"

@dataclass
class Screen10M20MNormCenterConfig(Screen10M20MConfig):
    """Q15 — Q-side Center norm (mean-only, no variance)."""
    q_norm_type: str = "center"

@dataclass
class Screen10M20MNormNoneConfig(Screen10M20MConfig):
    """Q16 — Q-side norm disabled. K still normed."""
    q_norm_type: str = "none"

# ---- Batch 5: learnable-param zoo (Q17-Q23) ----

@dataclass
class Screen10M20MQPerHeadBiasConfig(Screen10M20MConfig):
    """Q17 — Per-head bias. Q += b_h (per-head×channel) post-RoPE."""
    use_q_per_head_bias: bool = True

@dataclass
class Screen10M20MQPerChannelGainConfig(Screen10M20MConfig):
    """Q18 — Per-channel gain. Q *= g_d post-RoPE."""
    use_q_per_channel_gain: bool = True

@dataclass
class Screen10M20MQHDGainConfig(Screen10M20MConfig):
    """Q19 — Head×channel gain. Q *= g_hd post-RoPE."""
    use_q_hd_gain: bool = True

@dataclass
class Screen10M20MQNormGateConfig(Screen10M20MConfig):
    """Q20 — Norm-gate. per-head scalar σ(a_h·‖x‖+b_h) on Q."""
    use_q_norm_gate: bool = True

@dataclass
class Screen10M20MQLowRankRefineConfig(Screen10M20MConfig):
    """Q21 — Low-rank refine. Q += (W1·x)@W2 (zero-init)."""
    use_q_lowrank_refine: bool = True

@dataclass
class Screen10M20MQLayerScaleConfig(Screen10M20MConfig):
    """Q22 — LayerScale on Q. Q *= (1 + ls_d) per-channel post-RoPE."""
    use_q_layerscale: bool = True

@dataclass
class Screen10M20MQSoftplusGainConfig(Screen10M20MConfig):
    """Q23 — Softplus gain. Q *= softplus(g_h) per-head — always ≥ 0."""
    use_q_softplus_gain: bool = True

# ---- Batch 6: architecture/mixing (Q24-Q29) ----

@dataclass
class Screen10M20MQHeadMixConfig(Screen10M20MConfig):
    """Q24 — Head-mix. Q ← Q + Q @ M (M−I init 0) pre-attention."""
    use_q_head_mix: bool = True

@dataclass
class Screen10M20MQTimeConvConfig(Screen10M20MConfig):
    """Q25 — Time-conv. 1D conv k=3 over position axis, zero-init."""
    use_q_time_conv: bool = True

@dataclass
class Screen10M20MQEMASmoothConfig(Screen10M20MConfig):
    """Q26 — EMA-smooth over position. Q ← α·Q + (1−α)·Q_{t-1}."""
    use_q_ema_smooth: bool = True

@dataclass
class Screen10M20MQFeatureMapConfig(Screen10M20MConfig):
    """Q27 — Feature-map attention. NOT identity-init — needs own control."""
    use_q_feature_map: bool = True

@dataclass
class Screen10M20MQPerTokenRopeConfig(Screen10M20MConfig):
    """Q28 — Per-token RoPE. Each token's θ via small MLP on x."""
    use_q_per_token_rope: bool = True

@dataclass
class Screen10M20MQNoiseRegConfig(Screen10M20MConfig):
    """Q29 — Noise reg. Q += N(0, σ²) training only (learnable σ)."""
    use_q_noise_reg: bool = True


# =====================================================================
# Cautious-Muon recipes — appended here (not above) because they
# reference classes defined later in the file. Single-line addition to
# the Muon optimizer step (Liang et al. 2024, arXiv 2411.16085):
# zero out the orthogonalized update where its sign disagrees with the
# current gradient. See optimizers/muon.py for the implementation.
# =====================================================================


@dataclass
class Screen10M20MCautiousMuonConfig(Screen10M20MConfig):
    """Screen10M20M with cautious-muon sign-mask + small LR bump.

    A/B vs the screen20m control (4.8487) — should land ≤ 4.8387 for a pass.
    """
    use_cautious_muon: bool = True
    muon_lr: float = 0.025


@dataclass
class Screen10M20MVQGainSWAHighRoPECautiousMuonConfig(Screen10M20MVQGainSWAHighRoPEConfig):
    """V+q+SWA+HighRoPE best baseline + cautious-muon sign-mask.

    A/B vs the current screen20m best (4.6364) — tests whether cautious-Muon
    is additive on top of the V+q+SWA+HighRoPE plateau. Multi-seed confirm
    if single-seed wins.
    """
    use_cautious_muon: bool = True
    muon_lr: float = 0.025


# =====================================================================
