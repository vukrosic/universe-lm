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
class Screen10M20MSwiGLUConfig(Screen10M20MConfig):
    """Screen10M20M with SwiGLU feed-forward blocks."""
    ffn_variant: str = "swiglu"


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
