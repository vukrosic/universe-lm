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
    # ⚠️ WARNING: For simplicity, I recomend not changing max_seq_len
    # If you change max_seq_len, you MUST re-run data preparation!
    # The data preparation script chunks data at this exact length, and the RoPE
    # cache is initialized with this value. Mismatches will cause runtime errors.
    # Run: python data/prepare_mix_data.py --target_tokens 25_000_000
    # you may change the number of tokens
    max_seq_len: int = 2048  # check the warning above
    vocab_size: int = 49152  
    
    # Base Training Defaults
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
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class ResearchConfig(LLMConfig):
    """Legacy research preset: 25,366,272 parameters."""

    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1536
    n_kv_heads: int = 4
    max_seq_len: int = 1024
    train_tokens: int = 25_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class FastResearchConfig(LLMConfig):
    """Fast smoke-test preset: 14,026,240 parameters."""

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 1024
    n_kv_heads: int = 2
    max_seq_len: int = 512
    train_tokens: int = 1_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class UniverseSmokeConfig(LLMConfig):
    """v0.0 smoke release: ~15M params, short MacBook-friendly run."""

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    n_kv_heads: int = 2
    max_seq_len: int = 1024
    train_tokens: int = 50_000_000
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5
    batch_size: int = 4
    compile_model: bool = False


@dataclass
class FiveMillionConfig(LLMConfig):
    """Tiny pipeline preset: 6,652,800 parameters."""

    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 512
    n_kv_heads: int = 1
    max_seq_len: int = 2048
    train_tokens: int = 134_000_000  # 20x params
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class TwentyFiveMillionConfig(LLMConfig):
    """Small scaling-law preset: 25,366,272 parameters."""

    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1536
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    train_tokens: int = 507_000_000  # 20x params
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class FiftyMillionConfig(LLMConfig):
    """Mid scaling-law preset: 48,244,224 parameters."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    train_tokens: int = 965_000_000  # 20x params
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class HundredMillionConfig(LLMConfig):
    """Large scaling-law preset: 100,169,472 parameters."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 26
    d_ff: int = 2048
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    train_tokens: int = 2_000_000_000  # 20x params
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


# ============================================================================
# Release ladder (135M -> 1B). Same architecture at every size: dense decoder,
# RoPE + GQA + RMSNorm + squared-ReLU + Muon. Scaling these is a hyperparameter
# + engineering problem, NOT an architecture change. Shape follows fixed ratios
# (head_dim 64, d_ff = 4x d_model, GQA ~4:1) so larger sizes are "just numbers".
# Verified param counts use tied embeddings (vocab 49,152).
# ============================================================================


@dataclass
class OneThirtyFiveMillionConfig(LLMConfig):
    """Release-target preset: ~134.5M params (SmolLM2-135M class).

    Sized to train compute-optimal (~2.7B tokens) on a single rented GPU and
    benchmark head-to-head vs SmolLM2-135M / Gemma-3-270M.
    """

    d_model: int = 576
    n_heads: int = 9          # head_dim 64
    n_layers: int = 30
    d_ff: int = 2304          # 4x d_model
    n_kv_heads: int = 3       # 3:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 2_700_000_000  # ~20x params (Chinchilla-optimal)
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class FiveHundredMillionConfig(LLMConfig):
    """Scaling ladder preset: ~0.5B params. Needs distributed training to reach
    compute-optimal (~10B tokens) -- here for the config ladder, not yet trained."""

    d_model: int = 1280
    n_heads: int = 20         # head_dim 64
    n_layers: int = 26
    d_ff: int = 5120          # 4x d_model
    n_kv_heads: int = 5       # 4:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 10_000_000_000  # ~20x params
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


@dataclass
class OneBillionConfig(LLMConfig):
    """Scaling ladder preset: ~1B params. Requires multi-GPU FSDP + FlashAttention
    to be practical (~20B tokens compute-optimal). Config-only for now."""

    d_model: int = 1536
    n_heads: int = 24         # head_dim 64
    n_layers: int = 36
    d_ff: int = 6144          # 4x d_model
    n_kv_heads: int = 6       # 4:1 GQA
    max_seq_len: int = 2048
    train_tokens: int = 20_000_000_000  # ~20x params
    activation_variant: str = "squared_relu"
    activation_slope: float = 0.5


def make_config(d_model: int, n_layers: int, *, head_dim: int = 64,
                gqa_ratio: int = 4, ff_mult: float = 4.0,
                max_seq_len: int = 2048, vocab_size: int = 49152,
                **overrides) -> LLMConfig:
    """Generate a config from a few shape knobs, deriving the rest from ratios.

    The future-proofing primitive: new sizes (0.5B, 1B, ...) are just different
    (d_model, n_layers) here -- heads/KV/FFN follow fixed ratios, so the named
    presets above are simply pinned points on this curve. Any LLMConfig field
    can be set via **overrides (e.g. train_tokens=..., muon_lr=...).
    """
    assert d_model % head_dim == 0, "d_model must be divisible by head_dim"
    n_heads = d_model // head_dim
    n_kv_heads = max(1, n_heads // gqa_ratio)
    while n_heads % n_kv_heads != 0:   # GQA requires n_kv_heads | n_heads
        n_kv_heads -= 1
    cfg = LLMConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=int(ff_mult * d_model), n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len, vocab_size=vocab_size,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
