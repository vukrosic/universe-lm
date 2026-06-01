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
    per_group_lr: bool = False
    embedding_lr_multiplier: float = 2.0
    scalar_lr_multiplier: float = 1.5

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
class Screen3M32KConfig(LLMConfig):
    """Fast screen — ~3.2M params · 32k tokens · 8 steps. Plumbing + sign/basin in seconds.
    """
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    n_kv_heads: int = 1
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 32_768
    compile_model: bool = False
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(8))


@dataclass
class Screen10M20MConfig(LLMConfig):
    """Screen — ~10M params · 20M tokens · ~4880 steps. Confirms sign survives more tokens.
    """
    d_model: int = 144
    n_heads: int = 6
    n_layers: int = 3
    d_ff: int = 576
    n_kv_heads: int = 2
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 20_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 4880, 200))


@dataclass
class Screen10M20MPerGroupLRConfig(Screen10M20MConfig):
    """Screen with per-group LR enabled for embeddings / scalar params."""
    per_group_lr: bool = True
    embedding_lr_multiplier: float = 2.0
    scalar_lr_multiplier: float = 1.5


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
    """Ladder — ~10M params · 200M tokens (20x) · ~48,800 steps.

    Same shape as Screen10M20MConfig but trained to the 20x regime — the cheapest
    transfer-valid point, runnable locally. First rung of the release ladder.
    """
    d_model: int = 144
    n_heads: int = 6
    n_layers: int = 3
    d_ff: int = 576
    n_kv_heads: int = 2
    max_seq_len: int = 2048
    batch_size: int = 2
    train_tokens: int = 200_000_000
    compile_model: bool = False
    warmup_ratio: float = 0.02
    schedule_type: str = "warmup_decay_to_zero"
    eval_milestones: Optional[Tuple[int, ...]] = tuple(range(0, 48800, 2000))


@dataclass
class Full10M200MPerGroupLRConfig(Full10M200MConfig):
    """Full 10M ladder run with per-group LR enabled."""
    per_group_lr: bool = True
    embedding_lr_multiplier: float = 2.0
    scalar_lr_multiplier: float = 1.5


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
