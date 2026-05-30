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
class TwoStepDebugConfig(LLMConfig):
    """Two-step debug preset: smallest possible run to test plumbing fast.

    Not for science — just verifies the pipeline end-to-end (on a MacBook or
    anywhere) in seconds. batch_size=1 + compile off keeps memory/startup
    minimal. Floor is train_tokens=2048 (one step at seq 2048, which the data is
    chunked at); 4096 = 2 steps so you can see loss move. 500 tokens = 0 steps.
    """

    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    n_kv_heads: int = 1
    max_seq_len: int = 2048   # must match the pre-chunked data; do not lower
    train_tokens: int = 4096  # 2 steps at batch_size=1
    batch_size: int = 1
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


# ============================================================================
# Release target (135M). Same architecture as every smaller size: dense decoder,
# RoPE + GQA + RMSNorm + squared-ReLU + Muon. Scaling is a hyperparameter
# + engineering problem, NOT an architecture change. Shape follows fixed ratios
# (head_dim 64, d_ff = 4x d_model, GQA ~4:1) so larger sizes are "just numbers".
# Verified param counts use tied embeddings (vocab 49,152).
# ============================================================================


@dataclass
class OneHundredThirtyFiveMillionConfig(LLMConfig):
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
