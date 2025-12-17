from dataclasses import dataclass
from .llm_config import MoEModelConfig

@dataclass
class SmolLM2_135M_Pow2_Config(MoEModelConfig):
    # Architecture params - Powers of 2 optimized
    d_model: int = 512       # 2^9
    n_heads: int = 8         # 2^3
    n_layers: int = 32       # 2^5 (Increased from 30 to match param count approx)
    d_ff: int = 2048         # 2^11
    
    # GQA params
    n_kv_heads: int = 4      # 2^2 (GQA group size 2)
    
    # Dense model settings
    use_moe: bool = False
    
    # Data params
    max_seq_len: int = 2048  # 2^11
    vocab_size: int = 49152  # Not power of 2, kept from tokenizer
    
    # Training defaults
    batch_size: int = 4
    
    def __post_init__(self):
        super().__post_init__()
