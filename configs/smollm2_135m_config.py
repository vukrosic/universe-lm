from dataclasses import dataclass
from .llm_config import MoEModelConfig

@dataclass
class SmolLM2_135M_Config(MoEModelConfig):
    # Architecture params for SmolLM2-135M
    d_model: int = 576
    n_heads: int = 9
    n_layers: int = 30
    d_ff: int = 2304  # Adjusted to match ~135M params with standard MLP
    
    # GQA params
    n_kv_heads: int = 3  # 9/3 = 3 groups
    
    # Dense model settings
    use_moe: bool = False
    
    # Data params
    max_seq_len: int = 2048
    vocab_size: int = 49152
    
    # Training defaults (can be overridden)
    batch_size: int = 4  # Adjusted for small model
    
    def __post_init__(self):
        super().__post_init__()
