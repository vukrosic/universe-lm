from .llm_config import Blueberry24GBConfig
from dataclasses import dataclass

@dataclass
class SFTConfig(Blueberry24GBConfig):
    # SFT Specifics
    # Lower LR to preserve pre-training knowledge
    muon_lr: float = 0.0003      # 10x lower than pretrain
    adamw_lr: float = 0.00003
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 12
    
    # Optimization
    compile_model: bool = True
    
    # Epochs (handled by max_steps in trainer usually, but conceptual)
    # We will likely explicitly set max_steps based on dataset size for 1 epoch.
    warmup_ratio: float = 0.03
    
    # use_moe: bool = False (Already in base)
