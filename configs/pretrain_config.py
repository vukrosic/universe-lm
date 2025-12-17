from .llm_config import SmolLM2_135M_Pow2_Config
from dataclasses import dataclass

@dataclass
class PretrainConfig(SmolLM2_135M_Pow2_Config):
    # Training
    # For H100: batch_size=128, grad_accum=16 -> 2048 seqs -> ~4M tokens
    # For 4090: batch_size=4, grad_accum=12 -> 48 seqs -> ~100k tokens
    
    # We set defaults for H100 SCALE (assuming user might change manually for 4090 if needed, 
    # but let's stick to the 4090 settings as defaults since that's where we test).
    
    # 4090 Defaults
    batch_size: int = 4
    gradient_accumulation_steps: int = 12 
    
    # H100 Settings (Uncomment to use)
    # batch_size: int = 128
    # gradient_accumulation_steps: int = 16
    
    # Optimization
    compile_model: bool = True
    
    # Checkpointing
    save_every: int = 1000
    
    # Learning Rate (Aggressive for pre-training)
    muon_lr: float = 0.003       # Slightly lower than extreme, safe start
    adamw_lr: float = 0.0003
    warmup_ratio: float = 0.01  # Fast warmup
    
    # Dataset
    # We will pass the specific parsed dataset path at runtime
