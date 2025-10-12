"""
Common utilities for benchmarking models
Shared across all benchmark scripts
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer


def load_model_from_checkpoint(checkpoint_path, device='cuda', dtype=torch.bfloat16):
    """
    Load a model from checkpoint - works with any experiment
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        dtype: Data type (bf16/fp16 for Flash Attention)
    
    Returns:
        model, config, tokenizer
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Detect experiment type from path and import appropriate classes
    exp_path = checkpoint_path.parent.parent
    exp_name = exp_path.name
    
    # Import experiment-specific modules
    import sys
    import os
    root_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(root_dir))
    
    if 'exp7' in exp_name or 'exp6' in exp_name:
        # Exp6/7 use FLA DeltaNet
        if 'exp7' in exp_name:
            from experiments.exp7_hybrid_deltanet_ablation.models import GatedDeltaNetWrapper
            from experiments.exp7_hybrid_deltanet_ablation.config import ExperimentConfig
        else:
            from experiments.exp6_gated_deltanet_training.models import GatedDeltaNetWrapper
            from experiments.exp6_gated_deltanet_training.config import ExperimentConfig
        
        # Load checkpoint
        torch.serialization.add_safe_globals([ExperimentConfig])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        config = checkpoint['config']
        
        # Print model info
        if hasattr(config, 'attn_config') and config.attn_config is not None:
            print(f"✓ Hybrid model: Attention on layers {config.attn_config.get('layers', [])}")
        else:
            print(f"✓ Pure DeltaNet model")
        
        # Create and load model
        model = GatedDeltaNetWrapper(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        
    else:
        raise ValueError(f"Unknown experiment type: {exp_name}")
    
    # Print summary
    print(f"✅ Model loaded successfully!")
    print(f"   Experiment: {exp_name}")
    print(f"   Steps trained: {checkpoint.get('global_step', 'N/A')}")
    print(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Precision: {dtype}")
    
    return model, config, tokenizer


def get_model_info(checkpoint_path):
    """Get basic info about a checkpoint without loading the full model"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    info = {
        'global_step': checkpoint.get('global_step', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
        'epoch': checkpoint.get('epoch', 'N/A'),
    }
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        info['hidden_size'] = getattr(config, 'hidden_size', 'N/A')
        info['num_layers'] = getattr(config, 'num_hidden_layers', 'N/A')
        info['num_heads'] = getattr(config, 'num_attention_heads', 'N/A')
        
        if hasattr(config, 'attn_config') and config.attn_config is not None:
            info['model_type'] = 'Hybrid'
            info['attention_layers'] = config.attn_config.get('layers', [])
        else:
            info['model_type'] = 'Pure DeltaNet'
    
    return info


def get_device_and_dtype():
    """Get appropriate device and dtype for evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16 precision")
        else:
            dtype = torch.float16
            print("Using float16 precision")
    else:
        dtype = torch.float32
        print("Using float32 precision (CPU mode)")
    
    return device, dtype

