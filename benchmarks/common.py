
import torch
from pathlib import Path
from transformers import AutoTokenizer
import sys
import os

# Add root directory to path
# benchmarks/common.py -> benchmarks/ -> root/
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.append(str(root_dir))

from models.llm import MinimalLLM
from configs.llm_config import LLMConfig

def load_model_from_checkpoint(checkpoint_path, device='cuda', dtype=torch.bfloat16):
    """
    Load a model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        dtype: Data type
    
    Returns:
        model, config, tokenizer
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    # Safe globals to allow loading custom config class
    torch.serialization.add_safe_globals([LLMConfig])
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: Config not found in checkpoint, using default LLMConfig")
        config = LLMConfig()

    # Ensure max_seq_len is large enough for benchmarks
    # RoPE can handle extrapolation/longer sequences to some extent, but cache must be initialized
    if hasattr(config, 'max_seq_len') and config.max_seq_len < 2048:
        print(f"Adjusting max_seq_len from {config.max_seq_len} to 2048 for evaluation")
        config.max_seq_len = 2048

    # Create model
    # Note: the config object should already have the right parameters
    model = MinimalLLM(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback if checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # Load tokenizer - assuming SmolLM-135M as used in training
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Precision: {dtype}")
    
    return model, config, tokenizer

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


class HFModelAdapter(torch.nn.Module):
    """Wrap a HuggingFace CausalLM so the benchmark suite can call it exactly
    like a universe-lm MinimalLLM.

    The benchmarks call the model two ways:
      - ARC:       outputs = model(input_ids, labels=input_ids); outputs.loss
      - HellaSwag: logits  = model(input_ids, return_aux_loss=False)
    HF models support `labels=` natively and return `.logits`; we just absorb
    the extra `return_aux_loss` kwarg and return raw logits when no labels.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, input_ids, labels=None, return_aux_loss=False, **kwargs):
        outputs = self.hf(input_ids=input_ids, labels=labels)
        if labels is not None:
            return outputs            # has .loss (HF computes the shift internally)
        return outputs.logits         # raw logits tensor


def load_hf_model(model_name, device='cuda', dtype=torch.bfloat16):
    """Load a HuggingFace baseline (e.g. 'HuggingFaceTB/SmolLM2-135M') as a
    drop-in model for the benchmark suite.

    Returns (model, config, tokenizer) — same contract as
    load_model_from_checkpoint, so compare_models can treat it identically.
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading HF baseline: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    hf_model = hf_model.to(device).eval()

    model = HFModelAdapter(hf_model)
    config = hf_model.config  # HF config exposes hidden_size / num_hidden_layers / num_attention_heads

    print(f"✅ Baseline loaded: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in hf_model.parameters()):,}")
    print(f"   Precision: {dtype}")

    return model, config, tokenizer


def model_size_info(config):
    """Pull (hidden_size, num_layers, num_heads) from either a universe-lm
    LLMConfig (d_model / n_layers / n_heads) or a HF config
    (hidden_size / num_hidden_layers / num_attention_heads)."""
    hidden = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', 'N/A')
    layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layers', 'N/A')
    heads = getattr(config, 'num_attention_heads', None) or getattr(config, 'n_heads', 'N/A')
    return {'hidden_size': hidden, 'num_layers': layers, 'num_heads': heads}
