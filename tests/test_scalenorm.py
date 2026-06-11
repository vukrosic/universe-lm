"""Tests for the 051 ScaleNorm lever."""
import torch

from configs.llm_config import Tiny1M3MScaleNormConfig
from models.layers import ScaleNorm, make_norm
from models.llm import MinimalLLM


def test_factory_builds_scalenorm():
    norm = make_norm(8, "scalenorm")
    assert isinstance(norm, ScaleNorm)
    assert norm.weight.shape == torch.Size([]), "ScaleNorm gain must be scalar"


def test_scalenorm_is_identity_at_init():
    torch.manual_seed(42)
    norm = ScaleNorm(8)
    x = torch.randn(2, 3, 8)
    y = norm(x)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    assert torch.allclose(y, x / rms), "ScaleNorm should be identity at init"


def test_tiny_config_routes_residual_norms():
    torch.manual_seed(42)
    model = MinimalLLM(Tiny1M3MScaleNormConfig()).eval()
    for i, block in enumerate(model.transformer_blocks):
        assert isinstance(block.norm1, ScaleNorm), f"block {i} norm1 is not ScaleNorm"
        assert isinstance(block.norm2, ScaleNorm), f"block {i} norm2 is not ScaleNorm"
