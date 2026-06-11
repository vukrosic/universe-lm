"""Tests for the 038 SWAN lever."""
import torch

from configs.llm_config import Tiny1M3MSWANConfig, Tiny1M3MConfig
from models.llm import MinimalLLM
from optimizers.swan import SWAN
from training.trainer import setup_muon_optimizer


def test_swan_optimizer_whitens_2d_updates():
    p = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    opt = SWAN([p], lr=0.1)
    p.grad = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    before = p.detach().clone()
    opt.step()
    assert not torch.equal(p.detach(), before), "SWAN should update matrix params"
    assert len(opt.state) == 0, "SWAN should remain stateless"


def test_trainer_routes_matrix_params_to_swan():
    torch.manual_seed(42)
    cfg = Tiny1M3MSWANConfig()
    model = MinimalLLM(cfg)
    optimizers = setup_muon_optimizer(model, cfg)
    names = [type(o).__name__ for o in optimizers if o is not None]
    assert "SWAN" in names, f"SWAN optimizer missing from {names}"


def test_tiny_config_uses_swan_flag():
    cfg = Tiny1M3MSWANConfig()
    assert cfg.use_swan is True
    assert Tiny1M3MConfig().use_swan is False
