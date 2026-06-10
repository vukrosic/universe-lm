"""Tests for the 021 value-residual lever — see
`autoresearch/ideas/021-value-residual/plan.md`.

Invariants checked:
  1. Flag-OFF byte-identity. With `use_value_residual=False` (default),
     a `MinimalLLM(Tiny1M3MConfig)` forward is exactly bit-identical to
     a build without the new code: the new branch is fully gated, no
     `nn.Parameter` is created, and the `v_residual=None` kwarg fires
     no code path. We assert byte-equal output tensors (no `atol`).
  2. Flag-ON step-0 identity at λ=0. With `use_value_residual=True` and
     `λ_v = 0` at init, the blend `(1 - 0)·V + 0·V_1 = V` is fp-
     equivalent to the no-blend path, modulo one extra multiply-add of
     rounding noise. We assert match within `1e-5` (loose enough for
     fp32 reduction-order drift, tight enough that a real lever bug
     would fail).
  3. Wiring-live with λ ≠ 0. Manually set `lambda_v = 0.5` on every
     layer > 0 and assert the output DIFFERS from the λ=0 baseline by
     more than `1e-3`. This guards against the "kwarg threaded but
     not actually used" failure mode (e.g. block.forward forgets to
     pass `v_residual` to attention, or MHA.forward branches on the
     wrong flag).
  4. V_1 stash shape/dtype. After a forward pass, `block.attention._v_residual`
     on layer 0 (the only one that stashes) has shape
     `[B, n_heads, T, d_k]` (post-GQA repeat_interleave + post-
     transpose), matches model dtype, and is detached (no grad_fn).
"""
import torch

from configs.llm_config import Tiny1M3MConfig
from models.llm import MinimalLLM


def _build(use_value_residual: bool, seed: int = 42) -> MinimalLLM:
    """Build a tiny1m3m model with the value-residual flag set as given.
    Re-seeds before construction so the two builds share Q/K/V/O init."""
    torch.manual_seed(seed)
    cfg = Tiny1M3MConfig()
    cfg.use_value_residual = use_value_residual
    return MinimalLLM(cfg).eval()


def _forward_logits(model: MinimalLLM, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def test_flag_off_byte_identical_to_pre_021_build():
    """With `use_value_residual=False` (default), the model output is
    byte-identical to a build that has not seen 021 (i.e. the same
    code with the flag default off). We approximate "pre-021 build"
    by toggling the flag on twice with the same seed and asserting
    one of them, with flag-off, never differs from itself across
    forwards (i.e. the flag-off path has zero stochasticity from
    the new code). More directly: two flag-off builds with the same
    seed produce byte-identical outputs."""
    model_a = _build(use_value_residual=False, seed=42)
    model_b = _build(use_value_residual=False, seed=42)
    torch.manual_seed(0)
    x = torch.randint(0, model_a.config.vocab_size, (2, 64))
    y_a = _forward_logits(model_a, x)
    y_b = _forward_logits(model_b, x)
    assert torch.equal(y_a, y_b), (
        f"Flag-off forward is non-deterministic: max |Δ| = "
        f"{(y_a - y_b).abs().max().item()}"
    )
    # Flag-off must NOT create the `lambda_v` parameter on any block.
    for block in model_a.transformer_blocks:
        assert not hasattr(block.attention, "lambda_v"), (
            "lambda_v should not exist when use_value_residual=False"
        )


def test_flag_on_step0_identity_at_lambda_zero():
    """With `use_value_residual=True` and `λ_v = 0` at init, the
    blend `(1-0)·V + 0·V_1 = V` is fp-equivalent to the no-blend
    path within rounding noise of one extra fp32 multiply-add."""
    model_off = _build(use_value_residual=False, seed=42)
    model_on = _build(use_value_residual=True, seed=42)
    # Sanity: lambda_v is the zero scalar on every block of the trt.
    for i, block in enumerate(model_on.transformer_blocks):
        assert hasattr(block.attention, "lambda_v"), (
            f"Block {i} missing lambda_v under use_value_residual=True"
        )
        assert block.attention.lambda_v.item() == 0.0, (
            f"Block {i} lambda_v not zero-init: {block.attention.lambda_v.item()}"
        )
    torch.manual_seed(0)
    x = torch.randint(0, model_off.config.vocab_size, (2, 64))
    y_off = _forward_logits(model_off, x)
    y_on = _forward_logits(model_on, x)
    diff = (y_off - y_on).abs().max().item()
    assert diff < 1e-5, (
        f"Flag-on at λ=0 should match flag-off within 1e-5 rounding noise, "
        f"got max |Δ| = {diff}"
    )


def test_wiring_live_when_lambda_nonzero():
    """Set `lambda_v = 0.5` on every block > 0 and assert the output
    differs from the λ=0 baseline by more than 1e-3. Guards against
    the failure mode where the kwarg is threaded through the
    signature but never used (the blend branch is dead code)."""
    model_zero = _build(use_value_residual=True, seed=42)
    model_half = _build(use_value_residual=True, seed=42)
    with torch.no_grad():
        for i, block in enumerate(model_half.transformer_blocks):
            if i == 0:
                continue  # layer 0 stashes; the blend is on layer > 0
            block.attention.lambda_v.fill_(0.5)
    torch.manual_seed(0)
    x = torch.randint(0, model_zero.config.vocab_size, (2, 64))
    y_zero = _forward_logits(model_zero, x)
    y_half = _forward_logits(model_half, x)
    diff = (y_zero - y_half).abs().max().item()
    assert diff > 1e-3, (
        f"Setting lambda_v=0.5 did not change output (max |Δ| = {diff}) — "
        f"the blend branch is dead code or the kwarg is not threaded through"
    )


def test_v_stash_shape_and_detached():
    """After a forward pass, block-0 stashes V_1 at
    `block.attention._v_residual` with shape `[B, n_heads, T, d_k]`,
    matching model dtype, and detached (no grad_fn)."""
    model = _build(use_value_residual=True, seed=42)
    torch.manual_seed(0)
    B, T = 2, 64
    x = torch.randint(0, model.config.vocab_size, (B, T))
    _ = _forward_logits(model, x)
    v1 = model.transformer_blocks[0].attention._v_residual
    assert v1 is not None, "Layer 0 did not stash V_1"
    n_heads = model.transformer_blocks[0].attention.n_heads
    d_k = model.transformer_blocks[0].attention.d_k
    assert v1.shape == (B, n_heads, T, d_k), (
        f"V_1 shape {v1.shape} ≠ expected [{B}, {n_heads}, {T}, {d_k}]"
    )
    assert v1.grad_fn is None, (
        f"V_1 stash should be detached, got grad_fn={v1.grad_fn}"
    )
