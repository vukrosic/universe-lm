"""Tests for `models.fox.FoX` — see
`autoresearch/ideas/020-forgetting-attn/plan.md`.

Invariants checked:
  1. No NaN/Inf on a non-trivial random input.
  2. Causal: D[i, j] = 0 for j > i (strictly lower-triangular).
  3. Identity at init: W_f=0, b_f=+10 → D is within 9% of all-ones
     over the full T context (math-derived in `models/fox.py`).
  4. MHA smoke: `use_fox=False` and `use_fox=True` (with the identity
     init) produce attention outputs within `1e-2` of each other on a
     freshly-initialized tiny1m3m-style model — the step-0 row-renorm
     cancels the small D perturbation when the softmax is roughly
     uniform (a freshly-init Q,K gives a near-uniform post-softmax
     distribution).
  5. With a non-zero W_f perturbation, D becomes meaningfully
     different from identity (the wiring is live).
"""
import torch
from models.fox import FoX, FOX_BF_INIT


def test_no_nan_or_inf():
    """Random input → finite output."""
    torch.manual_seed(42)
    fox = FoX(d_model=64, n_heads=6)
    x = torch.randn(2, 32, 64)
    D = fox(x)
    assert torch.isfinite(D).all(), "D has NaN/Inf"
    assert D.shape == (2, 6, 32, 32), f"Shape mismatch: {D.shape}"


def test_causal_lower_triangular():
    """D[i, j] = 0 for j > i (strictly causal)."""
    torch.manual_seed(42)
    fox = FoX(d_model=64, n_heads=6)
    x = torch.randn(2, 32, 64)
    D = fox(x)  # [B, H, T, T]
    T = D.size(-1)
    ar = torch.arange(T)
    causal = ar[None, :] <= ar[:, None]
    upper = D[..., ~causal]
    assert upper.abs().max().item() < 1e-6, (
        f"Non-causal entries in D: max |D[upper]| = {upper.abs().max().item()}"
    )


def test_identity_init_close_to_ones():
    """With W_f=0, b_f=+10, D[i, j] is within 9% of 1 for all (i, j).

    Math (at init): f = sigmoid(10) ≈ 0.99995, log f ≈ −4.54e-5.
    D[i, j] = exp((i − j + 1) · log f) for j ≤ i. Worst case is
    D[0, T-1] = exp(T · log f) = exp(−T · 4.54e-5). For T=64
    (test-time, not the real 2048) that's exp(−0.0029) ≈ 0.9971 —
    trivially close to 1. For T=2048 (real tier) it's exp(−0.0929) ≈
    0.911 — ≤ 9% decay. The test asserts the conservative bound for
    a smaller T to keep it fast.
    """
    torch.manual_seed(42)
    T = 64  # small T keeps the test fast; real tier is 2048
    fox = FoX(d_model=64, n_heads=6)
    # Init must be the identity init: W_f=0 (the default), b_f=+10
    # (FOX_BF_INIT). FoX.__init__ applies both.
    x = torch.zeros(1, T, 64)  # zero input → W_f·x = 0 → only b_f fires
    D = fox(x)  # [1, H, T, T]
    # D is causal, so the upper triangle is exactly 0. The lower
    # triangle should be ≤ 1 (decay) and ≥ D_min = exp(T·log f).
    # Compute the expected D_min via the actual gate value.
    f_init = torch.sigmoid(torch.tensor(FOX_BF_INIT)).item()
    log_f = torch.log(torch.tensor(f_init)).item()
    D_min = float(torch.tensor(T * log_f).exp())
    # Lower triangle (D[upper] is 0 by construction).
    ar = torch.arange(T)
    causal = ar[None, :] <= ar[:, None]
    D_lower = D[0, 0][causal]  # head 0, batch 0
    # All values should be in [D_min, 1] (causal monotone non-incr).
    assert D_lower.max().item() <= 1.0 + 1e-6, (
        f"D[i, j] > 1 for some (i, j): max = {D_lower.max().item()}"
    )
    assert D_lower.min().item() >= D_min - 1e-6, (
        f"D[i, j] below expected min {D_min}: min = {D_lower.min().item()}"
    )
    # Diagonal entries are f_init (~0.99995) — within 1e-3 of 1.
    diag = D[0, 0].diagonal()
    assert (diag - 1.0).abs().max().item() < 1e-3, (
        f"Diagonal not close to 1: max |D_diag - 1| = "
        f"{(diag - 1.0).abs().max().item()}"
    )


def test_wiring_live_with_Wf_perturbation():
    """A non-zero W_f changes D from its init. Confirms the projection
    actually flows into the kernel (i.e. the module is wired)."""
    torch.manual_seed(42)
    fox_a = FoX(d_model=64, n_heads=6)
    fox_b = FoX(d_model=64, n_heads=6)
    # Perturb head 0's W_f with a large weight.
    with torch.no_grad():
        fox_b.gate_w[0] = fox_b.gate_w[0] + 1.0
    x = torch.randn(2, 16, 64)
    D_a = fox_a(x)
    D_b = fox_b(x)
    # Head 0: must differ.
    d0 = (D_a[:, 0] - D_b[:, 0]).abs().max().item()
    assert d0 > 1e-4, f"Head 0 unchanged: max diff = {d0}"
    # Heads 1..5: must be identical (up to fp noise).
    for h in range(1, 6):
        dh = (D_a[:, h] - D_b[:, h]).abs().max().item()
        assert dh < 1e-5, f"Head {h} changed: max diff = {dh}"


def test_step0_attention_output_unchanged():
    """With use_fox=True at the identity init, the MHA output is within
    1e-2 of the use_fox=False baseline.

    Caveat from `idea.md:54-56` / review.md nit: the row-renorm on a
    near-uniform softmax (random-init Q,K → near-uniform post-softmax)
    cancels most of the D perturbation, so the actual diff is well
    inside 1e-2 — but the test pins a defensible 1e-2 ceiling (any
    tighter tolerance would risk flaking on a different init seed).
    """
    from models.layers import MultiHeadAttention
    torch.manual_seed(42)
    d_model, n_heads, max_seq_len, T = 64, 8, 64, 32
    # Two MHAs with the SAME parameter init (we copy state_dict after
    # building each so the only difference is the FoX module).
    torch.manual_seed(42)
    mha_no = MultiHeadAttention(
        d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len,
        dropout=0.0, use_fire_pe=False, use_fox=False,
    )
    mha_no.eval()
    torch.manual_seed(42)
    mha_fox = MultiHeadAttention(
        d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len,
        dropout=0.0, use_fire_pe=False, use_fox=True,
    )
    mha_fox.eval()
    # Sync all shared parameters (mha_no and mha_fox were built with
    # the same seed; verify state_dicts match for the shared keys).
    sd_no = mha_no.state_dict()
    sd_fox = mha_fox.state_dict()
    shared_keys = set(sd_no.keys()) & set(sd_fox.keys())
    for k in shared_keys:
        if sd_no[k].shape == sd_fox[k].shape:
            sd_fox[k] = sd_no[k].clone()
    mha_fox.load_state_dict(sd_fox, strict=False)
    # Run the same input through both.
    x = torch.randn(2, T, d_model)
    with torch.no_grad():
        y_no = mha_no(x)
        y_fox = mha_fox(x)
    diff = (y_no - y_fox).abs().max().item()
    assert diff < 1e-2, (
        f"Step-0 MHA output drifted by {diff} (tolerance 1e-2). "
        f"Check FoX identity init (W_f=0, b_f=+10)."
    )
