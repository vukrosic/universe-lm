"""Tests for `models.fox.FoX` — see
`autoresearch/ideas/020-forgetting-attn/plan.md`.

`FoX.forward(x)` returns `log_D: [B, H, T, T]` — the pre-softmax additive
bias on attention logits (recode r2: log-add formulation, math-equivalent
to the r1 post-softmax multiply but numerically stable).

Invariants checked:
  1. No NaN/Inf on a non-trivial random input.
  2. Causal: log_D[i, j] = 0 for j > i (strictly lower-triangular).
  3. Identity at init: W_f=0, b_f=+10 → exp(log_D) is within 9% of 1
     over the full T context (math-derived in `models/fox.py`).
  4. MHA smoke: `use_fox=False` and `use_fox=True` (with the identity
     init) produce attention outputs within `1e-2` of each other on a
     freshly-initialized tiny1m3m-style model — the softmax is
     near-invariant to a small additive bias when the underlying score
     distribution is roughly uniform. Recode r3: parametrized over
     `use_fire_pe ∈ {False, True}` so the FIRE-branch FoX hookup at
     `models/layers.py:1632-1633` is also covered (r2 nit closed).
  5. With a non-zero W_f perturbation, log_D becomes meaningfully
     different from identity (the wiring is live).
"""
import pytest
import torch
from models.fox import FoX, FOX_BF_INIT


def test_no_nan_or_inf():
    """Random input → finite output."""
    torch.manual_seed(42)
    fox = FoX(d_model=64, n_heads=6)
    x = torch.randn(2, 32, 64)
    log_D = fox(x)
    assert torch.isfinite(log_D).all(), "log_D has NaN/Inf"
    assert log_D.shape == (2, 6, 32, 32), f"Shape mismatch: {log_D.shape}"


def test_causal_lower_triangular():
    """log_D[i, j] = 0 for j > i (upper triangle masked to 0; lower
    triangle is the cumulative log-decay ≤ 0)."""
    torch.manual_seed(42)
    fox = FoX(d_model=64, n_heads=6)
    x = torch.randn(2, 32, 64)
    log_D = fox(x)  # [B, H, T, T]
    T = log_D.size(-1)
    ar = torch.arange(T)
    causal = ar[None, :] <= ar[:, None]
    upper = log_D[..., ~causal]
    assert upper.abs().max().item() < 1e-6, (
        f"Non-zero entries in log_D upper-tri: "
        f"max |log_D[upper]| = {upper.abs().max().item()}"
    )


def test_identity_init_close_to_ones():
    """With W_f=0, b_f=+10, exp(log_D[i, j]) is within 9% of 1 for all
    (i, j) in the causal lower triangle.

    Math (at init): f = sigmoid(10) ≈ 0.99995, log f ≈ −4.54e-5.
    log_D[i, j] = (i − j + 1) · log f for j ≤ i. Worst case is
    log_D[0, T-1] = T · log f = −T · 4.54e-5. For T=64 (test-time, not
    the real 2048) that's −0.0029 → exp ≈ 0.9971 — trivially close to 1.
    For T=2048 (real tier) it's −0.0929 → exp ≈ 0.911 — ≤ 9% decay.
    The test asserts the conservative bound for a smaller T to keep it
    fast.
    """
    torch.manual_seed(42)
    T = 64  # small T keeps the test fast; real tier is 2048
    fox = FoX(d_model=64, n_heads=6)
    # Init must be the identity init: W_f=0 (the default), b_f=+10
    # (FOX_BF_INIT). FoX.__init__ applies both.
    x = torch.zeros(1, T, 64)  # zero input → W_f·x = 0 → only b_f fires
    log_D = fox(x)  # [1, H, T, T]
    D = log_D.exp()
    # log_D is lower-tri-only (upper is 0 → exp(0) = 1, but we only
    # check the causal region).
    f_init = torch.sigmoid(torch.tensor(FOX_BF_INIT)).item()
    log_f = torch.log(torch.tensor(f_init)).item()
    D_min = float(torch.tensor(T * log_f).exp())
    ar = torch.arange(T)
    causal = ar[None, :] <= ar[:, None]
    D_lower = D[0, 0][causal]  # head 0, batch 0
    # All values in [D_min, 1] (causal monotone non-incr from diag).
    assert D_lower.max().item() <= 1.0 + 1e-6, (
        f"exp(log_D)[i, j] > 1 for some (i, j): max = {D_lower.max().item()}"
    )
    assert D_lower.min().item() >= D_min - 1e-6, (
        f"exp(log_D)[i, j] below expected min {D_min}: "
        f"min = {D_lower.min().item()}"
    )
    # Diagonal entries: log_D[i, i] = log f_init → exp ≈ 0.99995, within
    # 1e-3 of 1.
    diag = D[0, 0].diagonal()
    assert (diag - 1.0).abs().max().item() < 1e-3, (
        f"Diagonal not close to 1: max |exp(log_D)_diag - 1| = "
        f"{(diag - 1.0).abs().max().item()}"
    )


def test_wiring_live_with_Wf_perturbation():
    """A non-zero W_f changes log_D from its init. Confirms the
    projection actually flows into the kernel (i.e. the module is
    wired)."""
    torch.manual_seed(42)
    fox_a = FoX(d_model=64, n_heads=6)
    fox_b = FoX(d_model=64, n_heads=6)
    # Perturb head 0's W_f with a large weight.
    with torch.no_grad():
        fox_b.gate_w[0] = fox_b.gate_w[0] + 1.0
    x = torch.randn(2, 16, 64)
    log_D_a = fox_a(x)
    log_D_b = fox_b(x)
    # Head 0: must differ.
    d0 = (log_D_a[:, 0] - log_D_b[:, 0]).abs().max().item()
    assert d0 > 1e-4, f"Head 0 unchanged: max diff = {d0}"
    # Heads 1..5: must be identical (up to fp noise).
    for h in range(1, 6):
        dh = (log_D_a[:, h] - log_D_b[:, h]).abs().max().item()
        assert dh < 1e-5, f"Head {h} changed: max diff = {dh}"


def test_step0_attention_output_unchanged():
    """With use_fox=True at the identity init, the MHA output is within
    1e-2 of the use_fox=False baseline, in BOTH the SDPA path
    (use_fire_pe=False) and the FIRE branch (use_fire_pe=True).

    The identity-init log_D has |log_D| ≤ ~T · 4.54e-5 in the causal
    triangle (0.0029 at T=64); softmax is differentiably small for small
    additive shifts, so the resulting attention probabilities — and the
    output `attn @ V` — drift by an order of magnitude less than this.
    1e-2 is the defensible ceiling chosen per the r2 reviewer nit; the
    real drift at T=32 is well below it.

    Recode r3: parametrized over `use_fire_pe ∈ {False, True}` — at
    `use_fire_pe=True` both sides exercise the FIRE branch
    (`models/layers.py:1574-1652`), so the FoX hookup at
    `models/layers.py:1632-1633` is covered. At `use_fire_pe=False`
    the ctrl runs SDPA and the trt runs the manual branch (FoX forces
    `_use_manual=True` via `models/layers.py:1659`); the SDPA-vs-manual
    drift sits inside the 1e-2 ceiling (measured ~9e-6 in r1 codereview).
    """
    from models.layers import MultiHeadAttention
    for use_fire_pe in (False, True):
        torch.manual_seed(42)
        d_model, n_heads, max_seq_len, T = 64, 8, 64, 32
        # Two MHAs with the SAME parameter init (we copy state_dict after
        # building each so the only difference is the FoX module).
        torch.manual_seed(42)
        mha_no = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len,
            dropout=0.0, use_fire_pe=use_fire_pe, use_fox=False,
        )
        mha_no.eval()
        torch.manual_seed(42)
        mha_fox = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len,
            dropout=0.0, use_fire_pe=use_fire_pe, use_fox=True,
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
            f"Step-0 MHA output drifted by {diff} (tolerance 1e-2) "
            f"at use_fire_pe={use_fire_pe}. "
            f"Check FoX identity init (W_f=0, b_f=+10)."
        )


def test_trained_gate_does_not_blow_up():
    """Regression for the r1 NaN: with gate_b at a *trained-like* value
    (lower b_f → larger decay → log_D[0, T-1] very negative), the
    pre-softmax add must remain finite and the resulting softmax must
    sum to 1 per row. The r1 implementation's post-softmax multiply +
    renorm produced NaN here (row-sum underflowed to ~0).
    """
    torch.manual_seed(42)
    T = 2048  # real tier seq_len — the spot where r1 NaN'd
    fox = FoX(d_model=64, n_heads=6, b_f=-3.0)  # gate ~0.047, big decay
    x = torch.randn(1, T, 64)
    log_D = fox(x)
    # log_D is finite (no Inf despite very negative cumsum at this T).
    assert torch.isfinite(log_D).all(), "log_D has NaN/Inf at trained-like init"
    # Worst-case log_D[0, T-1] is very negative — but finite (fp32 can
    # represent ~ -1e38 before underflow to -inf).
    worst = log_D.min().item()
    assert worst > -1e30, f"log_D underflowed to -inf: min = {worst}"
    # Simulate the pre-softmax add: random logits + log_D → softmax →
    # row-sum should be 1 (within fp32 epsilon).
    logits = torch.randn(1, 6, T, T) + log_D
    # Mask non-causal positions (the caller does this — softmax must
    # handle it).
    ar = torch.arange(T)
    causal = (ar[None, :] <= ar[:, None]).view(1, 1, T, T)
    logits = logits.masked_fill(~causal, -1e9)
    attn = torch.softmax(logits, dim=-1)
    assert torch.isfinite(attn).all(), "attn has NaN/Inf after softmax(scores + log_D)"
    row_sums = attn.sum(dim=-1)
    assert (row_sums - 1.0).abs().max().item() < 1e-4, (
        f"Softmax rows don't sum to 1: max |sum - 1| = "
        f"{(row_sums - 1.0).abs().max().item()}"
    )
