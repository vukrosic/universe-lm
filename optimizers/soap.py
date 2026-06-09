"""SOAP — Shampoo + Adam (Vyas et al. 2024, arXiv 2409.11321).

Runs Adam updates inside the eigenbasis of the Shampoo preconditioner.
The eigenbasis is refreshed only every `precondition_frequency` steps
(one new HP), making the cost comparable to AdamW while inheriting
Shampoo's curvature benefits.

For 1D parameters, falls back to plain AdamW math (eigendecomposition
is meaningless on 1D). On the first step of a 2D param, the eigenbasis
is initialized to the identity, so the projected grad equals the raw
grad and the update is bit-identical to AdamW's first step.

When `use_soap=False` (default), this class is never instantiated —
the trainer's routing skip-path keeps every param on its existing
optimizer. See `training/trainer.py` for the gate.
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class SOAP(Optimizer):
    """SOAP optimizer.

    Per-tensor dispatch:
      - ndim < 2: pure AdamW (eigendecomp is meaningless; Shampoo's
        preconditioner is a 2-D object).
      - ndim == 2: Adam in the eigenbasis of the running
        preconditioner `L = E[G G^T]`, `R = E[G^T G]`.

    State per 2D param: `L`, `R` (running preconditioners), `Q_L`,
    `Q_R` (eigenbasis), `exp_avg`, `exp_avg_sq` (Adam moments in the
    eigenbasis). The eigenbasis is identity on step 0 → first step is
    bit-identical to AdamW.

    Parameters
    ----------
    params : iterable
    lr : float
    betas : (β1, β2) for Adam
    eps : float — Adam denominator
    weight_decay : float — decoupled (AdamW style)
    precondition_frequency : int — refresh eigenbasis every K steps
    precondition_eps : float — diagonal shift on L, R before eigh
        (regularizer; paper recommends 1e-6)
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, precondition_frequency=10,
                 precondition_eps=1e-6):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        precondition_eps=precondition_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            K = group["precondition_frequency"]
            pre_eps = group["precondition_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state)

                state["step"] += 1
                step_t = state["step"]

                # Decoupled weight decay (AdamW style).
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                if p.ndim < 2:
                    # 1D path: pure AdamW. Eigendecomp is meaningless.
                    self._adamw_step(p, grad, state, step_t,
                                     beta1, beta2, lr, eps)
                    continue

                # 2D path: update running preconditioner, then
                # Adam in the eigenbasis.
                L = state["L"]
                R = state["R"]
                # L = β2 L + (1-β2) G G^T  (d_out × d_out)
                L.mul_(beta2).add_(grad @ grad.t(), alpha=1 - beta2)
                # R = β2 R + (1-β2) G^T G  (d_in × d_in)
                R.mul_(beta2).add_(grad.t() @ grad, alpha=1 - beta2)

                # Periodically refresh the eigenbasis.
                if step_t % K == 0:
                    Q_L, _ = _eigh(L, pre_eps)
                    Q_R, _ = _eigh(R, pre_eps)
                    state["Q_L"] = Q_L
                    state["Q_R"] = Q_R

                Q_L = state["Q_L"]
                Q_R = state["Q_R"]

                # Project grad into eigenbasis: G' = Q_L^T G Q_R.
                grad_proj = Q_L.t() @ grad @ Q_R

                # Adam moments on the projected grad.
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t
                exp_avg.mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_proj, grad_proj,
                                                value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                update_proj = exp_avg / denom / bc1

                # Project update back: u = Q_L u' Q_R^T.
                update = Q_L @ update_proj @ Q_R.t()

                p.add_(update, alpha=-lr)

        return loss

    def _init_state(self, p, state):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        if p.ndim >= 2:
            d_out, d_in = p.shape[0], p.shape[1]
            # Preconditioner stats in fp32 — the running averages are
            # the source of truth, and bf16 in this product is
            # lossy enough to skew the eigenbasis.
            state["L"] = torch.zeros(d_out, d_out, dtype=torch.float32,
                                     device=p.device)
            state["R"] = torch.zeros(d_in, d_in, dtype=torch.float32,
                                     device=p.device)
            # Eigenbasis — identity at step 0 so the first step is
            # bit-identical to AdamW (Q^T G Q = G when Q = I).
            state["Q_L"] = torch.eye(d_out, dtype=torch.float32,
                                     device=p.device)
            state["Q_R"] = torch.eye(d_in, dtype=torch.float32,
                                     device=p.device)

    def _adamw_step(self, p, grad, state, step_t, beta1, beta2, lr, eps):
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        bc1 = 1 - beta1 ** step_t
        bc2 = 1 - beta2 ** step_t
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
        update = exp_avg / denom / bc1
        p.add_(update, alpha=-lr)


def _eigh(M, eps):
    """Stable eigendecomposition of a (near-)symmetric matrix.

    Symmetrize (float error may have broken it) and add a small
    identity shift for numerical stability (paper's trick — keeps
    eigh well-conditioned and the eigenvalues strictly positive).
    """
    M = (M + M.t()) / 2
    M = M + eps * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    eigvals, eigvecs = torch.linalg.eigh(M)
    # Defensive clamp — after the shift this should be a no-op, but
    # a negative eigenvalue would mean sqrt blows up downstream.
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs, eigvals
