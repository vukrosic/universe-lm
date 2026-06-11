"""SWAN — SGD with Normalization and Whitening.

Stateless optimizer for matrix parameters. The treatment in this repo
applies a per-step whitening transform to 2-D gradients, while 1-D,
norm, and embedding parameters stay on AdamW in the trainer routing.
"""
import math

import torch
from torch.optim.optimizer import Optimizer


class SWAN(Optimizer):
    """Stateless gradient whitening for 2-D parameters.

    The implementation keeps no momentum / second-moment state. For a
    matrix gradient G, it computes a whitened update by normalizing the
    row covariance `G G^T` and applying the inverse square root to the
    gradient rows.
    """

    def __init__(self, params, lr=0.024, weight_decay=0.0, whiten_eps=1e-6):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, weight_decay=weight_decay, whiten_eps=whiten_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            whiten_eps = group["whiten_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SWAN does not support sparse gradients")

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                if grad.ndim < 2:
                    p.add_(grad, alpha=-lr)
                    continue

                g = grad.float()
                if g.ndim > 2:
                    g = g.view(g.shape[0], -1)

                # Normalize the raw matrix gradient before whitening.
                g = g / g.norm().clamp_min(1e-12)

                cov = (g @ g.t()) / max(1, g.shape[1])
                cov = (cov + cov.t()) * 0.5
                eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
                eigvals, eigvecs = torch.linalg.eigh(cov + whiten_eps * eye)
                inv_sqrt = eigvecs @ torch.diag(torch.clamp(eigvals, min=whiten_eps).rsqrt()) @ eigvecs.t()
                update = inv_sqrt @ g

                if grad.ndim > 2:
                    update = update.view_as(grad.float())

                p.add_(update.to(p.dtype), alpha=-lr)

        return loss
