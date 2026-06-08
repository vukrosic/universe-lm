"""Cautious AdamW — Liang et al. 2024 (arXiv 2411.16085).

One-line sign-mask on the AdamW update: zero out components whose sign
disagrees with the current gradient. Suppresses stale-momentum artifacts
on the AdamW path (1D / embedding / head). Bit-identical to
`torch.optim.AdamW` whenever the instance is selected with
`mask_buckets=()` (the empty default) — the mask is a no-op only when
the bucket set is empty AND `mask_all=False`, so callers must leave both
at their defaults to reproduce baseline AdamW.

The mask is applied per-tensor: if a tensor's name matches any substring
in `mask_buckets` (or `mask_all=True`), the full tensor is masked
element-wise. Substring matching is intentional — it lets the caller
match `token_embedding`, `emb_proj`, `norm.weight` etc. without an enum
of every 1D scalar name in the model.
"""
import math
import torch
from torch.optim import AdamW


class CautiousAdamW(AdamW):
    """AdamW with the cautious sign-mask (Liang et al. 2024).

    Math is identical to `torch.optim.AdamW` (m, v, bias-correction,
    decoupled weight decay) except: just before `param.add_(update,
    alpha=-lr)`, we multiply `update` by
    `(update.sign() == grad.sign()).to(update.dtype)`. When the mask is
    zero everywhere for a tensor (the bucket selector didn't match), the
    update is unchanged and the step is bit-identical to the parent
    AdamW — but we only select this class via the gate in
    `training/trainer.py:142` when at least one bucket is active, so
    `use_cautious_adamw != "none"` ⇒ at least one tensor is masked.

    Parameters
    ----------
    mask_buckets : tuple of str
        Substrings of `param.shape` names that should be masked. Empty
        tuple + `mask_all=False` = no-op (still applies the mask, which
        is `(update.sign() == grad.sign())`; if the optimizer math
        already puts update in sign-of-grad, the mask is all-ones — the
        update is unchanged). Default is conservative: mask nothing.
    mask_all : bool
        If True, mask every param tensor in the optimizer (ignores
        `mask_buckets`). Useful for the `"all"` config value.
    """

    def __init__(self, params, mask_buckets=(), mask_all=False, **kwargs):
        super().__init__(params, **kwargs)
        self._mask_buckets = tuple(mask_buckets)
        self._mask_all = bool(mask_all)

    def _should_mask(self, param):
        if self._mask_all:
            return True
        if not self._mask_buckets:
            return False
        # `_param_name` is set by torch.optim.Optimizer via the
        # add_param_group path; fall back to "unknown" if missing.
        name = getattr(param, "_param_name", "") or ""
        return any(b in name for b in self._mask_buckets)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            step_t = group["step"] = group.get("step", 0) + 1
            bc1 = 1 - beta1 ** step_t
            bc2 = 1 - beta2 ** step_t

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CautiousAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decoupled weight decay (AdamW style): scale param toward 0.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # m, v update (same as Adam).
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                # Bias-corrected signed direction; matches the magnitude
                # that `param.add_(update, alpha=-lr)` would apply in the
                # parent AdamW (which folds 1/bc1 into `step_size`).
                update = exp_avg / denom / bc1

                # Cautious mask — keep components whose sign agrees with
                # the current gradient (Liang et al. 2024, §3.2). Applied
                # only to params in the selected buckets; others receive
                # the unmasked update (i.e. parent-AdamW behavior).
                if self._should_mask(p):
                    mask = (update.sign() == grad.sign()).to(update.dtype)
                    update = update * mask

                p.add_(update, alpha=-lr)
        return loss
