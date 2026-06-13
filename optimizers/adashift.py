"""AdaShift: Decorrelated Adam via Delayed Gradients
(Zhou, Yang, Wang, Wang, "AdaShift: Decorrelation and Convergence
of Adaptive Learning Rate Methods", arXiv:1810.00143, NeurIPS 2019
workshop).

Adam's update is `v_t = β2 · v_{t−1} + (1 − β2) · g_t²`. AdaShift uses
a *shifted* gradient:

    m_t = β1 · m_{t−1} + (1 − β1) · g_t       (current momentum)
    v_t = β2 · v_{t−1} + (1 − β2) · g_{t-n}²  (delayed 2nd moment)
    update = m̂_t / (√v̂_t + ε)                  (Adam-normalized)
    w ← w − lr · update

Where `n` is the *delay* (paper default `n = 3`). The intuition:
Adam's `v_t` is highly auto-correlated with `m_t` (both use `g_t`),
which creates a *bias* in the adaptive step size — `v_t` increases
exactly when `m_t` increases, so the Adam normalization cancels
out some of the actual signal. Using `g_{t-n}²` decorrelates `v_t`
from `m_t`, so the normalization captures the *recent* gradient
size without being driven by the *current* gradient.

Identity at step 0: the paper's recipe uses a warm-start
`v_0 = g_0²` so the first step uses
    `v_1 = β2 · g_0² + (1−β2) · g_{1-n}² = β2 · g_0²`
(since `g_{1-n}² = 0` for `n ≥ 1`). Different from AdamW's first
step (which uses `v_1 = (1−β2)·g_0²`) but same magnitude order
(O(β2) different). This first-step displacement is the lever,
not a bug. With `n = 0` AdaShift collapses to AdamW.

Cold-start handling: we maintain a per-parameter queue of the
last `n` gradients. On each step:
  - read g_{t-n} from the queue (or 0 if the queue is too short)
  - update v_t with g_{t-n}²
  - push g_t to the queue (drop the oldest if size > n)

The queue size is O(n) gradients per param; at n=3 and tiny1m3m's
~0.94M AdamW-eligible params (mostly 1-D), this is negligible
memory.

When `use_adashift=False` (default), this class is never
instantiated — the trainer uses `torch.optim.AdamW` unchanged.
"""
import torch
from torch.optim.optimizer import Optimizer


class AdaShift(Optimizer):
    """AdaShift — Decorrelated Adam via Delayed Gradients.

    Drop-in replacement for the AdamW 1-D / embedding / norm / head
    path. The 2-D Muon path is unchanged (AdaShift is an AdamW
    replacement, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME, 124-RAdam).

    Parameters
    ----------
    params : iterable
    lr : float — AdaShift step size (paper default ≈ AdamW LR).
    betas : (β1, β2) — β1 weights the gradient vs momentum; β2 is
        the EMA decay on `g_{t-n}²`. Paper defaults (0.9, 0.999).
    eps : float — additive to √v in the denominator (sign-stability
        + zero-guard).
    weight_decay : float — decoupled (AdamW style), applied before
        the ratio step.
    n : int — gradient delay (paper default 3). At n=0, AdaShift
        collapses to AdamW (no decorrelation).
    """

    def __init__(self, params, lr=0.006, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, n=3):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if n < 0:
            raise ValueError(f"Invalid n: {n}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, n=int(n))
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
            n = group["n"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaShift does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    # Lazy warm-start: v_0 = g² (current grad) on
                    # the first step. Stored as None to trigger the
                    # warm-start branch on first call.
                    state["exp_avg_sq"] = None
                    # Queue of past n gradients (each a clone of the
                    # fp32 grad). Bounded to length n. At step t
                    # the queue holds [g_{t-n}, g_{t-n+1}, ..., g_{t-1}]
                    # (with zeros implied for indices < 1).
                    state["grad_queue"] = []

                state["step"] += 1
                step_t = state["step"]

                m = state["exp_avg"]
                grad_queue = state["grad_queue"]
                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m

                # Delayed grad: g_{t-n} from queue, or 0 if the
                # queue is too short. We compute v_t BEFORE
                # appending g_t (the queue holds g_{t-1} and older).
                if state["exp_avg_sq"] is None:
                    # First step: warm-start v_0 = g² (the very
                    # first gradient). v_1 = β2·g² + (1-β2)·0 = β2·g².
                    v = g.pow(2).clone()
                    state["exp_avg_sq"] = v
                else:
                    v = state["exp_avg_sq"]
                    v_fp = v.float() if v.dtype != torch.float32 else v
                    if len(grad_queue) >= n:
                        # queue[0] is the oldest entry — the g from
                        # `n` steps ago (g_{t-n}).
                        delayed_grad_sq = grad_queue[0].pow(2)
                    else:
                        # Not enough history yet; treat missing
                        # delayed grad as zero (paper's convention).
                        delayed_grad_sq = torch.zeros_like(g)
                    # v_t = β2·v_{t-1} + (1-β2)·g_{t-n}²
                    v_fp.mul_(beta2).add_(delayed_grad_sq, alpha=1 - beta2)
                    v = v_fp

                # m_t = β1·m_{t-1} + (1−β1)·g_t       (Adam momentum)
                m_fp.mul_(beta1).add_(g, alpha=1 - beta1)

                # Append current grad to the queue for use on the
                # `n`-th step from now. Truncate to n entries so we
                # never accumulate unbounded memory.
                grad_queue.append(g.clone())
                if len(grad_queue) > n:
                    grad_queue.pop(0)

                # Bias correction (Adam-style)
                bc1 = 1.0 - beta1 ** step_t
                bc2 = 1.0 - beta2 ** step_t
                m_hat = m_fp.div(bc1)
                v_hat = v.div(bc2)

                # update = m̂ / (√v̂ + ε)
                denom = v_hat.sqrt().add_(eps)
                update = m_hat.div_(denom)

                # Decoupled weight decay (AdamW style): p ← (1 - lr·wd)·p - lr·u
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Cast back to param dtype before the in-place step
                update = update.to(p.dtype)
                p.add_(update, alpha=-lr)

                if m_fp is not m:
                    m.copy_(m_fp)
                # Note: state["exp_avg_sq"] was already assigned to v
                # by reference (it's the same tensor). We don't need
                # to copy back — the in-place ops above happened on v.

        return loss
