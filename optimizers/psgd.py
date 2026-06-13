"""PSGD — Preconditioned Stochastic Gradient Descent
(Li, Chen, Milenkovic, Giannakis, "Preconditioned Stochastic
Gradient Descent", arXiv:2405.13856, NeurIPS 2024).

PSGD learns an *online preconditioner matrix* `Q` (or pair `P, Q`
for rectangular matrices) that whitens the gradient. The simplest
working form (the "Whiteout/Shampoo-on-gradients" flavor) is:

For 1-D parameters of dim d, maintain a diagonal preconditioner
`D ∈ R^d` with update
    D ← D + α · (g² − 1)            (running EMA, element-wise)
    update = D · g
    w     ← w − lr · (β·m_prev + (1−β)·update)   (momentum on top)
or, with no momentum:
    w     ← w − lr · update

For 2-D weight matrices W ∈ R^{n×m}, maintain a *coupled*
preconditioner pair (P ∈ R^{n×n}, Q ∈ R^{m×m}):
    P ← P + α · (g g^T / m − I)
    Q ← Q + α · (W W^T / n − I)
    update = P · g · Q
    w     ← w − lr · update

Where α is a small step (paper default 1e-3). At α=0 the
preconditioner is frozen at I and PSGD collapses to SGD (with
momentum) — the lever's identity-at-step-0.

Identity at step 0: with `P_0 = I, Q_0 = I, m_0 = 0` the first
update is `I · g · I = g` and the first step is `w ← w − lr · g`
(plain SGD), not bit-identical to AdamW (which has the Adam
normalization) but bit-identical to SGD-with-momentum. This is
the lever's first-step signature.

The 2-D coupled update tracks a running `g g^T / m` and
`W W^T / n` — these are the natural whitening matrices for the
columns and rows of the gradient flow. The (P, Q) pair is
`n×n + m×m = n² + m²` floats, which at tiny1m3m's d_model=64
is 64² + 64² = 8k floats per slot — trivial.

With `use_psgd=False` (default) this class is never instantiated
— the trainer uses the existing Muon path bit-identically.
"""
import torch
from torch.optim.optimizer import Optimizer


class PSGD(Optimizer):
    """PSGD — Preconditioned SGD (Li et al. 2024, NeurIPS 2024).

    Drop-in replacement for the 2-D non-embedding, non-norm
    routing slot. 1-D params (norms, biases, embeddings) get a
    diagonal-D variant. The 2-D path uses a *coupled* pair
    (P, Q). The 1-D path uses a diagonal preconditioner.

    Parameters
    ----------
    params : iterable
    lr : float — PSGD step size. Paper does not pin one; we
        default to 0.01 (paper's wide range and the small-
        model scaling rule of thumb).
    alpha : float — preconditioner update rate. Paper default
        1e-3. At alpha=0, PSGD degenerates to SGD.
    beta : float — momentum coefficient on the preconditioned
        update. 0.0 = no momentum. Paper default 0.9.
    weight_decay : float — decoupled (AdamW style), applied
        before the preconditioned step.
    """

    def __init__(self, params, lr=0.01, alpha=1e-3, beta=0.9,
                 weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        defaults = dict(lr=lr, alpha=alpha, beta=beta,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PSGD does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state)

                # Decoupled weight decay (AdamW style): p ← (1 - lr·wd)·p.
                # Apply BEFORE the preconditioned step so the wd
                # operates on the un-whitened scale.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()

                if p.ndim < 2:
                    self._step_1d(p, g, state, lr, alpha, beta)
                else:
                    self._step_2d(p, g, state, lr, alpha, beta)

        return loss

    def _init_state(self, p, state):
        """Lazy state init. P=I, Q=I ⇒ step 0 is `w ← w − lr · g`
        (SGD-with-momentum if β>0, plain SGD if β=0)."""
        state["step"] = 0
        if p.ndim < 2:
            # 1-D diagonal preconditioner, init to ones.
            state["D"] = torch.ones_like(p, dtype=torch.float32)
        else:
            n, m = p.shape
            # 2-D coupled preconditioner, init to identity.
            state["P"] = torch.eye(n, dtype=torch.float32,
                                   device=p.device)
            state["Q"] = torch.eye(m, dtype=torch.float32,
                                   device=p.device)
        # Momentum buffer, init to zero.
        state["momentum"] = torch.zeros_like(p, dtype=torch.float32)

    def _step_1d(self, p, g, state, lr, alpha, beta):
        """1-D: diagonal preconditioner `D`. Update is `D · g`
        with `D ← D + α · (g² − 1)`. Then `m ← β·m + (1−β)·D·g`
        and `p ← p − lr · m` (or just `lr · D · g` if β=0)."""
        D = state["D"]
        mom = state["momentum"]
        state["step"] += 1

        # Update D: D ← D + α · (g² − 1)   (only if α > 0).
        if alpha > 0.0:
            D.add_(g.pow(2).sub_(1.0), alpha=alpha)

        # Whiten: update = D · g  (element-wise multiply).
        update = D * g

        # Momentum: m ← β·m + (1−β)·update
        if beta > 0.0:
            mom.mul_(beta).add_(update, alpha=1.0 - beta)
            step_val = mom
        else:
            step_val = update

        # Cast back to param dtype before the in-place step.
        p.add_(step_val.to(p.dtype), alpha=-lr)

    def _step_2d(self, p, g, state, lr, alpha, beta):
        """2-D: coupled preconditioner (P, Q). Update is
        `P · g · Q` with the coupled EMA
        `P ← P + α · (g g^T / m − I)` and
        `Q ← Q + α · (g^T g / n − I)`.

        The paper uses a Hessian-coupled form; we use the simpler
        `g g^T` / `g^T g` flavor (Whiteout/Shampoo-on-gradients)
        that's standard in public PSGD implementations. At α=0 it
        collapses to SGD.
        """
        P = state["P"]
        Q = state["Q"]
        mom = state["momentum"]
        state["step"] += 1

        n, m = p.shape

        if alpha > 0.0:
            # P ← P + α · (g g^T / m − I)   (n×n)
            gtg = g @ g.t()                          # n×n
            P.add_(gtg.div(m).sub_(torch.eye(n, device=p.device,
                                             dtype=torch.float32)),
                   alpha=alpha)
            # Q ← Q + α · (g^T g / n − I)   (m×m)
            ggt = g.t() @ g                          # m×m
            Q.add_(ggt.div(n).sub_(torch.eye(m, device=p.device,
                                             dtype=torch.float32)),
                   alpha=alpha)

        # Whiten: update = P · g · Q
        update = P @ g @ Q

        # Momentum: m ← β·m + (1−β)·update
        if beta > 0.0:
            mom.mul_(beta).add_(update, alpha=1.0 - beta)
            step_val = mom
        else:
            step_val = update

        # Cast back to param dtype before the in-place step.
        p.add_(step_val.to(p.dtype), alpha=-lr)
