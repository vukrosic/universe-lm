"""GaLore — Gradient Low-Rank Projection (Zhao et al. 2024, arXiv:2403.03507).

For each 2-D weight matrix W ∈ R^{n×m}, maintain two projection
matrices P ∈ R^{n×r}, Q ∈ R^{m×r} with r ≪ min(n,m). The gradient G
is projected into a low-rank subspace before AdamW is applied:

    G̃       = P^T @ G @ Q                      (shape r×r)
    update   = AdamW(G̃)                        (state r×r not n×m)
    W       ← W − lr · P @ update @ Q^T        (shape n×m)

Every `proj_every` steps, P, Q are refreshed from the SVD of a running
gradient EMA `G_ema ← γ·G_ema + (1−γ)·G`. With `proj_every=200` (paper
default) the projection is mostly frozen, so AdamW sees a stable
low-rank view of the gradient.

1-D params get pure AdamW (eigendecomposition / SVD is meaningless on
1-D). The routing in `training/trainer.py:setup_muon_optimizer` keeps
embeddings, norms, and 1-D scalars on AdamW; only 2-D non-embed, non-
norm params take this path.

Identity at step 0: the FORWARD pass is unchanged (no model-graph
change), so the val score at step 0 (computed before any optimizer
step) is bit-identical to the baseline. The first optimizer step
itself differs from AdamW's first step (GaLore's step is
`P @ sign(P^T G Q) @ Q^T`, not `sign(G)`), but the model is identical
at eval step 0.
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class GaLoreAdamW(Optimizer):
    """GaLore AdamW — AdamW on a low-rank projection of the gradient.

    Per-tensor dispatch:
      - ndim < 2: pure AdamW (projection is meaningless on 1-D).
      - ndim == 2: AdamW in the rank-`rank` subspace spanned by
        `P @ Q^T`, with periodic SVD refresh from the gradient EMA.

    State per 2-D param: `P`, `Q` (n×r and m×r, the projection
    matrices), `grad_ema` (n×m, the running gradient for SVD basis
    refresh), `exp_avg`, `exp_avg_sq` (r×r, the Adam moments in the
    projected space). The Adam state is r×r — much smaller than the
    full n×m state of vanilla AdamW, hence the "memory-efficient"
    half of the paper's claim.

    Parameters
    ----------
    params : iterable
    lr : float
    betas : (β1, β2) for Adam
    eps : float — Adam denominator
    weight_decay : float — decoupled (AdamW style)
    rank : int — projection rank r (paper sweet spot 4-256, default 4)
    proj_every : int — refresh P, Q from SVD of grad EMA every K
        steps (paper default 200)
    proj_ema : float — EMA decay for the gradient before SVD
        (paper default 0.95 — `G_ema ← 0.95·G_ema + 0.05·G`)
    """

    def __init__(self, params, lr=0.006, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, rank=4, proj_every=200, proj_ema=0.95):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        rank=rank, proj_every=proj_every, proj_ema=proj_ema)
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
            rank = group["rank"]
            proj_every = group["proj_every"]
            proj_ema = group["proj_ema"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GaLore does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state, rank)

                state["step"] += 1
                step_t = state["step"]

                # Decoupled weight decay (AdamW style).
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                if p.ndim < 2:
                    # 1-D: pure AdamW (GaLore is meaningless on 1-D).
                    self._adamw_step(p, grad, state, step_t,
                                     beta1, beta2, lr, eps)
                    continue

                # Update running gradient EMA. Used as the SVD source
                # for projection refresh, so the projection tracks a
                # smoothed view of the gradient flow direction.
                state["grad_ema"].mul_(proj_ema).add_(grad, alpha=1 - proj_ema)

                # Refresh P, Q from SVD of the gradient EMA every
                # `proj_every` steps. SVD can fail on near-singular
                # matrices; in that case we keep the previous
                # projection (the AdamW path is unchanged).
                if proj_every > 0 and step_t % proj_every == 0:
                    self._refresh_projection(p, grad, state, rank)

                P = state["P"]
                Q = state["Q"]

                # Project: G̃ = P^T @ G @ Q  (r×r in projected space).
                # Casting to the param's dtype keeps the math
                # numerically compatible with bf16 training; the
                # running EMA already lives in the param's dtype.
                grad_proj = P.t() @ grad @ Q

                # AdamW on the projected gradient.
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t
                exp_avg.mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_proj, grad_proj,
                                                value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                update_proj = exp_avg / denom / bc1

                # Project back: W ← W − lr · P @ update_proj @ Q^T.
                update = P @ update_proj @ Q.t()
                p.add_(update, alpha=-lr)

        return loss

    def _init_state(self, p, state, rank):
        """Initialize per-param state. P, Q are random orthonormal via
        QR of a random normal — a well-conditioned basis with no
        preferred axis. The projection is frozen for the first
        `proj_every` steps, so the first batch of updates share one
        basis."""
        state["step"] = 0
        if p.ndim < 2:
            # 1-D: pure AdamW state. No projection / SVD machinery.
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            return

        n, m = p.shape
        rank_actual = min(rank, n, m)

        # Initial P, Q: random orthonormal via QR. Stays frozen until
        # the first SVD refresh (step `proj_every`). `torch.linalg.qr`
        # on the Vast CUDA build lacks bf16/fp16 kernels (`geqrf_cuda
        # not implemented for BFloat16`), so the QR is *always* run in
        # float32 and cast back to the param dtype — the projection is
        # a random orthonormal basis, not a numerical copy, so the
        # cast has no downstream effect. (Round-2 fix: previous
        # conditional cast failed when p.dtype was float16; round-1
        # failed when p.dtype was bfloat16 — use float32 unconditionally.)
        P = torch.randn(n, rank_actual, device=p.device, dtype=torch.float32)
        P, _ = torch.linalg.qr(P)
        Q = torch.randn(m, rank_actual, device=p.device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(Q)
        state["P"] = P.to(dtype=p.dtype)
        state["Q"] = Q.to(dtype=p.dtype)

        # Running gradient EMA — source for the SVD basis refresh.
        state["grad_ema"] = torch.zeros_like(p)

        # Adam moments in the projected space (r×r). Zero-init ⇒
        # step-1 update is `sign(G̃)` projected back, the canonical
        # AdamW first-step behavior — just on a different basis.
        state["exp_avg"] = torch.zeros(rank_actual, rank_actual,
                                        device=p.device, dtype=p.dtype)
        state["exp_avg_sq"] = torch.zeros_like(state["exp_avg"])

    def _refresh_projection(self, p, grad, state, rank):
        """Refresh P, Q from the SVD of `state["grad_ema"]`. Resets
        the AdamW state on `G̃` so moments in the old basis don't
        leak across the basis change. Same bf16 → float32 promotion
        as the init path: `torch.linalg.svd` on the Vast CUDA build
        also lacks a bf16 kernel, and the SVD is a basis-extraction
        op whose result is a *projection* — not a numerical copy —
        so the cast is safe."""
        G_ema = state["grad_ema"]
        try:
            # `torch.linalg.svd` on the Vast CUDA build lacks bf16/fp16
            # kernels. Always promote to float32 — the SVD result is a
            # basis (U, Vh), not a numerical copy of G_ema, so the cast
            # has no downstream effect. (Round-2 fix.)
            G_svd = G_ema.to(dtype=torch.float32) if G_ema.dtype != torch.float32 \
                else G_ema
            U, S, Vh = torch.linalg.svd(G_svd, full_matrices=False)
        except Exception:
            # SVD can fail on near-singular / non-finite inputs (rare
            # in practice; the EMA has a 0.95 decay so it's well-
            # conditioned after a few steps). Keep the previous
            # projection — the AdamW path is unchanged and the next
            # refresh attempt will retry.
            return
        n, m = p.shape
        rank_actual = min(rank, n, m, S.shape[0])
        if rank_actual < 1:
            return
        P = U[:, :rank_actual].contiguous().to(dtype=p.dtype)
        Q = Vh[:rank_actual, :].t().contiguous().to(dtype=p.dtype)
        state["P"] = P
        state["Q"] = Q
        # Reset AdamW moments — they live in the OLD basis and are
        # stale. Zero-init reverts to the canonical step-1 sign-of-
        # grad behavior on the new basis.
        state["exp_avg"] = torch.zeros(rank_actual, rank_actual,
                                        device=p.device, dtype=p.dtype)
        state["exp_avg_sq"] = torch.zeros_like(state["exp_avg"])

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
