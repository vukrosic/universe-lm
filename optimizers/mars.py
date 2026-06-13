"""MARS: Variance-Reduced AdamW (Yuan et al. 2024, arXiv:2401.03855).

A drop-in replacement for `torch.optim.AdamW` that operates on the
*gradient* (not the per-parameter state) to add a variance-reduction
correction. For each parameter, MARS maintains a ring buffer of past
first-moment snapshots `m_history[2*lag]`. On each step, it reads two
historical m snapshots at indices `-lag` and `-2*lag` from the buffer
and constructs a corrected gradient

    g̃_t = g_t + mix_coef * (m_{t-lag} − m_{t-2*lag})

which is then fed to the standard AdamW `(m, v, bias-correction,
decoupled-WD)` step. The per-parameter `v` is *untouched* — MARS only
operates on the gradient input.

Identity at step 0: the ring buffer is empty for the first `2*lag`
steps, so the correction term is undefined and `g̃_t = g_t` ⇒ the
first `2*lag` optimizer steps are bit-identical to plain
`torch.optim.AdamW`. After that the lag correction engages.

When `use_mars=False` (default), this class is never instantiated —
the trainer uses `torch.optim.AdamW` unchanged. See
`training/trainer.py:setup_muon_optimizer` for the gate.
"""
import torch
from torch.optim import AdamW


class MARSAdamW(AdamW):
    """AdamW with MARS variance-reduced gradient (Yuan et al. 2024).

    Subclass of `torch.optim.AdamW`. The parent's `.step()` is called
    with a *corrected* gradient `g̃_t` (raw `g_t` + a lag-based
    correction) and the parent's `(m, v, bias-correction, decoupled
    WD)` math runs unchanged on it. The first-moment buffer `exp_avg`
    is then updated by the parent on the *corrected* gradient, and
    MARS snapshots the new `exp_avg` into the ring buffer for use on
    the next step.

    Parameters
    ----------
    params : iterable
    lr : float
    betas : (β1, β2) — Adam
    eps : float — Adam denominator
    weight_decay : float — decoupled (AdamW style)
    lag : int — lookback window. Paper default 10. MARS needs
        snapshots at `t-lag` and `t-2*lag` ⇒ the ring buffer is
        `2*lag` deep and the first `2*lag` steps are bit-identical
        to baseline AdamW.
    mix_coef : float — scale on the correction. Paper default 0.5.
        At mix_coef=0, MARS degenerates to plain AdamW.
    lr_scale : float — optional LR multiplier applied to the parent
        `lr` (paper does not require re-tuning; default 1.0).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, lag=10, mix_coef=0.5, lr_scale=1.0):
        if lag < 1:
            raise ValueError(f"MARS lag must be >= 1, got {lag}")
        if not 0.0 <= mix_coef <= 1.0:
            raise ValueError(f"MARS mix_coef should be in [0, 1], got {mix_coef}")
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        # Stash MARS-specific knobs in the default group dict so
        # `.state_dict()` / `.load_state_dict()` round-trip them.
        for group in self.param_groups:
            group["lag"] = int(lag)
            group["mix_coef"] = float(mix_coef)
            group["lr_scale"] = float(lr_scale)
            # If lr_scale != 1.0, the parent's `lr` is scaled so the
            # downstream `super().step()` consumes the scaled value
            # in-place. This keeps the `.state_dict()` clean (one
            # canonical `lr` field).
            if group["lr_scale"] != 1.0:
                group["lr"] = group["lr"] * group["lr_scale"]
        self._mars_initialized = False

    def _init_mars_state(self):
        """No-op placeholder. The ring buffer is allocated lazily
        in pass 3 of `.step()` (AFTER the parent has initialized
        `exp_avg` on the first call). This avoids the chicken-and-
        egg: if we pre-allocate `m_history` in state, the parent's
        `_init_group` sees a non-empty state and SKIPS its own
        `exp_avg` / `exp_avg_sq` lazy init, then dereferences
        `state["exp_avg"]` and KeyError's.
        """
        self._mars_initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        # Lazy state allocation — just sets a flag, the actual
        # ring buffer is allocated in pass 3 below.
        if not self._mars_initialized:
            self._init_mars_state()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Pass 1: read the lag buffer, optionally swap `p.grad` to
        # the corrected gradient. We do this BEFORE the parent's
        # step runs so the parent sees the corrected grad when it
        # updates `exp_avg` (the m_t at the end of this step is the
        # EMA of the *corrected* gradient — the paper's recipe).
        # On the first call, `m_history` does not exist yet (it is
        # allocated in pass 3 on the previous step — which doesn't
        # exist for the very first call). Both m_old and m_older
        # are None ⇒ no correction is applied. This makes the
        # first call bit-identical to plain AdamW.
        grad_backups = []
        try:
            for group in self.param_groups:
                lag = group["lag"]
                mix_coef = group["mix_coef"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "MARSAdamW does not support sparse gradients"
                        )
                    state = self.state[p]
                    if "m_history" not in state:
                        # Ring buffer not yet allocated (this is
                        # the first call for this param, or a
                        # param with no grad last step). No
                        # correction possible.
                        continue
                    m_history = state["m_history"]
                    head = state["m_head"]
                    old_idx = (head - lag) % (2 * lag)
                    older_idx = (head - 2 * lag) % (2 * lag)
                    m_old = m_history[old_idx]
                    m_older = m_history[older_idx]

                    if m_old is not None and m_older is not None and mix_coef != 0.0:
                        # MARS correction:
                        #   g̃_t = g_t + mix_coef * (m_{t-lag} − m_{t-2*lag})
                        # Clone p.grad (to avoid mutating the
                        # autograd graph) and replace it with the
                        # corrected grad. We back up so we can
                        # restore after the parent's `.step()`.
                        grad_backup = grad.detach().clone(
                            memory_format=torch.preserve_format
                        )
                        grad_backups.append((p, grad_backup))
                        # In-place: subtract, scale, add (memory-friendly).
                        p.grad = m_old.sub(m_older).mul_(mix_coef).add_(grad)

            # Pass 2: delegate to the parent AdamW with the (possibly
            # corrected) gradients in place. The parent updates
            # `exp_avg`, `exp_avg_sq`, applies decoupled WD, and
            # subtracts the AdamW update from p. This single call
            # iterates every param in every group exactly once.
            # On the first call, the parent lazily creates
            # `exp_avg` / `exp_avg_sq` / `step` (we did NOT pre-
            # allocate these in state to avoid colliding with the
            # parent's lazy-init `len(state) == 0` check).
            super().step(closure=None)

            # Pass 3: allocate the ring buffer (first call only)
            # and snapshot the freshly-updated `exp_avg` (which is
            # now the m_t for this step) into the ring buffer for
            # use on the next step. This is what makes the buffer
            # "previous-step" m snapshots on the next call.
            for group in self.param_groups:
                lag = group["lag"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "exp_avg" not in state:
                        # Parent didn't initialize this param (it
                        # had no grad). Skip.
                        continue
                    if "m_history" not in state:
                        # First call for this param: allocate the
                        # ring buffer. m_history is all-None; the
                        # next 2*lag-1 calls will fill it in.
                        state["m_history"] = [None] * (2 * lag)
                        state["m_head"] = 0
                    head = state["m_head"]
                    state["m_history"][head] = state["exp_avg"].detach().clone(
                        memory_format=torch.preserve_format
                    )
                    state["m_head"] = (head + 1) % (2 * lag)
        finally:
            # Restore all backed-up p.grad tensors so the next
            # forward's autograd graph is unaffected by the swap.
            for p, grad_backup in grad_backups:
                p.grad = grad_backup
        return loss
