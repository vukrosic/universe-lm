import torch
import torch.nn.functional as F

# coeffs for polar express 
# not pre_computed, same as modded-nanoGPT 
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

def _maybe_compile(fn):
    # torch.compile/Inductor crashes on MPS for this kernel; CUDA is fine.
    if torch.cuda.is_available():
        return torch.compile(fn)
    return fn


@_maybe_compile
def zeropower_polar_express(G:torch.Tensor, steps: int = 5, coeffs_mode: str = "polar_express"):
    """Orthogonalize a matrix via Newton-Schulz / Polar Express iteration.

    coeffs_mode:
        "polar_express" — modded-nanogpt 5-step schedule (default,
            byte-identical to the original kernel).
        "newton_schulz" — classic NS quintic; uses a single
            (a, b, c) = (3.4445, -4.7750, 2.0315) at every step.
    """
    assert G.ndim >= 2
    if coeffs_mode == "polar_express":
        assert steps <= len(coeffs_list)
        coeffs = coeffs_list[:steps]
    elif coeffs_mode == "newton_schulz":
        ns_coeff = (3.4445, -4.7750, 2.0315)
        coeffs = [ns_coeff] * steps
    else:
        raise ValueError(f"Unknown coeffs_mode: {coeffs_mode!r}")

    X = G.bfloat16()
    # X = G.half()

    transpose_needed = G.size(-2) > G.size(-1) # transposing if tall matrix
    if transpose_needed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7) # safety factor

    for a , b, c in coeffs:
        A = X @ X.mT
        A2 = A @ A
        B = b * A + c * A2
        X = a * X + B @ X  # Right-multiplication for left polar factor

    if transpose_needed:
        X = X.mT

    return X # orthogonalized




class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Polar Express / Newton Schulz.

    Optional cautious-update mask (Liang et al. 2024, arXiv 2411.16085):
    zero out the orthogonalized update component when its sign disagrees
    with the current gradient. Suppresses stale-momentum artifacts. The
    masked components are zero, so the effective step norm shrinks ~10-20%
    on average — caller is expected to bump lr slightly to compensate
    (the project default bump is 0.024 → 0.025, +4%). Bit-identical to
    baseline when cautious=False (default).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, orthogonalize=True, coeffs_mode="polar_express", shape_scale=True, scale_mode="shape_aspect", adamw_lr=0.006, lazy_ortho_steps=1, cautious=False, moonlight_c=0.2):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, orthogonalize=orthogonalize, coeffs_mode=coeffs_mode, shape_scale=shape_scale, scale_mode=scale_mode, adamw_lr=adamw_lr, lazy_ortho_steps=lazy_ortho_steps, cautious=cautious, moonlight_c=moonlight_c)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=torch.float32)

                buf = state["momentum_buffer"]
                if buf.dtype != torch.float32:
                    buf = buf.float()
                    state["momentum_buffer"] = buf

                g = g.float()
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                # M13 lever — LazyOrtho: orthogonalize every N steps, reuse the
                # cached polar-express output in between. With lazy_ortho_steps=1
                # (default) the ortho re-runs every step — byte-identical to the
                # pre-M13 baseline. With N>1 we cut the polar-express cost ~Nx;
                # the momentum buffer is still updated every step, so the cached
                # "old" orthogonalized g is applied to a "new" momentum buffer.
                # The plan claims this is loss-neutral; speed lever. See
                # docs/research/muon/plan.md (Batch 5).
                if group["orthogonalize"]:
                    if "ortho_step" not in state:
                        state["ortho_step"] = -1
                        state["cached_ortho_g"] = None
                    state["ortho_step"] += 1
                    if (
                        state["cached_ortho_g"] is None
                        or state["ortho_step"] % group["lazy_ortho_steps"] == 0
                    ):
                        g = zeropower_polar_express(g, steps=group["ns_steps"], coeffs_mode=group["coeffs_mode"]) # steps are 5 for both ns and pe
                        state["cached_ortho_g"] = g
                    else:
                        g = state["cached_ortho_g"]
                g = g.to(p.dtype)
                # M5 lever — NoShapeScale: drop the max(1, fanout/fanin)**0.5 factor.
                # When shape_scale=True (default) the scale is unchanged; when False,
                # the update uses a flat lr. Changes effective step size → must sweep
                # muon_lr. See docs/research/muon/plan.md, Batch 2.
                #
                # M6 lever — SpectralScale: under scale_mode="spectral" use the
                # modded-nanogpt 0.2·sqrt(max(dims)) scale instead. A different
                # principled update-norm target. Default scale_mode="shape_aspect"
                # is byte-identical to the previous behavior. Also changes
                # effective step size → must sweep muon_lr. See
                # docs/research/muon/plan.md, Batch 2.
                #
                # M7 lever — RMSMatchScale: under scale_mode="rms_match" rescale
                # the orthogonalized update g so its RMS equals AdamW's typical
                # per-step RMS (≈ adamw_lr, since Adam normalizes the update by
                # sqrt(v) ≈ 1). Makes Muon/AdamW steps commensurate — fairer LR
                # coupling. Changes effective step size → must sweep muon_lr.
                # See docs/research/muon/plan.md, Batch 2.
                if not group["shape_scale"]:
                    scale = 1.0
                elif group["scale_mode"] == "shape_aspect":
                    scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                elif group["scale_mode"] == "spectral":
                    scale = 0.2 * (max(p.size(-2), p.size(-1)) ** 0.5)
                elif group["scale_mode"] == "moonlight":
                    # #15 Moonlight Muon (Kimi / Moonshot AI, arXiv:2502.16982).
                    # Per-tensor RMS rescale `c·sqrt(max(d_in, d_out))` so every
                    # 2-D weight has an approximately unit-RMS element-wise
                    # update — geometric calibration across matrix shapes
                    # (1:1 attention heads vs 1:4 FFN up). Mathematically the
                    # same family as `spectral` but with the paper's tuned
                    # constant carried on the group as `moonlight_c` (default
                    # 0.2) and a paper-named key for traceability. See
                    # autoresearch/ideas/015-moonlight-muon-rms/plan.md.
                    c = float(group.get("moonlight_c", 0.2))
                    scale = c * (max(p.size(-2), p.size(-1)) ** 0.5)
                elif group["scale_mode"] == "rms_match":
                    adamw_lr = group.get("adamw_lr", 0.006)
                    g_flat = g.view_as(p)
                    mu = g_flat.float().pow(2).mean().sqrt().clamp_min(1e-12)
                    scale = adamw_lr / mu
                else:
                    raise ValueError(f"Unknown scale_mode: {group['scale_mode']!r}")
                # Cautious update mask (Liang et al. 2024). When the
                # orthogonalized update component disagrees with the
                # current gradient's sign, the orthogonalized direction
                # is likely a stale-momentum artifact — zero it out.
                # Saves work on the agreeing components. Default off.
                if group.get("cautious", False):
                    grad_for_mask = p.grad.to(g.dtype) if p.grad is not None else None
                    if grad_for_mask is not None:
                        mask = (g * grad_for_mask > 0).to(g.dtype)
                        g = g * mask
                p.add_(g.view_as(p), alpha=-group["lr"] * scale)
