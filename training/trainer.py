import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _torch_ckpt
import math
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional, Callable, Dict, Any
from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from optimizers.muon import Muon
from optimizers.swan import SWAN
from optimizers.cautious_adamw import CautiousAdamW
from optimizers.soap import SOAP
from optimizers.schedule_free_adamw import ScheduleFreeAdamW
from optimizers.lion import Lion
from optimizers.tiger import Tiger
from optimizers.mars import MARSAdamW
from optimizers.galore import GaLoreAdamW
from optimizers.sam import AdamSAM
from optimizers.looksam import LookSAM
from optimizers.dadaptation import DAdaptAdamW
from optimizers.came import CAME
from optimizers.radam import RAdam
from optimizers.psgd import PSGD
from optimizers.adashift import AdaShift
from optimizers.spectral_decoupling import SDAdamW
from optimizers.adapnm import AdaPNM
from optimizers.adamp import AdamP
from optimizers.adabelief import AdaBelief
from optimizers.sophia import Sophia
from training.checkpointing import (
    capture_git_metadata,
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from training.device import resolve_device
from training.evaluation import evaluate_model
from utils.helpers import set_seed, format_time


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 30, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_step = 0
        
    def __call__(self, val_loss: float, step: int) -> bool:
        """Returns True if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = step
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered at step {step}")
                print(f"   Best loss: {self.best_loss:.4f} at step {self.best_step}")
                return True
            return False


def _swap_optimizers_eval_mode(optimizers, mode: str) -> None:
    """Swap Schedule-Free AdamW optimizers between train (y) and eval (x)
    iterates. The optimizer's `train_mode` group flag is idempotent, so
    repeated calls are safe. Other optimizers (Muon, AdamW, Cautious,
    SOAP) are no-ops here — they don't have an eval/train swap.

    The Schedule-Free paper requires eval at the averaged iterate `x`,
    not the gradient-following iterate `y`. We call `optimizer.eval()`
    before `evaluate_model(...)` and `optimizer.train()` after.
    """
    for opt in optimizers:
        if hasattr(opt, "eval") and hasattr(opt, "train"):
            getattr(opt, mode)()


def load_exact_step_baseline(config: LLMConfig) -> Dict[int, float]:
    """Load exact-step baseline values for log-only comparisons."""
    if config.train_tokens < 150_000_000:
        return {}
    baseline_path = Path("baselines/10m_baseline.json")
    if not baseline_path.exists():
        return {}
    with open(baseline_path) as f:
        baseline = json.load(f)
    return dict(zip(baseline.get("steps", []), baseline.get("val_losses", [])))


def default_metrics_history() -> Dict[str, list]:
    return {
        "steps": [],
        "val_losses": [],
        "val_accuracies": [],
        "val_perplexities": [],
        "elapsed_times": [],
        "learning_rates": [],
    }


class Lookahead:
    """Lookahead Optimizer Wrapper (Zhang et al. 2019, arXiv:1907.08610).

    Maintains a "slow" EMA copy of the model parameters alongside the
    live "fast" weights. Every `k` inner optimizer steps, pulls slow
    halfway toward fast (`slow ← slow + α·(fast − slow)`) and resets
    fast to slow. The inner optimizer's state dict (momentum buffers,
    AdamW exp_avg/exp_avg_sq, Muon momentum_buffer) is cleared at the
    outer step so the next inner step doesn't see stale gradients
    from before the slow reset — otherwise the next inner step would
    overshoot.

    Identity at step 0: `slow = θ_init`, first inner step uses the
    baseline Muon/AdamW path. The lookahead sync only fires at step
    `k`; before that the wrapper is a no-op.

    Args:
        optimizers: list of inner optimizers (e.g. [Muon, AdamW]).
        model: the model whose parameters to track.
        k: inner cycle length (paper default 5-10).
        alpha: slow step size (paper default 0.5).
    """

    def __init__(self, optimizers, model, k, alpha):
        self.optimizers = optimizers
        self.model = model
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        # Snapshot of initial slow weights (clones of fast at wrap time)
        self.slow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def step(self):
        # 1. Let the inner optimizers do their normal update
        for opt in self.optimizers:
            opt.step()
        self.step_count += 1
        # 2. Outer step: every k inner steps, sync fast to slow
        if self.step_count % self.k == 0:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if not p.requires_grad or n not in self.slow:
                        continue
                    # slow <- slow + alpha * (fast - slow)
                    self.slow[n].add_(p.detach() - self.slow[n], alpha=self.alpha)
                    # fast <- slow
                    p.data.copy_(self.slow[n])
            # 3. Clear inner optimizer state to avoid stale momentum
            #    carrying across the slow reset. Each inner optimizer's
            #    `state` is a defaultdict — `.clear()` keeps the type
            #    but drops every entry.
            for opt in self.optimizers:
                if hasattr(opt, "state") and opt.state is not None:
                    opt.state.clear()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()


class ModelEMA:
    """110 — Polyak-Ruppert Weight EMA (Polyak 1990).

    Maintains a shadow copy of every trainable parameter. After each
    optimizer step, the EMA is updated as
        θ_ema ← μ·θ_ema + (1−μ)·θ_live
    where μ is the *current* (post-ramp) decay:
        μ = ema_decay * min(1, step / ema_warmup_steps)
    During the warm-up window `μ` ramps linearly from 0 to `ema_decay`,
    so at step 0 the update is `θ_ema ← 0·θ_ema + 1·θ_live = θ_live`
    ⇒ byte-identical to the live model. Once the ramp saturates
    (`step ≥ ema_warmup_steps`) the EMA becomes a long-horizon
    Polyak average of the trajectory.

    The EMA is **not** a parameter of the model — it's a sibling
    tensor buffer with its own `.state_dict()` for checkpointing.
    When `ema_eval_only=True` (default) the live `θ_live` is the
    saved/resumed model and the EMA is only swapped in around
    `evaluate_model(...)` via `apply_to(model)` / `restore_from(model)`.

    Args:
        model: the live model whose named_parameters to shadow.
        decay: target decay μ (paper default 0.999).
        warmup_steps: ramp length; decay ramps 0 → `decay` over this
            many optimizer steps.
    """

    def __init__(self, model, decay, warmup_steps):
        self.decay = float(decay)
        self.warmup_steps = max(1, int(warmup_steps))
        self.step_count = 0
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    @torch.no_grad()
    def update_from(self, model):
        """Update the EMA from the current live weights of `model`.
        Call once after every optimizer.step(). Identity at step 0
        (μ=0 ⇒ θ_ema ← θ_live)."""
        self.step_count += 1
        t = min(1.0, self.step_count / self.warmup_steps)
        mu = self.decay * t
        for n, p_live in model.named_parameters():
            if not p_live.requires_grad or n not in self.shadow:
                continue
            p_ema = self.shadow[n]
            p_ema.mul_(mu).add_(p_live.detach(), alpha=1.0 - mu)

    @torch.no_grad()
    def apply_to(self, model):
        """Swap EMA weights into `model`'s live parameters for eval.
        Returns a dict of backup tensors so `restore_from(model, backup)`
        can put the live weights back. Used as:
            backup = ema.apply_to(model)
            try: evaluate_model(model, ...)
            finally: ema.restore_from(model, backup)
        """
        backup = {}
        for n, p_live in model.named_parameters():
            if not p_live.requires_grad or n not in self.shadow:
                continue
            backup[n] = p_live.detach().clone()
            p_live.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore_from(self, model, backup):
        """Restore live weights from a backup dict (the inverse of
        `apply_to`). Always run in a `finally` so a val crash doesn't
        leave the model on the EMA copy."""
        for n, p_live in model.named_parameters():
            if n in backup:
                p_live.data.copy_(backup[n])

    def state_dict(self):
        return {"step_count": self.step_count, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.step_count = int(sd.get("step_count", 0))
        shadow = sd.get("shadow", {})
        for n, p in shadow.items():
            if n in self.shadow:
                self.shadow[n].copy_(p)


def _rdrop_loss(logits_1, logits_2, shift_labels, vocab_size, alpha):
    """115 — R-Drop (Liang et al. 2021, arXiv:2106.14448) loss head.

    Computes the per-token CE on each of two forward passes (with
    different dropout masks) and the symmetric KL between their
    softmax distributions. Returns:

        (mean CE of two passes,  KL penalty scalar to add to the loss)

    `alpha` is the *current* KL weight — already linearly warmed by
    the caller from 0 → target over `rdrop_warmup_steps`. With
    `alpha = 0` the KL term vanishes (bit-identical to averaging two
    dropout-noisy CEs); the KL itself is always >= 0.

    Memory: at B=2, T=2048, V=49152 the naive symmetric KL
    materialises four [N_valid, V] fp32 tensors (~3.2 GiB total) which
    OOMs on a 12.5 GiB RTX 3060 once activations + the second forward's
    logits are accounted for. We chunk along the N dimension
    (`_RDROP_KL_CHUNK=512`) so the peak intermediate is one
    [512, 49152] fp32 tensor (~100 MB) plus two log_softmax scratch
    tensors. `batchmean` ≡ `sum / N_valid` so we accumulate the sum
    and divide at the end (mathematically equivalent, byte-equivalent
    in fp32 modulo the order of additions within a chunk-size window).

    Args:
        logits_1, logits_2: [B, T, V] tensors from two forward passes.
        shift_labels:       [B, T] int64 with -100 on masked positions.
        vocab_size:         V (used to flatten).
        alpha:              scalar weight on the KL term (>= 0).
    """
    N = logits_1.shape[0] * logits_1.shape[1]
    flat_y = shift_labels.view(-1)
    # CE on full flatten — PyTorch's cross_entropy is memory-efficient
    # (does NOT materialise the [N, V] log-softmax). We keep the logits
    # in their original dtype (bf16 under autocast) to halve the
    # memory pressure; cross_entropy handles the upcast internally.
    ce = 0.5 * (
        F.cross_entropy(
            logits_1.reshape(N, vocab_size), flat_y, ignore_index=-100,
        )
        + F.cross_entropy(
            logits_2.reshape(N, vocab_size), flat_y, ignore_index=-100,
        )
    )
    # Chunked symmetric KL. `batchmean` is `sum / N_valid`, so we
    # accumulate the per-chunk sum and divide by the count of valid
    # positions at the end. We never materialise the full
    # [N_valid, V] tensors — each chunk is [chunk, V] in fp32
    # (~100 MB at chunk=512).
    CHUNK = _RDROP_KL_CHUNK
    flat_1_view = logits_1.reshape(N, vocab_size)
    flat_2_view = logits_2.reshape(N, vocab_size)
    kl_sum_12 = logits_1.new_zeros(())
    kl_sum_21 = logits_1.new_zeros(())
    n_valid_total = 0
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk_y = flat_y[start:end]
        valid = chunk_y != -100
        n_v = int(valid.sum().item())
        if n_v == 0:
            continue
        c1 = flat_1_view[start:end][valid].float()
        c2 = flat_2_view[start:end][valid].float()
        log_p1 = F.log_softmax(c1, dim=-1)
        log_p2 = F.log_softmax(c2, dim=-1)
        # F.kl_div(input=log_p, target=p) with reduction='sum'
        # already returns the per-chunk sum over all elements of
        # the [chunk, V] tensor. Summing over chunks gives the full
        # total sum; dividing by n_valid_total at the end yields
        # the equivalent of `batchmean` over the full [N_valid, V].
        kl_sum_12 = kl_sum_12 + F.kl_div(log_p1, log_p2.exp(), reduction='sum')
        kl_sum_21 = kl_sum_21 + F.kl_div(log_p2, log_p1.exp(), reduction='sum')
        n_valid_total += n_v
    if n_valid_total > 0:
        kl = 0.5 * (kl_sum_12 + kl_sum_21) / n_valid_total
    else:
        kl = logits_1.new_zeros(())
    return ce, alpha * kl


# R-Drop KL chunk size. 512 positions × V=49152 × 4 bytes ≈ 96 MiB
# per chunk (in fp32). At B*T = 4096 this gives 8 chunks per step.
# Tuned for tiny1m3m (B=2, T=2048) on RTX 3060 12.5 GiB — should be
# reduced further on smaller VRAM. Constant (not a config flag) to
# keep the LoC delta minimal and the step-0 path bit-identical.
_RDROP_KL_CHUNK = 512


class BornAgainTeacher:
    """132 — Born-Again Self-Distillation Teacher (Furlanello et al. 2018,
    arXiv:1805.04770).

    Maintains a shadow copy of every trainable parameter. After each
    optimizer step, the shadow is updated as
        θ_teacher ← (1 − β) · θ_teacher + β · θ_student
    where `β = born_again_beta` (paper default 0.999; high β ⇒ teacher
    tracks the student closely). Unlike `ModelEMA`, there is **no
    warmup ramp**: at step 0 the shadow is a clone of the live init,
    so `apply_to(model)` puts the original init into `model` ⇒ a
    forward with the swapped params is byte-identical to the
    student forward at step 0 ⇒ the KL distillation loss is exactly
    0 and the total loss equals the baseline CE.

    The teacher is **only** swapped in for the no-grad distillation
    forward (called once per step from the train loop); it does NOT
    participate in eval or checkpoint save. With `use_born_again=
    False` (default) the trainer never builds a teacher and the
    baseline path is bit-identical.

    Args:
        model: the live student model whose named_parameters to shadow.
        beta: EMA "speed" (paper default 0.999).
    """

    def __init__(self, model, beta):
        self.beta = float(beta)
        # Clone every trainable parameter at construction so the
        # shadow starts as a deep copy of the live init. We move the
        # clone onto the same device as the live param so the swap
        # below is device-mismatch-free on CUDA/MPS/CPU.
        self.shadow = {
            n: p.detach().clone().to(p.device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def apply_to(self, model):
        """Swap shadow weights into `model`'s live parameters for the
        teacher distillation forward. Returns a backup dict so
        `restore_from(model, backup)` can put the live weights back.
        Always restore in a `finally` so a teacher-forward crash
        never leaves the model on the teacher copy."""
        backup = {}
        for n, p_live in model.named_parameters():
            if not p_live.requires_grad or n not in self.shadow:
                continue
            backup[n] = p_live.detach().clone()
            p_live.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore_from(self, model, backup):
        """Restore live weights from a backup dict (the inverse of
        `apply_to`). Always run in a `finally`."""
        for n, p_live in model.named_parameters():
            if n in backup:
                p_live.data.copy_(backup[n])

    @torch.no_grad()
    def update_from(self, model):
        """EMA update after each optimizer step:
            θ_teacher ← (1−β)·θ_teacher + β·θ_student.
        Called once per optimizer step from the train loop."""
        one_minus_beta = 1.0 - self.beta
        for n, p_live in model.named_parameters():
            if not p_live.requires_grad or n not in self.shadow:
                continue
            p_teacher = self.shadow[n]
            p_teacher.mul_(one_minus_beta).add_(p_live.detach(), alpha=self.beta)

    def state_dict(self):
        return {"beta": self.beta, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.beta = float(sd.get("beta", self.beta))
        shadow = sd.get("shadow", {})
        for n, p in shadow.items():
            if n in self.shadow:
                self.shadow[n].copy_(p.to(self.shadow[n].device))


def _born_again_distill_kl(student_logits, model, x, teacher, T, alpha, vocab_size):
    """132 — compute the Born-Again distillation KL term.

        KL(softmax(teacher/T) ‖ softmax(student/T)) · T² · α

    Swaps the teacher weights into the live model, runs a no-grad
    forward to get teacher logits, restores the live weights, and
    computes the (student-only) KL divergence. The KL is multiplied
    by `T²` (Hinton 2015 convention) so gradient magnitude is
    preserved when `T ≠ 1`. With identical student/teacher logits
    (the step-0 case) the KL is exactly 0 and the term contributes
    nothing to the loss.

    Args:
        student_logits: [B, T, V] from the live student forward (with grad).
        model:          the live model — used for the teacher no-grad forward.
        x:              the input batch (same as the student forward).
        teacher:        a `BornAgainTeacher` instance.
        T:              distillation temperature (> 0).
        alpha:          KL weight on top of CE.
        vocab_size:     V (used to flatten).
    """
    backup = teacher.apply_to(model)
    try:
        with torch.no_grad():
            teacher_logits = model(x)
    finally:
        teacher.restore_from(model, backup)
    # KL on flattened logits, in fp32 for stability under bf16 AMP.
    s = student_logits.view(-1, vocab_size).float() / T
    t = teacher_logits.view(-1, vocab_size).float() / T
    # F.kl_div(input=log_p_student, target=p_teacher) = KL(p_teacher ‖ p_student)
    log_p_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(t, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return alpha * kl


def setup_muon_optimizer(model: nn.Module, config: LLMConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    swan_params = []
    adamw_params = []
    soap_params = []
    lion_params = []
    tiger_params = []
    galore_params = []
    psgd_params = []

    # Lion (Chen et al. 2023) — see autoresearch/ideas/011-cautious-lion.
    # Replaces Muon on the 2-D non-embedding, non-norm routing slot when
    # `use_lion=True`. Lion's sign-update with a fixed LR has known
    # divergence risk on the embedding, so the routing keeps `token_embedding`
    # / `emb_proj` / `*.norm.weight` / 1-D scalars on AdamW — same 2-D / 1-D
    # split as Muon. Default off → Muon path is unchanged.
    use_lion = getattr(config, "use_lion", False)
    # Tiger (Chen et al. 2024, arXiv:2401.16691) — see
    # autoresearch/ideas/122-tiger. Replaces Muon on the same 2-D
    # non-embedding, non-norm routing slot when `use_tiger=True`.
    # Same 1-D / embedding / norm stay-on-AdamW split as Lion
    # (Tiger's magnitude-scaled sign update can be aggressive on
    # the embedding; the paper recommends AdamW for it). Default
    # off → Muon path is bit-identical.
    use_tiger = getattr(config, "use_tiger", False)
    use_swan = getattr(config, "use_swan", False)
    use_cautious_lion = getattr(config, "use_cautious_lion", False)
    # GaLore (Zhao et al. 2024) — see autoresearch/ideas/113-galore.
    # Replaces Muon on the 2-D non-embedding, non-norm routing slot when
    # `use_galore=True`. AdamW state lives in the rank-r×rank-r projected
    # space, refresh from SVD of grad EMA every `galore_proj_every` steps.
    # 1-D / embedding / norm stay on AdamW (same split as Muon). Default
    # off → Muon path is unchanged.
    use_galore = getattr(config, "use_galore", False)
    # 125 — PSGD: Preconditioned Stochastic Gradient Descent
    # (Li, Chen, Milenkovic, Giannakis 2024, arXiv:2405.13856,
    # NeurIPS 2024). Replaces Muon on the 2-D non-embedding, non-norm
    # routing slot when `use_psgd=True`. Maintains a coupled (P, Q)
    # whitening preconditioner for 2-D params and a diagonal D for
    # 1-D params; 1-D / embedding / norm stay on AdamW per the
    # paper's default. Identity at step 0: P=I, Q=I, m=0 ⇒ first
    # update is `I · g · I = g` ⇒ first step is `w ← w − lr · g`
    # (SGD, not AdamW). With `use_psgd=False` (default) the existing
    # Muon path is bit-identical. See
    # `autoresearch/ideas/125-psgd/idea.md`.
    use_psgd = getattr(config, "use_psgd", False)

    for name, param in model.named_parameters():
        is_muon_candidate = (
            param.ndim == 2
            and 'token_embedding' not in name
            and 'norm' not in name
            and param.requires_grad
        )
        # R1 lever — MuonFor1DNorm: route 1-D norm params (e.g. `norm.weight`)
        # to Muon instead of AdamW. Default off → step-0 identical to baseline.
        # See docs/research/optimizer_routing/plan.md.
        if (
            getattr(config, "muon_for_1d_norm", False)
            and param.ndim == 1
            and 'norm' in name
            and param.requires_grad
        ):
            is_muon_candidate = True
        # R2 lever — MuonForEmbed: route `token_embedding` (and `emb_proj`,
        # since they're related) to Muon instead of AdamW. The embedding is
        # ~91% of params at vocab=50k — does it want orthogonalized updates?
        # Default off → step-0 identical to baseline. See
        # docs/research/optimizer_routing/plan.md.
        if (
            getattr(config, "muon_for_embed", False)
            and ('token_embedding' in name or 'emb_proj' in name)
            and param.ndim == 2
            and param.requires_grad
        ):
            is_muon_candidate = True
        # R3 lever — MuonForOutput: route the attention output projection
        # (`out_proj` / `W_O`) to Muon instead of AdamW. The output projection
        # is 2-D but might be deliberately kept in AdamW. Default off →
        # step-0 identical to baseline. See docs/research/optimizer_routing/plan.md.
        if (
            getattr(config, "muon_for_output", False)
            and 'out_proj' in name
            and param.ndim == 2
            and param.requires_grad
        ):
            is_muon_candidate = True
        # R4 lever — SOAP: route 2-D non-Muon AdamW params to the SOAP
        # optimizer (Adam in Shampoo's eigenbasis). Only 2-D params
        # (`token_embedding`, `emb_proj`, `out_proj`) — 1-D scalars
        # and `*.norm.weight` stay on plain AdamW (eigendecomp is
        # meaningless on 1-D). Default off → step-0 identical to
        # baseline. See autoresearch/ideas/003-soap/plan.md.
        if (
            getattr(config, "use_soap", False)
            and param.ndim == 2
            and not is_muon_candidate
            and param.requires_grad
        ):
            soap_params.append(param)
            continue
        if is_muon_candidate:
            # Lion routing: when use_lion=True, the 2-D non-embedding,
            # non-norm slot goes to Lion (sign-based optimizer) instead
            # of Muon. The 1-D / embedding path stays on AdamW — Lion's
            # fixed-LR sign update is known to diverge on the embedding
            # (Chen et al. 2023 §5). Default off → bit-identical to the
            # Muon-AdamW path.
            if use_lion:
                lion_params.append(param)
            elif use_tiger:
                tiger_params.append(param)
            elif use_swan:
                swan_params.append(param)
            elif use_galore:
                # GaLore routing: when use_galore=True, the 2-D
                # non-embedding, non-norm slot goes to GaLoreAdamW
                # (low-rank projected AdamW) instead of Muon. 1-D /
                # embedding / norm stay on AdamW below. Default off →
                # Muon path is bit-identical to baseline.
                galore_params.append(param)
            elif use_psgd:
                # 125 — PSGD routing: when use_psgd=True, the 2-D
                # non-embedding, non-norm slot goes to PSGD
                # (coupled-whitening preconditioner) instead of
                # Muon. 1-D / embedding / norm stay on AdamW
                # below. Default off → Muon path is bit-identical
                # to baseline. See autoresearch/ideas/125-psgd/idea.md.
                psgd_params.append(param)
            else:
                muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  SWAN parameters: {sum(p.numel() for p in swan_params):,}")
    print(f"  Lion parameters: {sum(p.numel() for p in lion_params):,}")
    print(f"  Tiger parameters: {sum(p.numel() for p in tiger_params):,}")
    print(f"  GaLore parameters: {sum(p.numel() for p in galore_params):,}")
    print(f"  PSGD parameters: {sum(p.numel() for p in psgd_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")
    print(f"  SOAP parameters: {sum(p.numel() for p in soap_params):,}")

    if use_lion:
        # Lion replaces Muon on the 2-D non-embedding, non-norm slot
        # when use_lion=True. The 1-D / embedding / head path stays on
        # AdamW below. `use_cautious_lion` mirrors the cautious-Muon
        # flag — Liang et al. 2024 sign-mask on the sign-update with
        # `1 / mask.mean().clamp(min=0.1)` rescale. Default cautious=False
        # → bare Lion, bit-identical to Chen et al. 2023. See
        # autoresearch/ideas/011-cautious-lion/plan.md.
        lion_optimizer = Lion(
            lion_params,
            lr=getattr(config, "lion_lr", 3e-4),
            betas=(getattr(config, "lion_beta1", 0.9),
                   getattr(config, "lion_beta2", 0.98)),
            weight_decay=config.weight_decay,
            cautious=use_cautious_lion,
        )
        swan_optimizer = None
        galore_optimizer = None
        muon_optimizer = None
        tiger_optimizer = None
    elif use_tiger:
        # Tiger (Chen et al. 2024, arXiv:2401.16691) replaces Muon
        # on the 2-D non-embedding, non-norm slot when use_tiger=True.
        # The 1-D / embedding / head path stays on AdamW below —
        # Tiger's magnitude-scaled sign update can be aggressive on
        # the embedding (paper recommends AdamW for it). Cold-start
        # `m_0 = 0`, `v_0 = 0` ⇒ first step update = 0/ε = 0 ⇒ step-0
        # val_loss is bit-identical to baseline. `tiger_lr=1e-3` ≈
        # `adamw_lr / 6` (paper-recommended for tiny models; tune
        # in tandem if at all). See autoresearch/ideas/122-tiger/idea.md.
        tiger_optimizer = Tiger(
            tiger_params,
            lr=getattr(config, "tiger_lr", 1e-3),
            betas=(getattr(config, "tiger_beta1", 0.9),
                   getattr(config, "tiger_beta2", 0.999)),
            eps=getattr(config, "tiger_eps", 1e-8),
            weight_decay=config.weight_decay,
        )
        lion_optimizer = None
        swan_optimizer = None
        galore_optimizer = None
        muon_optimizer = None
    elif use_galore:
        # GaLore (Zhao et al. 2024) replaces Muon on the 2-D
        # non-embedding, non-norm slot when use_galore=True. AdamW
        # state lives in the rank-`galore_rank` × rank-`galore_rank`
        # projected space; P, Q are refreshed from SVD of the
        # gradient EMA every `galore_proj_every` steps. 1-D /
        # embedding / norm stay on AdamW below. The forward graph
        # is unchanged, so step-0 val_loss is bit-identical to
        # baseline. See autoresearch/ideas/113-galore/idea.md.
        galore_optimizer = GaLoreAdamW(
            galore_params,
            lr=getattr(config, "galore_lr", 0.006),
            betas=(getattr(config, "galore_beta1", 0.9),
                   getattr(config, "galore_beta2", 0.999)),
            eps=getattr(config, "galore_eps", 1e-8),
            weight_decay=config.weight_decay,
            rank=int(getattr(config, "galore_rank", 4)),
            proj_every=int(getattr(config, "galore_proj_every", 200)),
        )
        muon_optimizer = None
        tiger_optimizer = None
    elif use_psgd:
        # 125 — PSGD (Li et al. 2024, arXiv:2405.13856, NeurIPS
        # 2024) replaces Muon on the 2-D non-embedding, non-norm
        # slot when use_psgd=True. Maintains a coupled (P, Q)
        # whitening preconditioner per 2-D param; 1-D / embedding
        # / norm stay on AdamW below. Identity at step 0: P=I, Q=I,
        # m=0 ⇒ first update is `I · g · I = g` ⇒ first step is
        # `w ← w − lr · g` (SGD-with-momentum, not AdamW). At
        # `psgd_alpha=0` PSGD collapses to SGD-with-momentum.
        # With `use_psgd=False` (default) the existing Muon path
        # is bit-identical. See `optimizers/psgd.py` and
        # `autoresearch/ideas/125-psgd/idea.md`.
        psgd_optimizer = PSGD(
            psgd_params,
            lr=getattr(config, "psgd_lr", 0.01),
            alpha=getattr(config, "psgd_alpha", 1e-3),
            beta=getattr(config, "psgd_beta", 0.9),
            weight_decay=config.weight_decay,
        )
        lion_optimizer = None
        swan_optimizer = None
        galore_optimizer = None
        muon_optimizer = None
        tiger_optimizer = None
    elif use_swan:
        # SWAN replaces Muon on the same 2-D non-embedding, non-norm
        # slot. The routing keeps 1-D / embedding / norm on AdamW so the
        # delta isolates whitening vs Newton-Schulz orthogonalization.
        swan_optimizer = SWAN(
            swan_params,
            lr=config.muon_lr,
            weight_decay=config.weight_decay,
        )
        galore_optimizer = None
        muon_optimizer = None
        tiger_optimizer = None
    else:
        lion_optimizer = None
        swan_optimizer = None
        galore_optimizer = None
        tiger_optimizer = None
        psgd_optimizer = None
        muon_optimizer = Muon(
            muon_params,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            ns_steps=getattr(config, "muon_ns_steps", 5),
            orthogonalize=getattr(config, "muon_orthogonalize", True),
            coeffs_mode=getattr(config, "muon_coeffs_mode", "polar_express"),
            shape_scale=getattr(config, "muon_shape_scale", True),
            # #15 Moonlight Muon RMS rescale: when use_moonlight_muon=True,
            # override the default `shape_aspect` scale with the paper's
            # `c·sqrt(max(d_in, d_out))` formula. The Muon optimizer's
            # `moonlight_c` defaults to 0.2 (the paper's tuned constant)
            # and `c` can be overridden via `LLMConfig.moonlight_muon_c`.
            # Default off → `scale_mode="shape_aspect"` bit-identical
            # baseline. See autoresearch/ideas/015-moonlight-muon-rms/plan.md.
            scale_mode=(
                "moonlight"
                if getattr(config, "use_moonlight_muon", False)
                else getattr(config, "muon_scale_mode", "shape_aspect")
            ),
            adamw_lr=getattr(config, "adamw_lr", 0.006),
            nesterov=getattr(config, "muon_nesterov", True),
            lazy_ortho_steps=getattr(config, "muon_lazy_ortho_steps", 1),
            cautious=getattr(config, "use_cautious_muon", False),
            moonlight_c=getattr(config, "moonlight_muon_c", 0.2),
        )
    device = resolve_device(getattr(config, "device", "auto"))
    # Cautious-AdamW gate (Liang et al. 2024) — see
    # autoresearch/ideas/002-cautious-adamw/plan.md. "none" = baseline
    # `torch.optim.AdamW` (bit-identical to today); other values select
    # which AdamW bucket(s) the sign-mask fires on.
    _cautious_mode = getattr(config, "use_cautious_adamw", "none")
    # Schedule-Free AdamW (Defazio et al. 2024) — see
    # autoresearch/ideas/006-schedule-free-adamw/plan.md. Drop-in
    # replacement for the AdamW path only; Muon path is unchanged.
    # `use_schedule_free_adamw=True` overrides `use_cautious_adamw`
    # (SF has no sign-mask variant yet; the cautious mask and SF
    # averaging are independent levers and can be co-tested in a
    # future PR). The LR schedule must be flat — see comment in
    # `train_minimal_llm` below.
    use_sf = getattr(config, "use_schedule_free_adamw", False)
    # 119 — SAM: Sharpness-Aware Minimization (Foret et al. 2020,
    # arXiv:2010.01412). Replaces the AdamW path with `AdamSAM`
    # when `use_sam=True`. SAM does an adversarial ascent step
    # `w ← w + ρ · ∇L(w) / ‖∇L(w)‖` then applies AdamW to the
    # perturbed-point gradient. The trainer handles the
    # first_step/closure/second_step dance; see the optimizer
    # step block in `train_model` below. With `use_sam=False`
    # (default) plain `torch.optim.AdamW` is used — bit-identical
    # to the baseline. With `use_sam=True, sam_rho=0.0` the
    # first_step is a no-op and the SAM path collapses to plain
    # AdamW (per-param perturbation is zero). See
    # `optimizers/sam.py` and `autoresearch/ideas/119-sam/idea.md`.
    use_sam = getattr(config, "use_sam", False)
    sam_rho = float(getattr(config, "sam_rho", 0.05))
    # 121 — Prodigy: parameter-free AdamW (Mishchenko & Defazio
    # 2023, arXiv:2306.06101). Replaces the AdamW path with
    # `Prodigy` when `use_prodigy=True`. Maintains a group-level
    # step-size estimate `D_t` updated each step from a continuous
    # Adam-style gradient similarity, with a 10-step
    # displacement-based warm-start. With `use_prodigy=False`
    # (default) plain `torch.optim.AdamW` is used — bit-identical
    # to the baseline. See `optimizers/prodigy.py` and
    # `autoresearch/ideas/121-prodigy/idea.md`.
    use_prodigy = getattr(config, "use_prodigy", False)
    # 138 — LookSAM: periodic SAM (Du et al. 2022, ICLR 2023,
    # arXiv:2205.13539). Compute-efficient variant of SAM (119):
    # the SAM-style 2-backward ascent-descent step fires only
    # every K steps; the K-1 steps in between are plain AdamW.
    # Mutex with `use_sam`: if both are on, `use_sam` wins (full
    # SAM is the more aggressive variant). With `use_looksam=
    # False` (default) the trainer uses plain `AdamW` unchanged —
    # the LookSAM class is never instantiated, baseline path
    # bit-identical. Identity at step 0: with K=5 the first 4
    # steps are plain AdamW (`step_count=0..3`); the first SAM
    # ascent fires at `step_count=4`. So LookSAM is *more*
    # bit-identical at step 0 than full SAM (119), which always
    # runs the ascent on the first step. See `optimizers/
    # looksam.py` and `autoresearch/ideas/138-looksam/idea.md`.
    use_looksam = getattr(config, "use_looksam", False)
    looksam_k = int(getattr(config, "looksam_k", 5))
    looksam_rho = float(getattr(config, "looksam_rho", 0.05))
    if use_sam:
        # 119 — SAM replaces the AdamW path. The trainer handles
        # the ascent/closure/descent flow (see the optimizer step
        # block in `train_model`). The optimizer itself only
        # exposes `first_step` / `second_step` for the trainer to
        # call. `sam_rho` controls the perturbation radius; paper
        # default 0.05 for Adam-SAM.
        adamw_optimizer = AdamSAM(
            adamw_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            rho=sam_rho,
        )
    elif use_looksam:
        # 138 — LookSAM replaces the AdamW path with periodic
        # SAM. The trainer routes LookSAM into either the SAM
        # group (when `next_is_sam` is True) or the non-SAM
        # group (when False) on each step — see the optimizer
        # step block in `train_model` below. On non-SAM steps
        # the trainer calls `LookSAM.step()` which is the
        # baseline AdamW step on the w-grad (no ascent, no
        # closure). On SAM steps it calls `first_step` →
        # closure → `second_step` exactly like AdamSAM.
        # `looksam_k` is the period between SAM steps (paper
        # default 5 → ~1.2x compute vs. SAM's 2x); `looksam_rho`
        # is the perturbation radius (paper default 0.05).
        adamw_optimizer = LookSAM(
            adamw_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            rho=looksam_rho,
            k=looksam_k,
        )
    elif use_sf:
        # SF replaces the AdamW path entirely. It carries its own
        # internal averaging schedule (`c = 1/(k-warmup+1)` after
        # warmup), so the external LR schedule must be constant.
        adamw_optimizer = ScheduleFreeAdamW(
            adamw_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            warmup_steps=0,
        )
    elif use_prodigy:
        # 121 — Prodigy parameter-free AdamW (Mishchenko &
        # Defazio 2023, arXiv:2306.06101). Replaces the AdamW
        # path entirely. Maintains a group-level D estimate
        # updated from a continuous Adam-style gradient
        # similarity (smooth ramp-up) plus a 10-step
        # displacement-based warm-start. `prodigy_d0=0.01`
        # (re-code 2026-06-13; was 1.0) is the warm-start
        # scalar; the production LR is `D_t` which Prodigy
        # discovers. `prodigy_d_max=1.0` (paper §3.1) bounds
        # the discovery loop into a stable band at tiny1m3m.
        # `prodigy_update_clip=1.0` is the per-param max-norm
        # safety net on `delta = eff_lr · adam_update`. The
        # 2-D Muon path is unchanged. With `use_prodigy=False`
        # (default) the trainer uses plain `torch.optim.AdamW`
        # — baseline bit-identical.
        from optimizers.prodigy import Prodigy
        adamw_optimizer = Prodigy(
            adamw_params,
            lr=1.0,  # unit conversion (paper default)
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            d0=float(getattr(config, "prodigy_d0", 0.01)),
            warmup_steps=int(getattr(config, "prodigy_warmup_steps", 10)),
            beta3=float(getattr(config, "prodigy_beta3", 0.01)),
            d_max=float(getattr(config, "prodigy_d_max", 1.0)),
            min_d=float(getattr(config, "prodigy_min_d", 1e-6)),
            update_clip=float(getattr(config, "prodigy_update_clip", 1.0)),
        )
    elif getattr(config, "use_mars", False):
        # 114 — MARS Variance-Reduced AdamW (Yuan et al. 2024,
        # arXiv:2401.03855). Subclass of `torch.optim.AdamW` that
        # adds a lag-based correction to the *gradient* input. The
        # per-parameter `v` is untouched; only the gradient input
        # is modified. Ring buffer of past `exp_avg` snapshots of
        # length `2*lag` is maintained per param. Identity at step
        # 0: the buffer is empty for the first `2*lag` steps ⇒
        # correction undefined ⇒ g̃_t = g_t ⇒ bit-identical to
        # plain AdamW. Default off → plain `torch.optim.AdamW`
        # path is used, baseline bit-identical. See
        # `autoresearch/ideas/114-mars/idea.md`.
        adamw_optimizer = MARSAdamW(
            adamw_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            lag=int(getattr(config, "mars_lag", 10)),
            mix_coef=float(getattr(config, "mars_mix_coef", 0.5)),
            lr_scale=float(getattr(config, "mars_lr_scale", 1.0)),
        )
    elif getattr(config, "use_dadapt", False):
        # 120 — D-Adaptation: Automatic LR Discovery (Defazio 2023,
        # arXiv:2301.11933 / arXiv:2201.11941, ICML 2023). Thin
        # subclass of `torch.optim.AdamW` that maintains a per-group
        # scalar `D` (log-LR lower bound) and derives the effective
        # LR as `lr_t = D_t / ‖g_t‖`. The 1st/2nd moments of AdamW
        # are retained intact — only the outer LR scaling is
        # replaced. The 2-D Muon path is unchanged (D-Adapt is
        # ortho to Muon, lives only on the AdamW bucket). At step 0
        # `D = 1e-6` warm-start ⇒ `lr_0 ≈ 1e-6 / ‖g_0‖` (essentially
        # zero); after ~10–20 steps `D` reaches a typical AdamW-
        # equivalent value. With `use_dadapt=False` (default) plain
        # `torch.optim.AdamW` is used — baseline bit-identical. With
        # `D` frozen at its initial value the lever collapses to
        # AdamW. **Numerical-stability guards (2026-06-13 re-code):**
        # `D` is clamped to `[min_lr, d_max]` (paper §3.1) to prevent
        # the unbounded `D ← D · exp(η·(c_+−c_-))` growth that caused
        # the previous GPU run to blow up (val 10.81 → 36.89 at step
        # 50 → 7.04e15 final). The derived `lr_t` is also clamped to
        # `d_max`, the gradient norm is floored at `1e-12`, and a
        # NaN/Inf guard on the gradient / momentum falls back to the
        # base lr without poisoning `D`. See `optimizers/dadaptation.py`
        # and `autoresearch/ideas/120-dadaptation/idea.md`.
        adamw_optimizer = DAdaptAdamW(
            adamw_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            d0_lr=float(getattr(config, "dadapt_d0_lr", 1.0)),
            d_init=1e-6,
            min_lr=float(getattr(config, "dadapt_min_lr", 0.0)),
            d_max=float(getattr(config, "dadapt_d_max", 1.0)),
        )
    elif getattr(config, "use_came", False):
        # 123 — CAME: Confidence-guided Adaptive Memory Efficient
        # Optimization (Luo et al. 2023, arXiv:2307.02085, NeurIPS
        # 2023). Replaces the AdamW 1-D / embedding / norm / head
        # path with `CAME`. The 2-D Muon path is unchanged (CAME
        # is an AdamW replacement, not a Muon replacement, like
        # 119-SAM, 120-DAdapt, 121-Prodigy, 114-MARS). The update
        # adds a confidence rescaling `conf_t = max(res_t, 0) + ε`
        # where `res_t = (m_t − g_t) / (√v_t + ε)`, so when the
        # gradient agrees with the running momentum the update is
        # tiny and when they disagree the update is residual-shaped.
        # Cold-start `m_0 = 0`, `v_0 = 0` ⇒ first-step residual is
        # negative, clipped to 0, `conf_0 = ε`, update ≈ 0 ⇒
        # baseline path byte-identical at step 0. With
        # `use_came=False` (default) plain `torch.optim.AdamW` is
        # used — baseline bit-identical. See `optimizers/came.py`
        # and `autoresearch/ideas/123-came/idea.md`.
        adamw_optimizer = CAME(
            adamw_params,
            lr=getattr(config, "came_lr", 0.006),
            betas=(getattr(config, "came_beta1", 0.9),
                   getattr(config, "came_beta2", 0.999)),
            eps=getattr(config, "came_eps", 1e-8),
            weight_decay=config.weight_decay,
            update_clip=getattr(config, "came_update_clip", 10.0),
        )
    elif getattr(config, "use_radam", False):
        # 124 — RAdam: Rectified Adam (Liu et al. 2019,
        # arXiv:1908.03265, ICLR 2020). Replaces the AdamW 1-D /
        # embedding / norm / head path with `RAdam`. The 2-D Muon
        # path is unchanged (RAdam is an AdamW replacement, like
        # 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME).
        # The update applies a variance-bounded correction `ρ_t` to
        # Adam's bias-corrected step: when the variance of
        # `1/(1−β2^t)` is high (early steps), RAdam falls back to
        # an SGD-only `m̂_t` step (no `v̂_t`); once `ρ_t > 4`
        # (≈ `t > 4/(1−β2)`) it switches to the full Adam-normalized
        # update with the variance-aware `√ρ_t` rescale. This
        # *removes the manual warmup knob* — RAdam auto-detects
        # when the effective LR is safe. At step 0 (t=1) `ρ_1 ≪ 4`
        # ⇒ SGD-fallback path ⇒ `update = (1−β1)·g_0`. NOT
        # bit-identical to AdamW's first step (which uses the full
        # Adam-normalized update), but the magnitude is comparable
        # (O(β1) smaller). This first-step divergence is the lever,
        # not a bug. With `use_radam=False` (default) plain
        # `torch.optim.AdamW` is used — baseline bit-identical.
        # `radam_lr=0.006` matches `adamw_lr` (paper does not
        # require re-tuning). See `optimizers/radam.py` and
        # `autoresearch/ideas/124-radam/idea.md`.
        adamw_optimizer = RAdam(
            adamw_params,
            lr=getattr(config, "radam_lr", 0.006),
            betas=(getattr(config, "radam_beta1", 0.9),
                   getattr(config, "radam_beta2", 0.999)),
            eps=getattr(config, "radam_eps", 1e-8),
            weight_decay=config.weight_decay,
        )
    elif getattr(config, "use_adashift", False):
        # 126 — AdaShift: Decorrelated Adam via Delayed Gradients
        # (Zhou et al. 2019, arXiv:1810.00143, NeurIPS 2019
        # workshop). Replaces the AdamW 1-D / embedding / norm /
        # head path with `AdaShift`. The 2-D Muon path is unchanged
        # (AdaShift is an AdamW replacement, like 114-MARS,
        # 119-SAM, 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam).
        # The update uses a delayed gradient g_{t-n}² for the
        # 2nd moment, decorrelating v_t from m_t:
        #     m_t = β1·m_{t-1} + (1-β1)·g_t
        #     v_t = β2·v_{t-1} + (1-β2)·g_{t-n}²
        #     update = m̂_t / (√v̂_t + ε)
        # Per-parameter state keeps a queue of past `n` gradients
        # (fp32 clones, length bounded by n). The paper's
        # warm-start `v_0 = g_0²` makes `v_1 = β2·g_0²` — NOT
        # bit-identical to AdamW's first step (`v_1 = (1-β2)·g_0²`)
        # but same magnitude order (O(β2) different). The
        # first-step displacement is the lever, not a bug. With
        # `use_adashift=False` (default) plain `torch.optim.AdamW`
        # is used — baseline path bit-identical. `adashift_lr=0.006`
        # matches `adamw_lr` (paper does not require re-tuning).
        # See `optimizers/adashift.py` and
        # `autoresearch/ideas/126-adashift/idea.md`.
        adamw_optimizer = AdaShift(
            adamw_params,
            lr=getattr(config, "adashift_lr", 0.006),
            betas=(getattr(config, "adashift_beta1", 0.9),
                   getattr(config, "adashift_beta2", 0.999)),
            eps=getattr(config, "adashift_eps", 1e-8),
            weight_decay=config.weight_decay,
            n=int(getattr(config, "adashift_n", 3)),
        )
    elif getattr(config, "use_adan", False):
        # 135 — Adan: Adaptive Nesterov Momentum with N-Step Lookback
        # (Xie et al. 2022, arXiv:2208.06677, TPAMI 2022 / ICLR 2023
        # workshop). Replaces the AdamW 1-D / embedding / norm /
        # head path with `Adan`. The 2-D Muon path is unchanged
        # (Adan is an AdamW replacement, like 114-MARS, 119-SAM,
        # 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam, 126-AdaShift,
        # 127-GC, 128-SD). The update is (paper Algorithm 1):
        #     g_la = g_t + β_la · (g_t − g_{t-1})      (Nesterov)
        #     m_t  = β1·m + (1−β1)·g_la
        #     v_t  = β2·v + (1−β2)·mean(g²_{t..t-N+1})  (N-step)
        #     update = m_t / (√v_t + ε)                (no bias-correction)
        # `adan_n_lookback=4` is the paper's default N.
        # `adan_lookahead_beta=0.5` is the paper's default Nesterov
        # coefficient. The first optimizer step has `update_0 ≈
        # sign(g_0)` (no bias correction, queue length 1) — NOT
        # bit-identical to AdamW's first step (which uses bias-
        # corrected `m̂/√v̂`), but the magnitude is similar and the
        # N=4 lookback ramps in over the first 4 steps. The forward
        # graph is unchanged, so step-0 `val_loss` (no optimizer
        # step yet) is bit-identical to baseline. The first optimizer
        # step itself differs from AdamW's first step by an O(1/N)
        # correction in the `v_t` estimate — this is the lever's
        # signature, not a bug. With `use_adan=False` (default)
        # plain `torch.optim.AdamW` is used — baseline path
        # bit-identical. See `optimizers/adan.py` and
        # `autoresearch/ideas/135-adan/idea.md`.
        from optimizers.adan import Adan
        adamw_optimizer = Adan(
            adamw_params,
            lr=getattr(config, "adan_lr", 0.006),
            betas=(getattr(config, "adan_beta1", 0.9),
                   getattr(config, "adan_beta2", 0.999)),
            eps=getattr(config, "adan_eps", 1e-8),
            weight_decay=config.weight_decay,
            lookahead_beta=float(getattr(config, "adan_lookahead_beta", 0.5)),
            n_lookback=int(getattr(config, "adan_n_lookback", 4)),
        )
    elif getattr(config, "use_sophia", False):
        # 140 — Sophia: Scalable Stochastic Second-order Optimizer
        # (Liu, Wang, et al. 2023, arXiv:2305.14342, ICML 2023).
        # Replaces the AdamW 1-D / embedding / norm / head path with
        # `Sophia`. The 2-D Muon path is unchanged (Sophia is an
        # AdamW replacement, like 114-MARS, 119-SAM, 121-Prodigy,
        # 135-Adan). The update is the diagonal-Hessian-aware
        # preconditioned step
        #     m_t = β1·m + (1−β1)·g_t
        #     h_t = β2·h + (1−β2)·h_hat_t     (h_hat every k=10)
        #     update = clip(g, ±ρ) / max(h, ε)
        #     θ   ← θ − lr·(update + λ·θ)     (decoupled WD)
        # The diagonal Hessian `h_hat` is sampled via Hutchinson's
        # trace estimator: `u ~ Rademacher(±1)` per parameter, then
        # `h_hat = u · ∇(g·u)` (one extra backward on the scalar
        # `g·u`, amortized ~1.1× backward cost at k=10 and 92
        # update steps). The trainer handles the extra backward +
        # h_hat feed-in; see the Hutchinson block in
        # `train_model` below. Defaults match the paper's 125M
        # model: lr=6e-3, β1=0.965, β2=0.99, ρ=0.04, k=10. The
        # per-param `update_clip=1.0` safety guard bounds the
        # cold-start `h_t≈0` amplification to a single AdamW-
        # magnitude step. Cold-start `m_0 = h_0 = 0` ⇒ first-step
        # update magnitude bounded by `lr · 1.0`; first Hutchinson
        # sample at step `k−1` and `h_t` becomes `O(g²)` thereafter.
        # NOT bit-identical to AdamW's first step (the diagonal
        # preconditioner IS the lever), but the first-step
        # magnitude matches AdamW by construction. With
        # `use_sophia=False` (default) plain `torch.optim.AdamW`
        # is used — baseline path bit-identical. The Hutchinson
        # extra backward is also gated by `use_sophia=True` so
        # the baseline training cost is unchanged when the flag
        # is off. See `optimizers/sophia.py` and
        # `autoresearch/ideas/140-sophia/idea.md`.
        from optimizers.sophia import Sophia
        adamw_optimizer = Sophia(
            adamw_params,
            lr=getattr(config, "sophia_lr", 6e-3),
            betas=(getattr(config, "sophia_beta1", 0.965),
                   getattr(config, "sophia_beta2", 0.99)),
            eps=getattr(config, "sophia_eps", 1e-8),
            weight_decay=config.weight_decay,
            rho=float(getattr(config, "sophia_rho", 0.04)),
            hessian_update_freq=int(getattr(config, "sophia_hessian_freq", 10)),
            update_clip=float(getattr(config, "sophia_update_clip", 1.0)),
        )
    elif getattr(config, "use_adapnm", False):
        # 136 — AdaPNM: Adaptive Positive-Negative Momentum
        # (Ding, Zhou, Zhu, Ye, Jiao 2019, arXiv:1906.01520,
        # NeurIPS 2019). Replaces the AdamW 1-D / embedding / norm /
        # head path with `AdaPNM`. The 2-D Muon path is unchanged
        # (AdaPNM is an AdamW replacement, like 114-MARS, 119-SAM,
        # 120-DAdapt, 121-Prodigy, 123-CAME, 124-RAdam, 126-AdaShift,
        # 135-Adan, 127-GC, 128-SD). The mechanism maintains TWO
        # parallel momentum buffers — `m+` for the positive part of
        # the gradient `max(g, 0)` and `m-` for the negative part
        # `max(-g, 0)` — and combines them as `m = m+ − m-`. The
        # combined direction is algebraically equal to AdamW's EMA
        # (`m+ − m- = g` element-wise at every step), so today's
        # AdaPNM degenerates to a no-bias-correction AdamW; the
        # factored-state representation is what makes the lever a
        # "structured disagreement with itself" option for future
        # per-side processing. Cold-start `m+_0 = m-_0 = v_0 = 0`
        # ⇒ first-step update = `(1−β1)·g_0 / (√((1−β2)·g_0²) + ε)`
        # (NOT bit-identical to AdamW's bias-corrected first step,
        # but within `O(β1)` factor — the lever's signature, not a
        # bug). With `use_adapnm=False` (default) plain
        # `torch.optim.AdamW` is used — baseline path bit-identical.
        # See `optimizers/adapnm.py` and
        # `autoresearch/ideas/136-adapnm/idea.md`.
        adamw_optimizer = AdaPNM(
            adamw_params,
            lr=getattr(config, "adapnm_lr", 0.006),
            betas=(getattr(config, "adapnm_beta1", 0.9),
                   getattr(config, "adapnm_beta2", 0.999)),
            eps=getattr(config, "adapnm_eps", 1e-8),
            weight_decay=config.weight_decay,
        )
    elif getattr(config, "use_adabelief", False):
        # 141 — AdaBelief: Adapting Stepsizes by the Belief in
        # Observed Gradients (Zhuang, Liu, Tran, Hoang, Chang, et
        # al. 2020, arXiv:2010.07468, NeurIPS 2020). Replaces the
        # AdamW 1-D / embedding / norm / head path with
        # `AdaBelief`. The 2-D Muon path is unchanged (AdaBelief
        # is an AdamW replacement, like 114-MARS, 119-SAM, 120-
        # DAdapt, 121-Prodigy, 123-CAME, 124-RAdam, 126-AdaShift,
        # 127-GC, 128-SD, 135-Adan, 136-AdaPNM, 137-AdamP). The
        # mechanism replaces AdamW's 2nd moment `v_t = E[g²]` with
        # the *residual* variance `s_t = E[(g_t − m_t)²] + ε` —
        # large step when the current gradient agrees with the
        # momentum (we trust the direction), small step when they
        # disagree. AdamW does the *opposite* (large `g²` shrinks
        # the step). At step 0 `m_0 = 0`, `s_0 = ε`; first-step
        # residual is `g_0 − (1−β1)·g_0 = β1·g_0` ⇒ `s_1 = (1−β2)
        # ·β1²·g_0² + ε` ⇒ `update_0 ≈ 3.5·sign(g_0)`. NOT
        # bit-identical to AdamW's first step (which gives
        # `sign(g_0)`), but the magnitude is the same order — the
        # lever's signature, not a bug. With `use_adabelief=False`
        # (default) plain `torch.optim.AdamW` is used — baseline
        # bit-identical. `adabelief_lr=0.006` matches `adamw_lr`
        # (paper does not require re-tuning). See
        # `optimizers/adabelief.py` and
        # `autoresearch/ideas/141-adabelief/idea.md`.
        adamw_optimizer = AdaBelief(
            adamw_params,
            lr=getattr(config, "adabelief_lr", 0.006),
            betas=(getattr(config, "adabelief_beta1", 0.9),
                   getattr(config, "adabelief_beta2", 0.999)),
            eps=getattr(config, "adabelief_eps", 1e-8),
            weight_decay=config.weight_decay,
        )
    elif getattr(config, "use_adamp", False):
        # 137 — AdamP: Adam with Projection-Based Update
        # (He, Liu, Mao, Chen, Zhang 2020, arXiv:2006.08217, NeurIPS
        # 2020). Replaces the AdamW 1-D / embedding / norm / head
        # path with `AdamP`. The 2-D Muon path is unchanged (AdamP
        # is an AdamW replacement, like 114-MARS, 119-SAM, 120-
        # DAdapt, 121-Prodigy, 123-CAME, 124-RAdam, 126-AdaShift,
        # 127-GC, 128-SD, 135-Adan, 136-AdaPNM). The mechanism
        # projects the Adam update `Δ = m̂/√v̂` onto the orthogonal
        # complement of `w` (removes the component of Δ along w,
        # leaving only the perpendicular component) so the update
        # rotates direction without changing magnitude. The L2 reg
        # is applied as the paper's `λ · ‖w‖ · ŵ` (pure magnitude
        # shrinkage, no rotation). Identity at step 0: for symmetric
        # inits the projection removes an `O(1/√d)` component of Δ_0,
        # so the first AdamP step ≈ the first AdamW step modulo
        # that small correction. With `adamp_lambda=0.0` the
        # projection is fully inert and `AdamP` collapses to plain
        # AdamW — bit-identical baseline. With `use_adamp=False`
        # (default) plain `torch.optim.AdamW` is used — baseline
        # path bit-identical. See `optimizers/adamp.py` and
        # `autoresearch/ideas/137-adamp/idea.md`.
        adamw_optimizer = AdamP(
            adamw_params,
            lr=getattr(config, "adamp_lr", 0.006),
            betas=(getattr(config, "adamp_beta1", 0.9),
                   getattr(config, "adamp_beta2", 0.999)),
            eps=getattr(config, "adamp_eps", 1e-8),
            weight_decay=config.weight_decay,
            adamp_lambda=float(getattr(config, "adamp_lambda", 1.0)),
        )
    elif getattr(config, "use_sd", False):
        # 128 — Spectral Decoupling (Yong, Pehlivan, Morariu,
        # Tsang 2022, arXiv:2202.05380, NeurIPS 2022). Replaces
        # the AdamW 1-D / embedding / norm / head path with
        # `SDAdamW` — a thin subclass of `torch.optim.AdamW` that
        # projects each per-param gradient perpendicular to the
        # weight direction (`g ← g − (⟨g,w⟩/‖w‖²)·w`) before
        # delegating to AdamW's `.step()`. Decoupled WD `λ·w` is
        # unchanged (it acts along w — magnitude-shrinking role
        # is preserved). The 2-D Muon path is unchanged (SD is an
        # AdamW replacement, like 119-SAM, 120-DAdapt, 121-
        # Prodigy, 114-MARS, 123-CAME, 124-RAdam, 126-AdaShift,
        # 127-GC). Identity at step 0: with symmetric inits
        # `⟨g_0, w_0⟩ ≈ 0` so the projection removes an `O(1/n)`
        # component of `g_0`. With `sd_lambda=0.0` the projection
        # is fully inert and `SDAdamW` collapses to plain
        # AdamW — bit-identical baseline. With `use_sd=False`
        # (default) plain `torch.optim.AdamW` is used — baseline
        # bit-identical. See `optimizers/spectral_decoupling.py`
        # and `autoresearch/ideas/128-spectral-decoupling/idea.md`.
        adamw_optimizer = SDAdamW(
            adamw_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            sd_lambda=float(getattr(config, "sd_lambda", 1.0)),
        )
    elif _cautious_mode != "none":
        if _cautious_mode == "embedding":
            _mask_buckets = ("token_embedding", "emb_proj")
        elif _cautious_mode == "gain":
            _mask_buckets = ("norm.weight",)
        elif _cautious_mode == "all":
            _mask_buckets = ()
            _mask_all = True
        else:
            raise ValueError(f"Unknown use_cautious_adamw: {_cautious_mode!r}")
        if _cautious_mode != "all":
            _mask_all = False
        adamw_optimizer = CautiousAdamW(
            adamw_params,
            mask_buckets=_mask_buckets,
            mask_all=_mask_all,
            lr=config.adamw_lr,
            weight_decay=config.weight_decay,
            fused=device.type == "cuda",
        )
    else:
        # 127 — Gradient Centralization (Yong et al. 2020, arXiv:2004.01461).
        # When `use_gc=True`, route the AdamW path through `GCAdamW` —
        # a thin subclass of `torch.optim.AdamW` that mean-centers
        # each gradient along `gc_axis` before the AdamW update. The
        # per-parameter `(m, v)` state is untouched. Compositional
        # with the plain AdamW baseline only — does NOT compose with
        # other AdamW replacements (SAM, SF, DAdapt, Prodigy, MARS,
        # CAME, RAdam, AdaShift have their own `.step()` contracts;
        # stacking GC on top would need its own wiring). With
        # `use_gc=False` (default) plain `torch.optim.AdamW` is used —
        # baseline path bit-identical. See
        # `optimizers/grad_centralization.py` and
        # `autoresearch/ideas/127-grad-centralization/idea.md`.
        if getattr(config, "use_gc", False):
            from optimizers.grad_centralization import GCAdamW
            adamw_optimizer = GCAdamW(
                adamw_params,
                lr=config.adamw_lr,
                weight_decay=config.weight_decay,
                gc_axis=int(getattr(config, "gc_axis", 1)),
            )
        else:
            adamw_optimizer = torch.optim.AdamW(
                adamw_params,
                lr=config.adamw_lr,
                weight_decay=config.weight_decay,
                fused=device.type == "cuda",
            )

    optimizers = ([lion_optimizer, adamw_optimizer] if use_lion
                  else [tiger_optimizer, adamw_optimizer] if use_tiger
                  else [galore_optimizer, adamw_optimizer] if use_galore
                  else [psgd_optimizer, adamw_optimizer] if use_psgd
                  else [swan_optimizer, adamw_optimizer] if use_swan
                  else [muon_optimizer, adamw_optimizer])
    # SOAP gate (Vyas et al. 2024) — see
    # autoresearch/ideas/003-soap/plan.md. `use_soap=True` swaps the
    # eligible 2-D params (`token_embedding`, `emb_proj`, `out_proj`)
    # to a SOAP optimizer; 1-D params stay on the existing AdamW.
    # `use_soap=False` (default) → `soap_params=[]` → no SOAP
    # optimizer instantiated, bit-identical to baseline.
    if getattr(config, "use_soap", False) and soap_params:
        soap_optimizer = SOAP(
            soap_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            precondition_frequency=config.use_soap_precondition_freq,
            precondition_eps=1e-6,
        )
        optimizers.append(soap_optimizer)

    return optimizers


def train_model(
    model: nn.Module,
    config: LLMConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizers: List[torch.optim.Optimizer],
    schedulers: Optional[List] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    early_stopper: Optional[EarlyStopping] = None,
    output_dir: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    log_every: int = 100,
    lookahead: Optional["Lookahead"] = None,
    ema: Optional["ModelEMA"] = None,
    born_again: Optional["BornAgainTeacher"] = None,
) -> Any:
    """
    Generic training function that can be used by experiments.
    
    Args:
        model: Model to train
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizers: List of optimizers
        schedulers: Optional list of learning rate schedulers
        early_stopper: Optional early stopping handler
        output_dir: Optional directory to save outputs
        extra_config: Optional dict of extra config to save with metrics
    
    Returns:
        model, final_metrics, metrics_history
    """
    device = resolve_device(getattr(config, "device", "auto"))
    model = model.to(device, dtype=torch.bfloat16) if device.type == "cuda" else model.to(device)
    
    if schedulers is None:
        schedulers = []

    current_loss_val = float(checkpoint_state.get("metrics", {}).get("train_loss", 0.0)) if checkpoint_state else 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_start_time = time.time()
    metrics_history = default_metrics_history()
    if checkpoint_state and checkpoint_state.get("metrics_history"):
        loaded_history = checkpoint_state["metrics_history"]
        metrics_history = {
            "steps": list(loaded_history.get("steps", [])),
            "val_losses": list(loaded_history.get("val_losses", [])),
            "val_accuracies": list(loaded_history.get("val_accuracies", [])),
            "val_perplexities": list(loaded_history.get("val_perplexities", [])),
            "elapsed_times": list(loaded_history.get("elapsed_times", [])),
            "learning_rates": list(loaded_history.get("learning_rates", [])),
        }

    # Training loop
    model.train()
    step = int(checkpoint_state.get("step", 0)) if checkpoint_state else 0
    tokens_seen = int(checkpoint_state.get("tokens_seen", 0)) if checkpoint_state else 0
    desc = "Training"
    pbar = tqdm(total=config.train_tokens, desc=desc, unit="tokens", initial=min(tokens_seen, config.train_tokens))
    
    stopped_early = False
    gated = False  # set True when we pause at a pre-registered gate step for human review
    gate_step = getattr(config, "stop_at_step", None)
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    exact_step_baseline = load_exact_step_baseline(config)

    def write_live_metrics(current_metrics: Dict[str, Any], current_step: int, current_tokens: int) -> None:
        if not output_path:
            return
        live_metrics = {
            'final_metrics': current_metrics,
            'total_time_minutes': (time.time() - train_start_time) / 60,
            'stopped_early': stopped_early,
            'actual_steps': current_step,
            'tokens_seen': current_tokens,
            'history': metrics_history,
            # #65 self-identifying run (forward-only; live writes
            # also stamp identity so the latest .json always answers
            # "which run am I?" even mid-training)
            'run_name': output_path.name,
            'config_name': config.__class__.__name__,
            'seed': getattr(config, 'seed', None),
            'flags': config.active_flags(),
            **capture_git_metadata(),
        }
        if extra_config:
            live_metrics['experiment_config'] = extra_config
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(live_metrics, f, indent=2)

    while tokens_seen < config.train_tokens:
        for batch_idx, batch in enumerate(train_loader):
            if tokens_seen >= config.train_tokens:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch.get("attention_mask")
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, attention_mask, y = batch
                elif len(batch) == 2:
                    x, y = batch
                    attention_mask = None
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            x, y = x.to(device), y.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Count tokens in this batch (approx: batch_size * seq_len)
            batch_tokens = x.numel()

            # A10 EntropyReg aux loss: collect stashed terms from every
            # MultiHeadAttention that computed one this forward. Sum across
            # layers; treat λ as a per-layer coefficient (tune for depth).
            # Default (no flag set): no module sets _entropy_reg_loss, so
            # this stays 0 and CE behavior is unchanged.
            def _collect_entropy_reg(m):
                total = next(m.parameters()).new_zeros(())
                # 158 — GAU: when `use_gau=True`, `m.transformer_blocks`
                # is None (the GAU block replaces the standard stack
                # entirely; see `models/llm.py:752`). Skip the iteration
                # rather than letting `for ... in None` raise
                # TypeError. GAU does not stash `_entropy_reg_loss` on
                # a sub-module (no MHA sub-block), so the empty result
                # is correct. Baseline path (`use_gau=False`) is
                # untouched: `transformer_blocks` is still an
                # `nn.ModuleList` and the loop runs as before.
                for block in (m.transformer_blocks or ()):
                    term = getattr(block.attention, "_entropy_reg_loss", None)
                    if term is not None:
                        total = total + term
                        del block.attention._entropy_reg_loss  # avoid double-count next step
                return total

            # OH1 ZLoss (output-head ablation #1): aux term = λ·mean(logsumexp(logits)²).
            # Train-only; eval stays plain CE (see reporting rule in
            # docs/research/output_head/plan.md). λ=0 → no-op. Uses
            # getattr so the base LLMConfig is unchanged.
            z_loss_lambda = getattr(config, "z_loss_lambda", 0.0)
            use_z_loss = getattr(config, "use_z_loss", False) and z_loss_lambda > 0.0

            # OH2 LabelSmooth (output-head ablation #2): CE with smoothing ε.
            # Train-only; eval stays plain CE (see reporting rule in
            # docs/research/output_head/plan.md). ε=0 → no-op (PyTorch treats
            # 0.0 as no smoothing, but the conditional is cleaner and keeps
            # the call site byte-identical when the flag is off / absent).
            label_smooth = getattr(config, "label_smooth", 0.0)

            # Forward pass (optimized to avoid large contiguous copies of logits)
            if config.use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 133 — SeqMix (Guo, Mao, Zhang 2019, arXiv:1908.02951):
                    # when use_seqmix=True, swap the standard
                    # `model(x)` + `F.cross_entropy` for `seqmix_forward`
                    # which mixes a paired sequence's token embeddings
                    # (λ ~ Beta(α, α)) and returns the mixed CE loss
                    # directly. With use_seqmix=False (default) the
                    # seqmix path is never invoked → baseline path
                    # byte-identical.
                    if getattr(config, "use_seqmix", False):
                        seqmix_loss, logits, _seqmix_lam = model.seqmix_forward(
                            x, y, alpha=float(getattr(config, "seqmix_alpha", 0.4)),
                        )
                        ce_loss = seqmix_loss
                        # Keep shift_labels defined so the rdrop branch
                        # below sees a non-stale variable if rdrop is also
                        # enabled (it isn't by default; rdrop + seqmix is
                        # not currently supported and would assert).
                        shift_labels = None
                    else:
                        logits = model(x)
                        # Shift labels instead of logits to save ~3GB VRAM
                        # We set the last token to -100 so cross_entropy ignores it
                        shift_labels = torch.full_like(y, -100)
                        shift_labels[:, :-1] = y[:, 1:]

                        ce_loss = F.cross_entropy(
                            logits.view(-1, config.vocab_size),
                            shift_labels.view(-1),
                            ignore_index=-100,
                            label_smoothing=label_smooth if label_smooth > 0 else 0.0,
                        )
                    # 115 — R-Drop (Liang et al. 2021, arXiv:2106.14448).
                    # Re-run the forward with a fresh dropout mask; replace
                    # the single-CE loss with the mean of two CEs and add a
                    # symmetric KL penalty scaled by the warmup alpha.
                    # At step 0 alpha=0 → loss == mean of two CEs (well within
                    # run-to-run variance of single-CE baseline). With
                    # `use_rdrop=False` (default) the helper is skipped.
                    rdrop_alpha_step = 0.0
                    if getattr(config, "use_rdrop", False):
                        rdrop_target = float(getattr(config, "rdrop_alpha", 1.0))
                        rdrop_warmup = int(getattr(config, "rdrop_warmup_steps", 1000))
                        if rdrop_warmup > 0:
                            rdrop_alpha_step = rdrop_target * min(1.0, step / rdrop_warmup)
                        else:
                            rdrop_alpha_step = rdrop_target
                        if rdrop_alpha_step > 0:
                            # R-Drop does 2 forward passes per step; on small
                            # VRAM (e.g. RTX 3060 11.6 GiB) the two full
                            # activation graphs OOM (~768 MiB alloc fails).
                            # Wrap the *second* forward in
                            # `torch.utils.checkpoint.checkpoint` so its
                            # activations are recomputed during backward.
                            # Math is identical (same RNG state, same
                            # dropout mask on the recompute), only the
                            # memory profile changes.
                            def _rdrop_fwd(x_):
                                return model(x_)
                            logits_2 = _torch_ckpt.checkpoint(
                                _rdrop_fwd, x, use_reentrant=False,
                            )
                            rdrop_ce, rdrop_kl = _rdrop_loss(
                                logits, logits_2, shift_labels,
                                config.vocab_size, rdrop_alpha_step,
                            )
                            ce_loss = rdrop_ce
                        else:
                            rdrop_kl = logits.new_zeros(())
                    else:
                        rdrop_kl = logits.new_zeros(())
                    entropy_reg_loss = _collect_entropy_reg(model)
                    z_loss = (
                        z_loss_lambda * (logits.logsumexp(dim=-1) ** 2).mean()
                        if use_z_loss
                        else logits.new_zeros(())
                    )
                    # OH3 ConfPenalty (output-head ablation #3): aux term = -β·H(softmax(logits)).
                    # Anti-overconfidence regularizer. Train-only; eval stays plain CE
                    # (see reporting rule in docs/research/output_head/plan.md).
                    # β=0 → no-op (zero scalar added). Uses getattr so the base
                    # LLMConfig is unchanged.
                    conf_penalty_beta = getattr(config, "conf_penalty_beta", 0.0)
                    if conf_penalty_beta > 0:
                        probs = logits.float().softmax(-1)
                        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                        conf_penalty = -conf_penalty_beta * ent
                    else:
                        conf_penalty = logits.new_zeros(())
                    # OH4 PolyLoss (output-head ablation #4): train-only
                    # correction L_poly = L_CE + ε₁·(1 - p_t) — the j=1
                    # Taylor term in the `-log p_t` expansion. Mask respects
                    # ignore_index=-100. See autoresearch/ideas/010-polyloss.
                    # ε₁=1.0 is the paper's "strong default" (Leng et al. 2022,
                    # arXiv:2204.12511). Train-only; eval stays plain CE.
                    # Flag off / ε₁=0 → no-op (zero scalar added).
                    use_poly_loss = getattr(config, "use_poly_loss", False)
                    poly_eps1 = getattr(config, "poly_eps1", 1.0)
                    if use_poly_loss and poly_eps1 > 0.0:
                        flat_logits = logits.view(-1, config.vocab_size).float()
                        flat_labels = shift_labels.view(-1)
                        poly_mask = (flat_labels != -100).float()
                        safe_labels = flat_labels.clamp(min=0)
                        p_t = flat_logits.softmax(-1).gather(
                            -1, safe_labels.unsqueeze(-1)
                        ).squeeze(-1)
                        n_valid = poly_mask.sum().clamp_min(1.0)
                        poly_loss = poly_eps1 * ((1.0 - p_t) * poly_mask).sum() / n_valid
                    else:
                        poly_loss = logits.new_zeros(())
                    # 132 — Born-Again self-distillation KL. Teacher is
                    # a no-grad EMA copy of the student; with `born_again=None`
                    # (default — use_born_again=False) the term is exactly 0
                    # and the baseline CE-only loss path is bit-identical.
                    if born_again is not None:
                        ba_kl = _born_again_distill_kl(
                            logits, model, x, born_again,
                            float(getattr(config, "born_again_temp", 2.0)),
                            float(getattr(config, "born_again_alpha", 1.0)),
                            config.vocab_size,
                        )
                    else:
                        ba_kl = logits.new_zeros(())
                    loss = (ce_loss + entropy_reg_loss + z_loss + conf_penalty + poly_loss + rdrop_kl + ba_kl) / config.gradient_accumulation_steps
                loss.backward()
            else:
                # 133 — SeqMix: mirror of the AMP branch above.
                # See the AMP comment for the rationale; same code path,
                # same baseline byte-identical guarantee when the flag
                # is off.
                if getattr(config, "use_seqmix", False):
                    seqmix_loss, logits, _seqmix_lam = model.seqmix_forward(
                        x, y, alpha=float(getattr(config, "seqmix_alpha", 0.4)),
                    )
                    ce_loss = seqmix_loss
                    shift_labels = None
                else:
                    logits = model(x)
                    shift_labels = torch.full_like(y, -100)
                    shift_labels[:, :-1] = y[:, 1:]

                    ce_loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        label_smoothing=label_smooth if label_smooth > 0 else 0.0,
                    )
                # 115 — R-Drop (Liang et al. 2021, arXiv:2106.14448).
                # CPU/non-AMP path mirror of the AMP branch above.
                # See the AMP comment for the rationale; same code path,
                # same step-0 invariance. The second forward is also wrapped
                # in `torch.utils.checkpoint.checkpoint` for consistency
                # with the AMP path (cheap on CPU, prevents OOM on small
                # VRAM when run with AMP disabled).
                rdrop_alpha_step = 0.0
                if getattr(config, "use_rdrop", False):
                    rdrop_target = float(getattr(config, "rdrop_alpha", 1.0))
                    rdrop_warmup = int(getattr(config, "rdrop_warmup_steps", 1000))
                    if rdrop_warmup > 0:
                        rdrop_alpha_step = rdrop_target * min(1.0, step / rdrop_warmup)
                    else:
                        rdrop_alpha_step = rdrop_target
                    if rdrop_alpha_step > 0:
                        def _rdrop_fwd_cpu(x_):
                            return model(x_)
                        logits_2 = _torch_ckpt.checkpoint(
                            _rdrop_fwd_cpu, x, use_reentrant=False,
                        )
                        rdrop_ce, rdrop_kl = _rdrop_loss(
                            logits, logits_2, shift_labels,
                            config.vocab_size, rdrop_alpha_step,
                        )
                        ce_loss = rdrop_ce
                    else:
                        rdrop_kl = logits.new_zeros(())
                else:
                    rdrop_kl = logits.new_zeros(())
                entropy_reg_loss = _collect_entropy_reg(model)
                z_loss = (
                    z_loss_lambda * (logits.logsumexp(dim=-1) ** 2).mean()
                    if use_z_loss
                    else logits.new_zeros(())
                )
                # OH3 ConfPenalty (output-head ablation #3): aux term = -β·H(softmax(logits)).
                # Anti-overconfidence regularizer. Train-only; eval stays plain CE
                # (see reporting rule in docs/research/output_head/plan.md).
                # β=0 → no-op (zero scalar added). Uses getattr so the base
                # LLMConfig is unchanged.
                conf_penalty_beta = getattr(config, "conf_penalty_beta", 0.0)
                if conf_penalty_beta > 0:
                    probs = logits.float().softmax(-1)
                    ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                    conf_penalty = -conf_penalty_beta * ent
                else:
                    conf_penalty = logits.new_zeros(())
                # OH4 PolyLoss (output-head ablation #4): train-only
                # correction L_poly = L_CE + ε₁·(1 - p_t) — the j=1
                # Taylor term in the `-log p_t` expansion. Mask respects
                # ignore_index=-100. See autoresearch/ideas/010-polyloss.
                # Flag off / ε₁=0 → no-op (zero scalar added).
                use_poly_loss = getattr(config, "use_poly_loss", False)
                poly_eps1 = getattr(config, "poly_eps1", 1.0)
                if use_poly_loss and poly_eps1 > 0.0:
                    flat_logits = logits.view(-1, config.vocab_size).float()
                    flat_labels = shift_labels.view(-1)
                    poly_mask = (flat_labels != -100).float()
                    safe_labels = flat_labels.clamp(min=0)
                    p_t = flat_logits.softmax(-1).gather(
                        -1, safe_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    n_valid = poly_mask.sum().clamp_min(1.0)
                    poly_loss = poly_eps1 * ((1.0 - p_t) * poly_mask).sum() / n_valid
                else:
                    poly_loss = logits.new_zeros(())
                # 132 — Born-Again self-distillation KL (CPU/MPS branch).
                # Same identity-at-step-0 invariant as the AMP branch above:
                # with teacher == student init the KL term is exactly 0.
                if born_again is not None:
                    ba_kl = _born_again_distill_kl(
                        logits, model, x, born_again,
                        float(getattr(config, "born_again_temp", 2.0)),
                        float(getattr(config, "born_again_alpha", 1.0)),
                        config.vocab_size,
                    )
                else:
                    ba_kl = logits.new_zeros(())
                loss = (ce_loss + entropy_reg_loss + z_loss + conf_penalty + poly_loss + rdrop_kl + ba_kl) / config.gradient_accumulation_steps
                loss.backward()

            # Detach z-loss to a python float for logging only
            z_loss_val = z_loss.detach().item()

            # Detach entropy reg to a python float for logging only (graph no longer needed)
            entropy_reg_val = entropy_reg_loss.detach().item()

            # Detach conf penalty to a python float for logging only (graph no longer needed)
            conf_penalty_val = conf_penalty.detach().item()

            # Detach poly loss to a python float for logging only (graph no longer needed)
            poly_loss_val = poly_loss.detach().item()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                # 140 — Sophia: Hutchinson trace estimator for the
                # diagonal Hessian (Liu, Wang, et al. 2023,
                # arXiv:2305.14342, Algorithm 1). Re-run the forward
                # pass with `create_graph=True` so we can take the
                # SECOND-ORDER gradient `∇(g·u)` (the diagonal
                # Hessian sample, scaled by Rademacher u). Then
                # `h_hat = u · ∇(g·u)`. The trainer passes `h_hat`
                # to `Sophia.update_hessian` so the optimizer's
                # `h_t` EMA can use the fresh sample in this
                # step's `.step()` call. Fires every
                # `sophia_hessian_freq` optimizer steps (paper
                # default 10) — amortized ~1.1× forward+backward
                # cost. When `use_sophia=False` (default) this
                # block is skipped entirely and the baseline
                # training cost is unchanged. The Sophia instance
                # is identified by `isinstance(opt, Sophia)`, same
                # pattern as AdamSAM / LookSAM routing below.
                sophia_opts = [opt for opt in optimizers if isinstance(opt, Sophia)]
                if sophia_opts:
                    sophia_opt = sophia_opts[0]
                    hessian_freq = int(getattr(config, "sophia_hessian_freq", 10))
                    # `_step_count` is incremented at the END of
                    # Sophia.step(), so it equals the number of
                    # completed optimizer steps. We sample when
                    # `_step_count % hessian_freq == 0` — at step
                    # 0 (count=0) and every k steps thereafter,
                    # matching Liu et al. 2023 Algorithm 1.
                    if hessian_freq > 0 and sophia_opt._step_count % hessian_freq == 0:
                        # Sample Rademacher u per AdamW param
                        # (same device/dtype as the param so the
                        # `param * u` ops stay in-graph).
                        with torch.enable_grad():
                            u_list = [
                                torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1)
                                if p.requires_grad else None
                                for p in adamw_params
                            ]
                            # Re-forward the model with a fresh
                            # graph so we can take second-order
                            # grads. Use the same AMP context as
                            # the main training step (mirrors the
                            # SAM closure). The loss is the same
                            # CE used in the main step (no aux
                            # terms — keeps the second-order
                            # signal clean, matching the paper's
                            # recipe). The hessian sample uses the
                            # gradient of the *same* loss, just
                            # computed with a different graph so
                            # `grad` carries a `grad_fn`.
                            if config.use_amp and device.type == "cuda":
                                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    _hess_logits = model(x)
                                    _hess_shift = torch.full_like(y, -100)
                                    _hess_shift[:, :-1] = y[:, 1:]
                                    _hess_loss = F.cross_entropy(
                                        _hess_logits.view(-1, config.vocab_size),
                                        _hess_shift.view(-1),
                                        ignore_index=-100,
                                    )
                            else:
                                _hess_logits = model(x)
                                _hess_shift = torch.full_like(y, -100)
                                _hess_shift[:, :-1] = y[:, 1:]
                                _hess_loss = F.cross_entropy(
                                    _hess_logits.view(-1, config.vocab_size),
                                    _hess_shift.view(-1),
                                    ignore_index=-100,
                                )
                            # First-order grad WITH a graph (so
                            # we can take a second-order grad).
                            # `allow_unused=True` because the
                            # `use_softmax` 1-D scalars may be
                            # unused on a given minibatch.
                            g_list = torch.autograd.grad(
                                _hess_loss, adamw_params,
                                create_graph=True, allow_unused=True,
                            )
                            # Build the scalar g·u and take its
                            # gradient. This gives `H·u` for each
                            # param without mutating any `.grad`.
                            g_u = sum(
                                (g * u).sum() for g, u in zip(g_list, u_list)
                                if g is not None and u is not None
                            )
                            hv_list = torch.autograd.grad(
                                g_u, adamw_params,
                                allow_unused=True,
                            )
                        # Build h_hat = u · (H·u). Convert None
                        # entries (unused params) to None so the
                        # optimizer's update_hessian skips them.
                        h_hat_list = [
                            (u * hv).detach() if (u is not None and hv is not None) else None
                            for u, hv in zip(u_list, hv_list)
                        ]
                        # Feed h_hat into the optimizer's h_t EMA.
                        # The trainer only sets one Sophia instance
                        # (the adamw slot is single-occupant), so
                        # `sophia_opt` is the same as the one the
                        # .step() call below will invoke.
                        beta2 = float(getattr(config, "sophia_beta2", 0.99))
                        sophia_opt.update_hessian(h_hat_list, beta2=beta2)
                # 119 — SAM (Foret et al. 2020) ascent/closure/descent
                # interleaving. SAM replaces the AdamW path with
                # `AdamSAM`, which exposes `first_step` (ascent to
                # w + ε̂) and `second_step` (restore w, then AdamW on
                # the perturbed grad). The closure between them
                # re-runs forward+backward at the perturbed point.
                # The non-SAM optimizers (Muon) step on the w-grad
                # (the first backward) — they don't see the
                # perturbation. Order: (1) non-SAM step on w-grad,
                # (2) SAM ascent + zero AdamSAM grad, (3) second
                # forward+backward at w+ε̂, (4) SAM descent on
                # perturbed grad, (5) zero remaining grads.
                sam_opts = [opt for opt in optimizers if isinstance(opt, AdamSAM)]
                non_sam_opts = [opt for opt in optimizers if not isinstance(opt, AdamSAM)]
                # 138 — LookSAM is an AdamSAM subclass, but it only
                # needs the SAM dance every K steps. On non-SAM
                # steps LookSAM.step() is a plain AdamW step (the
                # trainer's `opt.step()` path). On SAM steps
                # LookSAM joins the SAM group and runs the
                # first_step → closure → second_step dance. Route
                # each LookSAM into the right group per step using
                # its `next_is_sam` property. Identity at step 0:
                # step_count=0 ⇒ next_is_sam=False ⇒ LookSAM goes
                # to non_sam_opts ⇒ plain AdamW path ⇒ bit-identical
                # to baseline AdamW at step 0.
                for opt in list(non_sam_opts):
                    if isinstance(opt, LookSAM) and opt.next_is_sam:
                        non_sam_opts.remove(opt)
                        sam_opts.append(opt)
                # (1) Non-SAM optimizers step on the w-grad.
                for optimizer in non_sam_opts:
                    optimizer.step()
                    optimizer.zero_grad()
                # (2) SAM ascent: w[AdamSAM] ← w + ε̂, zero AdamSAM grad.
                for optimizer in sam_opts:
                    optimizer.first_step(zero_grad=True)
                # (3) Second forward+backward at the perturbed point.
                # The closure runs in the same AMP context as the
                # first forward so the perturbed grad is on the same
                # scale. Loss is CE-only (no aux terms) — this is
                # the canonical SAM pattern; the perturbed-grad
                # direction is dominated by CE at this scale.
                if sam_opts:
                    def _sam_closure():
                        if config.use_amp and device.type == "cuda":
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                logits2 = model(x)
                                shift_labels2 = torch.full_like(y, -100)
                                shift_labels2[:, :-1] = y[:, 1:]
                                ce_loss2 = F.cross_entropy(
                                    logits2.view(-1, config.vocab_size),
                                    shift_labels2.view(-1),
                                    ignore_index=-100,
                                )
                            ce_loss2.backward()
                        else:
                            logits2 = model(x)
                            shift_labels2 = torch.full_like(y, -100)
                            shift_labels2[:, :-1] = y[:, 1:]
                            ce_loss2 = F.cross_entropy(
                                logits2.view(-1, config.vocab_size),
                                shift_labels2.view(-1),
                                ignore_index=-100,
                            )
                            ce_loss2.backward()
                    _sam_closure()
                # (4) SAM descent: restore w, AdamW on perturbed grad.
                for optimizer in sam_opts:
                    optimizer.second_step(zero_grad=True)
                # 112 — Lookahead outer step. Fires every k inner steps;
                # inert when use_lookahead=False. Operates on the live
                # `model` so the next forward sees the slow-pulled weights.
                if lookahead is not None:
                    lookahead.step()
                # 110 — Weight EMA. Updates the shadow copy AFTER the
                # inner optimizers (so the EMA tracks the freshly-stepped
                # weights). Linear ramp 0 → ema_decay over the first
                # `ema_warmup_steps` ⇒ at step 0 the EMA == live θ and
                # the swap-in eval reads the baseline model bit-identically.
                if ema is not None:
                    ema.update_from(model)
                # 132 — Born-Again teacher EMA. `θ_teacher ← (1−β)·θ_teacher
                # + β·θ_student` after each optimizer step. With β=0.999
                # and step 0's shadow == student init, the post-step
                # teacher ≈ student, so the next-step KL is small but
                # nonzero (the lever's signal). With `born_again=None`
                # (default — use_born_again=False) the helper is a
                # no-op and the baseline trajectory is bit-identical.
                if born_again is not None:
                    born_again.update_from(model)
                for scheduler in schedulers:
                    scheduler.step()

            # Track current loss as a scalar only every 100 steps to avoid sync bottleneck
            if step % 100 == 0 or step == 0:
                current_loss_val = ce_loss.item()
                
            # Logging
            if step % log_every == 0 or stopped_early:
                with torch.no_grad():
                    # Calculate accuracy using the shifted labels mask
                    predictions = logits.argmax(dim=-1)
                    mask = (shift_labels != -100)
                    accuracy = (predictions[mask] == shift_labels[mask]).float().mean().item()
                    
                    # Use the scalar value we polled above
                    perplexity = math.exp(min(current_loss_val if 'current_loss_val' in locals() else ce_loss.item(), 20))
                    current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']

                # Update progress bar
                tokens_per_step = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
                est_total_steps = config.train_tokens // tokens_per_step
                
                pbar.set_postfix({
                    'step': f'{step}/{est_total_steps}',
                    'loss': f'{current_loss_val:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ent': f'{entropy_reg_val:+.2e}',
                    'cp': f'{conf_penalty_val:+.2e}',
                    'pl': f'{poly_loss_val:+.2e}',
                    'lr': f'{current_lr:.5f}'
                })
                # Console print for visibility
                if step % (log_every * 10) == 0 or stopped_early:
                    print(f" [Step {step}] Loss: {current_loss_val:.4f} | EntReg: {entropy_reg_val:+.2e} | ConfPen: {conf_penalty_val:+.2e} | PolyLoss: {poly_loss_val:+.2e} | Acc: {accuracy:.3f} | LR: {current_lr:.6f}")
            
            pbar.update(batch_tokens)
            tokens_seen += batch_tokens

            if stopped_early:
                current_loss_val = ce_loss.item()
                break

            # Evaluation
            is_milestone = False
            if config.eval_milestones and step in config.eval_milestones:
                is_milestone = True
            elif config.eval_every is not None and step % config.eval_every == 0 and step > 0:
                is_milestone = True

            if is_milestone:
                # Schedule-Free AdamW: swap p.data from y (train) to x
                # (Polyak-Ruppert average) before eval, then back to y.
                # No-op for non-SF optimizers.
                _swap_optimizers_eval_mode(optimizers, "eval")
                # 110 — Weight EMA: when `ema_eval_only=True`, swap the
                # live parameters for the EMA copy before eval, then
                # restore on exit. `ema is None` when the flag is off.
                # The backup is taken inside the try / restored in finally
                # so a val crash never leaves the model on the EMA copy.
                ema_backup = ema.apply_to(model) if (ema is not None) else None
                try:
                    eval_metrics = evaluate_model(model, val_loader, config)
                finally:
                    if ema_backup is not None:
                        ema.restore_from(model, ema_backup)
                _swap_optimizers_eval_mode(optimizers, "train")
                elapsed_time = (time.time() - train_start_time) / 60
                current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
                
                # Track metrics
                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(eval_metrics['val_loss'])
                metrics_history['val_accuracies'].append(eval_metrics['val_accuracy'])
                metrics_history['val_perplexities'].append(eval_metrics['val_perplexity'])
                metrics_history['elapsed_times'].append(elapsed_time)
                metrics_history['learning_rates'].append(current_lr)
                
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"LR: {current_lr:.5f}")
                if step in exact_step_baseline:
                    baseline_loss = exact_step_baseline[step]
                    delta = eval_metrics['val_loss'] - baseline_loss
                    print(f"   Baseline@step {step}: {baseline_loss:.4f} | delta: {delta:+.4f}")

                live_metrics = {**eval_metrics, 'train_loss': current_loss_val, 'train/conf_penalty_loss': conf_penalty_val}
                write_live_metrics(live_metrics, step, tokens_seen)
                if output_path:
                    save_training_checkpoint(
                        output_path / "latest.pt",
                        model,
                        config,
                        optimizers,
                        schedulers,
                        live_metrics,
                        step,
                        tokens_seen,
                        metrics_history,
                    )
                    save_training_checkpoint(
                        output_path / f"checkpoint_step_{step}.pt",
                        model,
                        config,
                        optimizers,
                        schedulers,
                        live_metrics,
                        step,
                        tokens_seen,
                        metrics_history,
                    )
                
                # Early stopping check
                if early_stopper is not None:
                    if early_stopper(eval_metrics['val_loss'], step):
                        current_loss_val = ce_loss.item()
                        stopped_early = True
                        break

                # Gate stop: pause the run at a pre-registered step so a human can
                # judge the curve before spending the rest of the compute. We only
                # gate on a milestone (we just evaluated + checkpointed above), so
                # gate.pt is a full, resumable state with a fresh eval point.
                if gate_step is not None and step >= gate_step:
                    if output_path:
                        save_training_checkpoint(
                            output_path / "gate.pt",
                            model,
                            config,
                            optimizers,
                            schedulers,
                            live_metrics,
                            step,
                            tokens_seen,
                            metrics_history,
                        )
                        print(f"\n⏸️  Gate reached: stop_at_step={gate_step} (paused at step {step}). "
                              f"Saved resumable checkpoint for human review.\n"
                              f"    Resume the SAME run with: --load_checkpoint {output_path / 'gate.pt'}")
                    else:
                        print(f"\n⏸️  Gate reached at step {step}, but no --output_dir set, so no checkpoint was saved.")
                    gated = True
                    break

            step += 1
        
        # If we finished the inner loop but didn't stop early, 
        # ensure we have the most recent loss from the very last batch
        if not stopped_early and 'ce_loss' in locals():
            current_loss_val = ce_loss.item()

        if stopped_early or gated:
            break

    pbar.close()

    # Final evaluation (if not stopped early)
    if (not stopped_early and not gated) or tokens_seen >= config.train_tokens:
        # If the last in-loop milestone already evaluated these exact weights
        # (last completed step == step - 1, no optimizer step since), reuse that
        # result instead of re-evaluating — avoids a duplicate final data point.
        if metrics_history['steps'] and metrics_history['steps'][-1] == step - 1:
            final_eval = {
                'val_loss': metrics_history['val_losses'][-1],
                'val_accuracy': metrics_history['val_accuracies'][-1],
                'val_perplexity': metrics_history['val_perplexities'][-1],
                'train_loss': current_loss_val,
            }
        else:
            _swap_optimizers_eval_mode(optimizers, "eval")
            ema_backup = ema.apply_to(model) if (ema is not None) else None
            try:
                final_eval = evaluate_model(model, val_loader, config)
            finally:
                if ema_backup is not None:
                    ema.restore_from(model, ema_backup)
            _swap_optimizers_eval_mode(optimizers, "train")
            final_eval['train_loss'] = current_loss_val
            elapsed_time = (time.time() - train_start_time) / 60
            current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']

            metrics_history['steps'].append(step)
            metrics_history['val_losses'].append(final_eval['val_loss'])
            metrics_history['val_accuracies'].append(final_eval['val_accuracy'])
            metrics_history['val_perplexities'].append(final_eval['val_perplexity'])
            metrics_history['elapsed_times'].append(elapsed_time)
            metrics_history['learning_rates'].append(current_lr)
    else:
        # Use best metrics if stopped early
        if metrics_history['val_losses']:
            best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
            final_eval = {
                'val_loss': metrics_history['val_losses'][best_idx],
                'val_accuracy': metrics_history['val_accuracies'][best_idx],
                'val_perplexity': metrics_history['val_perplexities'][best_idx],
                'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
            }
        else:
            final_eval = {
                'val_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
                'val_accuracy': accuracy if 'accuracy' in locals() else 0.0,
                'val_perplexity': perplexity if 'perplexity' in locals() else 0.0,
                'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
            }
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time_seconds = time.time() - train_start_time
    
    if stopped_early:
        print(f"   ⚠️  Training stopped early at step {step}")
    
    # Save outputs if directory specified
    if output_dir:
        # Save metrics
        metrics_file = output_path / "metrics.json"
        metrics_data = {
            'final_metrics': final_eval,
            'total_time_minutes': total_time_seconds / 60,
            'stopped_early': stopped_early,
            'actual_steps': step,
            'history': metrics_history,
            **capture_git_metadata(),
        }
        if extra_config:
            metrics_data['experiment_config'] = extra_config
            
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"   📁 Metrics saved to {metrics_file}")
        
        
        # Save model checkpoint
        checkpoint_path = output_path / "model.pt"
        save_training_checkpoint(
            checkpoint_path,
            model,
            config,
            optimizers,
            schedulers,
            final_eval,
            step,
            tokens_seen,
            metrics_history,
        )
    
    return {
        'model': model,
        'final_metrics': final_eval,
        'metrics_history': metrics_history,
        'training_time': total_time_seconds,
        'steps': step,
        'tokens_seen': tokens_seen,
        'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
        'gated': gated,
    }



def warmup_compiled_kernels(
    model: nn.Module,
    config: LLMConfig,
    train_loader: DataLoader,
    device: torch.device,
    num_steps: int = 3
) -> None:
    """
    Warm up all compiled kernels (forward, backward, optimizer).
    Caller is responsible for resetting state afterwards.
    """
    print(f"🔥 Warming up kernels ({num_steps} steps)...")
    model.train()
    
    # Temporary optimizer to warm up optimizer kernels too
    temp_optimizers = setup_muon_optimizer(model, config)
    
    warmup_iter = iter(train_loader)
    
    for _ in range(num_steps):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            batch = next(warmup_iter)
        
        # Parse batch
        if isinstance(batch, dict):
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
        else:
            x, y = batch[0].to(device), batch[-1].to(device)
        
        # Forward + Backward
        if config.use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, config.vocab_size),
                    y[:, 1:].reshape(-1)
                )
            loss.backward()
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                y[:, 1:].reshape(-1)
            )
            loss.backward()
        
        # Optimizer step (warms up optimizer kernels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        for opt in temp_optimizers:
            opt.step()
            opt.zero_grad()
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
    # Cleanup temp optimizers
    del temp_optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print("✅ Kernels compiled and cached")

def train_minimal_llm(
    config: LLMConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    load_weights_path: Optional[str] = None,
    compare_baseline: bool = False,
    config_name: Optional[str] = None,
    run_seed: Optional[int] = None,
):
    print(f"\n🚀 Training dense model")
    setup_start = time.time()
    device = resolve_device(getattr(config, "device", "auto"))
    checkpoint_payload = None

    # ============================================
    # 1. Initialize model with the configured seed (varies init when --seed set)
    # ============================================
    set_seed(getattr(config, "seed", 42))
    model = MinimalLLM(config)
    model = model.to(device, dtype=torch.bfloat16) if device.type == "cuda" else model.to(device)
    
    # Load pretrained weights if specified
    if load_weights_path:
        print(f"Loading pretrained weights from {load_weights_path}...")
        checkpoint_payload = torch.load(load_weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        model.load_state_dict(state_dict, strict=False)

    # ============================================
    # 2. Save initial state BEFORE any forward pass (cloned to CPU)
    # ============================================
    initial_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # ============================================
    # 3. Compile model (if requested)
    # ============================================
    if config.compile_model:
        print("🚀 Compiling model with torch.compile...")
        # Keep a reference to the original model for state restoration
        orig_model = model
        try:
            model = torch.compile(model)
            print("✅ Model compiled successfully")
            
            # ============================================
            # 4. Warm up kernels (dirties model state)
            # ============================================
            warmup_compiled_kernels(model, config, train_loader, device, num_steps=3)
            
            # ============================================
            # 5. Reset model to initial state
            # ============================================
            # Restore state ensuring we use the original model keys to avoid calling load_state_dict on the wrapper
            orig_model.load_state_dict(initial_model_state)
            print("🔄 Model weights reset to initial state")
            
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
            print("Continuing in eager mode.")
            # Fallback to original model
            model = orig_model
            # Ensure state is clean
            model.load_state_dict(initial_model_state)
    
    # Free the backup
    del initial_model_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ============================================
    # 6. Create FRESH optimizers (no accumulated state)
    # ============================================
    optimizers = setup_muon_optimizer(model, config)

    # ============================================
    # 7. Create FRESH schedulers
    # ============================================
    # Tokens per optimization step
    tokens_per_opt = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
    total_steps = config.train_tokens // tokens_per_opt
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    schedule_type = getattr(config, 'schedule_type', 'cosine')
    # Schedule-Free AdamW (Defazio et al. 2024) carries its own internal
    # averaging schedule (`c = 1/(k-warmup+1)` after warmup). Forcing
    # the external schedule to constant keeps the LR flat — the
    # optimizer's internal averaging does the late-training stabilization
    # that the lr_lambda would otherwise provide. See
    # autoresearch/ideas/006-schedule-free-adamw/plan.md.
    if getattr(config, "use_schedule_free_adamw", False):
        schedule_type = 'constant'
    
    schedulers = []
    for optimizer in optimizers:
        if schedule_type == 'cosine':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_type == 'linear':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return max(0.1, 1.0 - progress)
        elif schedule_type == 'warmup_decay_to_zero':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return max(0.0, 1.0 - progress)
        else:  # constant
            def lr_lambda(current_step, warmup=warmup_steps):
                return current_step / warmup if current_step < warmup else 1.0
        
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))

    if checkpoint_payload and "optimizer_state_dicts" in checkpoint_payload:
        optimizer_states = checkpoint_payload.get("optimizer_state_dicts", [])
        scheduler_states = checkpoint_payload.get("scheduler_state_dicts", [])
        for optimizer, state_dict in zip(optimizers, optimizer_states):
            optimizer.load_state_dict(state_dict)
        for scheduler, state_dict in zip(schedulers, scheduler_states):
            scheduler.load_state_dict(state_dict)
        restore_rng_state(checkpoint_payload.get("rng_state"))

    # 112 — Lookahead Optimizer Wrapper. Sits OUTSIDE the per-step inner
    # optimizers and only fires every `lookahead_k` inner steps. With
    # use_lookahead=False (default) the wrapper is None → fully inert,
    # baseline path bit-identical. The wrapper's `slow` snapshot is taken
    # AFTER any checkpoint load so the slow weights match the live
    # model state at the start of this run.
    lookahead = None
    if getattr(config, "use_lookahead", False):
        lookahead = Lookahead(
            optimizers=optimizers,
            model=model,
            k=int(getattr(config, "lookahead_k", 5)),
            alpha=float(getattr(config, "lookahead_alpha", 0.5)),
        )
        print(f"  🔭 Lookahead wrapper active: k={lookahead.k}, α={lookahead.alpha}")

    # 110 — Polyak-Ruppert Weight EMA. Builds a shadow copy of every
    # trainable parameter and updates it after each optimizer step.
    # `ema_eval_only=True` (default) keeps the live `θ` as the saved
    # / resumed model and only swaps the EMA in for the val pass.
    # With `use_ema_eval=False` (default) the wrapper is None → fully
    # inert, baseline path bit-identical. The shadow snapshot is taken
    # AFTER any checkpoint load so the EMA starts from the resumed
    # weights, not from the random init.
    ema = None
    if getattr(config, "use_ema_eval", False):
        ema = ModelEMA(
            model=model,
            decay=float(getattr(config, "ema_decay", 0.999)),
            warmup_steps=int(getattr(config, "ema_warmup_steps", 100)),
        )
        eval_only = bool(getattr(config, "ema_eval_only", True))
        print(f"  📈 EMA shadow active: decay={ema.decay}, "
              f"warmup_steps={ema.warmup_steps}, eval_only={eval_only}")

    # 132 — Born-Again self-distillation teacher. Builds a shadow copy
    # of every trainable parameter at construction time (deep clone of
    # the live init) so step-0 KL is exactly 0. Updated each optimizer
    # step as `θ_teacher ← (1−β)·θ_teacher + β·θ_student`. With
    # `use_born_again=False` (default) the teacher is None → fully
    # inert, baseline path bit-identical. The shadow snapshot is taken
    # AFTER any checkpoint load so the teacher starts from the resumed
    # weights, not from the random init.
    born_again = None
    if getattr(config, "use_born_again", False):
        born_again = BornAgainTeacher(
            model=model,
            beta=float(getattr(config, "born_again_beta", 0.999)),
        )
        ba_alpha = float(getattr(config, "born_again_alpha", 1.0))
        ba_temp = float(getattr(config, "born_again_temp", 2.0))
        print(f"  🎓 Born-Again teacher active: β={born_again.beta}, "
              f"α={ba_alpha}, T={ba_temp}")

    # ============================================
    # 8. Reset RNG for reproducible training
    # ============================================
    if not (checkpoint_payload and "rng_state" in checkpoint_payload):
        set_seed(getattr(config, "seed", 42))
    
    setup_time = time.time() - setup_start
    print(f"⚙️ Setup & Compilation complete in {setup_time:.2f}s")
    print("-" * 70)

    # ============================================
    # 9. Train from scratch (fresh iterator created internally)
    # ============================================
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    train_start = time.time()
    
    results = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        checkpoint_state=checkpoint_payload,
        early_stopper=None,
        output_dir=output_dir,
        extra_config=None,
        log_every=getattr(config, 'log_every', 100),
        lookahead=lookahead,
        ema=ema,
        born_again=born_again,
    )
    
    total_training_time = results['training_time']
    total_wall_time = setup_time + total_training_time
    final_eval = results['final_metrics']
    metrics_history = results['metrics_history']
    step = results['steps']
    tokens_seen = results['tokens_seen']
    gated = results.get('gated', False)

    # ============================================
    # 10. Unified Saving & Reporting
    # ============================================
    # Save results to plots/ directory with timestamped names
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filenames
    base_filename = f"{config.train_tokens}_{timestamp}"
    metrics_file = plot_dir / f"metrics_{base_filename}.json"
    plot_file = plot_dir / f"val_loss_{base_filename}.png"
    output_path = Path(output_dir) if output_dir else None
    
    # Save comprehensive metrics to plots/
    metrics_data = {
        'final_metrics': final_eval,
        'setup_time_seconds': setup_time,
        'active_training_time_seconds': total_training_time,
        'total_wall_time_seconds': total_wall_time,
        'total_time_minutes': total_wall_time / 60,
        'actual_steps': step,
        'tokens_seen': tokens_seen,
        'train_tokens': config.train_tokens,
        'gated': gated,
        'history': metrics_history,
        # #65 self-identifying run: every metrics.json answers
        # "which run am I?" on its own. DESC in
        # runs/make_evidence_index.py is the curated prose;
        # these fields are the structured truth.
        'run_name': output_path.name if output_path else None,
        'config_name': config_name or config.__class__.__name__,
        'seed': run_seed if run_seed is not None else getattr(config, 'seed', None),
        'flags': config.active_flags(),
        **capture_git_metadata(),
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"   📊 Metrics saved to {metrics_file}")
        
    try:
        from utils.plot_loss import plot_loss
        
        baseline_file = None
        if compare_baseline:
            # Determine closest baseline file based on token count
            known_baselines = {
                8_000_000: "plots/8M.json",
                20_000_000: "plots/20M.json",
                100_000_000: "plots/100M.json"
            }
            
            # Find closest baseline
            closest_tokens = min(known_baselines.keys(), key=lambda x: abs(x - config.train_tokens))
            baseline_file = known_baselines[closest_tokens]
            
            # Verify it exists
            if not os.path.exists(baseline_file):
                print(f"      (Baseline file {baseline_file} not found locally)")
                baseline_file = None
            
        plot_loss(
            str(metrics_file), 
            str(plot_file), 
            title=f"Validation Loss - {config.train_tokens:,} Tokens",
            baseline_file=baseline_file
        )
        print(f"   📈 Plot saved to {plot_file}")
        if baseline_file:
            print(f"      (Compared against baseline: {baseline_file})")
    except Exception as e:
        print(f"   ⚠️ Failed to generate plot: {e}")
    
    # Also save to output_dir if specified (for backward compatibility)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics copy
        checkpoint_metrics = output_path / "metrics.json"
        with open(checkpoint_metrics, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        # Save model
        checkpoint_path = output_path / "model.pt"
        save_training_checkpoint(
            checkpoint_path,
            results['model'],
            config,
            optimizers,
            schedulers,
            final_eval,
            step,
            tokens_seen,
            metrics_history,
        )
        
    
    # Final Output
    print("\n" + "="*70)
    print(" TRAINING RESULTS")
    print("="*70)
    print(f"Warmup & Setup:                  {format_time(setup_time)}")
    print(f"Active Training Time:            {format_time(total_training_time)}")
    print(f"Total Tokens:                    {tokens_seen:,}")
    print("-" * 70)
    print(f"Final Train Loss:                {final_eval.get('train_loss', 0.0):.4f}")
    print(f"Final Val Loss:                  {final_eval['val_loss']:.4f}")
    print(f"Final Val Accuracy:              {final_eval['val_accuracy']:.4f}")
    print("="*70 + "\n")

    return {
        'model': results['model'],
        'metrics': final_eval,
        'history': metrics_history,
        'setup_time': setup_time,
        'training_time': total_training_time,
        'steps': step,
        'tokens_seen': tokens_seen,
        'gated': gated,
    }
