"""Toy research-pipeline smoke: ~1M-param model, ~30k real tokens, MacBook-fast.

Purpose: establish the end-to-end research loop (data -> model -> Muon/AdamW
train -> eval -> metrics.json -> loss curve PNG) so experiments are reproducible
without a GPU. NOT a real result; it's the harness we hang real runs on later.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from training.trainer import setup_muon_optimizer
from training.device import resolve_device
from utils.helpers import set_seed

# ---- knobs -------------------------------------------------------------
SEQ_LEN = 256
TRAIN_TOKENS = 30_000
EVAL_EVERY = 2            # steps
SRC_ROWS = 24            # 2048-token rows to slice (24*2048 ~= 49k tokens)
OUT = Path("experiments/results/toy")
# -----------------------------------------------------------------------


def build_config() -> LLMConfig:
    # vocab is fixed (49,152) so the tied embedding dominates params; shrink
    # d_model hard to land near ~1M total.
    return LLMConfig(
        d_model=24, n_heads=2, n_layers=2, d_ff=96, n_kv_heads=1,
        max_seq_len=SEQ_LEN, train_tokens=TRAIN_TOKENS,
        muon_lr=0.02, adamw_lr=0.006, batch_size=8,
        eval_every=EVAL_EVERY, dropout=0.0, compile_model=False, use_amp=False,
    )


def load_tiny_tokens() -> torch.Tensor:
    from datasets import load_from_disk
    ds = load_from_disk("processed_data/pretrain_mix_1000000000")["train"]
    flat = []
    for i in range(SRC_ROWS):
        flat.extend(ds[i]["input_ids"])
    t = torch.tensor(flat, dtype=torch.long)
    n = (t.numel() // SEQ_LEN) * SEQ_LEN
    return t[:n].view(-1, SEQ_LEN)  # [num_seq, SEQ_LEN]


def make_loaders(seqs: torch.Tensor, batch_size: int):
    n_val = max(1, int(0.1 * seqs.size(0)))
    val, train = seqs[:n_val], seqs[n_val:]
    return (
        DataLoader(TensorDataset(train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(val), batch_size=batch_size),
    )


def lm_loss(model, x):
    logits = model(x)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        x[:, 1:].reshape(-1),
    )


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for (x,) in loader:
        tot += lm_loss(model, x.to(device)).item()
        n += 1
    model.train()
    return tot / max(1, n)


def main():
    set_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = build_config()
    device = resolve_device("auto")

    seqs = load_tiny_tokens()
    train_loader, val_loader = make_loaders(seqs, cfg.batch_size)

    model = MinimalLLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,} | device: {device} | train seqs: {len(seqs)}")

    optims = setup_muon_optimizer(model, cfg)
    tokens_per_step = cfg.batch_size * SEQ_LEN

    hist = {"step": [], "train_loss": [], "eval_step": [], "val_loss": []}
    step, tokens, t0 = 0, 0, time.time()
    model.train()
    while tokens < TRAIN_TOKENS:
        for (x,) in train_loader:
            if tokens >= TRAIN_TOKENS:
                break
            x = x.to(device)
            loss = lm_loss(model, x)
            for o in optims:
                o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for o in optims:
                o.step()

            hist["step"].append(step)
            hist["train_loss"].append(loss.item())
            if step % EVAL_EVERY == 0:
                vl = eval_loss(model, val_loader, device)
                hist["eval_step"].append(step)
                hist["val_loss"].append(vl)
                print(f"step {step:3d} | train {loss.item():.4f} | val {vl:.4f}")
            step += 1
            tokens += tokens_per_step

    vl = eval_loss(model, val_loader, device)
    hist["eval_step"].append(step)
    hist["val_loss"].append(vl)
    elapsed = time.time() - t0
    print(f"done: {step} steps, {tokens:,} tokens in {elapsed:.1f}s | final val {vl:.4f}")

    summary = {
        "params": n_params, "seq_len": SEQ_LEN, "train_tokens": tokens,
        "steps": step, "wall_s": round(elapsed, 1),
        "final_train_loss": hist["train_loss"][-1], "final_val_loss": vl,
        "tokens_per_s": round(tokens / elapsed, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps({"summary": summary, "history": hist}, indent=2))

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["step"], hist["train_loss"], lw=1.2, alpha=0.8, label="train loss")
    plt.plot(hist["eval_step"], hist["val_loss"], "o-", lw=1.5, label="val loss")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title(f"Toy pipeline: {n_params/1e6:.2f}M params, {tokens//1000}k tokens, {elapsed:.0f}s on {device.type}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "toy_loss_curve.png", dpi=130)
    print(f"wrote {OUT/'toy_loss_curve.png'} and {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()
