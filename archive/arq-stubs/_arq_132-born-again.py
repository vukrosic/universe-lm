"""Autoresearch 132 — trt: Born-Again Networks: Self-Distillation
with EMA Teacher (Furlanello, Lipton, Tschiatschek, Prabhudesai,
Urbach 2018, arXiv:1805.04770).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306).
Maintains a shadow copy of the model
    `θ_teacher ← (1 − β) · θ_teacher + β · θ_student`
with `β=0.999` (paper default) and adds the per-step distillation
loss
    `L_distill = α · T² · KL(softmax(teacher/T) ‖ softmax(student/T))`
(`α=1.0`, `T=2.0`) on top of CE. The teacher forward is a
parameter-swap around `model(x)` under `torch.no_grad()` (no
separate module needed — dropout is 0.0 by default and RoPE
caches are deterministic from position ids).

Identity at step 0: the shadow is a clone of the live init ⇒
teacher forward == student forward ⇒ KL = 0 ⇒ total loss = CE
exactly. Byte-identical baseline path with `use_born_again=False`
(the default for `Tiny1M3MConfig`).

Transfer-risk: med. Validated at ≥100M (DistilBERT 66M, BERT-base
110M); the mechanism is scale-free (EMA teacher is well-defined
at any scale). At 0.94M with 92 steps the EMA is slow relative to
the run length, so the distillation signal is small but
consistent. See `autoresearch/ideas/132-born-again/idea.md`.
"""
import sys
from configs.llm_config import Tiny1M3MBornAgainConfig


class C(Tiny1M3MBornAgainConfig):
    pass


if __name__ == "__main__":
    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ]
    train_llm.main()
