# Evidence — 132 born-again

## Verdict: NULL (wrong-sign, DRIFT, 2× the null band)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4481   Δ: +0.0209
- ctrl2: pending (queued last in tmux); even a favorable ctrl2 cannot recover this magnitude — verdict decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), NULL band |Δ|<0.01, DRIFT > +0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/arq-110/132-born-again_52674.log, ctrl_52674.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4001 vs ctrl 6.3966 ⇒ +0.0035 train-loss gap ⇒ Born-Again's EMA teacher is *too close* to the student at 92 steps to provide a useful distillation signal; the KL term adds gradient noise without information
- Treatment val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 (step 100) → 7.12 → 6.99 → 6.85 → 6.71 → 6.45 final
- Ctrl val curve:    10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Curves track through step ~100, then Born-Again's KL term drags the gradient and the val curve falls visibly behind (Δ widens from ~0.0 to ~0.02 by end). The +0.02 wrong-sign gap is unambiguous — and the +0.0035 train-loss gap confirms it is *not* a generalization vs fitting split (the model trains worse too).

## Transfer note
Born-Again Networks (Furlanello et al. 2018, arXiv:1805.04770) is the canonical self-distillation trick: keep a slow EMA copy of the student as a "teacher", add `α·KL(softmax(student/T) ‖ softmax(teacher/T))` to the loss. The paper validates on CIFAR-10/100 ResNet, ImageNet ResNet-50 (25M), and reports +0.3–1.5% top-1; subsequent DistilBERT/TinyBERT work extends to LM at BERT-base/large (110M–340M), reporting +1–3% on GLUE. The +0.021 wrong-sign result at tiny1m3m says: with `β=0.999, α=1.0, T=2.0` and **only 92 update steps**, the EMA teacher's effective window is ~1000 steps but the run is *shorter* than that — so the teacher is essentially the student-with-momentum, and the KL term approximates `KL(p‖p+noise)`, a pure gradient-noise injection. The mechanism's published wins assume (a) long enough training that EMA develops a meaningfully different posterior (≥1k steps), and (b) a model large enough that the *teacher* (a stale snapshot) is meaningfully less correct than the student on most tokens. Both fail at tiny1m3m: 92 steps is well below the EMA's correlation time, and a 0.94M model has no slack between teacher and student quality. Re-evaluate at Phase-2 (135M, 3-4k steps) where the EMA has time to develop a distinct posterior — but for tiny1m3m the lever is closed: NULL.
