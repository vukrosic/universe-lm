# Make Far-Apart Words Ignore Each Other (and Train Better)

**Vuk Rosić**
[airesearchmastery.com](https://airesearchmastery.com/)

*0.94M params · 3M tokens · ~5 min/run on one RTX 3060.*

**ALiBi** gives a transformer a sense of *how far apart* two words are. For every pair of words it subtracts a penalty from their attention score, and that penalty grows with the distance between them: `penalty = slope × distance`. Here *distance* is simply how many words sit between the two, and *slope* is a small fixed number (one per attention head) that sets how harsh the penalty is – a bigger slope means distance is punished harder. The further apart two words are, the more the model is nudged to ignore the link between them. It's already the best trick on this model – so instead of "does it work?", I took it apart with three questions: does the *shape* of the penalty matter, does its *starting strength* matter, and how much does it buy in the first place?

A note on the score: it's validation loss, and **lower is better**. Differences of ~0.15 are large; differences under ~0.02 are just run-to-run noise.

---

## Q1 · Does the *shape* of the penalty matter?

Three shapes of the same idea: **linear** (penalty grows in a straight line – plain ALiBi), **curved** (grows along an upward curve), **concave** (grows fast at first, then flattens off). Here's what each one looks like as distance grows:

![The three penalty shapes: linear is a straight line, curved bends upward, concave rises fast then flattens.](/figures/alibi/fig0_shapes.svg)

Now the real test – does the choice of shape change how well the model trains? Here is the validation loss over training:

![Three shapes train on top of each other; concave falls behind at the very end.](/figures/alibi/fig1_loss_curves.svg)

All three train almost identically; only at the very end does concave fall behind.

![Linear and curved tie; concave finishes behind.](/figures/alibi/fig2_bakeoff_bars.svg)

| shape | final score | verdict |
|---|---|---|
| curved (poly) | **6.256** | tied |
| linear (plain ALiBi) | **6.258** | tied |
| concave (kerple-log) | 6.315 | **worse** |

Each run lands at a slightly different score depending on its random starting point, so a single run can't be trusted. The fix is to compare shapes *on the same starting point* and look at the difference – that cancels out the luck:

![Paired: curved ≈ linear; concave conclusively worse.](/figures/alibi/fig4_paired.svg)

- **concave vs linear: +0.056** – about 10× bigger than the noise, so this is real. Concave is genuinely worse.
- **curved vs linear: −0.003** – well inside the noise. They're the same.

**Answer:** the shape barely matters. A plain straight line is as good as anything fancier, and the one shape that's different – the flattening one – is worse.

---

## Q2 · Where should the penalty's strength start?

Each attention head has its own penalty strength (its "slope"). I compared four ways to set them: **learn from zero** (start at nothing, let training find them – the default), **geometric frozen** (start from textbook values, never change), **geometric ×2 learnable**, and **geometric learnable** (start from textbook values, keep adjusting). I ran this one to 6 seeds to be sure.

![Seeding the slopes is much steadier; the average lead is real but small.](/figures/alibi/fig3_rq2_init.svg)

- **The real win is consistency, not a better score.** Starting from zero is the jumpiest: its results swing **0.053** between its best and worst run. Starting from textbook values swings only **0.017** – about a third as much. Seed the slopes and you can trust the first run.
- **The score lead is small and shaky:** on average textbook-start finishes ~0.02 lower, but compared run-for-run the gap is well within the noise – 4 of 6 runs favor it, 2 don't. Not a reliable win.

**Answer:** seeding the slopes makes training much steadier, but doesn't reliably lower the score. It's a consistency knob, not a performance knob.

---

## Q3 · How much does the penalty buy at all?

The cleanest test: run the model with **no position penalty at all** and compare. Does the whole mechanism even earn its place?

![Turning the penalty off costs the whole run: +0.155 worse, conclusive.](/figures/alibi/fig5_ablation.svg)

| setup | final score | verdict |
|---|---|---|
| with ALiBi (linear) | **6.258** | the recipe |
| no position bias | 6.413 | **+0.155 worse** |

The two start together, but the no-penalty run never catches up and the gap widens. The cost of removing it – **+0.155** – is about 8× the noise and shows up on every single run. This is by far the biggest effect here, dwarfing every tweak from Q1 and Q2.

**Answer:** ALiBi is load-bearing – the model genuinely needs it.

---

## Takeaway

- **Having the penalty (+0.155):** the single biggest lever. Remove it and the model gets much worse.
- **Its shape (≈0):** doesn't matter – a straight line ties any curve. Only the flattening shape is worse.
- **Its starting strength (small):** changes consistency, not score – runs get steadier, but don't reliably score lower.

In one line: *whether* you penalize distance matters a lot; *how* you shape or tune that penalty barely matters at all. Get the mechanism in place, and the fine details aren't worth chasing.

---

*Every claim is averaged across multiple random seeds; any difference under the model's ±0.02 run-to-run wobble is reported as a tie, not a win. Full protocol in `SEED_PROTOCOL.md`.*

---

## Credits & references

The "distance penalty" studied here is **ALiBi**, and the curved/concave variants come from the **KERPLE** family. Full credit to that prior work:

- **ALiBi** — Ofir Press, Noah A. Smith, Mike Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- **KERPLE** (the curved/concave shapes) — Ta-Chung Chi, Ting-Han Fan, Peter J. Ramadge, Alexander I. Rudnicky. *KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation.* NeurIPS 2022. [arXiv:2205.09921](https://arxiv.org/abs/2205.09921)
- **RoPE** (rotary positions, the other dominant scheme) — Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Transformer / attention** — Ashish Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

Experiments, figures, and writeup by **Vuk Rosić** · [airesearchmastery.com](https://airesearchmastery.com/)
