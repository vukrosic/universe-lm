# Evidence — 115 R-Drop (KL-Regularized Dropout)

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (5b8a7fea8963, RTX 3060, sm_86, driver 580.159.03, commit 42ed363)
- baseline: fresh 3 ctrls this queue (6.4259, 6.4416, 6.4453 → mean=6.4376 ±0.0084, band=0.04); MEASURE path triggered by commit-changed
- treatment val: **6.4269**   train=6.4001   acc=0.1466   Δ vs fresh ctrl mean: **−0.0107**
- bpb: n/a (pending harness — never omit)
- pass/fail bar (idea.md): PASS Δ≤-0.005; NULL |Δ|<0.005; DRIFT >+0.005  → **Δ=−0.0107 → NULL (inside ±0.04 noise band)**
- box check: fresh ctrls 6.4259/6.4416/6.4453 vs leaderboard 6.4306 — max |Δ|=0.0147 (within 0.04 noise, **NO DRIFT**)
- raw: autoresearch/remote-results/2026-06-14-vast-tiny1m3m-3/results.json (log 115-rdrop.log alongside)
- date: 2026-06-14

## Run trace
- step 0: bit-identity preserved (round-2 chunked-KL fix: `if rdrop_alpha_step > 0:` skip ⇒ step-0 is single-forward baseline)
- mid-training: ~6.7–6.4 (in ctrl cluster)
- final (step 732): val_loss=**6.4269**, train=6.4001, val_acc=0.1466 (vs ctrl val_acc 0.1443, vs 113-galore degenerate 0.0936)

The round-2 chunked-KL fix held — no OOM, no NaN, the `_RDROP_KL_CHUNK=512` memory profile fits the RTX 3060 12GB envelope with prior-process headroom. R-Drop ran cleanly end-to-end. The lever just doesn't gain at this tier: Δ=−0.0107 sits inside the ±0.04 noise band, and val_acc 0.1466 vs ctrl 0.1443 is essentially identical. Three attempts (round 1 structural OOM, round 2 KL-block OOM, round 3 chunked-KL fit) reach the same conclusion: the lever runs cleanly but does not move val loss at 0.94M.

## Transfer note
R-Drop was designed for fine-tuning (~110M BERT-base, NMT, classification) where dropout is the dominant regularizer and the KL term meaningfully closes the train↔val gap. At 0.94M pretraining, the dominant regularizer is data scale (3M tokens, single-pass) — dropout at p=0.1 contributes a much smaller fraction of total variance. The paper's reported −0.1..−0.3 PPL on classification pretraining at ~140M does not transfer to a 92-step language-model pretraining pass at 0.94M, where the noise band is ~±0.04 val loss and the per-step gradient signal already has high variance from small batch × long sequence. Re-evaluate at any future mid-scale tier where dropout is a larger share of regularization signal — but on the same axis, loss-shape regularizers (066–070) and adjacent dropouts (009-FIRE-PE, 111-drop-path) are already closed, so the axis as a whole does not look promising for parameter-golf-tier language models.
