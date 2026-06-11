# X thread draft (post after md is confirmed)

```text
1/

I screened 8 LLM training ideas at 1M params on a single rented 3060.
2 wins, 4 nulls, 1 clear negative.

Headline: FIRE positional encoding.

val 6.3234 vs ctrls 6.3875 / 6.4050  (Δ −0.064 / −0.082)
```

```text
2/

Protocol — one seed (42), two ctrls run back-to-back for variance.
A win has to beat BOTH ctrls by more than |ctrl_a − ctrl_b|.

Cheap. Honest about what 1 GPU can pay for.
~2 min per A/B. ~$0.20 for the whole table.
```

```text
3/

The full table (Δ = treatment − mean(ctrls), session-local):

009 FIRE pos-enc          −0.0728  WIN
011 Cautious-Lion         −0.0316  WIN
005 Decoupled-QKV Muon    −0.0053  NULL
010 PolyLoss (eps=1)      -0.0083  NULL
001 Cautious-Muon         +0.0162  NULL
004 RetNet retention      +0.0199  NULL
006 Schedule-Free AdamW   +0.2034  CLEAR NEG
```

```text
4/

Nulls and the negative are in the chart on purpose.

If you only ever publish the wins, you can't tell whether I tested 1 idea
or 100. The denominator is what makes a screen a screen.

Schedule-Free AdamW (AlgoPerf 2024 winner) is +0.20 at this scale. Real.
```

```text
5/

FIRE: replaces RoPE's pure rotation with bias(i,j) = γ(i−j) · f(φ(x_i), φ(x_j)).
γ is a fixed distance-decay kernel; φ is a small learned MLP per token.
Position kernel fixed, content can re-weight it.

Drop-in for RoPE. ~30-50 LoC.
```

```text
6/

Open question: FIRE's headline upside in the paper is length extrapolation.
tiny1m3m is fixed-length (T=2048) so I haven't tested that here.

Does the val win survive at longer context? That's the next experiment.

(Day of the week: this is one of the weekly screens.)

---

I am writing this with my Skool community, if you want to get mentorship and write your own mini AI research paper like this every week join our Skool (free trial below) - https://www.skool.com/become-ai-researcher-2669/about
```
