# Taste log — 063 YaRN

## r1 — 2026-06-11 — verdict: reject
- Off-niche on mechanism vs tier: YaRN is a **context-extension** method — it rescales RoPE frequencies so phases trained at length L stay coherent at inference length L' > L. Our pipeline trains AND evaluates at fixed `max_seq_len=2048` (configs/llm_config.py: "do not change; matches the downloaded data"); there is no L' > L regime, so the YaRN frequency-rescaling lever never fires. Idea.md's "Why it's worth a slot" sentence ("YaRN tests whether a smarter RoPE stretch beats plain interpolation at tiny1m3m") misframes the bet — at constant train/eval length, YaRN's stretch is identity-ish vs vanilla RoPE.
- Info value: both outcomes are uninformative. A null = "no extrapolation regime to stretch into, as expected"; a win at tiny1m3m would be a measurement artifact, not transferable signal toward the 135M recipe (which also doesn't push beyond its trained context in the screen).
- Crowded family: 061-alibi-bias, 062-pos-interp, 063-yarn, 064-xpos, 065-bilevel-pe — five PE/length-extrapolation ideas back-to-back. Portfolio is already saturated with the same family, and the relevant axis (`RoPE base sweep — 500k winner`) is already in `closed.md` "Closed axes".
- Re-pitching cannot save it: the mechanism is definitionally about length-extension; there is no in-niche reframe at fixed 2048. Reject (not revise).
