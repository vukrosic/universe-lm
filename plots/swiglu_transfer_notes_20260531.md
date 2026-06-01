# SwiGLU Transfer Notes - 2026-05-31

## Result

SwiGLU did not transfer cleanly to the longer scheduled run.

| Point | SwiGLU val loss | Baseline val loss | Delta |
| --- | ---: | ---: | ---: |
| 49.152M tokens / baseline step 12000 | 4.8159375 | 4.8243750 | -0.0084375 |
| 81.92M tokens / baseline step 20000 | 4.7791 | 4.6996875 | +0.0794125 |

Interpretation: SwiGLU looked slightly better around 49M tokens, then fell behind the scheduled 200M baseline curve by the next matched checkpoint.

## Follow-up

Do not continue scaling SwiGLU for now.

The apparent 5M `resid_scale05` and `embed_resid01` wins are not reliable transfer candidates in the current code, because the current model path does not consume those config fields. Treat them as dead-knob/noise until implemented properly.

Next valid transfer check: `batch4`, because batch size is actually consumed by the training loop and was the best remaining implemented 5M lever after SwiGLU.
