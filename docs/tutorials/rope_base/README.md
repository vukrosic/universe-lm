Sweep the RoPE base (for faster LLM / Transformer training)

Many people just set 10,000 as the RoPE base - you should instead tell your codex / claude to sweep it.

The curve says it.

![RoPE base U-curve](images/hero_ucurve.png)

- short context / small model -> try lower bases: 10k, 25k, 50k, 100k
- long context / larger model -> try higher bases: 100k, 250k, 500k, 750k

RoPE base is one integer. It adds no parameters, changes no tensor shapes, and can still move validation loss.

Why was `10k` common? It came from the original sinusoidal/rotary convention and worked well enough for many short-context setups. It is a historical default, not a measured optimum for every model size and context length.

---

My tiny sweep: ~1M params, 3M tokens:
| RoPE base | Val loss |
| 125k | 6.3650 |
| 250k | 6.3506 | (best)
| 375k | 6.3656 |
| 500k | 6.3694 |
| 750k | 6.3769 |

You can our my LLM research kit, just tell your codex / claude:

Clone this repo and setup it up: https://github.com/vukrosic/universe-lm

If you want to learn to do AI research join our Skool: https://www.skool.com/become-ai-researcher-2669/about (funds our research)