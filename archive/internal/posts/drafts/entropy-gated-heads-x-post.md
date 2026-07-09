Gating attention heads by how confident they are does not help. If anything, rewarding the confident ones hurts.

I tried "entropy-gated heads": measure each head's attention entropy, then let low-entropy (confident) heads speak louder and high-entropy ones quieter.

Same model, same data, seed 42, tiny tier, 3M tokens on a V100.

baseline:            6.4216
attenuate (1-a*h):   6.4203 (tie, inside noise)
amplify confident:   6.4356 (worse, +0.0140)

The attenuate gate moved its parameter during training but landed inside variance. The amplify gate, which leans harder on confident heads, made things clearly worse.

So the confident heads are not the valuable ones. Turning them up costs you loss. Whatever the uncertain heads are doing, the model wants to keep it.

Negative result, but a clean one: head confidence is not a good signal for how much a head should contribute.

---

I am writing this with my Skool community, if you want to get mentorship and write your own mini AI research paper like this every week join our Skool (free trial below) - https://www.skool.com/become-ai-researcher-2669/about
