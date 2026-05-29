# Goals

## First milestone: beat SmolLM2-135M

Train and release a ~135M model that beats [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) on its own benchmark suite. SmolLM2-135M was trained on **2T tokens** — our efficiency angle is to match or beat it on fewer tokens / less compute.

### Benchmarks to beat (SmolLM2-135M base scores)

| Benchmark | SmolLM2-135M | What it tests |
|---|---|---|
| HellaSwag | 42.1 | commonsense / sentence completion |
| ARC (avg) | 43.9 | grade-school science reasoning |
| PIQA | 68.4 | physical commonsense |
| CommonsenseQA | 33.9 | commonsense QA |
| Winogrande | 51.3 | coreference / commonsense |
| OpenBookQA | 34.6 | open-book science QA |
| MMLU (cloze) | 31.5 | broad world knowledge |
| TriviaQA | 4.1 | factual recall |
| GSM8K (5-shot) | 1.4 | grade-school math |

Primary targets: HellaSwag, ARC, PIQA, Winogrande (where a 135M model can realistically move the needle). MMLU/TriviaQA/GSM8K are stretch.

## What a 135M model is good for

These are the real use cases — what we release the model *for*, mirroring how SmolLM is used:

- **On-device / edge** — runs on phone, laptop, CPU (~300MB). No cloud, private by default.
- **Text rewriting & summarization** — short-form, streaming, low latency.
- **Classification & extraction** — intent, sentiment, tagging.
- **Autocomplete** — local short-form generation.
- **Function calling** — lightweight tool/routing layer.

Sources: [SmolLM blog](https://huggingface.co/blog/smollm), [SmolLM2-135M card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
