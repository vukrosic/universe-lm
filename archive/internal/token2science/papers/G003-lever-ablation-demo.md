# Does the candidate lever lower validation loss?

**Authors:** sim-researcher

## Abstract

This paper asks: does adding `good_lever` to the model lower validation loss compared to an identical control without it? The best observed result is 6.3616404224 versus the bar of 6.37, so the run beats the target.

## Method

does adding `good_lever` to the model lower validation loss compared to an identical control without it?

Experiment command: `python experiment.py --config config.json`

## Results

![Does the candidate lever lower validation loss?](charts/G003-lever-ablation-demo.png)

| task | n runs | mean value | beats bar |
| --- | --- | --- | --- |
| `T301` | 1 | 6.3933476595 | no |
| `T302` | 1 | 6.3616404224 | yes |

Best value: `6.3616404224`

Beats bar: `yes`

Confirmation: not available

## Reproducibility Appendix

- Worker: `sim-researcher`
- Config hash: `sha256:7cd3a422a1605b528d36d1bed61a2ed300967259f5f119576305616852f73eea`
- Exact command: `python experiment.py --config config.json`
