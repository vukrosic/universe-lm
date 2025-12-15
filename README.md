# Looking for a name for this LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install -r requirements.txt

python train_moe.py
```

## Hyperparameter Tuning

To optimize training for your specific hardware or architecture changes, you can use the quick sweep tool:

```bash
python experiments/quick_sweep.py
```

This script will:
1.  Train the model with 3 different learning rates (0.5x, 1.0x, 2.0x) for 200 steps each.
2.  Clean up large checkpoint files automatically, keeping only metrics.
3.  Generate a comparison plot (`sweep_comparison.png`) to help you choose the best learning rate.

You can then apply the best learning rate in `configs/moe_config.py`.


## Getting Started

1. **Fork this repository** - Click the "Fork" button at the top right of this page to create your own copy
2. Clone your fork: `git clone URL_HERE`
3. Install dependencies: `pip install -r requirements.txt`
4. Read `CONTRIBUTING.md` for contribution guidelines
5. Create your own experiment and merge it
6. Explore the `experiments/` folder for ongoing research and inspiration
7. Once you finish with your research, create a pull request to merge it back to this repo

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project