# Token2Science Testbench

This directory provides a mock GPU/results backend that behaves like a real
experiment runner, but is just a deterministic oracle.

## Determinism

The result is a pure function of:

- `config["seed"]`
- `config["levers"]`

The backend derives a stable noise seed from those inputs, then uses
`random.Random` to generate the Gaussian noise. That means the same config
always produces the same metric value, which keeps reproduce checks valid.

## Files

- `effects.json` stores the ground truth for the mock world.
- `mock_backend.py` exposes the oracle as both a CLI and an HTTP server.
- `experiment_mock.py` is a tiny experiment wrapper that reads `config.json`
  and prints the required `RESULT` line.

## Modes

### In-process or CLI

Run the oracle directly:

```bash
python testbench/mock_backend.py --config testbench/example_config.json
```

This prints one final line in the format:

```text
RESULT metric=val_loss value=...
```

### HTTP server

Start the mock backend as a local service:

```bash
python testbench/mock_backend.py --serve 8000
```

Send `POST /run` with a JSON body shaped like:

```json
{"config": {"seed": 42, "levers": ["good_lever"]}}
```

The response is JSON with:

- `metric`
- `value`
- `curve`

## Pointing experiments at it

For a simulated task, place a `config.json` next to `experiment_mock.py` with
at least:

```json
{"seed": 42, "levers": ["good_lever"]}
```

Then run:

```bash
python testbench/experiment_mock.py
```

That makes the task look like a normal experiment while staying fully
deterministic.
