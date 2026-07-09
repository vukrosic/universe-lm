#!/usr/bin/env python3
"""Deterministic oracle for token2science test runs.

By default this is byte-for-byte deterministic for a given config. If the
`T2S_MOCK_NOISE` environment variable is set to a positive float, the oracle
adds an extra non-seeded Gaussian jitter with that standard deviation on top
of the deterministic value. This is meant to simulate GPU-style run-to-run
noise without changing the default reproduce behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import random
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EFFECTS_PATH = HERE / "effects.json"


def _load_effects() -> dict[str, Any]:
    with EFFECTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _noise_seed(config: dict[str, Any]) -> int:
    seed = config.get("seed")
    levers = sorted(config.get("levers", []))
    payload = json.dumps([seed, levers], separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def simulate(config: dict[str, Any]) -> dict[str, Any]:
    effects = _load_effects()
    lever_effects = effects["levers"]
    levers = config.get("levers", [])
    value = float(effects["base"])
    value += sum(float(lever_effects.get(name, 0.0)) for name in levers)

    rng = random.Random(_noise_seed(config))
    noise = rng.gauss(0.0, float(effects["noise_sd"]))
    value += noise

    mock_noise_sd = float(os.getenv("T2S_MOCK_NOISE", "0.0"))
    if mock_noise_sd > 0.0:
        # Use an unseeded generator so identical configs can land on slightly
        # different values across runs, like a noisy GPU kernel.
        nondeterministic_rng = random.Random()
        value += nondeterministic_rng.gauss(0.0, mock_noise_sd)

    start = value + 0.30
    curve = [start, value + 0.15, value + 0.07, value + 0.03, value]

    return {
        "metric": effects["metric"],
        "value": value,
        "curve": curve,
    }


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _handle_cli(args: argparse.Namespace) -> None:
    result = simulate(_load_config(args.config))
    print(f"RESULT metric={result['metric']} value={result['value']:.10f}")


class _RunHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
            result = simulate(body["config"])
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            self.send_error(400, "invalid request body")
            return

        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def _serve(port: int) -> None:
    server = http.server.HTTPServer(("127.0.0.1", port), _RunHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--serve", type=int)
    args = parser.parse_args()

    if args.serve is not None:
        _serve(args.serve)
        return
    if not args.config:
        parser.error("--config is required unless --serve is used")
    _handle_cli(args)


if __name__ == "__main__":
    main()
