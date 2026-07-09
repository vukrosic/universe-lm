#!/usr/bin/env python3
"""Local-only control dashboard for token2science."""

import html
import http.server
import json
import os
import socketserver
import subprocess


ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8899"))
BOARD_PATH = os.path.join(ROOT, "BOARD.md")
LEADERBOARD_PATH = os.path.join(ROOT, "LEADERBOARD.md")
REPUTATION_PATH = os.path.join(ROOT, "REPUTATION.md")
PERSONA_PATH = os.path.join(ROOT, "testbench", "contributor_sim", "persona_contributor.md")


ACTION_SPECS = {
    "Unit tests": {
        "kind": "commands",
        "commands": [["python3", "-m", "pytest", "tests/", "-q"]],
    },
    "Claim concurrency sim": {
        "kind": "commands",
        "commands": [["python3", "testbench/sim_claim.py", "--agents", "20", "--tasks", "5", "--rounds", "5"]],
    },
    "Mock backend determinism": {
        "kind": "mock_determinism",
    },
    "Runner: fill confirmations": {
        "kind": "commands",
        "commands": [["python3", "runner/runner.py", "--worker", "dash-runner", "--rounds", "1"]],
    },
    "Confirm T001": {
        "kind": "commands",
        "commands": [["python3", "verify/confirm.py", "--task", "T001", "--k", "2"]],
    },
    "Generate paper (G001)": {
        "kind": "commands",
        "commands": [["python3", "paper.py", "--goal", "G001-deterministic-demo", "--me", "dash-user"]],
    },
    "Refresh board+leaderboard+reputation": {
        "kind": "commands",
        "commands": [
            ["python3", "board.py"],
            ["python3", "leaderboard.py"],
            ["python3", "reputation.py"],
        ],
    },
    "Launch 1 simulated contributor": {
        "kind": "launch_contributor",
    },
}


def _read_text(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _escape_block(text):
    return html.escape(text, quote=False)


def _run_command(argv):
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        parts = []
        if exc.stdout:
            parts.append(str(exc.stdout).rstrip())
        if exc.stderr:
            parts.append(str(exc.stderr).rstrip())
        parts.append("[timeout after 180s]")
        return "\n".join(part for part in parts if part).strip() or "[timeout after 180s]"
    except OSError as exc:
        return f"[failed to run {' '.join(argv)}: {exc}]"

    parts = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(proc.stderr.rstrip())
    if proc.returncode != 0:
        parts.append(f"[exit code {proc.returncode}]")
    return "\n".join(part for part in parts if part).strip() or "(no output)"


def _mock_backend_determinism():
    cmd = ["python3", "testbench/mock_backend.py", "--config", "testbench/example_config.json"]
    result_lines = []
    for _ in range(2):
        output = _run_command(cmd)
        result_line = None
        for line in output.splitlines():
            if line.startswith("RESULT"):
                result_line = line.strip()
        result_lines.append(result_line or output.strip() or "(no output)")
    match = "yes" if result_lines[0] == result_lines[1] else "no"
    return "\n".join(
        [
            f"Run 1: {result_lines[0]}",
            f"Run 2: {result_lines[1]}",
            f"Match: {match}",
        ]
    )


def _launch_simulated_contributor():
    persona = _read_text(PERSONA_PATH)
    if persona is None:
        return "[missing persona_contributor.md]"
    prompt = persona.rstrip() + "\n\nYour handle: sim-user-dash.\n"
    env = os.environ.copy()
    env["T2S_SIM_MAX"] = "1"
    try:
        subprocess.Popen(
            [
                "bash",
                "testbench/contributor_sim/capped_launch.sh",
                "dash-contrib",
                prompt,
                "600",
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return f"[failed to launch contributor sim: {exc}]"
    return (
        "Launched tmux session simu-dash-contrib.\n"
        "Watch it with: tmux attach -t simu-dash-contrib"
    )


def run_action(action):
    spec = ACTION_SPECS.get(action)
    if spec is None:
        return False, f"unknown action: {action}"

    kind = spec["kind"]
    if kind == "commands":
        outputs = []
        for argv in spec["commands"]:
            outputs.append(_run_command(argv))
        return True, "\n\n".join(outputs)
    if kind == "mock_determinism":
        return True, _mock_backend_determinism()
    if kind == "launch_contributor":
        return True, _launch_simulated_contributor()
    return False, f"unsupported action kind: {kind}"


def _render_file_section(title, path):
    content = _read_text(path)
    if content is None:
        body = "(not generated yet)"
    else:
        body = content.rstrip() or "(not generated yet)"
    return (
        f"<section class=\"doc\"><h2>{html.escape(title)}</h2>"
        f"<pre>{_escape_block(body)}</pre></section>"
    )


def _render_page():
    buttons = []
    for action in ACTION_SPECS:
        safe = html.escape(action, quote=True)
        buttons.append(f'<button class="action" data-action="{safe}">{safe}</button>')

    sections = "".join(
        [
            _render_file_section("BOARD.md", BOARD_PATH),
            _render_file_section("LEADERBOARD.md", LEADERBOARD_PATH),
            _render_file_section("REPUTATION.md", REPUTATION_PATH),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>token2science control</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #121a2d;
      --panel-2: #0f1628;
      --text: #e5eefb;
      --muted: #98a6c7;
      --border: #26324d;
      --accent: #7cc4ff;
      --accent-2: #a1ffce;
      --danger: #ff8f8f;
    }}
    body {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      background:
        radial-gradient(circle at top left, rgba(124, 196, 255, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(161, 255, 206, 0.10), transparent 22%),
        linear-gradient(180deg, #09101f 0%, #0b1020 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
      letter-spacing: 0.02em;
    }}
    .note {{
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }}
    button.action {{
      appearance: none;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, #18233a, #121b2d);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }}
    button.action:hover:not(:disabled) {{
      transform: translateY(-1px);
      border-color: var(--accent);
    }}
    button.action:disabled {{
      cursor: wait;
      opacity: 0.7;
    }}
    .panel {{
      border: 1px solid var(--border);
      background: rgba(10, 16, 28, 0.88);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.25);
    }}
    #output {{
      min-height: 150px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0 0 18px 0;
    }}
    .docs {{
      display: grid;
      gap: 16px;
    }}
    .doc h2 {{
      font-size: 14px;
      margin: 0 0 8px 0;
      color: var(--accent-2);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .doc pre {{
      margin: 0;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      line-height: 1.45;
    }}
    .footer {{
      color: var(--muted);
      margin-top: 18px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>token2science control</h1>
    <div class="note">local only (127.0.0.1)</div>
    <div class="toolbar">
      {''.join(buttons)}
    </div>
    <pre id="output" class="panel">ready.</pre>
    <div class="docs">
      {sections}
    </div>
    <div class="footer">POST /run accepts only the fixed whitelist below.</div>
  </div>
  <script>
    const output = document.getElementById('output');
    const buttons = Array.from(document.querySelectorAll('button[data-action]'));

    async function runAction(button) {{
      const action = button.dataset.action;
      const original = button.textContent;
      button.disabled = true;
      button.textContent = 'running...';
      output.textContent = `running: ${{action}}`;
      try {{
        const response = await fetch('/run', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ action }}),
        }});
        let payload = null;
        try {{
          payload = await response.json();
        }} catch (err) {{
          payload = {{ ok: false, output: `invalid JSON response: ${{err}}` }};
        }}
        output.textContent = payload && typeof payload.output === 'string'
          ? payload.output
          : '(no output)';
      }} catch (err) {{
        output.textContent = `request failed: ${{err}}`;
      }} finally {{
        button.disabled = false;
        button.textContent = original;
      }}
    }}

    for (const button of buttons) {{
      button.addEventListener('click', () => runAction(button));
    }}
  </script>
</body>
</html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path != "/":
            self.send_error(404)
            return
        body = _render_page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path != "/run":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
            action = body["action"]
            if not isinstance(action, str):
                raise TypeError("action must be a string")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError):
            self._send_json(400, {"ok": False, "output": "invalid request body"})
            return

        ok, output = run_action(action)
        self._send_json(200 if ok else 400, {"ok": ok, "output": output})

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003
        return


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    if len(ACTION_SPECS) != 8:
        raise SystemExit(f"expected 8 actions, found {len(ACTION_SPECS)}")
    server = _Server(("127.0.0.1", PORT), _Handler)
    print(f"http://127.0.0.1:{PORT}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
