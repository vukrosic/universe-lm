#!/usr/bin/env python3
"""W&B-style GPU monitor for autoresearch — runs LOCALLY, polls box over SSH.

Stdlib only. No CDN, no external deps. Single-file.

What it shows:
  - GPU util / memory / temp / power (live from `nvidia-smi` on vast)
  - Current arq job (parsed from /root/arq/STATUS) + step / loss / tok-s / ETA
  - Val-loss line chart (parsed from the active job's tqdm stream)
  - Last 30 arq STATUS events (queue timeline)

Run:
    python3 tools/gpu_dashboard.py                  → http://localhost:8081
    PORT=8082 python3 tools/gpu_dashboard.py
    SSH_PORT=22179 python3 tools/gpu_dashboard.py
    SSH_HOST=root@box.example.com python3 tools/gpu_dashboard.py
"""

import http.server
import json
import os
import re
import socketserver
import subprocess
import time
from urllib.parse import urlparse

PORT = int(os.environ.get("PORT", "8081"))
SSH_HOST = os.environ.get("SSH_HOST", "root@81.45.65.189")
SSH_PORT = os.environ.get("SSH_PORT", "22179")
ARQ_DIR = "/root/arq"
SSH_TIMEOUT = 8

# --- remote helpers ---------------------------------------------------------


def _ssh(cmd, timeout=SSH_TIMEOUT):
    try:
        out = subprocess.run(
            ["ssh",
             "-p", SSH_PORT,
             "-o", "ConnectTimeout=2",
             "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes",
             SSH_HOST, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.returncode, out.stdout, out.stderr
    except (subprocess.TimeoutExpired, OSError) as e:
        return -1, "", f"ssh: {e}"


# tqdm line shapes (single line, carriage returns make multi-event lines):
#   Training:  41%|████      | 1236992/3000000 [01:39<11:25, 2571.25tokens/s, step=300/732, loss=6.6168, acc=0.123, ent=+0.00e+00, cp=+0.00e+00, pl=+0.00e+00, lr=0.01441]
_STEP_RE = re.compile(
    r"step=(\d+)/(\d+),\s*loss=([\d\.Na+n]+),\s*acc=([\d\.]+),"
    r"\s*ent=\S+,\s*cp=\S+,\s*pl=\S+,\s*lr=([\d\.eE+-]+)"
)
_TOKS_RE = re.compile(r"([\d\.]+)tokens/s")
_ETA_RE = re.compile(r"<([\d:]+),")
# Step XX: Val Loss: 6.8991, Val Acc: 0.1089, Val PPL: 991.34, LR: 0.01775
_VAL_RE = re.compile(
    r"^Step (\d+): Val Loss: ([\d\.]+), Val Acc: ([\d\.]+), Val PPL: ([\d\.]+), LR: ([\d\.eE+-]+)$"
)
# Final Val Loss:                  6.3419
# Final Val Accuracy:              0.1511
_FINAL_LOSS_RE = re.compile(r"^Final Val Loss:\s+([\d\.Na+n]+)\s*$")
_FINAL_ACC_RE = re.compile(r"^Final Val Accuracy:\s+([\d\.]+)\s*$")


def _parse_log(text):
    cur = None
    series = []
    final_loss = None
    final_acc = None
    for line in text.splitlines()[-2500:]:
        m = _STEP_RE.search(line)
        if m:
            cur = {
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": m.group(3),
                "acc": float(m.group(4)),
                "lr": float(m.group(5)),
            }
        m = _TOKS_RE.search(line)
        if m and cur is not None and "tok_s" not in cur:
            cur["tok_s"] = float(m.group(1))
        m = _ETA_RE.search(line)
        if m and cur is not None and "eta" not in cur:
            cur["eta"] = m.group(1)
        m = _VAL_RE.match(line.strip())
        if m:
            series.append({
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "acc": float(m.group(3)),
                "ppl": float(m.group(4)),
            })
        m = _FINAL_LOSS_RE.match(line.strip())
        if m:
            final_loss = m.group(1)
        m = _FINAL_ACC_RE.match(line.strip())
        if m:
            final_acc = float(m.group(1))
    if cur and final_loss is not None:
        cur["final_loss"] = final_loss
        cur["final_acc"] = final_acc
    return cur, series[-50:]


def _gather():
    out = {"ts": time.time(), "ssh_host": f"{SSH_HOST}:{SSH_PORT}"}

    # 1) GPU
    rc, txt, err = _ssh(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits"
    )
    if rc == 0 and txt.strip():
        idx, name, util, mem_u, mem_t, temp, pwr, pwr_m = [
            x.strip() for x in txt.strip().splitlines()[0].split(",")
        ]
        util_i, mem_u_i, mem_t_i, temp_i = int(util), int(mem_u), int(mem_t), int(temp)
        pwr_f, pwr_m_f = float(pwr), float(pwr_m)
        out["gpu"] = {
            "up": True,
            "index": int(idx),
            "name": name,
            "util": util_i,
            "mem_used_mib": mem_u_i,
            "mem_total_mib": mem_t_i,
            "mem_pct": round(100.0 * mem_u_i / mem_t_i, 1),
            "temp_c": temp_i,
            "power_w": round(pwr_f, 1),
            "power_max_w": round(pwr_m_f, 1),
            "power_pct": round(100.0 * pwr_f / pwr_m_f, 1),
        }
    else:
        out["gpu"] = {"up": False, "error": err.strip() or "no nvidia-smi"}

    # 2) arq STATUS
    rc, txt, _ = _ssh(f"cat {ARQ_DIR}/STATUS 2>/dev/null")
    history = []
    running = None
    if rc == 0 and txt.strip():
        for ln in txt.strip().splitlines():
            parts = ln.split(None, 2)
            if len(parts) < 3:
                continue
            kind, name, ts = parts
            history.append({"state": kind, "job": name, "ts": ts})
        if history and history[-1]["state"] == "START":
            last = history[-1]
            running = {"job": last["job"], "started": last["ts"], "ts": last["ts"]}
    out["history"] = history[-30:]
    out["running"] = running

    # 3) Active log (val loss)
    out["current"] = None
    out["series"] = []
    if running:
        rc, txt, _ = _ssh(f"cat {ARQ_DIR}/logs/{running['job']}.log 2>/dev/null | tail -2500")
        if rc == 0 and txt:
            cur, series = _parse_log(txt)
            out["current"] = cur
            out["series"] = series

    return out


# --- HTML -------------------------------------------------------------------

INDEX = """<!doctype html>
<html lang=en><head>
<meta charset=utf-8>
<title>autoresearch · GPU monitor</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
html,body{height:100%;margin:0;background:#0b0e14;color:#e6edf3;
  font:13px -apple-system,BlinkMacSystemFont,'SF Pro Text',sans-serif}
body{display:flex;flex-direction:column;padding:12px;gap:10px;height:100vh}
.hdr{display:flex;align-items:baseline;gap:14px;flex:0 0 auto}
.hdr h1{margin:0;font-size:14px;font-weight:600;letter-spacing:.2px}
.hdr h1 small{color:#6e7681;font-weight:400;margin-left:6px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot.ok{background:#3fb950;box-shadow:0 0 6px #3fb950}
.dot.stale{background:#d29922}
.dot.down{background:#f85149}
#ts{color:#6e7681;font-size:11px;margin-left:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.row{display:flex;gap:10px;flex:0 0 auto}
.row.grow{flex:1 1 auto;min-height:0}
.card{background:#13171f;border:1px solid #1f2630;border-radius:8px;padding:12px 14px;flex:1 1 0;min-width:0;display:flex;flex-direction:column;min-height:0}
.card h2{margin:0 0 8px;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;color:#6e7681;font-weight:600}
.metric{font:600 24px ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6edf3;line-height:1}
.metric small{font-size:11px;color:#6e7681;font-weight:400;margin-left:4px}
.bar{position:relative;height:5px;background:#1f2630;border-radius:3px;margin-top:8px;overflow:hidden}
.bar > i{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#3fb950 0%,#58a6ff 60%,#d29922 90%);border-radius:3px;transition:width .5s}
.sub{color:#6e7681;font-size:10.5px;margin-top:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.jname{font:600 14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#58a6ff}
.jname .pill{display:inline-block;background:#1f2630;color:#3fb950;padding:1px 7px;border-radius:9px;font-size:10px;margin-left:6px;text-transform:uppercase;letter-spacing:.4px;font-weight:600}
.jname .pill.idle{background:#1f2630;color:#6e7681}
.jname .pill.done{background:#1f2630;color:#d29922}
.grid2{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 14px;margin-top:10px}
.kv{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;color:#8b949e}
.kv b{color:#e6edf3;font-weight:600}
.kv.bad b{color:#f85149}
.chart{flex:1 1 auto;width:100%;min-height:0}
.chart .ax{stroke:#1f2630;stroke-width:1}
.chart .ln{fill:none;stroke:#58a6ff;stroke-width:1.6}
.chart .pt{fill:#58a6ff;stroke:#0b0e14;stroke-width:1}
.chart .gl{stroke:#1f2630;stroke-dasharray:2 3}
.chart .lb{font:9.5px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#6e7681}
.qtable{width:100%;border-collapse:collapse;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;flex:1 1 auto;display:block;overflow:auto}
.qtable thead{position:sticky;top:0;background:#13171f}
.qtable th{color:#6e7681;text-align:left;font-weight:600;padding:4px 8px;border-bottom:1px solid #1f2630;font-size:10px;text-transform:uppercase;letter-spacing:.4px}
.qtable td{padding:3px 8px;border-bottom:1px solid #1f2630}
.qtable tr.START td{color:#58a6ff}
.qtable tr.OK td{color:#3fb950}
.qtable tr.FAIL td{color:#f85149}
.qtable tr.QUEUE_DONE td{color:#d29922}
.warn{color:#d29922}
.bad{color:#f85149}
.muted{color:#6e7681}
.gpu-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
@media (max-width: 900px){.gpu-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class=hdr>
  <h1>autoresearch · GPU <small id=host></small></h1>
  <span id=st><span class="dot ok"></span> <span id=sttxt>live</span></span>
  <span id=ts></span>
</div>
<div class=row>
  <div class=card style="flex:0 0 24%">
    <h2>GPU util</h2>
    <div class=metric id=util>—<small>%</small></div>
    <div class=bar><i id=utilb style="width:0%"></i></div>
    <div class=sub id=utilsub>—</div>
  </div>
  <div class=card style="flex:0 0 24%">
    <h2>Memory</h2>
    <div class=metric id=mem>—<small>GiB</small></div>
    <div class=bar><i id=memb style="width:0%"></i></div>
    <div class=sub id=memsub>—</div>
  </div>
  <div class=card style="flex:0 0 24%">
    <h2>Temp / power</h2>
    <div class=metric id=temp>—<small>°C</small></div>
    <div class=bar><i id=tempb style="width:0%"></i></div>
    <div class=sub id=pwr>—</div>
  </div>
  <div class=card style="flex:1 1 auto">
    <h2>Current job</h2>
    <div class=jname id=jname>idle <span class="pill idle" id=jstatus>—</span></div>
    <div class=grid2>
      <div class=kv>step: <b id=jstep>—</b></div>
      <div class=kv>tok/s: <b id=jtoks>—</b></div>
      <div class=kv>loss: <b id=jloss>—</b></div>
      <div class=kv>acc: <b id=jacc>—</b></div>
      <div class=kv>lr: <b id=jlr>—</b></div>
      <div class=kv>eta: <b id=jeta>—</b></div>
    </div>
  </div>
</div>
<div class="row grow">
  <div class=card style="flex:1 1 60%">
    <h2>Val loss (live)</h2>
    <svg class=chart id=chart preserveAspectRatio="none"></svg>
    <div class=sub id=chartsub>—</div>
  </div>
  <div class=card style="flex:0 0 38%">
    <h2>arq queue</h2>
    <table class=qtable id=qtab>
      <thead><tr><th>state</th><th>job</th><th>ts</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<script>
const REFRESH_MS = 4000;
const hostEl = document.getElementById('host');
hostEl.textContent = '· ssh ' + (window.SSH_HOST || '');

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toISOString().replace('T', ' ').replace(/\\..*/, 'Z');
}
function setMetric(id, val, unit) {
  document.getElementById(id).innerHTML = val + (unit != null ? `<small>${unit}</small>` : '');
}
function setBar(id, pct, color) {
  const b = document.getElementById(id);
  b.style.width = Math.max(0, Math.min(100, pct)) + '%';
  if (color) b.style.background = color;
}
function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function isNum(x){ return typeof x === 'number' && isFinite(x); }

function render(data) {
  document.getElementById('ts').textContent = fmtTime(data.ts);
  const st = document.getElementById('st');
  const sttxt = document.getElementById('sttxt');
  const g = data.gpu || {};
  if (g.up) {
    st.firstElementChild.className = 'dot ok';
    sttxt.textContent = 'live';
    setMetric('util', g.util, '%');
    setBar('utilb', g.util);
    document.getElementById('utilsub').textContent = g.name;
    setMetric('mem', (g.mem_used_mib / 1024).toFixed(1), 'GiB');
    setBar('memb', g.mem_pct);
    document.getElementById('memsub').textContent = `${g.mem_used_mib} / ${g.mem_total_mib} MiB`;
    setMetric('temp', g.temp_c, '°C');
    setBar('tempb', Math.min(100, g.temp_c), g.temp_c > 80 ? 'linear-gradient(90deg,#3fb950,#d29922,#f85149)' : null);
    document.getElementById('pwr').textContent = `${g.power_w} / ${g.power_max_w} W (${g.power_pct}%)`;
  } else {
    st.firstElementChild.className = 'dot down';
    sttxt.textContent = 'GPU down';
    setMetric('util', '—', '%'); setBar('utilb', 0);
    setMetric('mem', '—', 'GiB'); setBar('memb', 0);
    setMetric('temp', '—', '°C'); setBar('tempb', 0);
    document.getElementById('utilsub').textContent = '';
    document.getElementById('memsub').textContent = '';
    document.getElementById('pwr').textContent = g.error || 'no nvidia-smi';
  }
  const r = data.running;
  const c = data.current || {};
  if (r) {
    document.getElementById('jname').innerHTML = esc(r.job) + ' <span class="pill">running</span>';
    document.getElementById('jstep').textContent = c.step != null ? `${c.step} / ${c.total || '?'}` : '—';
    document.getElementById('jtoks').textContent = c.tok_s != null ? Math.round(c.tok_s).toLocaleString() : '—';
    const lossEl = document.getElementById('jloss');
    lossEl.textContent = c.loss != null ? c.loss : '—';
    lossEl.parentElement.classList.toggle('bad', c.loss === 'nan' || c.loss === 'NaN');
    document.getElementById('jacc').textContent = c.acc != null ? c.acc.toFixed(4) : '—';
    document.getElementById('jlr').textContent = c.lr != null ? c.lr.toExponential(2) : '—';
    document.getElementById('jeta').textContent = c.eta || '—';
  } else {
    document.getElementById('jname').innerHTML = 'idle <span class="pill idle">queue empty</span>';
    ['jstep','jtoks','jloss','jacc','jlr','jeta'].forEach(id => document.getElementById(id).textContent = '—');
  }
  drawChart(data.series || []);
  const tbody = document.querySelector('#qtab tbody');
  tbody.innerHTML = '';
  (data.history || []).slice().reverse().forEach(e => {
    const tr = document.createElement('tr');
    tr.className = e.state;
    tr.innerHTML = `<td>${esc(e.state)}</td><td>${esc(e.job)}</td><td class=muted>${esc(e.ts)}</td>`;
    tbody.appendChild(tr);
  });
}

function drawChart(series) {
  const svg = document.getElementById('chart');
  const sub = document.getElementById('chartsub');
  if (!series.length) {
    svg.innerHTML = '<text x=20 y=90 class=lb>no val samples yet</text>';
    sub.textContent = '—';
    return;
  }
  const W = svg.clientWidth || 600, H = svg.clientHeight || 240;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const M = {l: 38, r: 10, t: 8, b: 22};
  const xs = series.map(s => s.step);
  const ys = series.map(s => s.loss);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys) - 0.05, ymax = Math.max(...ys) + 0.05;
  const xn = v => M.l + (W - M.l - M.r) * (xmax === xmin ? 0.5 : (v - xmin) / (xmax - xmin));
  const yn = v => H - M.b - (H - M.t - M.b) * (v - ymin) / (ymax - ymin);
  const pts = series.map(s => `${xn(s.step)},${yn(s.loss)}`).join(' ');
  let h = '';
  for (let i = 0; i <= 4; i++) {
    const yv = ymin + (ymax - ymin) * i / 4;
    const yp = yn(yv);
    h += `<line class=gl x1=${M.l} y1=${yp} x2=${W - M.r} y2=${yp}/>`;
    h += `<text class=lb x=2 y=${yp + 3}>${yv.toFixed(3)}</text>`;
  }
  h += `<line class=ax x1=${M.l} y1=${H - M.b} x2=${W - M.r} y2=${H - M.b}/>`;
  h += `<line class=ax x1=${M.l} y1=${M.t} x2=${M.l} y2=${H - M.b}/>`;
  const ix = [0, Math.floor(series.length/2), series.length-1];
  for (const i of ix) {
    const s = series[i]; if (!s) continue;
    h += `<text class=lb x=${xn(s.step)} y=${H - 4} text-anchor=middle>step ${s.step}</text>`;
  }
  h += `<polyline class=ln points="${pts}"/>`;
  const last = series[series.length - 1];
  h += `<circle class=pt cx=${xn(last.step)} cy=${yn(last.loss)} r=3/>`;
  svg.innerHTML = h;
  sub.textContent = `${series.length} eval pts · step ${last.step} · val ${last.loss.toFixed(4)} · acc ${last.acc.toFixed(4)} · ppl ${last.ppl.toFixed(2)}`;
}

let lastOk = 0;
async function tick() {
  try {
    const r = await fetch('/api/data', {cache: 'no-store'});
    const data = await r.json();
    render(data);
    lastOk = Date.now();
    document.getElementById('st').firstElementChild.className = 'dot ok';
    document.getElementById('sttxt').textContent = 'live';
  } catch (e) {
    document.getElementById('st').firstElementChild.className = 'dot stale';
    document.getElementById('sttxt').textContent = 'fetch error';
  }
}
window.addEventListener('resize', () => {
  // re-draw at current size
  fetch('/api/data', {cache: 'no-store'}).then(r => r.json()).then(render);
});
tick();
setInterval(tick, REFRESH_MS);
setInterval(() => {
  if (Date.now() - lastOk > 15000) {
    document.getElementById('st').firstElementChild.className = 'dot stale';
    document.getElementById('sttxt').textContent = 'stale';
  }
}, 2000);
</script>
</body></html>
"""


# --- HTTP server ------------------------------------------------------------


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        return

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            body = INDEX.replace(
                "window.SSH_HOST || ''",
                f"'{SSH_HOST}:{SSH_PORT}'"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/data":
            try:
                data = _gather()
                body = json.dumps(data).encode()
            except Exception as e:
                body = json.dumps({"ts": time.time(), "error": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"GPU dashboard → http://localhost:{PORT}  (ssh {SSH_HOST}:{SSH_PORT}, refresh 4s)")
    S(("0.0.0.0", PORT), H).serve_forever()
