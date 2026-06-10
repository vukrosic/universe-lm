#!/usr/bin/env python3
"""Zero-dependency board for autoresearch ideas, grouped by pipeline phase.
- Columns = the 5 phases (taste/definition/code/run/done) so it fits with no
  horizontal scroll; each card shows its exact sub-status as a badge.
- Click any idea to read its idea.md in the panel below. Click also reveals
  evidence.md (full text) plus the final treatment vs ctrl/ctrl2 numbers and
  an inline SVG line chart of per-step val_loss (if a series is locatable).
Reads live from autoresearch/ideas/*/idea.md and evidence.md, and walks
remote-results/ to find per-step val_loss series. Status vocab = PIPELINE.md.
Run:  python3 tools/status_dashboard.py   → http://localhost:8080
"""
import http.server, socketserver, glob, os, re, json, html, subprocess
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get("PORT", "8080"))
AUTORESEARCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autoresearch"))
IDEAS = os.path.join(AUTORESEARCH, "ideas")
BRIEF = os.path.join(AUTORESEARCH, "brief.md")
BRIEFS = os.path.join(AUTORESEARCH, "briefs")
REMOTE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "remote-results"))

# tmux sessions we never want to show as "MiniMax workers" on the board.
AGENT_EXCLUDE = {"agent", "dash-coder"}
# Substrings whose presence in the recent tail indicates Claude is actively
# generating/tool-using (covers spinner verbs and the "esticulating"/"searched" tokens).
BUSY_TOKENS = ("Puzzling", "Thinking", "Considering", "Calculating",
               "Crunch", "Cook", "esticulat", "searched")

# (label, color, [member statuses], one-line meaning)
GROUPS = [
    ("Taste gate",      "#d29922", ["needs-taste", "tasting", "needs-repitch", "repitching"],
     "Is the idea even worth testing?"),
    ("Definition gate", "#a371f7", ["needs-review", "reviewing", "needs-revision", "revising"],
     "Is the hypothesis / spec sound?"),
    ("Code gate",       "#f0883e", ["needs-plan", "planning", "needs-codereview", "codereviewing", "needs-recode", "recoding"],
     "Implement the code, then review it."),
    ("Queued",          "#2f81f7", ["needs-run"],
     "Approved and waiting for a GPU slot."),
    ("On GPU",          "#58a6ff", ["running"],
     "Currently training on the GPU (seed 42)."),
    ("Done",            "#3fb950", ["done"],
     "Finished — evidence.md written, WIN/NULL recorded."),
    ("Rejected",        "#6e7681", ["rejected"],
     "Killed in review — folder moved to _closed/."),
]

SLUG_RE = re.compile(r"\A[\w.\-]+\Z")

def read_brief():
    """Return research brief markdown (topic / question / scope), or None."""
    if not os.path.isfile(BRIEF):
        return None
    with open(BRIEF) as fh:
        return fh.read()

def _parse_frontmatter(text):
    """Extract YAML frontmatter key: value pairs from text. Returns dict."""
    out = {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return out
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = re.match(r"^(\w+):\s*(.+)$", line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip().strip('"').strip("'")
    return out

def _count_done_ideas():
    """Count ideas with status: done."""
    count = 0
    for f in glob.glob(os.path.join(IDEAS, "*", "idea.md")):
        try:
            with open(f) as fh:
                for line in fh:
                    m = re.match(r"\s*status:\s*(.+)", line, re.I)
                    if m:
                        if m.group(1).strip().strip('"').split()[0] == "done":
                            count += 1
                        break
        except OSError:
            continue
    return count

def read_campaigns():
    """Return list of campaign dicts from autoresearch/briefs/*/brief.md."""
    out = []
    if not os.path.isdir(BRIEFS):
        return out
    for f in sorted(glob.glob(os.path.join(BRIEFS, "*", "brief.md"))):
        parent = os.path.basename(os.path.dirname(f))
        if parent.startswith("_"):
            continue
        try:
            with open(f) as fh:
                text = fh.read()
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        if not fm:
            continue
        camp = {
            "id": fm.get("id", parent),
            "status": fm.get("status", "?"),
            "round": fm.get("round", ""),
            "updated": fm.get("updated", ""),
            "exit": fm.get("exit", ""),
        }
        if camp["status"] == "active":
            dm = re.search(r"(\d+)\s+done\s+ideas", camp["exit"], re.I)
            if dm:
                camp["done_target"] = int(dm.group(1))
                camp["done_count"] = _count_done_ideas()
            xm = re.search(r"(\d{4}-\d{2}-\d{2})", camp["exit"])
            if xm:
                camp["exit_date"] = xm.group(1)
        out.append(camp)
    return out

def read_activity(limit=40):
    """Return newest `limit` events from ideas/*/log.jsonl and briefs/*/log.jsonl."""
    events = []
    for pattern, kind in [
        (os.path.join(IDEAS, "*", "log.jsonl"), "idea"),
        (os.path.join(BRIEFS, "*", "log.jsonl"), "brief"),
    ]:
        for f in glob.glob(pattern):
            if os.path.basename(os.path.dirname(f)).startswith("_"):
                continue
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        ev["kind"] = kind
                        events.append(ev)
            except OSError:
                continue
    events.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return events[:limit]

def read_statuses():
    out = {}
    for f in glob.glob(os.path.join(IDEAS, "*", "idea.md")):
        slug = os.path.basename(os.path.dirname(f))
        st = "?"
        try:
            with open(f) as fh:
                for line in fh:
                    m = re.match(r"\s*status:\s*(.+)", line, re.I)
                    if m:
                        st = m.group(1).strip().strip('"').split()[0]
                        break
        except OSError:
            continue
        out[slug] = st
    return out

def _tmux_sessions():
    """Return a list of tmux session names, or [] if tmux is absent / empty."""
    try:
        r = subprocess.run(
            ["tmux", "ls", "-F", "#{session_name}"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]

def _tmux_capture(name):
    """Capture the last ~45 lines of session `name`, returning raw text or ''."""
    try:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", name, "-p", "-S", "-45"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if r.returncode != 0:
        return ""
    return r.stdout or ""

def read_agents():
    """List MiniMax-worker tmux sessions (excluding AGENT_EXCLUDE).
    Returns: [{name, busy}]"""
    out = []
    for name in _tmux_sessions():
        if name in AGENT_EXCLUDE:
            continue
        raw = _tmux_capture(name)
        non_blank = [ln for ln in raw.splitlines() if ln.strip()]
        # "busy" looks only at the most recent slice (last ~10 non-blank lines),
        # so a stale spinner from earlier doesn't lock the dot on green forever.
        recent = "\n".join(non_blank[-10:])
        busy = any(tok in recent for tok in BUSY_TOKENS)
        out.append({"name": name, "busy": busy})
    return out

def read_idea(slug):
    # sanitize: slug must be a single safe path segment
    if not SLUG_RE.fullmatch(slug or ""):
        return None
    p = os.path.join(IDEAS, slug, "idea.md")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return fh.read()

def read_evidence(slug):
    """Return full text of evidence.md (sanitized) or None."""
    if not SLUG_RE.fullmatch(slug or ""):
        return None
    p = os.path.join(IDEAS, slug, "evidence.md")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return fh.read()

def parse_evidence(text):
    """Pull final treatment val_loss + ctrl + ctrl2 out of evidence.md text.
    All fields are optional (None if not present). Never raises."""
    if not text:
        return {"treatment_val": None, "ctrl": None, "ctrl2": None}
    out = {"treatment_val": None, "ctrl": None, "ctrl2": None}
    # treatment val: 6.3234 (optionally with r1/r2 suffix)
    m = re.search(r"treatment\s+val\s*:\s*([0-9]*\.?[0-9]+)", text, re.I)
    if m:
        out["treatment_val"] = float(m.group(1))
    # ctrl=6.3875, ctrl2=6.4050
    m = re.search(r"(?<![A-Za-z0-9_])ctrl\s*=\s*([0-9]*\.?[0-9]+)", text, re.I)
    if m:
        out["ctrl"] = float(m.group(1))
    m = re.search(r"ctrl2\s*=\s*([0-9]*\.?[0-9]+)", text, re.I)
    if m:
        out["ctrl2"] = float(m.group(1))
    return out

def _idea_prefix(slug):
    """Leading digits of a slug, e.g. '009-fire-pe' -> '009'. '' if none."""
    m = re.match(r"(\d+)", slug or "")
    return m.group(1) if m else ""

def _series_from_json(path):
    """Parse a metrics.json-style file with parallel `steps` + `val_losses` arrays."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    # Common shapes: {history:{steps,val_losses}} or top-level {steps,val_losses}
    src = data.get("history") if isinstance(data, dict) and isinstance(data.get("history"), dict) else data
    if not isinstance(src, dict):
        return None
    steps = src.get("steps")
    vals = src.get("val_losses")
    if not (isinstance(steps, list) and isinstance(vals, list)):
        return None
    out = []
    for s, v in zip(steps, vals):
        try:
            si = int(s); vf = float(v)
        except (TypeError, ValueError):
            continue
        if si < 0 or vf != vf:  # skip NaNs
            continue
        out.append({"step": si, "val_loss": vf})
    return out

_STEP_VAL_RE = re.compile(
    r"(?:^|\s)Step\s+(\d+)[^:\n]*?:\s*Val\s*Loss\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.I | re.M,
)

def _series_from_log(path):
    """Parse a training log for 'Step N: Val Loss: X' lines."""
    try:
        with open(path) as fh:
            txt = fh.read()
    except OSError:
        return None
    found = []
    for m in _STEP_VAL_RE.finditer(txt):
        try:
            step = int(m.group(1)); v = float(m.group(2))
        except ValueError:
            continue
        if v != v:  # NaN
            continue
        found.append((step, v))
    if not found:
        return None
    # Dedupe by step, keep first occurrence (most logs only emit one per step).
    seen = {}
    for s, v in found:
        seen.setdefault(s, v)
    series = sorted(seen.items())
    return [{"step": s, "val_loss": v} for s, v in series]

def find_series(slug):
    """Locate a per-step val_loss series for this slug, scanning remote-results/.
    Prefers JSON (`steps` + `val_losses`) over log-parsed series.
    Returns [] if nothing reliably maps to this idea. Never fabricates."""
    if not SLUG_RE.fullmatch(slug or ""):
        return []
    if not os.path.isdir(REMOTE):
        return []
    prefix = _idea_prefix(slug)            # "009"
    name_no_num = re.sub(r"^\d+-?", "", slug)  # "fire-pe"
    # Candidate files: any path under remote-results/ whose name contains the slug,
    # the leading digits, or the human-readable part of the slug.
    candidates = []
    for root, _dirs, files in os.walk(REMOTE):
        for fn in files:
            lfn = fn.lower()
            base = lfn[:-5] if lfn.endswith(".json") else (lfn[:-4] if lfn.endswith(".log") else lfn)
            slug_hit = (slug.lower() in lfn)
            prefix_hit = (prefix and prefix in lfn)
            name_hit = (name_no_num and len(name_no_num) >= 3 and name_no_num.lower() in lfn)
            if slug_hit or prefix_hit or name_hit:
                candidates.append(os.path.join(root, fn))
    if not candidates:
        return []
    # Pass 1: any JSON file yielding a usable series
    for p in sorted(candidates):
        if p.lower().endswith(".json"):
            series = _series_from_json(p)
            if series and len(series) >= 2:
                return series
    # Pass 2: any log file yielding a usable series
    for p in sorted(candidates):
        if p.lower().endswith(".log"):
            series = _series_from_log(p)
            if series and len(series) >= 2:
                return series
    return []

def read_result(slug):
    """Bundle evidence.md + parsed finals + per-step series for a slug."""
    ev_text = read_evidence(slug)
    parsed = parse_evidence(ev_text)
    series = find_series(slug) if ev_text else []
    return {
        "slug": slug,
        "evidence": ev_text,   # full text or None
        "final": parsed,       # {treatment_val, ctrl, ctrl2}
        "series": series,      # [{step, val_loss}] or []
    }

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>autoresearch board</title><style>
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#0d1117;color:#e6edf3;display:flex;flex-direction:column;padding:14px;gap:12px}
h1{font-size:15px;margin:0;font-weight:600}
h1 small{color:#8b949e;font-weight:400;margin-left:8px}
.board{display:flex;gap:10px;align-items:stretch;flex:0 0 auto;max-height:48vh}
.col{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:9px;flex:1 1 0;min-width:0;display:flex;flex-direction:column;overflow:auto}
.col h2{font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:1px 1px 4px;display:flex;justify-content:space-between;align-items:center}
.col h2 .nm{font-weight:700}
.col h2 .ct{border-radius:11px;padding:1px 8px;color:#0d1117;font-weight:700}
.desc{font-size:10.5px;line-height:1.3;color:#8b949e;margin:0 1px 8px}
.card{background:#21262d;border:1px solid #30363d;border-left:3px solid #58a6ff;border-radius:6px;padding:8px 9px;margin:5px 0;cursor:pointer;font-weight:500}
.card:hover{background:#2d333b}
.card.sel{outline:2px solid #58a6ff}
.card .sub{display:block;font-size:10px;color:#8b949e;font-weight:600;margin-top:3px;text-transform:uppercase;letter-spacing:.4px}
.none{color:#484f58;font-size:12px;text-align:center;padding:4px 0}
.reader{flex:1 1 auto;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;overflow:auto;min-height:0;display:flex;flex-direction:column;gap:12px}
.reader h3{margin:0 0 10px;font-size:13px;color:#58a6ff}
.reader pre{white-space:pre-wrap;word-wrap:break-word;font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c9d1d9;margin:0}
.section{border:1px solid #30363d;border-radius:8px;padding:10px 12px;background:#0d1117}
.section h4{margin:0 0 8px;font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.final-row{display:flex;gap:14px;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.final-row .pill{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:4px 9px}
.final-row .pill b{color:#e6edf3}
.final-row .pill .v{color:#58a6ff}
.final-row .pill .d{color:#3fb950}
.final-row .pill .d.bad{color:#f85149}
.final-row .pill .v.bad{color:#f85149}
.series-note{color:#6e7681;font-size:12px;font-style:italic}
.hint{color:#6e7681}
svg .axis{stroke:#30363d;stroke-width:1}
svg .grid{stroke:#21262d;stroke-width:1;stroke-dasharray:2,3}
svg .line{fill:none;stroke:#58a6ff;stroke-width:1.6}
svg .label{fill:#8b949e;font:10px ui-monospace,SFMono-Regular,Menlo,monospace}
svg .pt{fill:#58a6ff}
/* markdown-rendered content */
.md{font:13px -apple-system,BlinkMacSystemFont,sans-serif;color:#c9d1d9;line-height:1.5}
.md h1,.md h2,.md h3,.md h4,.md h5,.md h6{margin:14px 0 8px;color:#e6edf3;font-weight:600;line-height:1.25}
.md h1{font-size:18px;border-bottom:1px solid #30363d;padding-bottom:6px}
.md h2{font-size:16px;border-bottom:1px solid #21262d;padding-bottom:4px}
.md h3{font-size:14px}
.md h4{font-size:13px;color:#c9d1d9}
.md h5{font-size:12.5px;color:#8b949e;text-transform:uppercase;letter-spacing:.4px}
.md h6{font-size:12px;color:#8b949e}
.md p{margin:0 0 10px}
.md ul,.md ol{margin:0 0 10px;padding-left:22px}
.md li{margin:2px 0}
.md li>p{margin:0 0 4px}
.md b,.md strong{color:#e6edf3;font-weight:700}
.md i,.md em{color:#c9d1d9;font-style:italic}
.md code{background:#161b22;border:1px solid #30363d;border-radius:4px;padding:1px 5px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6edf3}
.md pre{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px 12px;margin:0 0 10px;overflow:auto;white-space:pre;word-wrap:normal}
.md pre code{background:transparent;border:0;padding:0;color:#c9d1d9;font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;display:block}
.md a{color:#58a6ff;text-decoration:none}
.md a:hover{text-decoration:underline}
.md hr{border:0;border-top:1px solid #30363d;margin:14px 0}
/* MiniMax workers panel */
.workers{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;flex:0 0 auto}
.workers h2{font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:0 0 8px;color:#8b949e;display:flex;align-items:center;gap:10px}
.workers h2 .ct{background:#30363d;color:#e6edf3;border-radius:11px;padding:1px 8px;font-weight:700}
.workers .row{display:flex;gap:10px;flex-wrap:wrap}
.worker{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:8px 10px;flex:1 1 320px;min-width:300px;display:flex;flex-direction:column;gap:6px}
.worker .hdr{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:600;color:#e6edf3}
.worker .dot{width:9px;height:9px;border-radius:50%;background:#6e7681;flex:0 0 auto;box-shadow:0 0 0 2px #0d1117}
.worker .dot.busy{background:#3fb950;box-shadow:0 0 0 2px #0d1117,0 0 6px #3fb95080}
.worker .state{font-size:10px;color:#8b949e;font-weight:500;text-transform:uppercase;letter-spacing:.4px;margin-left:auto}
.workers .none{color:#484f58;font-size:12px;padding:4px 2px}
/* research brief — paper step 0 */
.brief{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;flex:0 0 auto}
.brief h2{font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:0;color:#8b949e;display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.brief h2 .tag{font-size:10px;color:#6e7681;font-weight:500;text-transform:none;letter-spacing:0}
.brief h2 .chev{color:#6e7681;font-size:10px;margin-left:auto;transition:transform .15s}
.brief.collapsed h2 .chev{transform:rotate(-90deg)}
.brief.collapsed .brief-body{display:none}
.brief-body{margin-top:10px;max-height:28vh;overflow:auto}
.brief .path{font-size:10px;color:#484f58;margin-bottom:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
/* campaigns pipeline panel */
.campaigns{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;flex:0 0 auto}
.campaigns h2{font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:0 0 8px;color:#8b949e}
.camp-table{width:100%;border-collapse:collapse;font-size:12px}
.camp-table td{padding:3px 10px 3px 0;border-bottom:1px solid #1c2129;vertical-align:middle}
.camp-table tr:last-child td{border-bottom:0}
.camp-hint{color:#d29922;font-size:11.5px;margin-top:8px;border-top:1px solid #30363d;padding-top:8px}
/* activity feed panel */
.activity{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;flex:0 0 auto}
.activity h2{font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin:0;color:#8b949e;display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.activity h2 .chev{color:#6e7681;font-size:10px;margin-left:auto;transition:transform .15s}
.activity.collapsed h2 .chev{transform:rotate(-90deg)}
.activity.collapsed .activity-body{display:none}
.activity-body{margin-top:10px;max-height:22vh;overflow:auto}
.act-row{display:flex;gap:8px;align-items:baseline;font-size:12px;padding:3px 0;border-bottom:1px solid #1c2129}
.act-row:last-child{border-bottom:0}
.act-ts{color:#484f58;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;flex:0 0 62px}
.act-agent{color:#8b949e;white-space:nowrap;flex:0 0 auto}
.act-slug{color:#e6edf3;font-weight:600;white-space:nowrap;flex:0 0 auto}
.act-trans{white-space:nowrap;flex:0 0 auto}
.act-note{color:#6e7681;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1 1 0;min-width:0}
</style></head><body>
<h1>autoresearch board <small id=meta>loading…</small></h1>
<div class=campaigns id=campaigns>
  <h2>Campaigns</h2>
  <div id=campaigns-body><span class=hint>loading…</span></div>
</div>
<div class=activity id=activity>
  <h2 onclick="toggleActivity()"><span>Activity</span><span class=chev>▼</span></h2>
  <div class=activity-body id=activity-body><span class=hint>loading…</span></div>
</div>
<div class=brief id=brief>
  <h2 onclick="toggleBrief()"><span>Research brief</span><span class=tag>topic · question · scope</span><span class=chev>▼</span></h2>
  <div class=brief-body><div class=path>autoresearch/brief.md</div><div class="md" id=brief-md><span class=hint>loading…</span></div></div>
</div>
<div class=workers id=workers><h2><span>MiniMax workers</span><span class=ct id=workers-ct>0</span></h2><div class=row id=workers-row><div class=none>loading…</div></div></div>
<div class=board id=board></div>
<div class=reader id=reader><span class=hint>Click an idea above to read its idea.md here.</span></div>
<script>
const GROUPS = __GROUPS__;
let sel = null;
let lastBoardJson = '';
let lastBriefText = '';
function toggleBrief(){
  document.getElementById('brief').classList.toggle('collapsed');
}
async function loadBrief(){
  let txt = '';
  try {
    const r = await fetch('api/brief');
    if(r.ok) txt = await r.text();
  } catch(e) { txt = ''; }
  if(txt === lastBriefText) return;
  lastBriefText = txt;
  const el = document.getElementById('brief-md');
  if(!txt.trim()){
    el.innerHTML = '<span class=hint>no brief yet — create autoresearch/brief.md</span>';
    return;
  }
  el.innerHTML = renderMarkdown(txt);
}
async function load(){
  const r = await fetch('api/board'); const data = await r.json();
  const byStatus = data.ideas;            // {slug: status}
  const json = JSON.stringify(byStatus);
  if(json === lastBoardJson) return;
  lastBoardJson = json;
  const total = Object.keys(byStatus).length;
  document.getElementById('meta').textContent = total + ' ideas · 7 phases · live';
  const board = document.getElementById('board'); board.innerHTML='';
  for(const g of GROUPS){
    const mine = Object.entries(byStatus).filter(([s,st])=>g.members.includes(st)).sort();
    const col = document.createElement('div'); col.className='col';
    col.innerHTML = `<h2><span class=nm style="color:${g.color}">${g.label}</span>`+
      `<span class=ct style="background:${g.color}">${mine.length}</span></h2>`+
      `<div class=desc>${g.desc}</div>`;
    if(!mine.length) col.insertAdjacentHTML('beforeend','<div class=none>—</div>');
    for(const [slug,st] of mine){
      const d=document.createElement('div'); d.className='card'+(slug===sel?' sel':'');
      d.style.borderLeftColor=g.color;
      d.innerHTML=`${slug}<span class=sub>${st}</span>`;
      d.onclick=()=>show(slug);
      col.appendChild(d);
    }
    board.appendChild(col);
  }
}
function fmt(v){ if(v==null) return '—'; return Number(v).toFixed(4); }
function deltaClass(d){ if(d==null) return ''; return d<0?' d':(d>0?' d bad':''); }
function valueClass(d, isTreatment){ if(d==null) return ''; if(!isTreatment) return ''; return d<0?'':' bad'; }
function renderFinal(final){
  if(!final) return '';
  const t = final.treatment_val, c = final.ctrl, c2 = final.ctrl2;
  const dc = (t!=null && c!=null) ? (t - c) : null;
  const dc2 = (t!=null && c2!=null) ? (t - c2) : null;
  let s = '<div class=final-row>';
  s += `<span class=pill>treatment <b>val</b>: <span class="v${valueClass(dc,true)}">${fmt(t)}</span></span>`;
  s += `<span class=pill>ctrl: <span class=v>${fmt(c)}</span></span>`;
  s += `<span class=pill>ctrl2: <span class=v>${fmt(c2)}</span></span>`;
  s += `<span class=pill>Δ vs ctrl: <span class="d${deltaClass(dc)}">${fmt(dc)}</span></span>`;
  s += `<span class=pill>Δ vs ctrl2: <span class="d${deltaClass(dc2)}">${fmt(dc2)}</span></span>`;
  s += '</div>';
  return s;
}
function renderChart(series){
  if(!series || !series.length) return '<div class=series-note>no per-step series available</div>';
  const W = 600, H = 180;
  const ml = 44, mr = 12, mt = 12, mb = 22;
  const iw = W - ml - mr, ih = H - mt - mb;
  const xs = series.map(p=>p.step), ys = series.map(p=>p.val_loss);
  const xmin = xs[0], xmax = xs[xs.length-1];
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if(ymin === ymax){ ymin -= 0.5; ymax += 0.5; }
  const pad = (ymax - ymin) * 0.08;
  ymin -= pad; ymax += pad;
  const sx = s => ml + (xmax===xmin ? 0 : (s - xmin) / (xmax - xmin)) * iw;
  const sy = v => mt + (1 - (v - ymin) / (ymax - ymin)) * ih;
  let pts = series.map(p => `${sx(p.step).toFixed(1)},${sy(p.val_loss).toFixed(1)}`).join(' ');
  // 4 x ticks, 4 y ticks
  const xt = [], yt = [];
  for(let i=0;i<=3;i++){ const xv = xmin + (xmax-xmin)*i/3; xt.push(xv); }
  for(let i=0;i<=3;i++){ const yv = ymin + (ymax-ymin)*i/3; yt.push(yv); }
  let svg = `<svg viewBox="0 0 ${W} ${H}" width=100% preserveAspectRatio="xMidYMid meet" role="img" aria-label="val_loss vs step">`;
  // gridlines + y labels
  for(const yv of yt){
    const y = sy(yv);
    svg += `<line class=grid x1=${ml} x2=${W-mr} y1=${y.toFixed(1)} y2=${y.toFixed(1)} />`;
    svg += `<text class=label x=${ml-4} y=${(y+3).toFixed(1)} text-anchor=end>${yv.toFixed(2)}</text>`;
  }
  // x labels
  for(const xv of xt){
    const x = sx(xv);
    svg += `<text class=label x=${x.toFixed(1)} y=${H-6} text-anchor=middle>step ${Math.round(xv)}</text>`;
  }
  // axes
  svg += `<line class=axis x1=${ml} x2=${W-mr} y1=${H-mb} y2=${H-mb} />`;
  svg += `<line class=axis x1=${ml} x2=${ml} y1=${mt} y2=${H-mb} />`;
  // line + points
  svg += `<polyline class=line points="${pts}" />`;
  for(const p of series){
    svg += `<circle class=pt cx=${sx(p.step).toFixed(1)} cy=${sy(p.val_loss).toFixed(1)} r=2 />`;
  }
  // y-axis title
  svg += `<text class=label x=4 y=${(mt+4).toFixed(1)}>val_loss</text>`;
  svg += `</svg>`;
  return svg;
}
function mdEsc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function mdSafeUrl(u){
  // Only allow http(s), relative, or in-page URLs. Reject javascript:, data:, etc.
  return /^(https?:\/|\/|#)/i.test(u) ? u : '#';
}
function mdInline(s){
  // Stash inline code behind placeholders so the asterisks inside are safe.
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (m, c) => { codes.push(c); return '\x00C' + (codes.length-1) + '\x00'; });
  // links: [text](url)
  s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (m, t, u) =>
    '<a href="' + mdSafeUrl(u) + '" target="_blank" rel="noopener noreferrer">' + t + '</a>');
  // bold (** or __) — non-greedy, no internal newlines
  s = s.replace(/(\*\*|__)([^*\n_][\s\S]*?)\1/g, '<b>$2</b>');
  // italic (* or _) — only when the delimiter isn't adjacent to alnum on the other side
  s = s.replace(/(^|[^*\w_])(\*|_)([^*_\n][\s\S]*?)\2(?!\w)/g, '$1<i>$3</i>');
  // restore inline code
  s = s.replace(/\x00C(\d+)\x00/g, (m, i) => '<code>' + codes[+i] + '</code>');
  return s;
}
function renderMarkdown(src){
  if(src == null) return '';
  // 1) Escape HTML FIRST so user content can never inject tags.
  const text = mdEsc(String(src));
  const lines = text.split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length){
    const line = lines[i];
    // Fenced code block: ``` ... ```
    if (/^```/.test(line)){
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])){ buf.push(lines[i]); i++; }
      i++; // skip closing fence (or EOF)
      out.push('<pre><code>' + buf.join('\n') + '</code></pre>');
      continue;
    }
    // Horizontal rule
    if (/^---+\s*$/.test(line) || /^\*\*\*+\s*$/.test(line)){
      out.push('<hr>');
      i++;
      continue;
    }
    // ATX heading
    const hm = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (hm){
      const lvl = hm[1].length;
      out.push('<h' + lvl + '>' + mdInline(hm[2]) + '</h' + lvl + '>');
      i++;
      continue;
    }
    // Blank line
    if (line.trim() === ''){ i++; continue; }
    // Bullet list
    if (/^[-*]\s+/.test(line)){
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])){
        items.push(mdInline(lines[i].replace(/^[-*]\s+/, '')));
        i++;
      }
      out.push('<ul><li>' + items.join('</li><li>') + '</li></ul>');
      continue;
    }
    // Numbered list
    if (/^\d+\.\s+/.test(line)){
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])){
        items.push(mdInline(lines[i].replace(/^\d+\.\s+/, '')));
        i++;
      }
      out.push('<ol><li>' + items.join('</li><li>') + '</li></ol>');
      continue;
    }
    // Paragraph: consume consecutive plain lines until blank/special.
    const buf = [];
    while (i < lines.length
        && lines[i].trim() !== ''
        && !/^```/.test(lines[i])
        && !/^#{1,6}\s+/.test(lines[i])
        && !/^[-*]\s+/.test(lines[i])
        && !/^\d+\.\s+/.test(lines[i])
        && !/^---+\s*$/.test(lines[i])
        && !/^\*\*\*+\s*$/.test(lines[i])){
      buf.push(lines[i]); i++;
    }
    const joined = buf.join('\n');
    // Single-line paragraph: just inline. Multi-line: keep line breaks as <br>.
    if (buf.length === 1) out.push('<p>' + mdInline(joined) + '</p>');
    else out.push('<p>' + mdInline(joined).replace(/\n/g, '<br>') + '</p>');
  }
  return out.join('\n');
}
async function show(slug){
  sel=slug;
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('sel', c.firstChild.textContent===slug));
  const reader=document.getElementById('reader');
  // 1) idea.md (existing behavior)
  const r=await fetch('api/idea?slug='+encodeURIComponent(slug));
  if(!r.ok){reader.innerHTML='<span class=hint>could not load '+slug+'</span>';return;}
  const txt=await r.text();
  // 2) result bundle (evidence + final + series)
  let result = null;
  try {
    const rr = await fetch('api/result?slug='+encodeURIComponent(slug));
    if(rr.ok) result = await rr.json();
  } catch(e) { result = null; }
  // Build the panel: idea block, then result block(s). The .md divs hold
  // rendered markdown; we fill them with innerHTML AFTER setting the
  // surrounding HTML, so the markdown HTML can never be re-escaped by accident.
  let html = '<div class=section><h3 style="margin:0 0 8px;font-size:13px;color:#58a6ff">'
           + htmlEscape(slug)+'/idea.md</h3><div class=md></div></div>';
  if(result){
    if(result.evidence){
      html += '<div class=section><h4>evidence.md</h4><div class=md></div></div>';
    }
    if(result.final && (result.final.treatment_val!=null || result.final.ctrl!=null || result.final.ctrl2!=null)){
      html += '<div class=section><h4>final val_loss</h4>'+renderFinal(result.final)+'</div>';
    }
    if(result.series && result.series.length){
      html += '<div class=section><h4>val_loss vs step ('+result.series.length+' points)</h4>'+renderChart(result.series)+'</div>';
    } else if(result.evidence){
      html += '<div class=section><h4>val_loss vs step</h4><div class=series-note>no per-step series available</div></div>';
    }
  }
  reader.innerHTML = html;
  // Populate the .md content divs (in source order: idea, then evidence).
  const mds = reader.querySelectorAll('.section > .md');
  if(mds[0]) mds[0].innerHTML = renderMarkdown(txt);
  if(mds[1] && result && result.evidence) mds[1].innerHTML = renderMarkdown(result.evidence);
}
function htmlEscape(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function loadAgents(){
  let agents = [];
  try {
    const r = await fetch('api/agents');
    if(r.ok) agents = await r.json();
  } catch(e) { agents = []; }
  const row = document.getElementById('workers-row');
  document.getElementById('workers-ct').textContent = agents.length;
  if(!agents.length){
    row.innerHTML = '<div class=none>no MiniMax workers (excluding agent, dash-coder)</div>';
    return;
  }
  row.innerHTML = '';
  for(const a of agents){
    const w = document.createElement('div');
    w.className = 'worker'; w.dataset.name = a.name;
    w.innerHTML = `<div class=hdr><span class="dot${a.busy?' busy':''}"></span>`+
      `<span>${htmlEscape(a.name)}</span>`+
      `<span class=state>${a.busy?'busy':'idle'}</span></div>`;
    row.appendChild(w);
  }
}
function statusPill(st){
  const map = {
    'active':          {bg:'#1a3a2a',fg:'#3fb950',bd:'#3fb950'},
    'needs-blessing':  {bg:'#3a2e00',fg:'#d29922',bd:'#d29922'},
    'retired':         {bg:'#1c1f24',fg:'#6e7681',bd:'#484f58'},
    'rejected':        {bg:'#1c1f24',fg:'#6e7681',bd:'#484f58'},
    'shelved':         {bg:'#1c1f24',fg:'#8b949e',bd:'#30363d'},
  };
  const c = map[st] || {bg:'#1c1f24',fg:'#8b949e',bd:'#30363d'};
  return `<span style="display:inline-block;border-radius:11px;padding:1px 7px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;background:${c.bg};color:${c.fg};border:1px solid ${c.bd}">${htmlEscape(st)}</span>`;
}
async function loadCampaigns(){
  let camps = [];
  try {
    const r = await fetch('api/campaigns');
    if(r.ok) camps = await r.json();
  } catch(e) { camps = []; }
  const body = document.getElementById('campaigns-body');
  if(!camps.length){
    body.innerHTML = '<span class=hint>no campaign briefs found — create autoresearch/briefs/NNN-slug/brief.md</span>';
    return;
  }
  const needsBlessing = camps.some(c => c.status === 'needs-blessing');
  let rows = '';
  for(const c of camps){
    let exitCell = htmlEscape(c.exit || '—');
    if(c.status === 'active' && c.done_count != null){
      exitCell = `<b style="color:#e6edf3">${c.done_count}/${c.done_target} done</b>`;
      if(c.exit_date) exitCell += ` · exit ${htmlEscape(c.exit_date)}`;
    }
    const upd = (c.updated||'').replace('T',' ').replace(/:\d{2}(\.\d+)?Z$/,'Z');
    rows += `<tr>`+
      `<td style="color:#e6edf3;font-weight:600;white-space:nowrap">${htmlEscape(c.id)}</td>`+
      `<td style="white-space:nowrap">${statusPill(c.status)}</td>`+
      `<td style="color:#8b949e">${exitCell}</td>`+
      `<td style="color:#484f58;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap">${htmlEscape(upd)}</td>`+
      `</tr>`;
  }
  let out = `<table class=camp-table><tbody>${rows}</tbody></table>`;
  if(needsBlessing) out += `<div class=camp-hint>⏳ awaiting blessing — see autoresearch/briefs/BLESSING.md</div>`;
  body.innerHTML = out;
}
function toggleActivity(){
  document.getElementById('activity').classList.toggle('collapsed');
}
function relTime(ts){
  if(!ts) return '?';
  const d = Date.now() - new Date(ts).getTime();
  if(isNaN(d)) return ts.slice(0,10);
  const s = Math.floor(d/1000);
  if(s < 60) return s+'s ago';
  const m = Math.floor(s/60);
  if(m < 60) return m+'m ago';
  const h = Math.floor(m/60);
  if(h < 24) return h+'h ago';
  const days = Math.floor(h/24);
  if(days === 1) return 'yesterday';
  return days+'d ago';
}
function toStatusColor(st){
  if(!st) return '#8b949e';
  if(st==='done'||st==='active') return '#3fb950';
  if(st==='rejected') return '#f85149';
  if(st==='needs-run') return '#2f81f7';
  if(/ing$/.test(st)) return '#d29922';
  return '#8b949e';
}
async function loadActivity(){
  let events = [];
  try {
    const r = await fetch('api/activity');
    if(r.ok) events = await r.json();
  } catch(e) { events = []; }
  const body = document.getElementById('activity-body');
  if(!events.length){
    body.innerHTML = '<span class=hint>no activity yet</span>';
    return;
  }
  let rows = '';
  for(const e of events){
    const slug = e.idea || e.brief || '';
    const from_ = e.from || '';
    const to_ = e.to || '';
    const noteRaw = e.note || '';
    const note = noteRaw.length > 60 ? noteRaw.slice(0,60)+'…' : noteRaw;
    rows += `<div class=act-row>`+
      `<span class=act-ts>${htmlEscape(relTime(e.ts))}</span>`+
      `<span class=act-agent>${htmlEscape(e.agent||'')}</span>`+
      `<span class=act-slug> ${htmlEscape(slug)}</span>`+
      `<span class=act-trans> ${htmlEscape(from_)} → <span style="color:${toStatusColor(to_)}">${htmlEscape(to_)}</span></span>`+
      `<span class=act-note> ${htmlEscape(note)}</span>`+
      `</div>`;
  }
  body.innerHTML = rows;
}
load(); loadAgents(); loadBrief(); loadCampaigns(); loadActivity();
setInterval(load, 30000);   // poll board only; skip DOM rebuild when statuses unchanged
setInterval(loadBrief, 60000);
setInterval(loadCampaigns, 60000);
setInterval(loadActivity, 30000);
</script></body></html>"""

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            groups = [{"label": l, "color": c, "members": m, "desc": d} for (l, c, m, d) in GROUPS]
            self._send(200, PAGE.replace("__GROUPS__", json.dumps(groups)))
        elif u.path == "/api/brief":
            txt = read_brief()
            if txt is None:
                self._send(404, "not found", "text/plain")
            else:
                self._send(200, txt, "text/plain; charset=utf-8")
        elif u.path == "/api/campaigns":
            self._send(200, json.dumps(read_campaigns()), "application/json")
        elif u.path == "/api/activity":
            self._send(200, json.dumps(read_activity()), "application/json")
        elif u.path == "/api/board":
            self._send(200, json.dumps({"ideas": read_statuses()}), "application/json")
        elif u.path == "/api/agents":
            self._send(200, json.dumps(read_agents()), "application/json")
        elif u.path == "/api/idea":
            slug = parse_qs(u.query).get("slug", [""])[0]
            txt = read_idea(slug)
            if txt is None:
                self._send(404, "not found", "text/plain")
            else:
                self._send(200, txt, "text/plain; charset=utf-8")
        elif u.path == "/api/result":
            slug = parse_qs(u.query).get("slug", [""])[0]
            if not SLUG_RE.fullmatch(slug or ""):
                self._send(400, "bad slug", "text/plain")
                return
            self._send(200, json.dumps(read_result(slug)), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
        print(f"serving autoresearch board on http://localhost:{PORT}")
        httpd.serve_forever()
