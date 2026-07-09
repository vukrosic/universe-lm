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
import http.server, socketserver, glob, os, re, json, html, subprocess, time, mimetypes
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get("PORT", "8080"))
AUTORESEARCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autoresearch"))
IDEAS = os.path.join(AUTORESEARCH, "ideas")
BRIEF = os.path.join(AUTORESEARCH, "brief.md")
BRIEFS = os.path.join(AUTORESEARCH, "briefs")
REMOTE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "remote-results"))
FLIP_SH = os.path.join(AUTORESEARCH, "bin", "flip.sh")
QUESTIONS = os.path.join(AUTORESEARCH, "questions.jsonl")
# token2science subproject (the "donate tokens -> reproducible science" system).
T2S = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "token2science"))
T2S_PAPERS = os.path.realpath(os.path.join(T2S, "papers"))

# --- GPU monitor config -----------------------------------------------------
# Polls a Vast box over SSH to surface the running arq job + val-loss curve.
# Override via env: SSH_HOST=user@box.example.com, SSH_PORT=22, ARQ_DIR=/root/arq
SSH_HOST = os.environ.get("SSH_HOST", "root@81.45.65.189")
SSH_PORT = os.environ.get("SSH_PORT", "22179")
ARQ_DIR  = os.environ.get("ARQ_DIR",  "/root/arq")
GPU_CACHE_TTL = 4.0  # seconds; cached across HTTP requests to avoid SSH storms
_gpu_cache = {"ts": 0.0, "data": None}

# Statuses where the user can approve/reject from the board (taste gate).
TASTE_STATUSES = {"needs-taste", "tasting", "needs-repitch", "repitching"}
# Approve = forward to definition gate; reject = kill the idea.
APPROVE_TO = "needs-review"
REJECT_TO = "rejected"

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


def flip_idea(slug, action):
    """Run autoresearch/bin/flip.sh for one slug. Returns (ok, msg)."""
    if not SLUG_RE.fullmatch(slug or ""):
        return False, "bad slug"
    if action not in ("approve", "reject"):
        return False, "bad action"
    folder = os.path.join(IDEAS, slug)
    if not os.path.isdir(folder):
        return False, "no such idea"
    # Guard: only flip from a taste-gate status so this UI can't trample
    # ideas that are mid-pipeline elsewhere.
    cur = read_statuses().get(slug)
    if cur not in TASTE_STATUSES:
        return False, f"refusing: status={cur} (taste-gate only)"
    to_status = APPROVE_TO if action == "approve" else REJECT_TO
    note = f"{action}d via board"
    try:
        r = subprocess.run(
            ["bash", FLIP_SH, slug, to_status, "dashboard", note],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"flip.sh error: {e}"
    if r.returncode != 0:
        return False, (r.stderr or r.stdout or "flip.sh failed").strip()
    return True, (r.stdout or "ok").strip()


# --- GPU monitor (lives in the "GPU" tab; polls box over SSH) --------------

def _ssh(cmd, timeout=8):
    """Run `cmd` on the configured box; return (rc, stdout, stderr). Never raises."""
    try:
        out = subprocess.run(
            ["ssh",
             "-p", SSH_PORT,
             "-o", "ConnectTimeout=3",
             "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes",
             SSH_HOST, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.returncode, out.stdout, out.stderr
    except (subprocess.TimeoutExpired, OSError) as e:
        return -1, "", f"ssh: {e}"


_STEP_RE = re.compile(
    r"step=(\d+)/(\d+),\s*loss=([\d\.Na+n]+),\s*acc=([\d\.]+),"
    r"\s*ent=\S+,\s*cp=\S+,\s*pl=\S+,\s*lr=([\d\.eE+-]+)"
)
_TOKS_RE = re.compile(r"([\d\.]+)tokens/s")
_ETA_RE  = re.compile(r"<([\d:]+),")
_VAL_RE  = re.compile(
    r"^Step (\d+): Val Loss: ([\d\.]+), Val Acc: ([\d\.]+), Val PPL: ([\d\.]+), LR: ([\d\.eE+-]+)$"
)
_FINAL_LOSS_RE = re.compile(r"^Final Val Loss:\s+([\d\.Na+n]+)\s*$")
_FINAL_ACC_RE  = re.compile(r"^Final Val Accuracy:\s+([\d\.]+)\s*$")


def _parse_log_text(text):
    """Parse the tail of an arq log into (cur_step_dict, val_loss_series)."""
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


def _gather_gpu():
    """Build the GPU-tab payload. Cached GPU_CACHE_TTL seconds per process."""
    now = time.time()
    if _gpu_cache["data"] and now - _gpu_cache["ts"] < GPU_CACHE_TTL:
        return _gpu_cache["data"]
    out = {"ts": now, "ssh_host": f"{SSH_HOST}:{SSH_PORT}"}

    # 1) GPU snapshot
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

    # 3) Active log (val-loss)
    out["current"] = None
    out["series"] = []
    if running:
        rc, txt, _ = _ssh(f"cat {ARQ_DIR}/logs/{running['job']}.log 2>/dev/null | tail -2500")
        if rc == 0 and txt:
            cur, series = _parse_log_text(txt)
            out["current"] = cur
            out["series"] = series

    _gpu_cache["ts"] = now
    _gpu_cache["data"] = out
    return out


def _t2s_sessions():
    """Live codex/sim tmux sessions for the token2science workflow."""
    try:
        out = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        n = line.split(":", 1)[0]
        if n.startswith("t2s-") or n.startswith("simu-"):
            names.append(n)
    return names


def _gather_t2s():
    """Read-only snapshot of the token2science testing workflow: what agents are
    active, goal progress, recent runs, confirmations, and generated papers."""
    data = {"ts": time.time(), "sessions": [], "goals": [], "runs": [],
            "papers": [], "totals": {}}
    if not os.path.isdir(T2S):
        data["error"] = "token2science/ not found"
        return data
    data["sessions"] = _t2s_sessions()

    # runs: scan runs/<task>/<run>/result.json, keep best value per goal + recent
    runfiles = glob.glob(os.path.join(T2S, "runs", "*", "*", "result.json"))
    by_goal = {}
    for rf in runfiles:
        try:
            r = json.load(open(rf))
        except Exception:
            continue
        by_goal.setdefault(r.get("goal_id"), []).append(r.get("value"))
    for rf in sorted(runfiles, key=os.path.getmtime, reverse=True)[:12]:
        try:
            r = json.load(open(rf))
        except Exception:
            continue
        data["runs"].append({"task": r.get("task_id"), "goal": r.get("goal_id"),
                             "worker": r.get("worker"), "metric": r.get("metric"),
                             "value": r.get("value"), "ts": os.path.getmtime(rf)})

    for gj in sorted(glob.glob(os.path.join(T2S, "goals", "*", "goal.json"))):
        try:
            g = json.load(open(gj))
        except Exception:
            continue
        gid = g.get("goal_id", os.path.basename(os.path.dirname(gj)))
        lib = bool(g.get("lower_is_better", True))
        bar = g.get("bar")
        tdir = os.path.join(os.path.dirname(gj), "tasks")
        ntasks = len(glob.glob(os.path.join(tdir, "*"))) if os.path.isdir(tdir) else 0
        vs = [v for v in by_goal.get(gid, []) if isinstance(v, (int, float))]
        best = (min(vs) if lib else max(vs)) if vs else None
        beats = None
        if best is not None and isinstance(bar, (int, float)):
            beats = best < bar if lib else best > bar
        data["goals"].append({"id": gid, "title": g.get("title", ""),
                              "metric": g.get("metric", ""), "bar": bar,
                              "lower_is_better": lib, "status": g.get("status", ""),
                              "tasks": ntasks, "nruns": len(vs), "best": best,
                              "beats": beats})

    nconf = 0
    for cf in glob.glob(os.path.join(T2S, "runs", "*", "confirmation.json")):
        try:
            if json.load(open(cf)).get("confirmed_value") is not None:
                nconf += 1
        except Exception:
            pass
    legacy_papers = [os.path.basename(p)
                     for p in sorted(glob.glob(os.path.join(T2S, "papers", "*.md")))]
    focused_papers = _t2s_papers_list()
    data["papers"] = legacy_papers
    data["focused_papers"] = focused_papers
    data["totals"] = {"goals": len(data["goals"]), "runs": len(runfiles),
                      "papers": len(legacy_papers), "focused_papers": len(focused_papers),
                      "confirmed": nconf, "active_sessions": len(data["sessions"])}
    return data

_T2S_SESSION_RE = re.compile(r"\A(?:t2s-|simu-)[A-Za-z0-9._-]+\Z")

def _t2s_papers_realpath(relpath):
    rel = (relpath or "").replace("\\", "/").strip().lstrip("/")
    if not rel or os.path.isabs(rel):
        return None
    real = os.path.realpath(os.path.join(T2S_PAPERS, rel))
    root = T2S_PAPERS + os.sep
    if real != T2S_PAPERS and not real.startswith(root):
        return None
    return real

def _t2s_papers_list():
    out = []
    if not os.path.isdir(T2S_PAPERS):
        return out
    for pj in sorted(glob.glob(os.path.join(T2S_PAPERS, "*", "paper.json"))):
        try:
            with open(pj, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        paper_id = str(data.get("paper_id") or os.path.basename(os.path.dirname(pj)))
        authors = data.get("authors") or []
        if not isinstance(authors, list):
            authors = [authors]
        authors = [str(a) for a in authors if str(a).strip()]
        mech = data.get("mechanism") if isinstance(data.get("mechanism"), dict) else {}
        experiments = data.get("experiments") if isinstance(data.get("experiments"), list) else []
        out.append({
            "paper_id": paper_id,
            "title": str(data.get("title") or paper_id),
            "authors": authors,
            "status": str(data.get("status") or ""),
            "mechanism_name": str(mech.get("name") or ""),
            "num_experiments": len(experiments),
            "manuscript": str(data.get("manuscript") or "manuscript.md"),
            "_created": str(data.get("created") or ""),
        })
    out.sort(key=lambda p: (p.get("_created", ""), p.get("paper_id", "")), reverse=True)
    for p in out:
        p.pop("_created", None)
    return out

def _t2s_paper_text(name):
    rel = (name or "").strip()
    if not rel:
        return None
    real = _t2s_papers_realpath(rel)
    if real is None:
        return None
    if os.path.isdir(real):
        meta = os.path.join(real, "paper.json")
        if not os.path.isfile(meta):
            return None
        try:
            with open(meta, encoding="utf-8") as fh:
                paper = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(paper, dict):
            return None
        manuscript = str(paper.get("manuscript") or "manuscript.md")
        real = _t2s_papers_realpath(os.path.join(rel, manuscript))
        if real is None:
            return None
    if not os.path.isfile(real):
        return None
    try:
        with open(real, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None

def _t2s_asset_bytes(relpath):
    if not relpath or os.path.isabs(relpath):
        return None, None, 400
    allowed_roots = [
        os.path.realpath(os.path.join(T2S, "papers")),
        os.path.realpath(os.path.join(T2S, "posts", "charts")),
    ]
    real = os.path.realpath(os.path.join(T2S, relpath))
    ok = False
    for root in allowed_roots:
        prefix = root + os.sep
        if real == root or real.startswith(prefix):
            ok = True
            break
    if not ok:
        return None, None, 400
    if not os.path.isfile(real):
        return None, None, 404
    ctype = mimetypes.guess_type(real)[0] or "application/octet-stream"
    try:
        with open(real, "rb") as fh:
            return fh.read(), ctype, 200
    except OSError:
        return None, None, 404

def _t2s_session_capture(name):
    if not _T2S_SESSION_RE.fullmatch(name or ""):
        return None
    try:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", name, "-p", "-S", "-80"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "(session ended)"
    if r.returncode != 0:
        return "(session ended)"
    return r.stdout or ""


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
/* taste-gate approve/reject toolbar */
.gate-tools{display:flex;gap:6px;margin:0 1px 6px}
.gate-tools button{flex:1;background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:5px;padding:4px 6px;font-size:11px;font-weight:600;cursor:pointer}
.gate-tools button:hover:not(:disabled){background:#2d333b}
.gate-tools button:disabled{opacity:.4;cursor:not-allowed}
.gate-tools button.appr{border-color:#238636;color:#3fb950}
.gate-tools button.rej{border-color:#6e2c2c;color:#f85149}
.card .ck{margin-right:6px;vertical-align:middle;accent-color:#58a6ff}
.flash{position:fixed;right:14px;bottom:14px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 12px;font-size:12px;max-width:380px;z-index:50}
.flash.ok{border-color:#238636;color:#3fb950}
.flash.err{border-color:#6e2c2c;color:#f85149}
.topbar{background:#d29922;color:#0d1117;border-radius:14px;padding:24px 28px;font-weight:700;font-size:15px;letter-spacing:.3px;flex:0 0 auto;min-height:180px;box-shadow:0 0 0 2px #f0c33a inset}
.topbar h2{margin:0 0 14px;font-size:20px;font-weight:800;letter-spacing:.5px}
.topbar ul{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(2,1fr);gap:8px 24px}
.topbar li{display:flex;align-items:center;gap:10px;cursor:pointer;padding:4px 6px;border-radius:6px}
.topbar li:hover{background:#0d111720}
.topbar input[type=checkbox]{width:18px;height:18px;accent-color:#0d1117;cursor:pointer;flex:0 0 auto}
.topbar label{cursor:pointer;flex:1}
.topbar li.done label{text-decoration:line-through;opacity:.55}
.tabs{display:flex;gap:4px;border-bottom:1px solid #30363d;flex:0 0 auto;margin-bottom:-4px}
.tab{background:transparent;color:#8b949e;border:1px solid transparent;border-bottom:0;border-radius:8px 8px 0 0;padding:8px 16px;font:600 13px -apple-system,BlinkMacSystemFont,sans-serif;cursor:pointer;letter-spacing:.3px}
.tab:hover{color:#e6edf3}
.tab.active{background:#161b22;color:#e6edf3;border-color:#30363d}
.tab-pane{display:none;flex-direction:column;gap:12px;flex:1 1 auto;min-height:0}
.tab-pane.active{display:flex}
.goal{background:linear-gradient(135deg,#d29922 0%,#e6a93a 100%);color:#0d1117;border-radius:14px;padding:24px 28px;flex:0 0 auto;box-shadow:0 0 0 2px #f0c33a inset}
.goal .tag{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;opacity:.75}
.goal h1{margin:6px 0 4px;font-size:28px;font-weight:800;letter-spacing:.3px}
.goal .sub{font-size:14px;font-weight:600;opacity:.8}
.threads{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px;flex:1 1 auto;min-height:0;overflow:auto}
.threads h2{margin:0 0 4px;font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.threads .hint{color:#6e7681;font-size:12px;margin-bottom:14px;display:block}
.thread{border:1px solid #30363d;border-left:4px solid #58a6ff;border-radius:8px;padding:12px 14px;margin:10px 0;background:#0d1117}
.thread .head{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.thread .num{color:#6e7681;font:700 12px ui-monospace,Menlo,monospace;flex:0 0 auto}
.thread .name{color:#e6edf3;font-weight:700;font-size:14px;flex:1}
.thread .st{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;border-radius:11px;padding:2px 8px;flex:0 0 auto}
.thread .st.open{background:#1c1f24;color:#8b949e;border:1px solid #30363d}
.thread .st.active{background:#1a3a2a;color:#3fb950;border:1px solid #3fb950}
.thread .st.blocked{background:#3a2e00;color:#d29922;border:1px solid #d29922}
.thread .q{color:#c9d1d9;font-size:13px;line-height:1.4;margin:0 0 6px}
.thread .why{color:#8b949e;font-size:12px;line-height:1.4;margin:0 0 6px}
.thread .why b{color:#d29922;font-weight:700;text-transform:uppercase;font-size:10px;letter-spacing:.4px;margin-right:6px}
.thread .ideas{font:11px ui-monospace,Menlo,monospace;color:#6e7681;display:flex;flex-wrap:wrap;gap:6px}
.thread .ideas span{background:#161b22;border:1px solid #30363d;border-radius:4px;padding:1px 6px;color:#8b949e}
.thread .ideas .none{border:0;background:transparent;color:#484f58;padding:0;font-style:italic}
/* tree visualization (vertical) */
.tree{position:relative;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px 20px;flex:1 1 auto;min-height:0;overflow:auto}
.tree-svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1}
.tree-svg path{fill:none;stroke-width:1.6;opacity:.55}
.tree-svg path:hover{opacity:1}
.tree-cols{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;position:relative;z-index:2;padding-top:8px}
.tree-anchor{grid-column:1 / -1;display:flex;justify-content:center;margin-bottom:36px}
.tree-anchor-node{background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 18px;font-size:12px;font-weight:700;color:#e6edf3;letter-spacing:.3px}
.tree-anchor-node .arrow{color:#6e7681;margin:0 8px}
.tree-col{display:flex;flex-direction:column;gap:6px;min-width:0}
.tree-th{background:#0d1117;border:1px solid #30363d;border-top:3px solid var(--c,#58a6ff);border-radius:7px;padding:8px 9px;cursor:default}
.tree-th .th-row{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.tree-th .th-num{font:700 10px ui-monospace,Menlo,monospace;color:#6e7681}
.tree-th .th-name{font-size:11.5px;font-weight:700;color:#e6edf3;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tree-th .th-st{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;border-radius:9px;padding:1px 6px;flex:0 0 auto}
.tree-th .th-st.open{background:#1c1f24;color:#8b949e;border:1px solid #30363d}
.tree-th .th-st.active{background:#1a3a2a;color:#3fb950;border:1px solid #3fb950}
.tree-th .th-st.blocked{background:#3a2e00;color:#d29922;border:1px solid #d29922}
.tree-th .th-q{font-size:10.5px;color:#8b949e;line-height:1.3}
.tree-leaves{display:flex;flex-direction:column;gap:4px;margin-top:6px}
.tree-leaf{background:#0d1117;border:1px solid #30363d;border-left:3px solid var(--lc,#6e7681);border-radius:5px;padding:5px 7px;font:11px ui-monospace,Menlo,monospace;color:#c9d1d9;display:flex;align-items:center;gap:6px;cursor:default;transition:background .12s}
.tree-leaf:hover{background:#1c2129}
.tree-leaf .leaf-slug{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tree-leaf .leaf-st{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;color:var(--lc,#6e7681);flex:0 0 auto}
.tree-leaves .empty{font:italic 10.5px -apple-system,sans-serif;color:#484f58;padding:4px 2px}
.tree-legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:18px;padding-top:12px;border-top:1px solid #30363d;font-size:11px;color:#8b949e}
.tree-legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:middle}
/* graph visualization (true graph: absolute nodes + SVG edges) */
.tree-legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:18px;padding-top:12px;border-top:1px solid #30363d;font-size:11px;color:#8b949e}
.tree-legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:middle}
.graph{position:relative;width:100%;min-height:520px;height:calc(70vh - 100px);margin-top:8px}
.pill{position:absolute;display:flex;align-items:center;gap:6px;border-radius:14px;font:600 11px -apple-system,BlinkMacSystemFont,sans-serif;white-space:nowrap;user-select:none;cursor:default;transition:filter .12s, transform .12s;z-index:2}
.pill:hover{filter:brightness(1.25);transform:translateY(-1px);z-index:3}
.pill .badge{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;opacity:.85}
.goal-pill{top:14px;background:linear-gradient(135deg,#21262d 0%,#1c2129 100%);border:1.5px solid #58a6ff;color:#e6edf7;padding:9px 18px;font-size:13px;box-shadow:0 0 0 1px #58a6ff40,0 0 14px #58a6ff30;border-radius:18px;font-weight:700;z-index:4}
.goal-pill .goal-arrow{margin:0 6px;color:#6e7681;font-weight:400}
.thread-pill{background:#161b22;border:1.5px solid var(--c,#58a6ff);color:#e6edf3;padding:5px 10px;border-radius:14px;box-shadow:0 0 0 1px var(--c,#58a6ff)20}
.thread-pill .num{color:var(--c,#58a6ff);font:700 10px ui-monospace,Menlo,monospace;margin-right:4px}
.thread-pill .st{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;border-radius:8px;padding:1px 5px;margin-left:4px}
.thread-pill .st.open{background:#1c1f24;color:#8b949e}
.thread-pill .st.active{background:#1a3a2a;color:#3fb950}
.thread-pill .st.blocked{background:#3a2e00;color:#d29922}
.idea-pill{background:#0d1117;border:1px solid #30363d;border-left:3px solid var(--lc,#6e7681);color:#c9d1d9;padding:3px 8px;border-radius:11px;font:500 10.5px ui-monospace,Menlo,monospace;letter-spacing:.2px}
.idea-pill .leaf-st{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;color:var(--lc,#6e7681);margin-left:4px}
.ghost-pill{background:transparent;border:1px dashed #484f58;color:#6e7681;padding:3px 8px;border-radius:11px;font:italic 10.5px -apple-system,sans-serif}
</style></head><body>
<div class=tabs>
  <button class=tab data-tab=board>Board</button>
  <button class=tab data-tab=brainstorm>Brainstorm</button>
  <button class=tab data-tab=gpu>GPU</button>
  <button class=tab data-tab=testing>Testing</button>
  <button class=tab data-tab=papers>Papers</button>
</div>
<div class="tab-pane" id=pane-brainstorm>
<div class=goal>
  <div class=tag>main goal · 2026</div>
  <h1>Ship a 135M LLM that beats SmolLM-135M</h1>
  <div class=sub>target: HellaSwag · PIQA · ARC-easy · MMLU-tiny, FLOPs-matched</div>
</div>
<div class=tree id=tree>
  <div class=graph id=graph>
    <svg class=tree-svg id=tree-svg></svg>
    <div id=graph-nodes></div>
  </div>
  <div class=tree-legend>
    <span><i style="background:#3fb950"></i>done</span>
    <span><i style="background:#58a6ff"></i>running</span>
    <span><i style="background:#2f81f7"></i>needs-run</span>
    <span><i style="background:#d29922"></i>needs-*</span>
    <span><i style="background:#6e7681"></i>rejected</span>
  </div>
</div>
</div>
<div class="tab-pane" id=pane-board>
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
</div>
<div class="tab-pane" id=pane-testing>
<style scoped>
  #pane-testing h1{margin:0;font-size:15px;font-weight:600}
  #pane-testing h1 small{color:#6e7681;font-weight:400;margin-left:8px;font-size:11px;font-family:ui-monospace,Menlo,monospace}
  .t2s-totals{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
  .t2s-chip{background:#161b22;border:1px solid #30363d;border-radius:20px;padding:4px 12px;font-size:12px;color:#e6edf3}
  .t2s-chip b{color:#58a6ff}
  .t2s-grid{display:flex;gap:10px;flex-wrap:wrap}
  .t2s-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px;flex:1 1 320px;min-width:0;margin-bottom:10px}
  .t2s-card h2{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#8b949e;font-weight:600}
  .t2s-sess{font:600 12.5px ui-monospace,Menlo,monospace;color:#3fb950;padding:3px 0;display:flex;align-items:center;gap:7px}
  .t2s-live{width:8px;height:8px;border-radius:50%;background:#3fb950;box-shadow:0 0 6px #3fb95080;flex:0 0 auto;animation:t2spulse 1.4s infinite}
  @keyframes t2spulse{0%,100%{opacity:1}50%{opacity:.35}}
  .t2s-goal{padding:5px 0;border-bottom:1px solid #1c2129;font-size:12.5px}
  .t2s-goal:last-child{border-bottom:0}
  .t2s-goal b{color:#e6edf3}
  .t2s-mut{color:#6e7681}
  .t2s-ok{color:#3fb950;font-weight:600}
  .t2s-below{color:#d29922}
  .t2s-run{font:11.5px ui-monospace,Menlo,monospace;color:#8b949e;padding:2px 0}
  .t2s-run code{color:#58a6ff}
  .t2s-sess-link{font:600 12.5px ui-monospace,Menlo,monospace;color:#3fb950}
  .t2s-livewrap{display:flex;flex-direction:column;gap:8px}
  .t2s-livebar{display:flex;align-items:center;gap:8px;justify-content:space-between}
  .t2s-livebar .label{color:#8b949e;font-size:11px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .t2s-livebar button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:4px 8px;font-size:11px;font-weight:600;cursor:pointer}
  .t2s-livebar button:hover:not(:disabled){background:#2d333b}
  .t2s-livebar button:disabled{opacity:.45;cursor:not-allowed}
  .t2s-session-box{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:10px 12px;min-height:220px;max-height:46vh;overflow:auto;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c9d1d9;white-space:pre-wrap;word-break:break-word;margin:0}
  .t2s-session-box .hint{display:block}
  #pane-testing .hint{color:#6e7681;font-size:12px}
  .md table{width:100%;border-collapse:collapse;margin:0 0 10px;background:#0d1117;border:1px solid #30363d;border-radius:6px;overflow:hidden;display:block}
  .md thead{background:#161b22}
  .md th,.md td{border-bottom:1px solid #30363d;border-right:1px solid #30363d;padding:6px 8px;vertical-align:top;text-align:left}
  .md th:last-child,.md td:last-child{border-right:0}
  .md tr:last-child td{border-bottom:0}
  .md img{max-width:100%;height:auto;display:block;margin:8px 0;border:1px solid #30363d;border-radius:6px;background:#0d1117}
</style>
<h1>token2science testing <small id=t2s-meta>loading…</small></h1>
<div class=t2s-totals id=t2s-totals></div>
<div class=t2s-grid>
  <div class=t2s-card><h2>Active agents</h2><div id=t2s-sessions><span class=hint>…</span></div></div>
  <div class=t2s-card><h2>Goals</h2><div id=t2s-goals><span class=hint>…</span></div></div>
</div>
<div class=t2s-card><h2>Recent runs</h2><div id=t2s-runs><span class=hint>…</span></div></div>
<div class=t2s-card><h2>Live session</h2><div class=t2s-livewrap><div class=t2s-livebar><span class=label id=t2s-session-label>click an agent session to follow it live</span><button id=t2s-session-stop disabled>Stop/close</button></div><pre class=t2s-session-box id=t2s-session-view><span class=hint>no session selected</span></pre></div></div>
</div>
<div class="tab-pane" id=pane-papers>
<style scoped>
  #pane-papers h1{margin:0;font-size:15px;font-weight:600}
  #pane-papers h1 small{color:#6e7681;font-weight:400;margin-left:8px;font-size:11px;font-family:ui-monospace,Menlo,monospace}
  .t2s-paper-meta{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 12px}
  .t2s-paper-chip{background:#161b22;border:1px solid #30363d;border-radius:20px;padding:4px 12px;font-size:12px;color:#e6edf3}
  .t2s-paper-chip b{color:#58a6ff}
  .t2s-paper-filters{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 12px}
  .t2s-author-chip{background:#161b22;border:1px solid #30363d;border-radius:16px;padding:4px 10px;font-size:11.5px;color:#8b949e;cursor:pointer;white-space:nowrap}
  .t2s-author-chip:hover{background:#21262d;color:#e6edf3}
  .t2s-author-chip.active{background:#1a3a2a;border-color:#3fb950;color:#3fb950}
  .t2s-paper-layout{display:flex;gap:10px;flex:1 1 auto;min-height:0}
  .t2s-paper-list{flex:0 0 40%;min-width:320px;display:flex;flex-direction:column;gap:10px;overflow:auto;padding-right:2px}
  .t2s-paper-cards{display:flex;flex-direction:column;gap:10px}
  .t2s-paper-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px;cursor:pointer;display:flex;flex-direction:column;gap:8px}
  .t2s-paper-card:hover{background:#21262d}
  .t2s-paper-card.active{outline:2px solid #58a6ff}
  .t2s-paper-title{font-size:13px;font-weight:700;color:#e6edf3;line-height:1.3}
  .t2s-paper-authors{display:flex;gap:6px;flex-wrap:wrap}
  .t2s-paper-author{background:#0d1117;border:1px solid #30363d;border-radius:999px;padding:2px 8px;font-size:11px;color:#8b949e}
  .t2s-paper-rows{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px;font-size:11.5px;color:#8b949e}
  .t2s-paper-rows div{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .t2s-paper-rows b{color:#e6edf3;font-weight:600}
  .t2s-paper-empty{color:#6e7681;font-size:12px;padding:6px 2px}
  .t2s-paper-reader{flex:1 1 auto;min-width:0;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:12px;overflow:auto;min-height:0}
  .t2s-paper-reader h2{margin:0;font-size:15px;color:#58a6ff}
  .t2s-paper-reader .meta{display:flex;gap:8px;flex-wrap:wrap;color:#8b949e;font-size:11.5px}
  .t2s-paper-reader .meta span{background:#0d1117;border:1px solid #30363d;border-radius:999px;padding:2px 8px}
  .t2s-paper-reader .meta b{color:#e6edf3;font-weight:600}
  .t2s-paper-reader .path{color:#6e7681;font-size:10.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .t2s-paper-reader .body{max-height:100%;overflow:auto;padding-right:2px}
  .t2s-paper-reader .hint{color:#6e7681;font-size:12px}
  .t2s-paper-reader .body .md table{width:100%;border-collapse:collapse;margin:0 0 10px;background:#0d1117;border:1px solid #30363d;border-radius:6px;overflow:hidden;display:block}
  .t2s-paper-reader .body .md thead{background:#161b22}
  .t2s-paper-reader .body .md th,.t2s-paper-reader .body .md td{border-bottom:1px solid #30363d;border-right:1px solid #30363d;padding:6px 8px;vertical-align:top;text-align:left}
  .t2s-paper-reader .body .md th:last-child,.t2s-paper-reader .body .md td:last-child{border-right:0}
  .t2s-paper-reader .body .md tr:last-child td{border-bottom:0}
  .t2s-paper-reader .body .md img{max-width:100%;height:auto;display:block;margin:8px 0;border:1px solid #30363d;border-radius:6px;background:#0d1117}
  @media (max-width: 980px){
    .t2s-paper-layout{flex-direction:column}
    .t2s-paper-list{flex:1 1 auto;min-width:0;max-height:42vh}
  }
</style>
<h1>token2science papers <small id=t2s-papers-meta>loading…</small></h1>
<div class=t2s-paper-meta id=t2s-paper-stats></div>
<div class=t2s-paper-filters id=t2s-paper-filters><span class=hint>loading authors…</span></div>
<div class=t2s-paper-layout>
  <div class=t2s-paper-list>
    <div class=t2s-paper-cards id=t2s-paper-cards><span class=t2s-paper-empty>loading…</span></div>
  </div>
  <div class=t2s-paper-reader id=t2s-paper-reader>
    <span class=hint>click a paper card to render its manuscript here</span>
  </div>
</div>
</div>
<div class="tab-pane" id=pane-gpu>
<style scoped>
  .gpuwrap{display:flex;flex-direction:column;gap:10px;flex:1 1 auto;min-height:0}
  .gputop{display:flex;align-items:baseline;gap:14px;flex:0 0 auto}
  .gputop h1{margin:0;font-size:15px;font-weight:600}
  .gputop h1 small{color:#8b949e;font-weight:400;margin-left:6px}
  .gputop .dot{width:9px;height:9px;border-radius:50%;display:inline-block;background:#6e7681;vertical-align:middle;margin-right:4px}
  .gputop .dot.ok{background:#3fb950;box-shadow:0 0 6px #3fb95080}
  .gputop .dot.stale{background:#d29922}
  .gputop .dot.down{background:#f85149}
  .gputop .ts{color:#6e7681;font-size:11px;margin-left:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .gpurow{display:flex;gap:10px;flex:0 0 auto}
  .gpurow.grow{flex:1 1 auto;min-height:0}
  .gpucard{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px;flex:1 1 0;min-width:0;display:flex;flex-direction:column;min-height:0}
  .gpucard h2{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#8b949e;font-weight:600}
  .gpumetric{font:600 24px ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6edf3;line-height:1}
  .gpumetric small{font-size:11px;color:#8b949e;font-weight:400;margin-left:4px}
  .gpubar{position:relative;height:5px;background:#21262d;border-radius:3px;margin-top:8px;overflow:hidden}
  .gpubar > i{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#3fb950 0%,#58a6ff 60%,#d29922 90%);border-radius:3px;transition:width .5s}
  .gpusub{color:#6e7681;font-size:11px;margin-top:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .gpujob{font:600 14px ui-monospace,SFMono-Regular,Menlo,monospace;color:#58a6ff}
  .gpujob .pill{display:inline-block;background:#21262d;color:#3fb950;padding:1px 7px;border-radius:9px;font-size:10px;margin-left:6px;text-transform:uppercase;letter-spacing:.4px;font-weight:600}
  .gpujob .pill.idle{background:#21262d;color:#6e7681}
  .gpujob .pill.done{background:#21262d;color:#d29922}
  .gpukv{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 14px;margin-top:10px}
  .gpukv div{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;color:#8b949e}
  .gpukv div b{color:#e6edf3;font-weight:600}
  .gpukv div.bad b{color:#f85149}
  .gpuchart{flex:1 1 auto;width:100%;min-height:0}
  .gpuchart .ax{stroke:#30363d;stroke-width:1}
  .gpuchart .ln{fill:none;stroke:#58a6ff;stroke-width:1.6}
  .gpuchart .pt{fill:#58a6ff;stroke:#0d1117;stroke-width:1}
  .gpuchart .gl{stroke:#21262d;stroke-dasharray:2 3}
  .gpuchart .lb{font:9.5px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#8b949e}
  .gpuqtable{width:100%;border-collapse:collapse;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;flex:1 1 auto;display:block;overflow:auto}
  .gpuqtable thead{position:sticky;top:0;background:#161b22}
  .gpuqtable th{color:#8b949e;text-align:left;font-weight:600;padding:4px 8px;border-bottom:1px solid #30363d;font-size:10px;text-transform:uppercase;letter-spacing:.4px}
  .gpuqtable td{padding:3px 8px;border-bottom:1px solid #1c2129}
  .gpuqtable tr.START td{color:#58a6ff}
  .gpuqtable tr.OK td{color:#3fb950}
  .gpuqtable tr.FAIL td{color:#f85149}
  .gpuqtable tr.QUEUE_DONE td{color:#d29922}
  .gpuhint{color:#6e7681}
</style>
<div class=gpuwrap>
  <div class=gputop>
    <h1>GPU monitor <small id=gpu-host></small></h1>
    <span><span class=dot id=gpu-dot></span> <span id=gpu-st>connecting…</span></span>
    <span class=ts id=gpu-ts>—</span>
  </div>
  <div class=gpurow>
    <div class=gpucard style="flex:0 0 24%">
      <h2>GPU util</h2>
      <div class=gpumetric id=gpu-util>—<small>%</small></div>
      <div class=gpubar><i id=gpu-utilb style="width:0%"></i></div>
      <div class=gpusub id=gpu-utilsub>—</div>
    </div>
    <div class=gpucard style="flex:0 0 24%">
      <h2>Memory</h2>
      <div class=gpumetric id=gpu-mem>—<small>GiB</small></div>
      <div class=gpubar><i id=gpu-memb style="width:0%"></i></div>
      <div class=gpusub id=gpu-memsub>—</div>
    </div>
    <div class=gpucard style="flex:0 0 24%">
      <h2>Temp / power</h2>
      <div class=gpumetric id=gpu-temp>—<small>°C</small></div>
      <div class=gpubar><i id=gpu-tempb style="width:0%"></i></div>
      <div class=gpusub id=gpu-pwr>—</div>
    </div>
    <div class=gpucard style="flex:1 1 auto">
      <h2>Current job</h2>
      <div class=gpujob id=gpu-jname>idle <span class="pill idle" id=gpu-jstatus>—</span></div>
      <div class=gpukv>
        <div>step: <b id=gpu-jstep>—</b></div>
        <div>tok/s: <b id=gpu-jtoks>—</b></div>
        <div>loss: <b id=gpu-jloss>—</b></div>
        <div>acc: <b id=gpu-jacc>—</b></div>
        <div>lr: <b id=gpu-jlr>—</b></div>
        <div>eta: <b id=gpu-jeta>—</b></div>
      </div>
    </div>
  </div>
  <div class="gpurow grow">
    <div class=gpucard style="flex:1 1 60%">
      <h2>Val loss (live)</h2>
      <svg class=gpuchart id=gpu-chart preserveAspectRatio="none"></svg>
      <div class=gpusub id=gpu-chartsub>—</div>
    </div>
    <div class=gpucard style="flex:0 0 38%">
      <h2>arq queue</h2>
      <table class=gpuqtable id=gpu-qtab>
        <thead><tr><th>state</th><th>job</th><th>ts</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>
</div>
<script>
const GROUPS = __GROUPS__;
const TASTE_STATUSES = __TASTE__;   // statuses where checkboxes appear
let sel = null;
let lastBoardJson = '';
let lastBriefText = '';
const picked = new Set();            // slugs currently checked (taste gate)
function isTasteGroup(g){ return g.members.some(m => TASTE_STATUSES.includes(m)); }
function flash(msg, ok){
  const el = document.createElement('div');
  el.className = 'flash ' + (ok?'ok':'err');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}
async function bulkFlip(action){
  if(!picked.size) return;
  const slugs = [...picked];
  const r = await fetch('api/flip', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({slugs, action}),
  });
  let data = null;
  try { data = await r.json(); } catch(e) {}
  if(!data){ flash('flip failed (no response)', false); return; }
  const okN = data.results.filter(x => x.ok).length;
  const errs = data.results.filter(x => !x.ok);
  flash(`${action}: ${okN}/${slugs.length} ok` + (errs.length?` · err: ${errs.map(e=>e.slug+': '+e.msg).join('; ')}`:''), errs.length===0);
  picked.clear();
  lastBoardJson = '';   // force re-render
  load();
}
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
    if(isTasteGroup(g)){
      const tools = document.createElement('div'); tools.className='gate-tools';
      tools.innerHTML = `<button class=appr id=appr-btn disabled>✓ Approve (<span id=appr-n>0</span>)</button>`+
                       `<button class=rej  id=rej-btn  disabled>✗ Reject (<span id=rej-n>0</span>)</button>`;
      col.appendChild(tools);
      tools.querySelector('#appr-btn').onclick = () => bulkFlip('approve');
      tools.querySelector('#rej-btn').onclick  = () => bulkFlip('reject');
    }
    if(!mine.length) col.insertAdjacentHTML('beforeend','<div class=none>—</div>');
    for(const [slug,st] of mine){
      const d=document.createElement('div'); d.className='card'+(slug===sel?' sel':'');
      d.style.borderLeftColor=g.color;
      if(isTasteGroup(g)){
        const checked = picked.has(slug) ? ' checked' : '';
        d.innerHTML=`<input type=checkbox class=ck data-slug="${slug}"${checked}>${slug}<span class=sub>${st}</span>`;
      } else {
        d.innerHTML=`${slug}<span class=sub>${st}</span>`;
      }
      d.dataset.slug = slug;
      d.onclick=(e)=>{
        if(e.target.classList.contains('ck')) return;   // checkbox handles itself
        show(slug);
      };
      const cb = d.querySelector('.ck');
      if(cb){
        cb.onclick = (e) => { e.stopPropagation(); };
        cb.onchange = (e) => {
          if(e.target.checked) picked.add(slug); else picked.delete(slug);
          updateGateButtons();
        };
      }
      col.appendChild(d);
    }
    board.appendChild(col);
  }
  // Drop picks that no longer exist or have left the taste gate.
  for(const s of [...picked]){
    if(!(s in byStatus) || !TASTE_STATUSES.includes(byStatus[s])) picked.delete(s);
  }
  updateGateButtons();
}
function updateGateButtons(){
  const n = picked.size;
  const a = document.getElementById('appr-btn'), r = document.getElementById('rej-btn');
  const an = document.getElementById('appr-n'), rn = document.getElementById('rej-n');
  if(a){ a.disabled = !n; an.textContent = n; }
  if(r){ r.disabled = !n; rn.textContent = n; }
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
function mdIsExternalUrl(u){
  return /^(?:https?:\/\/|\/|#|\/\/)/i.test(u);
}
function mdSafeUrl(u){
  // Only allow http(s), relative, or in-page URLs. Reject javascript:, data:, etc.
  return mdIsExternalUrl(u) ? u : '#';
}
function t2sNormalizeAssetPath(baseParts, rel){
  const parts = baseParts.slice();
  for(const seg of rel.split('/')){
    if(!seg || seg === '.') continue;
    if(seg === '..'){
      if(parts.length <= 1) return null;
      parts.pop();
      continue;
    }
    parts.push(seg);
  }
  return parts.join('/');
}
function t2sResolveAssetPath(src, opts){
  const raw = String(src || '').trim();
  if(!raw) return null;
  if(mdIsExternalUrl(raw)) return raw;
  const m = raw.match(/^([^?#]*)([?#].*)?$/);
  const rel = m ? m[1] : raw;
  const tail = m && m[2] ? m[2] : '';
  let path = null;
  if(rel === 'papers' || rel.startsWith('papers/') || rel === 'posts/charts' || rel.startsWith('posts/charts/')){
    path = t2sNormalizeAssetPath([], rel);
  } else if(opts && opts.paperAsset){
    let base = String(opts.paperBase || 'papers').replace(/\\/g, '/').trim();
    if(!base) base = 'papers';
    const baseParts = base.split('/').filter(Boolean);
    if(!baseParts.length || baseParts[0] !== 'papers') baseParts.unshift('papers');
    path = t2sNormalizeAssetPath(baseParts, rel);
  }
  if(!path) return null;
  if(!(path === 'papers' || path.startsWith('papers/') || path === 'posts/charts' || path.startsWith('posts/charts/'))) return null;
  return '/api/t2s/asset?path=' + encodeURIComponent(path) + tail;
}
function mdInline(s, opts){
  // Stash inline code behind placeholders so the asterisks inside are safe.
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (m, c) => { codes.push(c); return '\x00C' + (codes.length-1) + '\x00'; });
  // images: ![alt](src)
  s = s.replace(/!\[([^\]\n]*)\]\(([^)\s]+)\)/g, (m, alt, u) => {
    const src = (opts && opts.paperAsset) ? t2sResolveAssetPath(u, opts) : null;
    const finalSrc = src || mdSafeUrl(u);
    return '<img alt="' + alt + '" src="' + finalSrc + '">';
  });
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
function mdTableSep(line){
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}
function mdTableCells(line){
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
}
function renderMarkdown(src, opts){
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
    // GFM table: header row + separator row + body rows.
    if (i + 1 < lines.length && line.includes('|') && mdTableSep(lines[i + 1])){
      const head = mdTableCells(line);
      const aligns = mdTableCells(lines[i + 1]).map(cell => {
        const left = /^:/.test(cell);
        const right = /:$/.test(cell);
        if(left && right) return 'center';
        if(right) return 'right';
        if(left) return 'left';
        return '';
      });
      const rows = [];
      i += 2;
      while (i < lines.length
          && lines[i].trim() !== ''
          && lines[i].includes('|')
          && !/^```/.test(lines[i])
          && !/^#{1,6}\s+/.test(lines[i])
          && !/^[-*]\s+/.test(lines[i])
          && !/^\d+\.\s+/.test(lines[i])){
        rows.push(mdTableCells(lines[i]));
        i++;
      }
      let html = '<table><thead><tr>';
      for(let j = 0; j < head.length; j++){
        const st = aligns[j] ? ' style="text-align:' + aligns[j] + '"' : '';
        html += '<th' + st + '>' + mdInline(head[j], opts) + '</th>';
      }
      html += '</tr></thead><tbody>';
      for(const row of rows){
        html += '<tr>';
        for(let j = 0; j < head.length; j++){
          const st = aligns[j] ? ' style="text-align:' + aligns[j] + '"' : '';
          html += '<td' + st + '>' + mdInline(row[j] || '', opts) + '</td>';
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
      out.push(html);
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
      out.push('<h' + lvl + '>' + mdInline(hm[2], opts) + '</h' + lvl + '>');
      i++;
      continue;
    }
    // Blank line
    if (line.trim() === ''){ i++; continue; }
    // Bullet list
    if (/^[-*]\s+/.test(line)){
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])){
        items.push('<li>' + mdInline(lines[i].replace(/^[-*]\s+/, ''), opts) + '</li>');
        i++;
      }
      out.push('<ul>' + items.join('') + '</ul>');
      continue;
    }
    // Numbered list
    if (/^\d+\.\s+/.test(line)){
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])){
        items.push('<li>' + mdInline(lines[i].replace(/^\d+\.\s+/, ''), opts) + '</li>');
        i++;
      }
      out.push('<ol>' + items.join('') + '</ol>');
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
        && !(i + 1 < lines.length && lines[i].includes('|') && mdTableSep(lines[i + 1]))
        && !/^---+\s*$/.test(lines[i])
        && !/^\*\*\*+\s*$/.test(lines[i])){
      buf.push(lines[i]); i++;
    }
    const joined = buf.join('\n');
    // Single-line paragraph: just inline. Multi-line: keep line breaks as <br>.
    if (buf.length === 1) out.push('<p>' + mdInline(joined, opts) + '</p>');
    else out.push('<p>' + mdInline(joined, opts).replace(/\n/g, '<br>') + '</p>');
  }
  return out.join('\n');
}
async function show(slug){
  sel=slug;
  document.querySelectorAll('.card').forEach(c=>c.classList.toggle('sel', c.dataset.slug===slug));
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
// graph visualization: 1 goal -> 7 threads -> 16 ideas, absolute-positioned nodes + SVG bezier edges
(function(){
  const THREADS = [
    {id:'T1', name:'Optimizer', status:'active', color:'#a371f7',
     ideas:[
       {slug:'001-cautious-muon', st:'done'},
       {slug:'005-decoupled-qkv-muon', st:'done'},
       {slug:'015-moonlight-muon-rms', st:'done'},
       {slug:'011-cautious-lion', st:'done'},
     ]},
    {id:'T2', name:'Attention', status:'active', color:'#58a6ff',
     ideas:[
       {slug:'020-forgetting-attn', st:'needs-run'},
       {slug:'021-value-residual', st:'needs-run'},
       {slug:'022-softpick-attn', st:'needs-run'},
       {slug:'023-canon-conv', st:'needs-run'},
       {slug:'024-gated-attn', st:'needs-run'},
       {slug:'025-scalable-softmax', st:'needs-run'},
     ]},
    {id:'T3', name:'Norm', status:'open', color:'#3fb950',
     ideas:[
       {slug:'016-qk-norm', st:'done'},
       {slug:'017-sub-ln-sandwich', st:'done'},
     ]},
    {id:'T4', name:'Positional', status:'open', color:'#f0883e',
     ideas:[
       {slug:'009-fire-pe', st:'done'},
       {slug:'013-cope', st:'done'},
     ]},
    {id:'T5', name:'Scaling-law', status:'blocked', color:'#d29922', ideas:[]},
    {id:'T6', name:'Data', status:'blocked', color:'#f85149', ideas:[]},
    {id:'T7', name:'Eval', status:'open', color:'#6e7681', ideas:[]},
  ];
  const ST_COLOR = {
    'done':'#3fb950', 'running':'#58a6ff', 'needs-run':'#2f81f7',
    'rejected':'#6e7681',
  };
  function statusColor(st){
    if(ST_COLOR[st]) return ST_COLOR[st];
    if(st && /^needs-/.test(st)) return '#d29922';
    if(st && /ing$/.test(st)) return '#d29922';
    return '#6e7681';
  }
  // estimate pill width from text length (since pills are absolutely positioned, no flex auto-size)
  function pillW(kind, text){
    // goal ~ 13px, thread ~ 11px, idea ~ 10.5px monospace; 12px padding + 4px border each side + 6px badge gap
    if(kind === 'goal')   return 26 + text.length * 8.2;
    if(kind === 'thread') return 22 + text.length * 7.0 + 30;   // +30 for T# + status badge
    if(kind === 'idea')   return 18 + text.length * 6.4 + 28;   // +28 for status badge
    if(kind === 'ghost')  return 18 + text.length * 6.0;
    return 60;
  }
  const H_GOAL = 36, H_THREAD = 26, H_IDEA = 20, H_GHOST = 20;
  const Y_GOAL = 14, Y_THREAD = 110, Y_IDEA = 200;
  const G_IDEA_Y = 4;    // vertical gap between idea rows
  const IDEA_W_FIXED = 168;  // idea pills: fixed width so columns line up
  const COL_GAP = 6;

  function build(){
    const tree = document.getElementById('tree');
    const graph = document.getElementById('graph');
    const svg = document.getElementById('tree-svg');
    const nodesEl = document.getElementById('graph-nodes');
    if(!tree || !graph || !svg || !nodesEl) return;
    nodesEl.innerHTML = '';

    // measure container
    const W = graph.clientWidth || tree.clientWidth;
    const H = graph.clientHeight || 520;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', W);
    svg.setAttribute('height', H);

    // ----- GOAL (centered at top) -----
    const goalText = '🎯 Ship a 135M LLM that beats SmolLM-135M';
    const goalW = Math.min(pillW('goal', goalText), W - 40);
    const goalX = (W - goalW) / 2;
    const goalY = Y_GOAL;
    const goal = pill('goal-pill', goalText, goalX, goalY, goalW, H_GOAL);
    nodesEl.appendChild(goal);

    // ----- THREADS (7 across the width) -----
    const PAD = 30;
    const slotW = (W - 2 * PAD) / THREADS.length;
    // each thread pill: width determined by name; horizontal centering within slot
    const threadNodes = THREADS.map(t => {
      const text = `${t.id} ${t.name} · ${t.status}`;
      const w = pillW('thread', text);
      const x = PAD + THREADS.indexOf(t) * slotW + (slotW - w) / 2;
      const y = Y_THREAD;
      const node = pill('pill thread-pill', text, x, y, w, H_THREAD);
      node.style.setProperty('--c', t.color);
      nodesEl.appendChild(node);
      return {t, x, y, w, h:H_THREAD, cx: x + w/2, top: y, bottom: y + H_THREAD};
    });

    // ----- IDEAS (2 columns per thread, centered under each thread cx) -----
    const ideaNodes = [];
    for(const tn of threadNodes){
      const t = tn.t;
      if(t.ideas.length === 0){
        const gw = 110, gh = H_GHOST;
        const x = tn.cx - gw/2;
        const y = Y_IDEA;
        const g = pill('pill ghost-pill', '+ propose idea', x, y, gw, gh);
        nodesEl.appendChild(g);
        continue;
      }
      const n = t.ideas.length;
      const rows = Math.ceil(n / 2);
      const clusterW = 2 * IDEA_W_FIXED + COL_GAP;
      const x0 = tn.cx - clusterW / 2;
      t.ideas.forEach((i, k) => {
        const col = k % 2;
        const row = Math.floor(k / 2);
        const x = x0 + col * (IDEA_W_FIXED + COL_GAP);
        const y = Y_IDEA + row * (H_IDEA + G_IDEA_Y);
        const lc = statusColor(i.st);
        const p = document.createElement('div');
        p.className = 'pill idea-pill';
        p.style.left = x + 'px'; p.style.top = y + 'px';
        p.style.width = IDEA_W_FIXED + 'px'; p.style.height = H_IDEA + 'px';
        p.style.setProperty('--lc', lc);
        p.innerHTML = `<span>${i.slug}</span><span class=leaf-st>${i.st}</span>`;
        nodesEl.appendChild(p);
        ideaNodes.push({t, x, y, w: IDEA_W_FIXED, h: H_IDEA, cx: x + IDEA_W_FIXED/2, top: y, bottom: y + H_IDEA, i});
      });
    }

    // ----- EDGES (cubic bezier paths) -----
    const goalBottomY = goalY + H_GOAL;
    const threadTopY  = Y_THREAD;
    let paths = '';
    // goal -> thread
    for(const tn of threadNodes){
      const gx = goalX + goalW / 2;
      const gy = goalBottomY;
      const tx = tn.cx;
      const ty = tn.top;
      const midY = gy + (ty - gy) * 0.55;
      paths += `<path d="M ${gx} ${gy} C ${gx} ${midY}, ${tx} ${midY}, ${tx} ${ty}" stroke="${tn.t.color}" />`;
    }
    // thread -> idea (one curve per idea)
    for(const idea of ideaNodes){
      const tn = threadNodes.find(x => x.t.id === idea.t.id);
      if(!tn) continue;
      const tx = tn.cx;
      const ty = tn.bottom;
      const ix = idea.cx;
      const iy = idea.top;
      const midY = ty + (iy - ty) * 0.5;
      paths += `<path d="M ${tx} ${ty} C ${tx} ${midY}, ${ix} ${midY}, ${ix} ${iy}" stroke="${tn.t.color}" />`;
    }
    svg.innerHTML = paths;
  }

  function pill(cls, text, x, y, w, h){
    const p = document.createElement('div');
    p.className = cls;
    p.style.left = x + 'px';
    p.style.top  = y + 'px';
    p.style.width  = w + 'px';
    p.style.height = h + 'px';
    p.style.justifyContent = 'center';
    p.textContent = text;
    return p;
  }

  let resizeT;
  function reflow(){ clearTimeout(resizeT); resizeT = setTimeout(build, 60); }
  build();
  window.addEventListener('resize', reflow);
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    if(t.dataset.tab === 'brainstorm') setTimeout(build, 30);
  }));
  // observe container size changes (e.g. window resize on tab switch)
  if(window.ResizeObserver){
    const ro = new ResizeObserver(reflow);
    const tree = document.getElementById('tree');
    if(tree) ro.observe(tree);
  }
})();
// tabs: switch between Board and Brainstorm, persist active tab
(function(){
  const TAB_KEY = 'autoresearch-tab';
  const tabs = document.querySelectorAll('.tab');
  const panes = document.querySelectorAll('.tab-pane');
  function activate(name){
    tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    panes.forEach(p => p.classList.toggle('active', p.id === 'pane-' + name));
    try { localStorage.setItem(TAB_KEY, name); } catch(e) {}
  }
  tabs.forEach(t => t.addEventListener('click', () => activate(t.dataset.tab)));
  let initial = 'board';
  try { initial = localStorage.getItem(TAB_KEY) || 'board'; } catch(e) {}
  activate(initial);
})();
// topbar checklist: persist checks in localStorage, strike on check
(function(){
  const list = document.getElementById('topbar-list');
  if(!list) return;
  const KEY = 'autoresearch-checklist';
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(e) { saved = {}; }
  for(const cb of list.querySelectorAll('input[type=checkbox]')){
    if(saved[cb.id]) { cb.checked = true; cb.closest('li').classList.add('done'); }
    cb.addEventListener('change', () => {
      cb.closest('li').classList.toggle('done', cb.checked);
      saved[cb.id] = cb.checked;
      try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch(e) {}
    });
  }
})();
setInterval(load, 30000);   // poll board only; skip DOM rebuild when statuses unchanged
setInterval(loadBrief, 60000);
setInterval(loadCampaigns, 60000);
setInterval(loadActivity, 30000);

// --- GPU monitor (pane-gpu) ----------------------------------------------
(function(){
  const REFRESH_MS = 4000;
  const $ = id => document.getElementById(id);
  function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
  function setMetric(id, val, unit){
    $(id).innerHTML = val + (unit != null ? `<small>${unit}</small>` : '');
  }
  function setBar(id, pct, color){
    const b = $(id);
    b.style.width = Math.max(0, Math.min(100, pct)) + '%';
    if (color) b.style.background = color;
  }
  function drawChart(series){
    const svg = $('gpu-chart');
    const sub = $('gpu-chartsub');
    if (!series || !series.length){
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
    for (let i = 0; i <= 4; i++){
      const yv = ymin + (ymax - ymin) * i / 4;
      const yp = yn(yv);
      h += `<line class=gl x1=${M.l} y1=${yp} x2=${W - M.r} y2=${yp}/>`;
      h += `<text class=lb x=2 y=${yp + 3}>${yv.toFixed(3)}</text>`;
    }
    h += `<line class=ax x1=${M.l} y1=${H - M.b} x2=${W - M.r} y2=${H - M.b}/>`;
    h += `<line class=ax x1=${M.l} y1=${M.t} x2=${M.l} y2=${H - M.b}/>`;
    const ix = [0, Math.floor(series.length/2), series.length-1];
    for (const i of ix){
      const s = series[i]; if (!s) continue;
      h += `<text class=lb x=${xn(s.step)} y=${H - 4} text-anchor=middle>step ${s.step}</text>`;
    }
    h += `<polyline class=ln points="${pts}"/>`;
    const last = series[series.length - 1];
    h += `<circle class=pt cx=${xn(last.step)} cy=${yn(last.loss)} r=3/>`;
    svg.innerHTML = h;
    sub.textContent = `${series.length} eval pts · step ${last.step} · val ${last.loss.toFixed(4)} · acc ${last.acc.toFixed(4)} · ppl ${last.ppl.toFixed(2)}`;
  }
  function render(data){
    $('gpu-ts').textContent = new Date(data.ts * 1000).toISOString().replace('T',' ').replace(/\..*/, 'Z');
    $('gpu-host').textContent = '· ' + (data.ssh_host || '');
    const g = data.gpu || {};
    const dot = $('gpu-dot'), stTxt = $('gpu-st');
    if (g.up){
      dot.className = 'dot ok'; stTxt.textContent = 'live';
      setMetric('gpu-util', g.util, '%'); setBar('gpu-utilb', g.util);
      $('gpu-utilsub').textContent = g.name;
      setMetric('gpu-mem', (g.mem_used_mib / 1024).toFixed(1), 'GiB');
      setBar('gpu-memb', g.mem_pct);
      $('gpu-memsub').textContent = `${g.mem_used_mib} / ${g.mem_total_mib} MiB`;
      setMetric('gpu-temp', g.temp_c, '°C');
      setBar('gpu-tempb', Math.min(100, g.temp_c), g.temp_c > 80 ? 'linear-gradient(90deg,#3fb950,#d29922,#f85149)' : null);
      $('gpu-pwr').textContent = `${g.power_w} / ${g.power_max_w} W (${g.power_pct}%)`;
    } else {
      dot.className = 'dot down'; stTxt.textContent = 'GPU down';
      setMetric('gpu-util','—','%'); setBar('gpu-utilb', 0);
      setMetric('gpu-mem','—','GiB'); setBar('gpu-memb', 0);
      setMetric('gpu-temp','—','°C'); setBar('gpu-tempb', 0);
      $('gpu-utilsub').textContent = ''; $('gpu-memsub').textContent = '';
      $('gpu-pwr').textContent = g.error || 'no nvidia-smi';
    }
    const r = data.running, c = data.current || {};
    if (r){
      $('gpu-jname').innerHTML = esc(r.job) + ' <span class="pill">running</span>';
      $('gpu-jstep').textContent = c.step != null ? `${c.step} / ${c.total || '?'}` : '—';
      $('gpu-jtoks').textContent = c.tok_s != null ? Math.round(c.tok_s).toLocaleString() : '—';
      const lossEl = $('gpu-jloss');
      lossEl.textContent = c.loss != null ? c.loss : '—';
      lossEl.parentElement.classList.toggle('bad', c.loss === 'nan' || c.loss === 'NaN');
      $('gpu-jacc').textContent = c.acc != null ? c.acc.toFixed(4) : '—';
      $('gpu-jlr').textContent  = c.lr  != null ? c.lr.toExponential(2) : '—';
      $('gpu-jeta').textContent = c.eta || '—';
    } else {
      $('gpu-jname').innerHTML = 'idle <span class="pill idle">queue empty</span>';
      ['gpu-jstep','gpu-jtoks','gpu-jloss','gpu-jacc','gpu-jlr','gpu-jeta'].forEach(id => $(id).textContent = '—');
    }
    drawChart(data.series || []);
    const tbody = document.querySelector('#gpu-qtab tbody');
    tbody.innerHTML = '';
    (data.history || []).slice().reverse().forEach(e => {
      const tr = document.createElement('tr');
      tr.className = e.state;
      tr.innerHTML = `<td>${esc(e.state)}</td><td>${esc(e.job)}</td><td class=muted>${esc(e.ts)}</td>`;
      tbody.appendChild(tr);
    });
  }
  let lastOk = 0, inFlight = false;
  async function tick(){
    if (inFlight) return;
    inFlight = true;
    try {
      const r = await fetch('api/gpu', {cache: 'no-store'});
      const data = await r.json();
      render(data);
      lastOk = Date.now();
      $('gpu-dot').className = 'dot ok';
      $('gpu-st').textContent = 'live';
    } catch (e) {
      $('gpu-dot').className = 'dot stale';
      $('gpu-st').textContent = 'fetch error';
    } finally {
      inFlight = false;
    }
  }
  setInterval(() => {
    if (Date.now() - lastOk > 15000){
      $('gpu-dot').className = 'dot stale';
      $('gpu-st').textContent = 'stale';
    }
  }, 2000);
  window.addEventListener('resize', () => tick());
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    if (t.dataset.tab === 'gpu') tick();
  }));
  setInterval(tick, REFRESH_MS);
  tick();
})();

// --- Testing tab: token2science workflow snapshot -------------------------
(function(){
  const $ = id => document.getElementById(id);
  const num = v => (typeof v === 'number') ? v.toFixed(4) : '—';
  const sessionList = $('t2s-sessions');
  const sessionView = $('t2s-session-view');
  const sessionLabel = $('t2s-session-label');
  const sessionStop = $('t2s-session-stop');
  let activeSession = '';
  let sessionTimer = null;
  let sessionBusy = false;

  function setSessionText(text){
    sessionView.textContent = text || '';
    sessionView.scrollTop = sessionView.scrollHeight;
  }
  function stopSession(ended){
    if(sessionTimer){
      clearInterval(sessionTimer);
      sessionTimer = null;
    }
    sessionBusy = false;
    sessionStop.disabled = true;
    sessionLabel.textContent = ended ? 'session ended' : 'polling stopped';
    if(ended){
      activeSession = '';
    } else {
      activeSession = '';
    }
    renderSessionList((lastT2sData && lastT2sData.sessions) || []);
  }
  function openSession(name){
    if(!name) return;
    activeSession = name;
    sessionLabel.textContent = `live: ${name}`;
    sessionStop.disabled = false;
    setSessionText('loading…');
    renderSessionList((lastT2sData && lastT2sData.sessions) || []);
    if(sessionTimer){
      clearInterval(sessionTimer);
      sessionTimer = null;
    }
    tickSession();
    sessionTimer = setInterval(tickSession, 2000);
  }
  async function tickSession(){
    if(!activeSession || sessionBusy) return;
    sessionBusy = true;
    try {
      const r = await fetch(`/api/t2s/session?name=${encodeURIComponent(activeSession)}`, {cache:'no-store'});
      const txt = await r.text();
      if(!r.ok){
        setSessionText(txt || '(session ended)');
        stopSession(true);
        return;
      }
      const body = (txt || '').trim();
      if(body === '(session ended)'){
        setSessionText(body);
        stopSession(true);
        return;
      }
      setSessionText(txt || '');
    } catch(e) {
      setSessionText('(session fetch failed)');
    } finally {
      sessionBusy = false;
    }
  }
  function renderSessionList(sessions){
    const names = sessions && sessions.length ? sessions : [];
    sessionList.innerHTML = names.length
      ? names.map(s => `<a href="#" class="t2s-sess-link${s===activeSession?' active':''}" data-sess="${htmlEscape(s)}"><span class=t2s-live></span><span class=name>${htmlEscape(s)}</span></a>`).join('')
      : '<span class=hint>none active</span>';
  }
  let lastT2sData = null;
  sessionList.addEventListener('click', e => {
    const a = e.target.closest('[data-sess]');
    if(!a) return;
    e.preventDefault();
    openSession(a.dataset.sess);
  });
  sessionStop.addEventListener('click', () => {
    stopSession(false);
    sessionLabel.textContent = 'polling stopped';
  });
  function render(d){
    if(!d){ return; }
    lastT2sData = d;
    $('t2s-meta').textContent = d.error ? d.error : new Date(d.ts*1000).toLocaleTimeString();
    const T = d.totals || {};
    $('t2s-totals').innerHTML =
      `<span class=t2s-chip><b>${T.active_sessions||0}</b> active agents</span>`+
      `<span class=t2s-chip><b>${T.goals||0}</b> goals</span>`+
      `<span class=t2s-chip><b>${T.runs||0}</b> runs</span>`+
      `<span class=t2s-chip><b>${T.confirmed||0}</b> confirmed</span>`+
      `<span class=t2s-chip><b>${T.papers||0}</b> papers</span>`;
    renderSessionList(d.sessions || []);
    $('t2s-goals').innerHTML = (d.goals && d.goals.length) ? d.goals.map(g => {
      const dir = g.lower_is_better ? '&lt;' : '&gt;';
      const beat = g.beats==null ? '' : (g.beats ? '<span class=t2s-ok>beats bar</span>' : '<span class=t2s-below>below bar</span>');
      return `<div class=t2s-goal><b>${g.id}</b> <span class=t2s-mut>${g.metric} ${dir} ${g.bar}</span> · ${g.nruns} runs · best ${num(g.best)} ${beat}</div>`;
    }).join('') : '<span class=hint>none</span>';
    $('t2s-runs').innerHTML = (d.runs && d.runs.length) ? d.runs.map(r =>
      `<div class=t2s-run><code>${r.task}</code> <span class=t2s-mut>${r.worker}</span> ${r.metric}=${num(r.value)}</div>`
    ).join('') : '<span class=hint>none yet</span>';
  }
  async function t2sTick(){
    try { const r = await fetch('/api/t2s'); render(await r.json()); } catch(e){}
  }
  window.t2sTick = t2sTick;
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    if (t.dataset.tab === 'testing') t2sTick();
  }));
  setInterval(() => {
    const p = document.getElementById('pane-testing');
    if (p && p.classList.contains('active')) t2sTick();
  }, 5000);
  t2sTick();
})();

// --- Papers tab: structured token2science papers --------------------------
(function(){
  const $ = id => document.getElementById(id);
  const paperCards = $('t2s-paper-cards');
  const paperReader = $('t2s-paper-reader');
  const paperFilters = $('t2s-paper-filters');
  const paperMeta = $('t2s-papers-meta');
  const paperStats = $('t2s-paper-stats');
  let papers = [];
  let activePaper = '';
  let activeAuthor = '';
  let lastJson = '';
  let loading = false;

  function paperBasePath(p){
    const manuscript = String(p.manuscript || 'manuscript.md').replace(/\\/g, '/');
    const idx = manuscript.lastIndexOf('/');
    const subdir = idx >= 0 ? manuscript.slice(0, idx) : '';
    return subdir ? `papers/${p.paper_id}/${subdir}` : `papers/${p.paper_id}`;
  }
  function visiblePapers(){
    return papers.filter(p => !activeAuthor || (p.authors || []).includes(activeAuthor));
  }
  function setReaderHint(msg){
    delete paperReader.dataset.paperId;
    paperReader.innerHTML = `<span class=hint>${htmlEscape(msg)}</span>`;
  }
  function renderPaperReader(paper, txt){
    const base = paperBasePath(paper);
    const body = renderMarkdown(txt, {paperAsset:true, paperBase:base});
    paperReader.dataset.paperId = paper.paper_id;
    paperReader.innerHTML =
      `<h2>${htmlEscape(paper.title || paper.paper_id)}</h2>`+
      `<div class=meta>`+
      `<span><b>authors</b> ${htmlEscape((paper.authors || []).join(', ') || '—')}</span>`+
      `<span><b>mechanism</b> ${htmlEscape(paper.mechanism_name || '—')}</span>`+
      `<span><b>status</b> ${htmlEscape(paper.status || '—')}</span>`+
      `<span><b>experiments</b> ${htmlEscape(String(paper.num_experiments ?? 0))}</span>`+
      `</div>`+
      `<div class=path>${htmlEscape(`papers/${paper.paper_id}/${paper.manuscript || 'manuscript.md'}`)}</div>`+
      `<div class=body><div class=md>${body}</div></div>`;
    const bodyEl = paperReader.querySelector('.body');
    if(bodyEl) bodyEl.scrollTop = 0;
  }
  function renderAuthorFilters(){
    const authors = [...new Set(papers.flatMap(p => p.authors || []))].sort((a, b) => a.localeCompare(b));
    if(!authors.length){
      paperFilters.innerHTML = '<span class=hint>no authors found</span>';
      return;
    }
    const chips = [`<button type=button class="t2s-author-chip${activeAuthor ? '' : ' active'}" data-author="">All authors</button>`]
      .concat(authors.map(a => `<button type=button class="t2s-author-chip${a === activeAuthor ? ' active' : ''}" data-author="${htmlEscape(a)}">${htmlEscape(a)}</button>`));
    paperFilters.innerHTML = chips.join('');
  }
  function renderCards(){
    const list = visiblePapers();
    const total = papers.length;
    const filtered = list.length;
    const authorCount = [...new Set(papers.flatMap(p => p.authors || []))].length;
    paperMeta.textContent = `${total} papers · ${authorCount} authors${activeAuthor ? ` · filter: ${activeAuthor}` : ''}`;
    paperStats.innerHTML =
      `<span class=t2s-paper-chip><b>${filtered}</b> visible</span>`+
      `<span class=t2s-paper-chip><b>${total}</b> total</span>`+
      `<span class=t2s-paper-chip><b>${authorCount}</b> authors</span>`;
    if(!list.length){
      paperCards.innerHTML = '<div class=t2s-paper-empty>no papers match this author filter</div>';
      return;
    }
    paperCards.innerHTML = list.map(p => {
      const authors = (p.authors || []).map(a => `<button type=button class=t2s-paper-author data-author="${htmlEscape(a)}">${htmlEscape(a)}</button>`).join('');
      return `<div class="t2s-paper-card${p.paper_id === activePaper ? ' active' : ''}" data-paper-id="${htmlEscape(p.paper_id)}">`+
        `<div class=t2s-paper-title>${htmlEscape(p.title || p.paper_id)}</div>`+
        `<div class=t2s-paper-authors>${authors || '<span class=t2s-paper-empty>no authors</span>'}</div>`+
        `<div class=t2s-paper-rows>`+
          `<div><b>mechanism</b> ${htmlEscape(p.mechanism_name || '—')}</div>`+
          `<div><b>status</b> ${htmlEscape(p.status || '—')}</div>`+
          `<div><b>experiments</b> ${htmlEscape(String(p.num_experiments ?? 0))}</div>`+
          `<div><b>manuscript</b> ${htmlEscape(p.manuscript || '—')}</div>`+
        `</div>`+
      `</div>`;
    }).join('');
  }
  async function openPaper(paperId){
    const paper = papers.find(p => p.paper_id === paperId);
    if(!paper) return;
    activePaper = paperId;
    renderCards();
    paperReader.dataset.paperId = paper.paper_id;
    paperReader.innerHTML = `<h2>${htmlEscape(paper.title || paper.paper_id)}</h2><div class=body><span class=hint>loading…</span></div>`;
    try {
      const r = await fetch(`/api/t2s/paper?name=${encodeURIComponent(paper.paper_id)}`, {cache:'no-store'});
      if(!r.ok){
        paperReader.innerHTML = `<h2>${htmlEscape(paper.title || paper.paper_id)}</h2><div class=body><span class=hint>could not load paper</span></div>`;
        return;
      }
      const txt = await r.text();
      renderPaperReader(paper, txt);
    } catch(e) {
      paperReader.innerHTML = `<h2>${htmlEscape(paper.title || paper.paper_id)}</h2><div class=body><span class=hint>paper fetch failed</span></div>`;
    }
  }
  function render(data){
    if(!data) return;
    const json = JSON.stringify(data);
    if(json === lastJson) return;
    lastJson = json;
    papers = Array.isArray(data) ? data : [];
    if(activeAuthor && !papers.some(p => (p.authors || []).includes(activeAuthor))){
      activeAuthor = '';
    }
    if(activePaper && !papers.some(p => p.paper_id === activePaper)){
      activePaper = '';
      setReaderHint('click a paper card to render its manuscript here');
    }
    if(!papers.length){
      paperMeta.textContent = '0 papers';
      paperStats.innerHTML = '<span class=t2s-paper-chip><b>0</b> visible</span>';
      paperFilters.innerHTML = '<span class=hint>no structured papers found</span>';
      paperCards.innerHTML = '<div class=t2s-paper-empty>no structured papers found under token2science/papers/*/paper.json</div>';
      setReaderHint('no structured papers found yet');
      return;
    }
    renderAuthorFilters();
    renderCards();
    if(activePaper && paperReader.dataset.paperId !== activePaper){
      const paper = papers.find(p => p.paper_id === activePaper);
      if(paper) openPaper(activePaper);
    } else if(!activePaper && !paperReader.querySelector('.md')){
      setReaderHint('click a paper card to render its manuscript here');
    }
  }
  async function t2sPapersTick(){
    if(loading) return;
    loading = true;
    try {
      const r = await fetch('/api/t2s/papers', {cache:'no-store'});
      if(r.ok) render(await r.json());
    } catch(e) {}
    finally { loading = false; }
  }
  paperFilters.addEventListener('click', e => {
    const a = e.target.closest('[data-author]');
    if(!a) return;
    activeAuthor = a.dataset.author || '';
    renderAuthorFilters();
    renderCards();
  });
  paperCards.addEventListener('click', e => {
    const chip = e.target.closest('[data-author]');
    if(chip){
      e.preventDefault();
      e.stopPropagation();
      activeAuthor = chip.dataset.author || '';
      renderAuthorFilters();
      renderCards();
      return;
    }
    const card = e.target.closest('[data-paper-id]');
    if(!card) return;
    e.preventDefault();
    openPaper(card.dataset.paperId);
  });
  window.t2sPapersTick = t2sPapersTick;
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    if (t.dataset.tab === 'papers') t2sPapersTick();
  }));
  setInterval(() => {
    const p = document.getElementById('pane-papers');
    if (p && p.classList.contains('active')) t2sPapersTick();
  }, 5000);
  t2sPapersTick();
})();
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
            page = PAGE.replace("__GROUPS__", json.dumps(groups))
            page = page.replace("__TASTE__", json.dumps(sorted(TASTE_STATUSES)))
            self._send(200, page)
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
        elif u.path == "/api/t2s/papers":
            self._send(200, json.dumps(_t2s_papers_list()), "application/json")
        elif u.path == "/api/t2s/paper":
            name = parse_qs(u.query).get("name", [""])[0]
            txt = _t2s_paper_text(name)
            if txt is None:
                self._send(404, "not found", "text/plain")
            else:
                self._send(200, txt, "text/markdown; charset=utf-8")
        elif u.path == "/api/t2s/asset":
            relpath = parse_qs(u.query).get("path", [""])[0]
            body, ctype, code = _t2s_asset_bytes(relpath)
            if body is None:
                self._send(code, "not found", "text/plain")
            else:
                self._send(200, body, ctype)
        elif u.path == "/api/t2s/session":
            name = parse_qs(u.query).get("name", [""])[0]
            txt = _t2s_session_capture(name)
            if txt is None:
                self._send(400, "bad session", "text/plain")
            else:
                self._send(200, txt, "text/plain; charset=utf-8")
        elif u.path == "/api/gpu":
            try:
                data = _gather_gpu()
            except Exception as e:
                data = {"ts": time.time(), "error": str(e)}
            self._send(200, json.dumps(data), "application/json")
        elif u.path == "/api/t2s":
            try:
                data = _gather_t2s()
            except Exception as e:
                data = {"ts": time.time(), "error": str(e)}
            self._send(200, json.dumps(data), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/flip":
            self._send(404, "not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            req = json.loads(body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            self._send(400, json.dumps({"error": "bad json"}), "application/json")
            return
        slugs = req.get("slugs") or []
        action = req.get("action")
        if not isinstance(slugs, list) or action not in ("approve", "reject"):
            self._send(400, json.dumps({"error": "need {slugs:[...], action:approve|reject}"}), "application/json")
            return
        results = []
        for slug in slugs:
            ok, msg = flip_idea(str(slug), action)
            results.append({"slug": slug, "ok": ok, "msg": msg})
        self._send(200, json.dumps({"results": results}), "application/json")

    def log_message(self, *a):
        pass

class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded so one slow handler (e.g. the GPU SSH poll) can't freeze the UI."""
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    with _ThreadingServer(("127.0.0.1", PORT), H) as httpd:
        print(f"serving autoresearch board on http://localhost:{PORT}")
        httpd.serve_forever()
