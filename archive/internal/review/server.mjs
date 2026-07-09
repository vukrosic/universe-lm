import { createServer } from 'node:http';
import { readFile, writeFile, readdir, mkdir } from 'node:fs/promises';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const queueDir = join(rootDir, 'queue');
const ideasDir = join(rootDir, 'autoresearch', 'ideas');
const reviewsPath = join(queueDir, 'reviews.json');
const ideaReviewsPath = join(rootDir, 'autoresearch', 'idea-reviews.json');
const port = 4500;
const host = '127.0.0.1';
const allowedOriginPattern = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;

function corsHeaders(origin) {
  if (!origin || !allowedOriginPattern.test(origin)) {
    return {};
  }

  return {
    'access-control-allow-origin': origin,
    'access-control-allow-methods': 'GET, POST, OPTIONS',
    'access-control-allow-headers': 'content-type',
    'access-control-max-age': '86400',
    vary: 'Origin',
  };
}

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Queue Review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101112;
      --panel: #17181a;
      --panel-2: #1d1f22;
      --border: #2c2e33;
      --text: #faf9f6;
      --muted: #b7b1a6;
      --soft: #8d8a81;
      --good: #8ccf97;
      --bad: #e48787;
      --keep: #d0bc7a;
      --accent: #f0d9a8;
      --shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(240, 217, 168, 0.08), transparent 30%),
        radial-gradient(circle at right 10%, rgba(140, 207, 151, 0.06), transparent 22%),
        var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .wrap {
      width: min(1200px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: -0.04em;
      line-height: 1;
    }

    .sub {
      color: var(--muted);
      margin-top: 8px;
      font-size: 14px;
    }

    .statusline {
      color: var(--soft);
      font-size: 13px;
      white-space: nowrap;
    }

    .grid {
      display: grid;
      gap: 14px;
    }

    .card {
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01)), var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }

    .top {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }

    .id {
      font-size: 14px;
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.02em;
      margin-bottom: 6px;
    }

    .title {
      margin: 0;
      font-size: 20px;
      line-height: 1.15;
      letter-spacing: -0.02em;
    }

    .pill {
      flex: none;
      display: inline-flex;
      align-items: center;
      padding: 7px 10px;
      border-radius: 999px;
      background: var(--panel-2);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .buttons {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    button {
      border: 1px solid var(--border);
      background: #121316;
      color: var(--text);
      border-radius: 999px;
      padding: 9px 14px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }

    button:hover { transform: translateY(-1px); border-color: #3b3e44; }
    button:focus-visible, textarea:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }

    button.active[data-decision="approve"] {
      background: rgba(140, 207, 151, 0.18);
      border-color: rgba(140, 207, 151, 0.55);
      color: #d6f7dc;
    }
    button.active[data-decision="disapprove"] {
      background: rgba(228, 135, 135, 0.18);
      border-color: rgba(228, 135, 135, 0.55);
      color: #ffd6d6;
    }
    button.active[data-decision="keep"] {
      background: rgba(208, 188, 122, 0.18);
      border-color: rgba(208, 188, 122, 0.55);
      color: #fff2cb;
    }

    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      margin-top: 14px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #121316;
      color: var(--text);
      padding: 12px 13px;
      font: inherit;
      line-height: 1.4;
    }

    .footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-top: 12px;
      color: var(--soft);
      font-size: 12px;
    }

    .saved {
      min-height: 1em;
      color: var(--good);
    }

    .error {
      color: var(--bad);
      margin: 16px 0 0;
    }

    @media (max-width: 700px) {
      header, .top, .footer { flex-direction: column; align-items: flex-start; }
      .statusline { white-space: normal; }
      .wrap { width: min(100vw - 20px, 1200px); padding-top: 16px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Queue review</h1>
        <div class="sub">Review each spec, choose a decision, and leave a note. Saved state is kept in <code>queue/reviews.json</code>.</div>
      </div>
      <div class="statusline" id="summary">Loading specs…</div>
    </header>
    <div class="grid" id="app"></div>
    <div class="error" id="error"></div>
  </div>

  <script>
    const app = document.getElementById('app');
    const summary = document.getElementById('summary');
    const errorBox = document.getElementById('error');
    const decisions = ['approve', 'disapprove', 'keep'];
    const state = new Map();

    function esc(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function setError(message) {
      errorBox.textContent = message || '';
    }

    async function loadSpecs() {
      const res = await fetch('/api/specs');
      if (!res.ok) throw new Error('Failed to load specs');
      const specs = await res.json();
      render(specs);
    }

    function render(specs) {
      summary.textContent = specs.length + ' specs loaded';
      app.innerHTML = specs.map(spec => {
        const decision = spec.decision || 'keep';
        const note = spec.note || '';
        return \`
          <section class="card" data-id="\${esc(spec.id)}">
            <div class="top">
              <div>
                <div class="id">\${esc(spec.id)}</div>
                <h2 class="title">\${esc(spec.title)}</h2>
              </div>
              <div class="pill">\${esc(spec.status)}</div>
            </div>
            <div class="row">
              <div class="buttons" role="group" aria-label="Decision for \${esc(spec.id)}">
                \${decisions.map(choice => \`
                  <button type="button" class="\${choice === decision ? 'active' : ''}" data-decision="\${choice}">\${choice}</button>
                \`).join('')}
              </div>
              <button type="button" data-action="save">save</button>
              <span class="saved" data-saved>\${spec.updated_at ? 'saved ' + new Date(spec.updated_at).toLocaleString() : ''}</span>
            </div>
            <textarea placeholder="Notes for orchestrator…" data-note>\${esc(note)}</textarea>
            <div class="footer">
              <span>Decision and note are stored in the sidecar file only.</span>
              <span data-meta>\${spec.updated_at ? esc(spec.updated_at) : 'unsaved'}</span>
            </div>
          </section>
        \`;
      }).join('');

      app.querySelectorAll('.card').forEach(card => {
        const id = card.dataset.id;
        const noteEl = card.querySelector('[data-note]');
        const savedEl = card.querySelector('[data-saved]');
        const metaEl = card.querySelector('[data-meta]');
        const saveButton = card.querySelector('[data-action="save"]');
        let currentDecision = card.querySelector('button.active[data-decision]')?.dataset.decision || 'keep';
        let saveTimer = null;
        let inFlight = false;

        const setActive = nextDecision => {
          currentDecision = nextDecision;
          card.querySelectorAll('[data-decision]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.decision === nextDecision);
          });
          queueSave();
        };

        const writeSavedState = updatedAt => {
          savedEl.textContent = 'saved ' + new Date(updatedAt).toLocaleString();
          metaEl.textContent = updatedAt;
        };

        async function saveNow() {
          if (inFlight) return;
          inFlight = true;
          saveButton.textContent = 'saving…';
          try {
            const res = await fetch('/api/review', {
              method: 'POST',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({
                id,
                decision: currentDecision,
                note: noteEl.value
              })
            });
            if (!res.ok) throw new Error('Save failed');
            const updated = await res.json();
            writeSavedState(updated.updated_at);
            savedEl.textContent = 'saved';
            setTimeout(() => {
              if (savedEl.textContent === 'saved') {
                savedEl.textContent = 'saved ' + new Date(updated.updated_at).toLocaleString();
              }
            }, 1200);
          } finally {
            saveButton.textContent = 'save';
            inFlight = false;
          }
        }

        function queueSave() {
          clearTimeout(saveTimer);
          saveTimer = setTimeout(saveNow, 350);
        }

        card.querySelectorAll('[data-decision]').forEach(btn => {
          btn.addEventListener('click', () => setActive(btn.dataset.decision));
        });
        noteEl.addEventListener('input', queueSave);
        saveButton.addEventListener('click', saveNow);
        state.set(id, { setActive, queueSave });
      });
    }

    loadSpecs().catch(err => {
      setError(err.message || String(err));
      summary.textContent = 'load failed';
    });
  </script>
</body>
</html>`;

function parseField(source, field) {
  const match = source.match(new RegExp(`^${field}:\\s*(.+)$`, 'm'));
  if (!match) return '';
  let value = match[1].trim();
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    value = value.slice(1, -1);
  }
  return value;
}

function parseLatestVerdict(source) {
  const match = source.match(/^##\s+r\d+\s+—.*?verdict:\s*([^\n]+)$/m);
  if (!match) return 'pending';
  const verdict = match[1].trim().toLowerCase();
  if (verdict === 'approve') return 'accept';
  return verdict || 'pending';
}

function parseEvidenceVerdict(source) {
  const match = source.match(/^##\s+Verdict:\s*([^\n]+)$/m);
  return match ? match[1].trim() : null;
}

function parseTimeline(source) {
  const timeline = [];
  for (const line of source.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (!entry || typeof entry !== 'object') continue;
      timeline.push({
        ts: typeof entry.ts === 'string' ? entry.ts : '',
        from: typeof entry.from === 'string' ? entry.from : '',
        to: typeof entry.to === 'string' ? entry.to : '',
        agent: typeof entry.agent === 'string' ? entry.agent : '',
        note: typeof entry.note === 'string' ? entry.note : '',
      });
    } catch {
      // Ignore malformed lines and keep the rest of the history.
    }
  }
  return timeline;
}

function stripFrontmatter(source) {
  if (!source.startsWith('---')) {
    return source.replace(/^\uFEFF/, '');
  }

  const match = source.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/);
  return (match ? source.slice(match[0].length) : source).replace(/^\uFEFF/, '');
}

function extractIdeaName(source, fallback) {
  const body = stripFrontmatter(source);
  const match = body.match(/^\s*#\s+(.+?)\s*$/m);
  return (match ? match[1].trim() : fallback).trim() || fallback;
}

function extractIdeaSummary(source) {
  const body = stripFrontmatter(source).trim();
  if (!body) {
    return '';
  }

  const paragraphs = body
    .split(/\r?\n\s*\r?\n/)
    .map(paragraph => paragraph.trim())
    .filter(Boolean);

  const firstContent = paragraphs.find(paragraph => !/^#{1,6}\s+/.test(paragraph)) || body;
  const summary = firstContent.replace(/\s+/g, ' ').trim();
  return summary.length > 240 ? summary.slice(0, 240).trimEnd() : summary;
}

function normalizeSuggestionDecision(value) {
  if (value === 'approve' || value === 'disapprove' || value === 'request-review' || value === 'request-improvement') {
    return value;
  }
  return null;
}

function compareSuggestions(a, b) {
  const aUnreviewed = !a.decision;
  const bUnreviewed = !b.decision;
  if (aUnreviewed !== bUnreviewed) {
    return aUnreviewed ? -1 : 1;
  }

  if (a.round !== b.round) {
    return b.round - a.round;
  }

  const statusCompare = a.status.localeCompare(b.status);
  if (statusCompare !== 0) {
    return statusCompare;
  }

  return a.id.localeCompare(b.id, undefined, { numeric: true, sensitivity: 'base' });
}

async function loadSuggestionReviews() {
  try {
    const raw = await readFile(ideaReviewsPath, 'utf8');
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return {};
    }
  }
  return {};
}

async function saveSuggestionReviews(reviews) {
  await mkdir(dirname(ideaReviewsPath), { recursive: true });
  await writeFile(ideaReviewsPath, JSON.stringify(reviews, null, 2) + '\n', 'utf8');
}

async function listSuggestions() {
  let entries = [];
  try {
    entries = await readdir(ideasDir, { withFileTypes: true });
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return [];
    }
    throw error;
  }

  const reviews = await loadSuggestionReviews();
  const suggestions = await Promise.all(
    entries
      .filter(entry => entry.isDirectory())
      .map(async entry => {
        const ideaPath = join(ideasDir, entry.name, 'idea.md');
        try {
          const ideaMd = await readFile(ideaPath, 'utf8');
          const id = parseField(ideaMd, 'id') || entry.name;
          const status = parseField(ideaMd, 'status') || 'unknown';
          const round = Number.parseInt(parseField(ideaMd, 'round'), 10) || 0;
          const review = reviews[id] || {};
          return {
            id,
            name: extractIdeaName(ideaMd, entry.name),
            status,
            round,
            summary: extractIdeaSummary(ideaMd),
            decision: normalizeSuggestionDecision(review.decision),
            note: typeof review.note === 'string' ? review.note : '',
            updated_at: typeof review.updated_at === 'string' ? review.updated_at : '',
          };
        } catch (error) {
          if (error && error.code === 'ENOENT') {
            return null;
          }
          throw error;
        }
      })
  );

  return suggestions.filter(Boolean).sort(compareSuggestions);
}

async function readIdeaSummary(slug) {
  if (!/^[A-Za-z0-9._-]+$/.test(slug)) {
    return null;
  }

  const ideaPath = join(ideasDir, slug);

  try {
    const [ideaMd, tasteMd, reviewMd, codeMd, logJsonl] = await Promise.all([
      readFile(join(ideaPath, 'idea.md'), 'utf8'),
      readFile(join(ideaPath, 'taste.md'), 'utf8').catch(error => (error && error.code === 'ENOENT' ? '' : Promise.reject(error))),
      readFile(join(ideaPath, 'review.md'), 'utf8').catch(error => (error && error.code === 'ENOENT' ? '' : Promise.reject(error))),
      readFile(join(ideaPath, 'codereview.md'), 'utf8').catch(error => (error && error.code === 'ENOENT' ? '' : Promise.reject(error))),
      readFile(join(ideaPath, 'log.jsonl'), 'utf8').catch(error => (error && error.code === 'ENOENT' ? '' : Promise.reject(error))),
    ]);

    return {
      slug,
      status: parseField(ideaMd, 'status') || 'unknown',
      round: Number.parseInt(parseField(ideaMd, 'round'), 10) || 0,
      gates: {
        taste: tasteMd ? parseLatestVerdict(tasteMd) : 'pending',
        definition: reviewMd ? parseLatestVerdict(reviewMd) : 'pending',
        code: codeMd ? parseLatestVerdict(codeMd) : 'pending',
      },
      ran: false,
      evidence_verdict: null,
      timeline: parseTimeline(logJsonl),
    };
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

function deriveStatus(fileName) {
  const stem = basename(fileName, '.yaml');
  const suffix = stem.split('.').pop();
  return suffix || 'unknown';
}

const reportSectionSpecs = [
  { key: 'evidence', title: 'Evidence', file: 'evidence.md' },
  { key: 'plan', title: 'Plan', file: 'plan.md' },
  { key: 'review', title: 'Definition review', file: 'review.md' },
  { key: 'taste', title: 'Taste review', file: 'taste.md' },
  { key: 'codereview', title: 'Code review', file: 'codereview.md' },
  { key: 'idea', title: 'Idea', file: 'idea.md' },
];

async function readReport(slug) {
  if (!/^[A-Za-z0-9._-]+$/.test(slug)) {
    return null;
  }

  const ideaPath = join(ideasDir, slug);

  try {
    await readdir(ideaPath);
  } catch (error) {
    if (error && error.code === 'ENOENT') return null;
    throw error;
  }

  const sections = [];
  for (const section of reportSectionSpecs) {
    try {
      const markdown = await readFile(join(ideaPath, section.file), 'utf8');
      sections.push({
        key: section.key,
        title: section.title,
        markdown,
      });
    } catch (error) {
      if (error && error.code === 'ENOENT') {
        continue;
      }
      throw error;
    }
  }

  return {
    slug,
    sections,
  };
}

async function loadReviews() {
  try {
    const raw = await readFile(reviewsPath, 'utf8');
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch (error) {
    if (error && error.code === 'ENOENT') return {};
  }
  return {};
}

async function saveReviews(reviews) {
  await mkdir(queueDir, { recursive: true });
  await writeFile(reviewsPath, JSON.stringify(reviews, null, 2) + '\n', 'utf8');
}

async function listSpecs() {
  const reviews = await loadReviews();
  const files = (await readdir(queueDir))
    .filter(name => name.endsWith('.yaml'))
    .sort((a, b) => a.localeCompare(b));

  const specs = [];
  for (const file of files) {
    const fullPath = join(queueDir, file);
    const contents = await readFile(fullPath, 'utf8');
    const id = parseField(contents, 'id');
    const title = parseField(contents, 'title');
    const plain = parseField(contents, 'plain');
    const review = reviews[id] || {};
    specs.push({
      id,
      title,
      plain,
      status: deriveStatus(file),
      decision: review.decision || '',
      note: review.note || '',
      updated_at: review.updated_at || ''
    });
  }

  return specs;
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error('Payload too large'));
        req.destroy();
      }
    });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (error) {
        reject(error);
      }
    });
    req.on('error', reject);
  });
}

function send(res, statusCode, body, headers = {}, origin = '') {
  res.writeHead(statusCode, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    ...corsHeaders(origin),
    ...headers
  });
  res.end(JSON.stringify(body));
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || host}`);
    const origin = typeof req.headers.origin === 'string' ? req.headers.origin : '';

    if (req.method === 'OPTIONS' && (url.pathname === '/api/specs' || url.pathname === '/api/review' || url.pathname.startsWith('/api/idea/') || url.pathname.startsWith('/api/report/') || url.pathname === '/api/suggestions' || url.pathname === '/api/suggestion-review')) {
      res.writeHead(204, {
        'cache-control': 'no-store',
        ...corsHeaders(origin),
      });
      res.end();
      return;
    }

    if (req.method === 'GET' && url.pathname === '/') {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store', ...corsHeaders(origin) });
      res.end(html);
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/specs') {
      send(res, 200, await listSpecs(), {}, origin);
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/suggestions') {
      send(res, 200, await listSuggestions(), {}, origin);
      return;
    }

    const reportMatch = url.pathname.match(/^\/api\/report\/([^/]+)$/);
    if (req.method === 'GET' && reportMatch) {
      const report = await readReport(decodeURIComponent(reportMatch[1]));
      if (!report) {
        send(res, 404, {}, {}, origin);
        return;
      }

      send(res, 200, report, {}, origin);
      return;
    }

    const ideaMatch = url.pathname.match(/^\/api\/idea\/([^/]+)$/);
    if (req.method === 'GET' && ideaMatch) {
      const idea = await readIdeaSummary(decodeURIComponent(ideaMatch[1]));
      if (!idea) {
        send(res, 404, {}, {}, origin);
        return;
      }

      try {
        const evidencePath = join(ideasDir, idea.slug, 'evidence.md');
        const evidenceMd = await readFile(evidencePath, 'utf8');
        idea.ran = true;
        idea.evidence_verdict = parseEvidenceVerdict(evidenceMd);
      } catch (error) {
        if (!error || error.code !== 'ENOENT') {
          throw error;
        }
      }

      if (idea.evidence_verdict == null && idea.ran === false) {
        // no-op, explicit for readability
      }

      send(res, 200, idea, {}, origin);
      return;
    }

    if (req.method === 'POST' && url.pathname === '/api/suggestion-review') {
      const { id, decision, note } = await readJson(req);
      if (!id || !['approve', 'disapprove', 'request-review', 'request-improvement'].includes(decision)) {
        send(res, 400, { error: 'Invalid review payload' }, {}, origin);
        return;
      }

      const reviews = await loadSuggestionReviews();
      const entry = {
        id,
        decision,
        note: typeof note === 'string' ? note : '',
        updated_at: new Date().toISOString(),
      };
      reviews[id] = entry;
      await saveSuggestionReviews(reviews);
      send(res, 200, entry, {}, origin);
      return;
    }

    if (req.method === 'POST' && url.pathname === '/api/review') {
      const { id, decision, note } = await readJson(req);
      if (!id || !['approve', 'disapprove', 'keep'].includes(decision)) {
        send(res, 400, { error: 'Invalid review payload' }, {}, origin);
        return;
      }
      const reviews = await loadReviews();
      const entry = {
        id,
        decision,
        note: typeof note === 'string' ? note : '',
        updated_at: new Date().toISOString()
      };
      reviews[id] = entry;
      await saveReviews(reviews);
      send(res, 200, entry, {}, origin);
      return;
    }

    send(res, 404, { error: 'Not found' }, {}, origin);
  } catch (error) {
    send(res, 500, { error: error.message || 'Internal error' }, {}, typeof req?.headers?.origin === 'string' ? req.headers.origin : '');
  }
});

server.listen(port, host, () => {
  console.log(`Queue review server running at http://${host}:${port}`);
});
