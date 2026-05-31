PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS threads (
    name TEXT PRIMARY KEY,
    hypothesis TEXT,
    owner TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    priority INTEGER NOT NULL DEFAULT 0,
    notes_path TEXT,
    summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS queue_items (
    id TEXT PRIMARY KEY,
    thread_name TEXT NOT NULL,
    name TEXT NOT NULL,
    command TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    depends_on TEXT,
    gpu_class TEXT,
    estimated_minutes REAL,
    created_by TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    log_path TEXT,
    output_dir TEXT,
    decision TEXT,
    FOREIGN KEY(thread_name) REFERENCES threads(name) ON DELETE CASCADE,
    UNIQUE(thread_name, name)
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    queue_item_id TEXT,
    thread_name TEXT NOT NULL,
    name TEXT NOT NULL,
    command TEXT,
    code_commit TEXT,
    branch TEXT,
    config TEXT,
    seed INTEGER,
    tokens_target INTEGER,
    tokens_seen INTEGER,
    actual_steps INTEGER,
    status TEXT NOT NULL,
    verdict TEXT,
    output_dir TEXT,
    metrics_path TEXT,
    checkpoint_path TEXT,
    final_val_loss REAL,
    final_val_accuracy REAL,
    final_train_loss REAL,
    git_commit TEXT,
    git_branch TEXT,
    git_dirty INTEGER,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(queue_item_id) REFERENCES queue_items(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS eval_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    tokens INTEGER,
    val_loss REAL,
    val_accuracy REAL,
    val_perplexity REAL,
    learning_rate REAL,
    elapsed_seconds REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    baseline_name TEXT,
    baseline_run_id TEXT,
    matched_step INTEGER,
    matched_tokens INTEGER,
    baseline_val_loss REAL,
    run_val_loss REAL,
    delta_val_loss REAL,
    verdict TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_name TEXT,
    run_id TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    decided_by TEXT,
    decided_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS ideas (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    explanation TEXT,
    confidence TEXT,
    expected_gain TEXT,
    pros TEXT,
    cons TEXT,
    outcome TEXT,
    reference_url TEXT,
    thread_name TEXT,
    hypothesis TEXT,
    lever TEXT,
    rationale TEXT,
    expected_effect TEXT,
    scale_target TEXT,
    command TEXT,
    gpu_class TEXT,
    estimated_minutes REAL,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'proposed',
    proposed_by TEXT,
    reviewed_by TEXT,
    review_note TEXT,
    reviewed_at TEXT,
    queue_item_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(thread_name) REFERENCES threads(name) ON DELETE SET NULL,
    FOREIGN KEY(queue_item_id) REFERENCES queue_items(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status, priority);
CREATE INDEX IF NOT EXISTS idx_queue_items_thread_status ON queue_items(thread_name, status);
CREATE INDEX IF NOT EXISTS idx_runs_thread_status ON runs(thread_name, status);
CREATE INDEX IF NOT EXISTS idx_eval_points_run_step ON eval_points(run_id, step);
CREATE INDEX IF NOT EXISTS idx_comparisons_run ON comparisons(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_thread ON decisions(thread_name);
