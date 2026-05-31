"""Registry dashboard. Read-only views + an Ideas approval gate (write)."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

DB = Path(__file__).parent / "experiments.sqlite"


def get_db():
    return sqlite3.connect(DB)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def review_idea(idea_id, status, reviewed_by="human", note=None):
    """Write-mode: approve/reject an idea straight from the browser."""
    conn = get_db()
    conn.execute(
        "UPDATE ideas SET status=?, reviewed_by=?, review_note=?, reviewed_at=?, updated_at=? WHERE id=?",
        (status, reviewed_by, note, utc_now(), utc_now(), idea_id),
    )
    conn.commit()
    conn.close()


def make_link(path):
    if path:
        return f"[{path}](file://{path})"
    return ""


def delta_color(val):
    try:
        v = float(val)
        if v < 0:
            return "🟢"
        elif v > 0:
            return "🔴"
    except:
        pass
    return ""


def main():
    st.set_page_config(page_title="Experiment Registry", page_icon="🧪")
    st.title("🧪 Experiment Registry")

    conn = get_db()
    cur = conn.cursor()

    # Summary metrics
    cur.execute("SELECT COUNT(*) FROM runs")
    total_runs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM runs WHERE verdict = 'scale'")
    scaled = cur.fetchone()[0]
    cur.execute("SELECT AVG(delta_val_loss) FROM comparisons WHERE delta_val_loss IS NOT NULL")
    avg_delta = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM threads")
    total_threads = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ideas WHERE status = 'proposed'")
    pending = cur.fetchone()[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Runs", total_runs)
    col2.metric("Scaled", scaled, f"{scaled}/{total_runs}" if total_runs else "0")
    col3.metric("Avg Delta", f"{avg_delta:+.4f}")
    col4.metric("Threads", total_threads)
    col5.metric("Pending ideas", pending)

    conn.close()
    conn = get_db()
    cur = conn.cursor()

    tab_ideas, tab_negatives, tab_runs, tab_threads, tab_queue, tab_comparisons, tab_decisions = st.tabs(
        ["💡 Ideas", "❌ Negatives", "Runs", "Threads", "Queue", "Comparisons", "Decisions"]
    )

    # ── Ideas / Approvals (the human gate) ─────────────────────────────────────
    with tab_ideas:
        st.caption("Proposed → **approve/reject** → promote approved to the run queue. "
                   "Approve/reject writes to the DB immediately.")
        cur.execute("SELECT status, COUNT(*) FROM ideas GROUP BY status")
        _counts = {s: n for s, n in cur.fetchall()}
        _counts["all"] = sum(_counts.values())
        status_filter = st.radio(
            "Show", ["proposed", "approved", "rejected", "queued", "all"],
            horizontal=True, index=0,
            format_func=lambda s: f"{s} ({_counts.get(s, 0)})",
        )
        if status_filter == "all":
            cur.execute("SELECT * FROM ideas ORDER BY priority, created_at")
        else:
            cur.execute("SELECT * FROM ideas WHERE status=? ORDER BY priority, created_at", (status_filter,))
        cols = [d[0] for d in cur.description]
        ideas = [dict(zip(cols, row)) for row in cur.fetchall()]

        if not ideas:
            st.info(f"No ideas with status '{status_filter}'.")
        for idea in ideas:
            conf_row = (idea.get("confidence") or "").lower()
            # color the row dot by confidence so green/yellow/red is visible while collapsed
            badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                conf_row,
                {"proposed": "🟡", "approved": "🟢", "rejected": "🔴",
                 "queued": "🔵", "done": "⚪"}.get(idea["status"], ""))
            tag = f" · {conf_row}" if conf_row else ""
            gain_row = idea.get("expected_gain") or ""
            gain_tag = f" · {gain_row.split('(')[0].strip()}" if gain_row else ""
            with st.expander(f"{badge} {idea['title']}{tag}{gain_tag}  ·  p{idea['priority']}"):
                if idea.get("summary"):
                    st.markdown(f"### {idea['summary']}")
                conf = (idea.get("confidence") or "").lower()
                if conf:
                    dot = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                    gain = idea.get("expected_gain") or "?"
                    st.markdown(f"**My call:** {dot} **{conf.upper()}** confidence · "
                                f"est. val-loss change at 200M: **{gain}**")
                if idea.get("reference_url"):
                    st.markdown(f"🔗 [{idea['reference_url']}]({idea['reference_url']})")
                if idea.get("explanation"):
                    st.markdown("**How it works:**")
                    st.markdown(idea["explanation"])
                if idea.get("pros") or idea.get("cons"):
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.markdown("**✅ Pros**")
                        for line in (idea.get("pros") or "—").split("\n"):
                            st.markdown(f"- {line}" if not line.startswith("-") else line)
                    with pc2:
                        st.markdown("**⚠️ Cons**")
                        for line in (idea.get("cons") or "—").split("\n"):
                            st.markdown(f"- {line}" if not line.startswith("-") else line)
                st.markdown(f"**Thread:** {idea.get('thread_name') or '—'}  ·  "
                            f"**Est:** {idea.get('estimated_minutes') or '?'} min  ·  "
                            f"**GPU:** {idea.get('gpu_class') or '—'}")
                if idea.get("hypothesis"):
                    st.markdown(f"**Hypothesis:** {idea['hypothesis']}")
                if idea.get("lever"):
                    st.markdown(f"**Lever:** {idea['lever']}")
                if idea.get("rationale"):
                    st.markdown(f"**Rationale:** {idea['rationale']}")
                if idea.get("expected_effect"):
                    st.markdown(f"**Expected:** {idea['expected_effect']}")
                if idea.get("scale_target"):
                    st.markdown(f"**Scale target:** {idea['scale_target']}")
                if idea.get("command"):
                    st.code(idea["command"], language="bash")
                if idea.get("review_note"):
                    st.markdown(f"_Review note: {idea['review_note']}_")
                if idea["status"] == "proposed":
                    note = st.text_input("Review note (optional)", key=f"note_{idea['id']}")
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Approve", key=f"app_{idea['id']}"):
                        review_idea(idea["id"], "approved", note=note or None)
                        st.rerun()
                    if c2.button("❌ Reject", key=f"rej_{idea['id']}"):
                        review_idea(idea["id"], "rejected", note=note or None)
                        st.rerun()
                elif idea["status"] == "approved":
                    st.success("Approved. Promote to queue via CLI: "
                               f"`python scripts/experiment_registry.py idea promote --id {idea['id']}`")

    # ── Negatives (ran-and-lost) ───────────────────────────────────────────────
    with tab_negatives:
        st.caption("Experiments we actually ran that did NOT win. Kept so they are never "
                   "re-proposed. Distinct from ideas rejected before running.")
        cur.execute("SELECT * FROM ideas WHERE outcome='tested_negative' ORDER BY updated_at DESC")
        ncols = [d[0] for d in cur.description]
        negs = [dict(zip(ncols, row)) for row in cur.fetchall()]
        if not negs:
            st.info("No tested-negative experiments recorded yet.")
        for idea in negs:
            with st.expander(f"❌ {idea['title']}"):
                if idea.get("summary"):
                    st.markdown(f"**{idea['summary']}**")
                st.markdown(f"**Measured result:** {idea.get('expected_gain') or '—'}")
                if idea.get("review_note"):
                    st.markdown(idea["review_note"])
                if idea.get("reference_url"):
                    st.markdown(f"🔗 [{idea['reference_url']}]({idea['reference_url']})")

    # ── Runs ──────────────────────────────────────────────────────────────────
    with tab_runs:
        # Fetch filter options
        cur.execute("SELECT DISTINCT thread_name FROM runs")
        thread_opts = ["All"] + [r[0] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT verdict FROM runs")
        verdict_opts = ["All"] + [r[0] for r in cur.fetchall() if r[0]]

        col_filter, col_search = st.columns([1, 2])
        with col_filter:
            filter_thread = st.selectbox("Thread", thread_opts)
            filter_verdict = st.selectbox("Verdict", verdict_opts)
        with col_search:
            search = st.text_input("Search", placeholder="filter by name...")

        base_query = """
            SELECT id, thread_name, name, status, final_val_loss, final_train_loss,
                   verdict, tokens_seen, actual_steps, metrics_path, checkpoint_path
            FROM runs
        """
        conditions = []
        params = []
        if filter_thread != "All":
            conditions.append("thread_name = ?")
            params.append(filter_thread)
        if filter_verdict != "All":
            conditions.append("verdict = ?")
            params.append(filter_verdict)
        if search:
            conditions.append("(name LIKE ? OR thread_name LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        where = " AND ".join(conditions) if conditions else "1=1"
        cur.execute(f"{base_query} WHERE {where}", params)
        runs = cur.fetchall()

        rows = []
        for r in runs:
            rows.append({
                "ID": r[0],
                "Thread": r[1],
                "Name": r[2],
                "Status": r[3],
                "Val Loss": f"{r[4]:.4f}" if r[4] else None,
                "Train Loss": f"{r[5]:.4f}" if r[5] else None,
                "Verdict": (delta_color(r[6]) + " " + r[6]) if r[6] else "",
                "Tokens": r[7],
                "Steps": r[8],
                "Metrics": make_link(r[9]),
                "Checkpoint": make_link(r[10]),
            })

        st.dataframe(
            rows,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Thread": st.column_config.TextColumn("Thread", width="small"),
                "Name": st.column_config.TextColumn("Name"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Val Loss": st.column_config.NumberColumn("Val Loss", format="%.4f", width="small"),
                "Train Loss": st.column_config.NumberColumn("Train Loss", format="%.4f", width="small"),
                "Verdict": st.column_config.TextColumn("Verdict", width="small"),
                "Tokens": st.column_config.NumberColumn("Tokens", width="small"),
                "Steps": st.column_config.NumberColumn("Steps", width="small"),
                "Metrics": st.column_config.LinkColumn("Metrics", display_text="Open", width="medium"),
                "Checkpoint": st.column_config.LinkColumn("Checkpoint", display_text="Open", width="medium"),
            },
            use_container_width=True,
            hide_index=True,
        )

        # Val loss curves
        st.divider()
        st.subheader("Val Loss Curves")
        run_ids = [str(r[0]) for r in runs]
        if run_ids:
            selected_run = st.selectbox("Select run", run_ids)
            cur.execute(
                "SELECT step, val_loss, tokens FROM eval_points WHERE run_id = ? ORDER BY step",
                (selected_run,),
            )
            ep = cur.fetchall()
            if ep:
                chart_data = {"step": [e[0] for e in ep], "val_loss": [e[1] for e in ep]}
                st.line_chart(chart_data, x="step", y="val_loss")
            else:
                st.info("No eval points for this run.")
        else:
            st.info("No runs to chart.")

    # ── Threads ────────────────────────────────────────────────────────────────
    with tab_threads:
        cur.execute(
            "SELECT name, hypothesis, status, priority, notes_path, summary FROM threads"
        )
        threads = cur.fetchall()
        rows = []
        for t in threads:
            rows.append({
                "Name": t[0],
                "Hypothesis": t[1],
                "Status": t[2],
                "Priority": t[3],
                "Notes": make_link(t[4]),
                "Summary": t[5],
            })
        st.dataframe(
            rows,
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Hypothesis": st.column_config.TextColumn("Hypothesis"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Priority": st.column_config.NumberColumn("Priority", width="small"),
                "Notes": st.column_config.LinkColumn("Notes", display_text="Open", width="medium"),
                "Summary": st.column_config.TextColumn("Summary"),
            },
            use_container_width=True,
            hide_index=True,
        )

    # ── Queue ─────────────────────────────────────────────────────────────────
    with tab_queue:
        cur.execute(
            "SELECT id, thread_name, name, command, status, created_at, started_at, finished_at, output_dir FROM queue_items"
        )
        queue = cur.fetchall()
        rows = []
        for q in queue:
            rows.append({
                "ID": q[0],
                "Thread": q[1],
                "Name": q[2],
                "Command": q[3],
                "Status": q[4],
                "Created": q[5],
                "Started": q[6],
                "Finished": q[7],
                "Output": make_link(q[8]),
            })
        st.dataframe(
            rows,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Thread": st.column_config.TextColumn("Thread", width="small"),
                "Name": st.column_config.TextColumn("Name"),
                "Command": st.column_config.TextColumn("Command"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Created": st.column_config.TextColumn("Created", width="small"),
                "Started": st.column_config.TextColumn("Started", width="small"),
                "Finished": st.column_config.TextColumn("Finished", width="small"),
                "Output": st.column_config.LinkColumn("Output", display_text="Open", width="medium"),
            },
            use_container_width=True,
            hide_index=True,
        )

    # ── Comparisons ────────────────────────────────────────────────────────────
    with tab_comparisons:
        cur.execute(
            "SELECT run_id, baseline_name, baseline_val_loss, run_val_loss, delta_val_loss, verdict FROM comparisons"
        )
        comparisons = cur.fetchall()
        rows = []
        for c in comparisons:
            delta = c[4] if c[4] else 0
            rows.append({
                "Run ID": c[0],
                "Baseline": c[1],
                "Baseline Val Loss": f"{c[2]:.4f}" if c[2] else None,
                "Run Val Loss": f"{c[3]:.4f}" if c[3] else None,
                "Delta": f"{delta:+.4f}",
                "Verdict": delta_color(delta) + " " + (c[5] or ""),
            })
        st.dataframe(
            rows,
            column_config={
                "Run ID": st.column_config.TextColumn("Run ID"),
                "Baseline": st.column_config.TextColumn("Baseline"),
                "Baseline Val Loss": st.column_config.NumberColumn("Baseline Val Loss", format="%.4f"),
                "Run Val Loss": st.column_config.NumberColumn("Run Val Loss", format="%.4f"),
                "Delta": st.column_config.TextColumn("Delta"),
                "Verdict": st.column_config.TextColumn("Verdict"),
            },
            use_container_width=True,
            hide_index=True,
        )

    # ── Decisions ──────────────────────────────────────────────────────────────
    with tab_decisions:
        cur.execute(
            "SELECT id, thread_name, decision, reason, decided_by, decided_at FROM decisions"
        )
        decisions = cur.fetchall()
        st.dataframe(
            decisions,
            column_config={
                0: "ID",
                1: "Thread",
                2: "Decision",
                3: "Reason",
                4: "Decided By",
                5: "Decided At",
            },
            use_container_width=True,
            hide_index=True,
        )

    conn.close()


if __name__ == "__main__":
    main()