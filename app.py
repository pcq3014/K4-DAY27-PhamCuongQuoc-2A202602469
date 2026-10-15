"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.db import db_path, replay_events
from exercises.exercise_4_audit import build_graph

load_dotenv(override=True)


# ─── Database Helpers ───────────────────────────────────────────────────────
def get_recent_sessions(limit: int = 25) -> list[dict]:
    """Fetch recent review sessions from SQLite audit_events."""
    db = db_path()
    if not os.path.exists(db):
        return []
    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'")
            if not cur.fetchone():
                return []
            cur.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp)        AS started,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", page_icon="🤖", layout="wide")
st.title("🤖 HITL PR Review Agent")
st.caption("Human-in-the-Loop automated code review with confidence-based triage and SQLite audit trail.")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Recent Sessions")
    sessions = get_recent_sessions()
    if not sessions:
        st.info("No audit sessions yet. Run a review to see logs here.")
    else:
        for s in sessions:
            risk = s.get("worst_risk", "low")
            risk_icon = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(risk, "⚪")
            short_id = s["thread_id"][:8]
            label = f"{risk_icon} `{short_id}` ({s['events']} ev)"
            with st.expander(label):
                st.markdown(f"**PR:** [{s['pr_url']}]({s['pr_url']})")
                st.markdown(f"**Risk:** `{risk}` | **Events:** `{s['events']}`")
                st.caption(f"Last: {s['last_event']}")
                if st.button("Load Session", key=f"btn_{s['thread_id']}", use_container_width=True):
                    st.session_state.thread_id = s["thread_id"]
                    st.session_state.pr_url = s["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()

    if st.session_state.thread_id:
        st.divider()
        st.subheader("🔍 Active Session Timeline")
        st.code(st.session_state.thread_id, language="text")
        try:
            evs = asyncio.run(replay_events(st.session_state.thread_id))
            if evs:
                for ev in evs:
                    st.text(f"[{ev['action']}] conf={ev['confidence']:.2f} dec={ev['decision']}")
            else:
                st.caption("No events recorded for this thread yet.")
        except Exception:
            pass


# ─── Top form — start a new review ─────────────────────────────────────────
st.markdown("##### 💡 Hoặc chọn nhanh PR thử nghiệm của bài tập:")
col_demo1, col_demo2 = st.columns(2)
if col_demo1.button("📌 Demo PR #1 (Add task priority — human_approval)", use_container_width=True):
    st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.rerun()

if col_demo2.button("⚠️ Demo PR #2 (User login & cloud sync — escalate)", use_container_width=True):
    st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.rerun()

with st.form("start"):
    pr_url = st.text_input(
        "GitHub Pull Request URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        help="Định dạng bắt buộc: https://github.com/<owner>/<repo>/pull/<number>",
    )
    col_submit, _ = st.columns([1, 5])
    with col_submit:
        submitted = st.form_submit_button("Run review", type="primary", use_container_width=True)


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"⚖️ Human Approval Requested — Confidence {conf:.0%}")
    st.info(f"**LLM Reasoning:** {payload.get('confidence_reasoning', 'Review completed with medium confidence.')}")
    st.markdown(f"**Summary:** {payload['summary']}")

    comments = payload.get("comments", [])
    if comments:
        st.markdown("### 💬 Proposed Review Comments")
        for c in comments:
            sev = c.get("severity", "suggestion")
            badge = {
                "blocker": "🔴 Blocker",
                "issue": "🟠 Issue",
                "suggestion": "🔵 Suggestion",
                "nit": "⚪ Nit",
            }.get(sev, sev.upper())
            st.markdown(f"- **[{badge}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    diff_preview = payload.get("diff_preview", "")
    if diff_preview:
        with st.expander("📄 Diff Preview"):
            st.code(diff_preview, language="diff")

    st.markdown("### ✍️ Reviewer Action")
    feedback = st.text_input(
        "Feedback / Reviewer Note (optional for Approve, recommended for Reject/Edit)",
        key="approval_feedback",
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"🚨 Strong Escalation — Confidence {conf:.0%}")
    st.caption(f"Reasoning: {payload.get('confidence_reasoning', 'Low confidence due to critical risks.')}")

    if payload.get("risk_factors"):
        st.error("⚠️ **Identified Risk Factors:**\n\n" + "\n".join(f"- {rf}" for rf in payload["risk_factors"]))

    st.markdown(f"**Summary:** {payload['summary']}")

    with st.form("escalation"):
        st.markdown("### ❓ Clarifying Questions for Reviewer")
        st.write("Please answer the agent's questions to resolve ambiguity and generate a safe, refined review:")
        answers: dict[str, str] = {}
        questions = payload.get("questions", [])
        if not questions:
            questions = ["What is the intent of this PR?", "Are there migration or security concerns?"]
        for idx, q in enumerate(questions):
            answers[q] = st.text_area(f"**Q{idx + 1}: {q}**", key=f"q_{idx}")
        escalate_submit = st.form_submit_button("Submit answers & refine review", type="primary")
        if escalate_submit:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(
                Command(resume=resume_value), cfg
            )
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    # Validate PR URL format before invoking graph
    if "/pull/" not in pr_url:
        st.error(f"❌ Đường dẫn không phải là một Pull Request: `{pr_url}`")
        st.info("💡 **Gợi ý:** Bạn vừa nhập link của cả Repository thay vì link của một Pull Request. URL của PR cần có định dạng kết thúc bằng `/pull/<số>`, ví dụ: `https://github.com/VinUni-AI20k/PR-Demo/pull/1` hoặc `https://github.com/pcq3014/Track3-Day25-2A202602469-Pham-Cuong-Quoc/pull/1` (nếu repo đó có PR).")
    else:
        st.session_state.pr_url = pr_url
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.interrupt_payload = None
        st.session_state.final = None

        with st.spinner("Fetching PR + analyzing with LLM..."):
            try:
                result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
                if "__interrupt__" in result:
                    st.session_state.interrupt_payload = result["__interrupt__"][0].value
                else:
                    st.session_state.final = result
            except ValueError as e:
                st.error(f"❌ URL PR không hợp lệ: {e}")
                st.info("Vui lòng kiểm tra lại URL PR theo cấu trúc: `https://github.com/<owner>/<repo>/pull/<number>`")
            except Exception as e:
                st.error(f"Error during graph execution: {e}")

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent with your input..."):
            try:
                result = asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                ))
                if "__interrupt__" in result:
                    st.session_state.interrupt_payload = result["__interrupt__"][0].value
                else:
                    st.session_state.interrupt_payload = None
                    st.session_state.final = result
            except Exception as e:
                st.error(f"Error during resume: {e}")
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")

    # Display results per confidence bucket specification
    if action.startswith("auto"):
        st.success("### 🎉 Auto-Approved (> 72% Confidence)")
        if analysis:
            st.markdown(f"**Confidence:** `{analysis.confidence:.0%}`")
            st.markdown(f"**Summary:** {analysis.summary}")
        st.markdown(f"[🔗 View comment on GitHub]({st.session_state.pr_url})")
        st.info("The agent determined this PR is low-risk and automatically posted the review.")
    elif action.startswith("committed"):
        st.success(f"### ✅ Review Committed ({action})")
        if analysis:
            st.markdown(f"**Final Confidence:** `{analysis.confidence:.0%}`")
            st.markdown(f"**Summary:** {analysis.summary}")
        st.markdown(f"[🔗 View comment on GitHub]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning("### 🛑 PR Review Rejected")
        st.write("No review comment was posted to the PR based on reviewer choice.")
    else:
        st.info(f"### Outcome: `{action}`")

    st.caption(
        f"Thread ID: `{st.session_state.thread_id}`  ·  Replay command: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
