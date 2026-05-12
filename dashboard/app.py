"""
GraphRAG vs Basic RAG vs LLM-Only — IPL Knowledge Base
Streamlit comparison dashboard.

Run with:  streamlit run dashboard/app.py
"""

import sys
import json
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Project root on path so pipeline modules are importable ──────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG vs Basic RAG vs LLM-Only",
    page_icon="🏏",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 14px;
}
.graphrag-box { background: #1a3a2a; border: 1px solid #2d6a4f; }
.basicrag-box { background: #1a2a3a; border: 1px solid #2d4f6a; }
.llmonly-box  { background: #3a2a1a; border: 1px solid #6a4f2d; }
.answer-text  { font-size: 13px; line-height: 1.6; }
.best-badge   { background: #ffd700; color: #000; border-radius: 4px;
                padding: 2px 8px; font-weight: bold; font-size: 12px; }
.stat-highlight { font-size: 24px; font-weight: bold; color: #2d6a4f; }
.reduction-stat { font-size: 28px; font-weight: bold; color: #00c851; }
</style>
""", unsafe_allow_html=True)

# ── Result file loaders ───────────────────────────────────────────────────────

def load_p2_results() -> dict | None:
    """Load pipeline2_results.json — format: {summary:{}, results:[]}"""
    path = RESULTS_DIR / "pipeline2_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_p3_results() -> dict | None:
    """Load pipeline3_results.json — format: {results:[]} (no top-level summary)"""
    path = RESULTS_DIR / "pipeline3_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_p1_results() -> dict | None:
    """Load pipeline1_results.json — format: {summary:{}, results:[]}"""
    path = RESULTS_DIR / "pipeline1_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Pipeline loaders (cached so they only initialise once) ────────────────────

@st.cache_resource(show_spinner="Loading LLM-Only pipeline (Pipeline 1)…")
def load_pipeline1():
    try:
        from pipeline1_llm_only.pipeline import run_query as p1_run_query
        return p1_run_query
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner="Loading FAISS index for Basic RAG (Pipeline 2)…")
def load_pipeline2():
    try:
        from pipeline2_basic_rag.pipeline import (
            build_index, get_embed_model, run_query as p2_run_query,
        )
        index, metadata = build_index()
        embed_model = get_embed_model()
        return p2_run_query, index, metadata, embed_model
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner="Loading GraphRAG pipeline (Pipeline 3)…")
def load_pipeline3():
    try:
        from groq import Groq
        from pipeline3_graphrag.pipeline import Pipeline3, load_token, GROQ_API_KEY
        import os
        key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        token = load_token()          # reads tg_token.txt — no network call
        groq_client = Groq(api_key=key)
        pipeline = Pipeline3(token=token, groq_client=groq_client)
        return pipeline
    except Exception as exc:
        return exc


# ── Live query runners ────────────────────────────────────────────────────────

def run_p1_live(question: str) -> dict:
    p1 = load_pipeline1()
    if isinstance(p1, Exception):
        return {"error": str(p1), "answer": "", "total_tokens": 0,
                "latency_ms": 0, "cost_usd": 0,
                "prompt_tokens": 0, "completion_tokens": 0}
    try:
        res = p1(question)
        return res
    except Exception as exc:
        return {"error": str(exc), "answer": "", "total_tokens": 0,
                "latency_ms": 0, "cost_usd": 0,
                "prompt_tokens": 0, "completion_tokens": 0}


def run_p2_live(question: str) -> dict:
    p2 = load_pipeline2()
    if isinstance(p2, Exception):
        return {"error": str(p2), "answer": "", "metrics": {
            "total_tokens": 0, "latency_ms": 0, "cost_usd": 0,
            "prompt_tokens": 0, "completion_tokens": 0}}
    run_query, index, metadata, embed_model = p2
    try:
        res = run_query(question, index, metadata, embed_model)
        return res
    except Exception as exc:
        return {"error": str(exc), "answer": "", "metrics": {
            "total_tokens": 0, "latency_ms": 0, "cost_usd": 0,
            "prompt_tokens": 0, "completion_tokens": 0}}


def run_p3_live(question: str) -> dict:
    p3 = load_pipeline3()
    if isinstance(p3, Exception):
        return {"error": str(p3), "answer": "", "total_tokens": 0,
                "latency_ms": 0, "cost_usd": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "tg_unavailable": False}
    try:
        res = p3.run_query(question_id=0, question=question)
        return res
    except Exception as exc:
        return {"error": str(exc), "answer": "", "total_tokens": 0,
                "latency_ms": 0, "cost_usd": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "tg_unavailable": False}


# ── Metric extraction helpers (normalise different result schemas) ─────────────

def p1_tokens(res: dict) -> int:
    return res.get("total_tokens", 0)

def p1_latency(res: dict) -> float:
    return res.get("latency_ms", 0)

def p1_cost(res: dict) -> float:
    return res.get("cost_usd", 0)


def p2_tokens(res: dict) -> int:
    return res.get("metrics", {}).get("total_tokens", 0)

def p2_latency(res: dict) -> float:
    return res.get("metrics", {}).get("latency_ms", 0)

def p2_cost(res: dict) -> float:
    return res.get("metrics", {}).get("cost_usd", 0)


def p3_tokens(res: dict) -> int:
    return res.get("total_tokens", 0)

def p3_latency(res: dict) -> float:
    return res.get("latency_ms", 0)

def p3_cost(res: dict) -> float:
    return res.get("cost_usd", 0)


# ── Display helpers ───────────────────────────────────────────────────────────

def best_badge() -> str:
    return '<span class="best-badge">🏆 Best</span>'


def render_result_column(
    label: str,
    box_class: str,
    answer: str,
    tokens: int,
    latency_ms: float,
    cost_usd: float,
    is_best_tokens: bool,
    error: str | None = None,
    tg_unavailable: bool = False,
):
    st.markdown(f"#### {label}")
    badge = best_badge() if is_best_tokens else ""
    st.markdown(
        f'<div class="metric-box {box_class}">'
        f'<b>Tokens:</b> {tokens:,} {badge}&nbsp;&nbsp; '
        f'<b>Latency:</b> {latency_ms:,.0f} ms&nbsp;&nbsp; '
        f'<b>Cost:</b> ${cost_usd:.6f}'
        f'</div>',
        unsafe_allow_html=True,
    )
    if tg_unavailable:
        st.warning("⚠️ GraphRAG: TigerGraph unavailable, showing fallback answer")
    if error:
        st.error(f"Error: {error}")
    elif answer:
        with st.expander("Answer", expanded=True):
            st.markdown(f'<div class="answer-text">{answer}</div>',
                        unsafe_allow_html=True)
    else:
        st.info("No answer yet. Run a query above.")


# ── Plotly helpers ────────────────────────────────────────────────────────────

PIPELINE_COLORS = {
    "LLM-Only (P1)":    "#e07b39",
    "Basic RAG (P2)":   "#4a90d9",
    "GraphRAG (P3)":    "#2d9e5f",
}
PIPELINE_LABELS = ["LLM-Only (P1)", "Basic RAG (P2)", "GraphRAG (P3)"]


def token_bar_chart(tokens: list[int]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=PIPELINE_LABELS,
        y=tokens,
        marker_color=list(PIPELINE_COLORS.values()),
        text=[f"{t:,}" for t in tokens],
        textposition="outside",
    ))
    fig.update_layout(
        title="Tokens Used per Pipeline",
        yaxis_title="Tokens",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=320,
        margin=dict(t=50, b=20),
    )
    return fig


def cost_bar_chart(costs: list[float]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=PIPELINE_LABELS,
        y=costs,
        marker_color=list(PIPELINE_COLORS.values()),
        text=[f"${c:.6f}" for c in costs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Cost per Pipeline (USD)",
        yaxis_title="USD",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=320,
        margin=dict(t=50, b=20),
    )
    return fig


# ── History aggregate helpers ─────────────────────────────────────────────────

def p2_aggregate(data: dict) -> dict:
    """Summarise pipeline2 result file."""
    summary = data.get("summary", {})
    results = data.get("results", [])
    successful = [r for r in results if not r.get("error")]
    n = len(successful)
    if n == 0:
        return {}
    avg_tokens = summary.get("avg_total_tokens") or (
        sum(r.get("metrics", {}).get("total_tokens", 0) for r in successful) / n
    )
    avg_cost = (
        sum(r.get("metrics", {}).get("cost_usd", 0) for r in successful) / n
    )
    return {
        "total_questions": summary.get("total_questions", len(results)),
        "successful": summary.get("successful", n),
        "avg_tokens": avg_tokens,
        "avg_cost": avg_cost,
        "total_tokens": summary.get("total_tokens", 0),
        "total_cost": summary.get("total_cost_usd", 0),
    }


def p3_aggregate(data: dict) -> dict:
    """Summarise pipeline3 result file."""
    results = data.get("results", [])
    successful = [r for r in results if not r.get("error") and r.get("total_tokens", 0) > 0]
    n = len(successful)
    if n == 0:
        return {}
    avg_tokens = sum(r.get("total_tokens", 0) for r in successful) / n
    avg_cost   = sum(r.get("cost_usd", 0) for r in successful) / n
    return {
        "total_questions": len(results),
        "successful": n,
        "avg_tokens": avg_tokens,
        "avg_cost": avg_cost,
        "total_tokens": sum(r.get("total_tokens", 0) for r in successful),
        "total_cost": sum(r.get("cost_usd", 0) for r in successful),
    }


def p1_aggregate(data: dict) -> dict:
    """Summarise pipeline1 result file."""
    summary = data.get("summary", {})
    results = data.get("results", [])
    successful = [r for r in results if not r.get("error")]
    n = len(successful)
    if n == 0:
        return {}
    avg_tokens = summary.get("avg_tokens_per_query") or (
        sum(r.get("metrics", {}).get("total_tokens", 0) for r in successful) / n
    )
    avg_cost = (
        sum(r.get("metrics", {}).get("cost_usd", 0) for r in successful) / n
    )
    return {
        "total_questions": summary.get("total_questions", len(results)),
        "successful": summary.get("successful", n),
        "avg_tokens": avg_tokens,
        "avg_cost": avg_cost,
        "total_tokens": summary.get("total_tokens", 0),
        "total_cost": summary.get("total_cost_usd", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏏 GraphRAG vs Basic RAG vs LLM-Only — IPL Knowledge Base")
st.caption(
    "Compare three retrieval strategies on Indian Premier League questions. "
    "GraphRAG uses a TigerGraph knowledge graph; Basic RAG uses FAISS vector search; "
    "LLM-Only uses Gemini with no retrieval."
)

# ── Session-state defaults ────────────────────────────────────────────────────
for key, default in [
    ("p1_result", None), ("p2_result", None), ("p3_result", None),
    ("last_query", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Query input ───────────────────────────────────────────────────────────────
st.markdown("---")
query = st.text_input(
    "🔍 Ask an IPL question",
    placeholder="e.g. Which team won the most IPL titles?",
    label_visibility="visible",
)
run_button = st.button("▶ Run All Pipelines", type="primary", use_container_width=False)

# ── Run pipelines ─────────────────────────────────────────────────────────────
if run_button and query.strip():
    st.session_state["last_query"] = query.strip()

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.spinner("Pipeline 1 — LLM-Only (Gemini)…"):
            st.session_state["p1_result"] = run_p1_live(query.strip())

    with col2:
        with st.spinner("Pipeline 2 — Basic RAG (FAISS + Groq)…"):
            st.session_state["p2_result"] = run_p2_live(query.strip())

    with col3:
        with st.spinner("Pipeline 3 — GraphRAG (TigerGraph + Groq)…"):
            st.session_state["p3_result"] = run_p3_live(query.strip())

elif run_button and not query.strip():
    st.warning("Please enter a question first.")

# ── Three result columns ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Pipeline Results")

p1_res = st.session_state["p1_result"]
p2_res = st.session_state["p2_result"]
p3_res = st.session_state["p3_result"]

# Extract tokens for badge logic
tok1 = p1_tokens(p1_res) if p1_res else 0
tok2 = p2_tokens(p2_res) if p2_res else 0
tok3 = p3_tokens(p3_res) if p3_res else 0
min_tok = min(t for t in [tok1, tok2, tok3] if t > 0) if any([tok1, tok2, tok3]) else 0

col1, col2, col3 = st.columns(3)

with col1:
    render_result_column(
        label="🟠 Pipeline 1 — LLM-Only (Gemini)",
        box_class="llmonly-box",
        answer=p1_res.get("answer", "") if p1_res else "",
        tokens=tok1,
        latency_ms=p1_latency(p1_res) if p1_res else 0,
        cost_usd=p1_cost(p1_res) if p1_res else 0,
        is_best_tokens=(tok1 == min_tok and tok1 > 0),
        error=p1_res.get("error") if p1_res else None,
    )

with col2:
    render_result_column(
        label="🔵 Pipeline 2 — Basic RAG (FAISS + Groq)",
        box_class="basicrag-box",
        answer=p2_res.get("answer", "") if p2_res else "",
        tokens=tok2,
        latency_ms=p2_latency(p2_res) if p2_res else 0,
        cost_usd=p2_cost(p2_res) if p2_res else 0,
        is_best_tokens=(tok2 == min_tok and tok2 > 0),
        error=p2_res.get("error") if p2_res else None,
    )

with col3:
    render_result_column(
        label="🟢 Pipeline 3 — GraphRAG (TigerGraph + Groq)",
        box_class="graphrag-box",
        answer=p3_res.get("answer", "") if p3_res else "",
        tokens=tok3,
        latency_ms=p3_latency(p3_res) if p3_res else 0,
        cost_usd=p3_cost(p3_res) if p3_res else 0,
        is_best_tokens=(tok3 == min_tok and tok3 > 0),
        error=p3_res.get("error") if p3_res else None,
        tg_unavailable=p3_res.get("tg_unavailable", False) if p3_res else False,
    )

# ── Metrics panel (only when we have live results) ────────────────────────────
if any([tok1, tok2, tok3]):
    st.markdown("---")
    st.subheader("📊 Metrics Comparison")

    tokens = [tok1, tok2, tok3]
    costs  = [
        p1_cost(p1_res) if p1_res else 0,
        p2_cost(p2_res) if p2_res else 0,
        p3_cost(p3_res) if p3_res else 0,
    ]
    latencies = [
        p1_latency(p1_res) if p1_res else 0,
        p2_latency(p2_res) if p2_res else 0,
        p3_latency(p3_res) if p3_res else 0,
    ]

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(token_bar_chart(tokens), use_container_width=True)
    with chart_col2:
        st.plotly_chart(cost_bar_chart(costs), use_container_width=True)

    # Side-by-side table
    metrics_df = pd.DataFrame({
        "Pipeline":     PIPELINE_LABELS,
        "Tokens":       tokens,
        "Latency (ms)": [f"{l:,.0f}" for l in latencies],
        "Cost (USD)":   [f"${c:.6f}" for c in costs],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Big highlighted stat
    if tok2 > 0 and tok3 > 0:
        reduction = round((1 - tok3 / tok2) * 100, 1)
        sign = "fewer" if reduction > 0 else "more"
        abs_reduction = abs(reduction)
        st.markdown(
            f'<div style="text-align:center; padding: 20px;">'
            f'<div class="reduction-stat">GraphRAG uses {abs_reduction}% {sign} tokens than Basic RAG</div>'
            f'<div style="color:#aaa; margin-top:4px;">on this query: {tok3:,} vs {tok2:,} tokens</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS HISTORY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("📁 Results History (from saved runs)")

p1_data = load_p1_results()
p2_data = load_p2_results()
p3_data = load_p3_results()

p1_agg = p1_aggregate(p1_data) if p1_data else {}
p2_agg = p2_aggregate(p2_data) if p2_data else {}
p3_agg = p3_aggregate(p3_data) if p3_data else {}

any_history = any([p1_agg, p2_agg, p3_agg])

if not any_history:
    st.info(
        "No saved results found yet. "
        "Run `python run_pipeline2.py` or `python run_pipeline3.py` to generate them, "
        "or use the query box above."
    )
else:
    # Aggregate stats cards
    hist_col1, hist_col2, hist_col3 = st.columns(3)

    def agg_card(col, label, box_class, agg: dict):
        with col:
            if agg:
                n   = agg.get("total_questions", 0)
                ok  = agg.get("successful", 0)
                avg = agg.get("avg_tokens", 0)
                cost= agg.get("avg_cost", 0)
                st.markdown(
                    f'<div class="metric-box {box_class}">'
                    f'<b>{label}</b><br>'
                    f'Questions: {n} &nbsp;|&nbsp; Successful: {ok}<br>'
                    f'Avg tokens: <b>{avg:,.1f}</b><br>'
                    f'Avg cost: <b>${cost:.6f}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="metric-box" style="background:#1a1a1a; border:1px solid #333;">'
                    f'<b>{label}</b><br><i>No data</i>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    agg_card(hist_col1, "🟠 LLM-Only (P1)",  "llmonly-box",  p1_agg)
    agg_card(hist_col2, "🔵 Basic RAG (P2)", "basicrag-box", p2_agg)
    agg_card(hist_col3, "🟢 GraphRAG (P3)",  "graphrag-box", p3_agg)

    # Token reduction highlight (P3 vs P2)
    if p2_agg.get("avg_tokens") and p3_agg.get("avg_tokens"):
        avg2 = p2_agg["avg_tokens"]
        avg3 = p3_agg["avg_tokens"]
        reduction = round((1 - avg3 / avg2) * 100, 1)
        sign = "fewer" if reduction > 0 else "more"
        abs_r = abs(reduction)
        color = "#00c851" if reduction > 0 else "#ff4444"
        st.markdown(
            f'<div style="text-align:center; padding:24px;">'
            f'<div style="font-size:32px; font-weight:bold; color:{color};">'
            f'GraphRAG uses {abs_r}% {sign} tokens than Basic RAG'
            f'</div>'
            f'<div style="color:#aaa; margin-top:6px;">'
            f'Avg: {avg3:,.1f} vs {avg2:,.1f} tokens '
            f'(over {p2_agg.get("total_questions", 0)} questions)'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Aggregate comparison chart
    if any_history:
        hist_tokens = [
            p1_agg.get("avg_tokens", 0),
            p2_agg.get("avg_tokens", 0),
            p3_agg.get("avg_tokens", 0),
        ]
        hist_costs = [
            p1_agg.get("avg_cost", 0),
            p2_agg.get("avg_cost", 0),
            p3_agg.get("avg_cost", 0),
        ]

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = token_bar_chart(hist_tokens)
            fig.update_layout(title="Avg Tokens per Query (historical)")
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            fig = cost_bar_chart(hist_costs)
            fig.update_layout(title="Avg Cost per Query (historical, USD)")
            st.plotly_chart(fig, use_container_width=True)

    # Full summary table
    rows = []
    for lbl, agg in [("LLM-Only (P1)", p1_agg), ("Basic RAG (P2)", p2_agg), ("GraphRAG (P3)", p3_agg)]:
        if agg:
            rows.append({
                "Pipeline":          lbl,
                "Total Questions":   agg.get("total_questions", 0),
                "Successful":        agg.get("successful", 0),
                "Avg Tokens":        f"{agg.get('avg_tokens', 0):,.1f}",
                "Avg Cost (USD)":    f"${agg.get('avg_cost', 0):.6f}",
                "Total Tokens":      f"{agg.get('total_tokens', 0):,}",
                "Total Cost (USD)":  f"${agg.get('total_cost', 0):.6f}",
            })
    if rows:
        st.markdown("#### All-run summary")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Sample results from saved files
    with st.expander("📄 Show sample saved answers (first 5 questions)", expanded=False):
        sample_col1, sample_col2, sample_col3 = st.columns(3)

        def show_sample(col, label, results_list: list, tok_key_fn):
            with col:
                st.markdown(f"**{label}**")
                if not results_list:
                    st.write("—")
                    return
                for r in results_list[:5]:
                    q   = r.get("question", "")[:80]
                    ans = r.get("answer", "")[:200]
                    tok = tok_key_fn(r)
                    err = r.get("error")
                    st.markdown(f"**Q:** {q}")
                    if err:
                        st.markdown(f"_Error: {str(err)[:100]}_")
                    else:
                        st.markdown(f"**A:** {ans}")
                    st.markdown(f"`{tok:,} tokens`")
                    st.markdown("---")

        p1_results_list = (p1_data or {}).get("results", [])
        p2_results_list = (p2_data or {}).get("results", [])
        p3_results_list = (p3_data or {}).get("results", [])

        show_sample(sample_col1, "🟠 LLM-Only",  p1_results_list,
                    lambda r: r.get("metrics", {}).get("total_tokens", 0))
        show_sample(sample_col2, "🔵 Basic RAG", p2_results_list,
                    lambda r: r.get("metrics", {}).get("total_tokens", 0))
        show_sample(sample_col3, "🟢 GraphRAG",  p3_results_list,
                    lambda r: r.get("total_tokens", 0))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Pipeline 1: Gemini 2.5 Flash (no retrieval) · "
    "Pipeline 2: FAISS + all-MiniLM-L6-v2 + Groq llama-3.3-70b · "
    "Pipeline 3: TigerGraph 2-hop + community context + Groq llama-3.3-70b"
)
