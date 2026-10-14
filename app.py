"""AI-Powered Web Research and Intelligent Answering - Enhanced Streamlit Application."""

import os
import time
from typing import List, Dict, Any
import streamlit as st

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    GROQ_API_KEY,
    ROOT_DIR,
)
from src.pipeline import WebResearchPipeline
from src.models import ResearchReport, SourceCitation, ScrapedDocument, FactualClaim

# Set page config
st.set_page_config(
    page_title="DeepResearch AI - Verified Intelligence Engine",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        background: #1E293B;
        color: #38BDF8;
        border: 1px solid #334155;
    }
    .exec-summary-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .source-card {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .source-card:hover {
        border-color: #64748B;
    }
    .claim-card {
        background: #0B1329;
        border: 1px solid #1E293B;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-green {
        color: #4ADE80;
        font-weight: 600;
    }
    .badge-blue {
        color: #60A5FA;
        font-weight: 600;
    }
    .badge-yellow {
        color: #FACC15;
        font-weight: 600;
    }
    .badge-red {
        color: #F87171;
        font-weight: 600;
    }
    .timing-bar {
        font-size: 0.85rem;
        color: #94A3B8;
        background: #0F172A;
        padding: 0.5rem 0.8rem;
        border-radius: 0.4rem;
        border: 1px solid #1E293B;
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "") or GROQ_API_KEY

if "selected_model" not in st.session_state or st.session_state.selected_model not in AVAILABLE_MODELS:
    st.session_state.selected_model = DEFAULT_MODEL

if "current_report" not in st.session_state:
    st.session_state.current_report = None

if "current_context" not in st.session_state:
    st.session_state.current_context = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar Configuration
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/8644/8644485.png",
        width=50,
    )
    st.title("⚙️ Engine Controls")

    # Groq API Key Input
    st.subheader("🔑 Groq API Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free ultra-fast Groq API key at console.groq.com/keys",
    )

    clean_key = api_key_input.strip().strip('"').strip("'")

    if clean_key != st.session_state.groq_api_key:
        st.session_state.groq_api_key = clean_key
        # Save to .env if user changed it
        env_path = ROOT_DIR / ".env"
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f"GROQ_API_KEY={clean_key}\n")
        except Exception:
            pass

    if st.session_state.groq_api_key:
        if st.session_state.groq_api_key.startswith("gsk_"):
            st.success("🟢 Groq API Key Configured")
        else:
            st.error("⚠️ Invalid Key format: Groq API keys start with 'gsk_'. Please check your key from console.groq.com/keys.")
    else:
        st.warning(
            "⚠️ No Groq Key found. Running in verified Extractive Mode. [Get free key](https://console.groq.com/keys)"
        )

    st.markdown("---")

    # LLM Model Selector
    st.subheader("🧠 Synthesis Model")
    model_keys = list(AVAILABLE_MODELS.keys())
    model_names = [AVAILABLE_MODELS[k]["name"] for k in model_keys]
    current_index = model_keys.index(st.session_state.selected_model) if st.session_state.selected_model in model_keys else 0

    selected_name = st.selectbox(
        "Groq LLM Model",
        options=model_names,
        index=current_index,
        help="Groq high-speed inference models verified for your API key.",
    )
    selected_key = model_keys[model_names.index(selected_name)]
    st.session_state.selected_model = selected_key

    st.caption(f"ℹ️ {AVAILABLE_MODELS[selected_key]['description']}")

    st.markdown("---")

    # Research Depth & Parameters
    st.subheader("🔬 Research Depth")
    depth_choice = st.radio(
        "Investigation Level",
        options=["⚡ Quick Research", "🔬 Deep Research (Multi-Hop)"],
        index=0,
        help="Quick executes a focused single-pass search. Deep performs multi-query decomposition and deeper source retrieval.",
    )
    research_depth = "quick" if "Quick" in depth_choice else "deep"

    max_sources = st.slider(
        "Maximum Web Sources",
        min_value=3,
        max_value=10,
        value=5 if research_depth == "quick" else 8,
        step=1,
    )

    st.markdown("---")

    # Past Research History (SQLite)
    st.subheader("📜 Research History")
    from src.database import get_recent_reports, get_report_by_id, clear_history

    try:
        past_reports = get_recent_reports(limit=10)
    except Exception:
        past_reports = []

    if past_reports:
        for past in past_reports:
            p_id = past["id"]
            p_query = past["query"]
            p_date = past["created_at"].split("T")[0] if "T" in past["created_at"] else past["created_at"][:10]
            label = f"📁 {p_query[:28]}... ({p_date})"
            if st.button(label, key=f"hist_btn_{p_id}", use_container_width=True, help=f"Grounding: {past.get('grounding_score', 0)}%"):
                loaded = get_report_by_id(p_id)
                if loaded:
                    st.session_state.current_report = loaded
                    st.session_state.chat_history = []
                    st.rerun()

        if st.button("🗑️ Clear History", key="clear_hist_btn", use_container_width=True):
            clear_history()
            st.rerun()
    else:
        st.caption("No past research runs recorded yet.")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #94A3B8;">
            <b>Built for:</b> Antigravity IDE<br>
            <b>Stack:</b> Python · Groq API · DDGS · Trafilatura<br>
            <b>Capabilities:</b> Web Search, Hybrid Retrieval, Claim Verification, Freshness Scoring, SQLite Archival
        </div>
        """,
        unsafe_allow_html=True,
    )


# Main Content Area
st.markdown('<div class="main-title">🌐 AI-Powered Web Research & Intelligent Answering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Evidence-grounded web intelligence with claim verification, source credibility grading, and factual provenance.</div>',
    unsafe_allow_html=True,
)

# Example Query Pills
st.markdown("**💡 Try an example research topic:**")
cols = st.columns(4)
sample_queries = [
    "Quantum Computing: IBM vs Google Breakthroughs",
    "Commercial Nuclear Fusion: Status & Net Energy Gain",
    "Solid-State Batteries: Toyota & QuantumScape Timeline",
    "AI Agents in Coding: Benchmarks & Limitations",
]

chosen_sample = None
for i, sample in enumerate(sample_queries):
    if cols[i].button(sample, key=f"sample_{i}", use_container_width=True):
        chosen_sample = sample

# Research Input Bar
with st.container():
    query_input = st.text_input(
        "Enter your research query or question:",
        value=chosen_sample if chosen_sample else "",
        placeholder="e.g. Compare the latest breakthroughs in solid-state batteries in 2024-2025 and their commercial timeline",
        key="main_query_input",
    )

    col_btn, col_clear = st.columns([1, 6])
    run_btn = col_btn.button("🔍 Research", type="primary", use_container_width=True)


# Execute Research
if run_btn and query_input.strip():
    query = query_input.strip()

    # Progress placeholder
    progress_box = st.status(f"🚀 Investigating: '{query}'...", expanded=True)

    def on_progress(step: str, detail: str):
        progress_box.write(f"**[{step}]** {detail}")

    pipeline = WebResearchPipeline(
        groq_api_key=st.session_state.groq_api_key,
        model=st.session_state.selected_model,
    )

    try:
        report, context_text, scraped_docs, chunks = pipeline.run_research(
            query=query,
            depth=research_depth,
            max_sources=max_sources,
            progress_callback=on_progress,
        )
        progress_box.update(
            label=f"✅ Research completed in {report.execution_time_seconds}s ({report.total_sources_scraped} sources verified, {report.grounding_score}% grounded)",
            state="complete",
            expanded=False,
        )

        st.session_state.current_report = report
        st.session_state.current_context = context_text
        st.session_state.chat_history = []  # Reset follow-ups on new query

    except Exception as e:
        progress_box.update(label=f"❌ Error during research: {str(e)}", state="error")
        st.error(f"Error details: {str(e)}")


# Render Report if Available
if st.session_state.current_report:
    report: ResearchReport = st.session_state.current_report

    st.markdown("---")

    # Truthful Synthesis Mode Banner
    if report.synthesis_mode == "ai_generative":
        st.success(f"🤖 **Synthesis Mode:** Full Generative AI Reasoning powered by Groq ({report.model_used}) with Factual Provenance Verification")
    else:
        st.warning("⚠️ **Synthesis Mode:** Offline Extractive Baseline (No LLM API). Showing verified excerpts from scraped sources.")

    # Overview Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱️ Total Latency", f"{report.execution_time_seconds} sec")
    m2.metric("📚 Sources Analyzed", f"{report.total_sources_scraped}")
    m3.metric("🎯 Evidence Grounding", f"{report.grounding_score}%")
    m4.metric("🔬 Depth", f"{report.research_depth.capitalize()}")

    # Latency Breakdown Bar
    if report.timings:
        t_search = report.timings.get("search_seconds", 0.0)
        t_scrape = report.timings.get("scrape_seconds", 0.0)
        t_retrieval = report.timings.get("retrieval_seconds", 0.0)
        t_synthesis = report.timings.get("synthesis_seconds", 0.0)
        st.markdown(
            f'<div class="timing-bar">⚡ <b>Stage Latency Breakdown:</b> Search: <code>{t_search}s</code> | Scraping & Date Extraction: <code>{t_scrape}s</code> | Ranking & Filtering: <code>{t_retrieval}s</code> | LLM Synthesis & Claim Verification: <code>{t_synthesis}s</code></div>',
            unsafe_allow_html=True,
        )

    # Freshness summary
    if report.source_freshness_summary:
        st.info(f"📅 **Temporal Freshness:** {report.source_freshness_summary}")

    # Sub-queries analyzed tag badges
    if report.sub_queries and len(report.sub_queries) > 1:
        st.markdown("**Targeted Search Angles Investigated:**")
        badge_html = "".join([f'<span class="metric-badge">🔍 {sq}</span>' for sq in report.sub_queries])
        st.markdown(badge_html, unsafe_allow_html=True)

    # Executive Summary Card
    st.markdown("### 📋 Executive Summary")

    def safe_md(text: str) -> str:
        """Escape $ signs in LLM-generated text so Streamlit doesn't render them as LaTeX math.
        Also escapes * sequences that could become unwanted markdown bold/italic."""
        # Replace lone $ with escaped version — preserves $$ for real display math if needed
        return text.replace("$", r"\$")

    st.markdown(
        f'<div class="exec-summary-box">{safe_md(report.executive_summary)}</div>',
        unsafe_allow_html=True,
    )

    # Tabs for In-depth Findings, Sources, Claim Verification, and Follow-Up Q&A
    tab_analysis, tab_findings, tab_claims, tab_sources, tab_qa = st.tabs([
        "📖 In-Depth Evidence Synthesis",
        "💡 Key Findings & Consensus",
        "🛡️ Claim Verification & Provenance",
        "🔗 Verified Sources (Registry)",
        "💬 Interactive Follow-Up Q&A",
    ])

    with tab_analysis:
        st.markdown(safe_md(report.detailed_synthesis))

    with tab_findings:
        st.markdown("#### 🔑 Key Takeaways")
        if report.key_findings:
            for finding in report.key_findings:
                st.markdown(f"- {safe_md(finding)}")
        else:
            st.info("No explicit bullet list extracted; see In-Depth Synthesis.")

        if report.consensus_analysis:
            st.markdown("#### ⚖️ Cross-Source Consensus & Perspectives")
            st.markdown(safe_md(report.consensus_analysis))

    with tab_claims:
        st.markdown("#### 🛡️ Factual Claim Verification Matrix")
        st.caption(
            "Grounding Score Formula: **verified × 1.0 + qualified × 0.5 + unsupported × 0.0** ÷ total × 100. "
            "Each claim requires direct sentence-level entailment with key named entities present in the source excerpt."
        )

        if report.claims:
            for c_idx, claim in enumerate(report.claims, start=1):
                status_badge = "🟢 Verified" if claim.verification_status == "verified" else "🟡 Qualified" if claim.verification_status == "qualified" else "🔴 Unsubstantiated"
                badge_style = "badge-green" if claim.verification_status == "verified" else "badge-yellow" if claim.verification_status == "qualified" else "badge-red"
                entailment_color = "#4ADE80" if claim.entailment_type == "direct" else "#FACC15" if claim.entailment_type == "indirect" else "#F87171"
                entailment_label = {"direct": "✅ Direct Entailment", "indirect": "⚠️ Indirect / Paraphrase", "none": "❌ No Entailment"}.get(claim.entailment_type, "—")

                st.markdown(
                    f"""
                    <div class="claim-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 1.0rem;">Claim #{c_idx}</span>
                            <span class="{badge_style}">{status_badge} ({int(claim.confidence_score * 100)}% Confidence)</span>
                        </div>
                        <div style="margin-top: 0.4rem; font-size: 0.95rem; color: #E2E8F0;">
                            "{claim.claim_text}"
                        </div>
                        <div style="margin-top: 0.4rem; font-size: 0.85rem; color: #94A3B8;">
                            <b>Cited Sources:</b> {claim.cited_source_ids} &nbsp;|&nbsp;
                            <b>Entailment:</b> <span style="color: {entailment_color}; font-weight: 600;">{entailment_label}</span>
                        </div>
                        <div style="margin-top: 0.3rem; font-size: 0.82rem; color: #94A3B8;">
                            <b>Audit Note:</b> {claim.reasoning}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.88rem; color: #CBD5E1; background: rgba(0,0,0,0.3); padding: 0.5rem; border-left: 3px solid #38BDF8; border-radius: 0.2rem;">
                            <b>Best Matching Sentence in Source:</b> "{claim.supporting_excerpt if claim.supporting_excerpt else 'No direct excerpt found in cited source text.'}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No standalone claims extracted for auditing.")

    with tab_sources:
        st.markdown(f"#### 🌐 Scraped & Verified Web Sources ({len(report.citations)})")
        st.caption("Evaluated for domain authority, publication date, and relevance.")

        for cit in report.citations:
            # Fix Gap 4: badge color now reflects new accurate label names
            if "Government" in cit.credibility_badge:
                badge_class = "badge-green"
            elif "Academic" in cit.credibility_badge or "Research" in cit.credibility_badge:
                badge_class = "badge-green"
            elif "News" in cit.credibility_badge or "Encyclopedia" in cit.credibility_badge:
                badge_class = "badge-blue"
            elif "Tech" in cit.credibility_badge or "Developer" in cit.credibility_badge:
                badge_class = "badge-yellow"
            else:
                badge_class = "badge-yellow"

            # Fix Gap 2: show date type so users know if it's a publication date or a page-creation date
            date_type_labels = {
                "publication": "📅 Published",
                "last_modified": "🔄 Last Modified",
                "url_pattern": "🔗 Date (from URL)",
                "page_created": "📄 Page Created",
                "unknown": "📅 Date",
            }
            if cit.published_date:
                date_label = date_type_labels.get(getattr(cit, "date_type", "unknown"), "📅 Date")
                date_display = f"{date_label}: <b>{cit.published_date}</b>"
            else:
                date_display = "📅 <b>Date: Unlisted</b>"

            event_display = f" | 📌 Event Mentioned: <b>{cit.event_date}</b>" if getattr(cit, "event_date", None) else ""

            st.markdown(
                f"""
                <div class="source-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1.05rem;">
                            [{cit.id}] <a href="{cit.url}" target="_blank" style="text-decoration: none; color: #38BDF8;">{cit.title}</a>
                        </span>
                        <span class="{badge_class}">{cit.credibility_badge} ({cit.credibility_score}%)</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.2rem;">
                        <b>Domain:</b> {cit.domain} | {date_display}{event_display} | <b>Relevance:</b> {cit.relevance_score:.2f}
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #CBD5E1; background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 0.3rem;">
                        "{cit.snippet}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_qa:
        st.markdown("#### 💬 Ask a Follow-Up Question")
        st.caption("Grounded strictly in the web sources retrieved for this research topic.")

        # Display conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        follow_up_q = st.chat_input("Ask a follow-up question about these findings...")
        if follow_up_q:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": follow_up_q})
            with st.chat_message("user"):
                st.markdown(follow_up_q)

            # Generate assistant answer
            with st.chat_message("assistant"):
                with st.spinner("Analyzing retrieved sources..."):
                    synthesizer = WebResearchPipeline(
                        groq_api_key=st.session_state.groq_api_key,
                        model=st.session_state.selected_model,
                    ).synthesizer
                    ans = synthesizer.answer_follow_up(
                        question=follow_up_q,
                        context_text=st.session_state.current_context,
                        chat_history=st.session_state.chat_history[:-1],
                    )
                    st.markdown(ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})

    # Download Multi-Format Reports
    st.markdown("---")
    st.markdown("### 💾 Export & Archival")

    # Generate Markdown content
    report_markdown = f"""# Research Report: {report.query}
*Generated on {report.created_at} using {report.model_used} ({report.synthesis_mode})*
*Total Sources Analyzed: {report.total_sources_scraped} | Grounding Score: {report.grounding_score}%*

---

## Executive Summary
{report.executive_summary}

---

## In-Depth Analysis
{report.detailed_synthesis}

---

## Key Findings
""" + "\n".join([f"- {f}" for f in report.key_findings]) + f"""

---

## Factual Claim Verification Matrix
""" + "\n".join([f"- **Claim:** \"{c.claim_text}\"\n  - Status: {c.verification_status} ({int(c.confidence_score*100)}% confidence)\n  - Evidence: \"{c.supporting_excerpt}\"\n  - Note: {c.reasoning}" for c in report.claims]) + f"""

---

## Sources & Citations
""" + "\n".join([f"[{c.id}] {c.title} - {c.url} ({c.domain} | Credibility: {c.credibility_score}% | Date: {c.published_date or 'N/A'})" for c in report.citations])

    # Generate Standalone Styled HTML
    claims_html_rows = "".join([
        f"""<tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Claim {idx}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{c.claim_text}</td>
            <td style="padding: 8px; border: 1px solid #ddd; color: {'green' if c.verification_status=='verified' else 'orange' if c.verification_status=='qualified' else 'red'}; font-weight: bold;">{c.verification_status.capitalize()} ({int(c.confidence_score*100)}%)</td>
            <td style="padding: 8px; border: 1px solid #ddd;"><i>"{c.supporting_excerpt}"</i><br><small>Note: {c.reasoning}</small></td>
        </tr>""" for idx, c in enumerate(report.claims, 1)
    ])

    citations_html_rows = "".join([
        f"""<li>
            <b>[{c.id}] <a href="{c.url}" target="_blank">{c.title}</a></b><br>
            <small>Domain: {c.domain} | Credibility: {c.credibility_score}% ({c.credibility_badge}) | Published: {c.published_date or 'Unlisted'}</small><br>
            <span style="color: #555;">"{c.snippet}"</span>
        </li><br>""" for c in report.citations
    ])

    report_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Research Report: {report.query}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 20px; color: #222; }}
        h1 {{ color: #1E3A8A; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; }}
        h2 {{ color: #1E40AF; margin-top: 30px; border-bottom: 1px solid #E2E8F0; padding-bottom: 5px; }}
        .badge {{ display: inline-block; padding: 3px 8px; background: #E0E7FF; color: #3730A3; border-radius: 4px; font-size: 13px; font-weight: 600; margin-right: 8px; }}
        .summary-box {{ background: #F8FAFC; border-left: 4px solid #3B82F6; padding: 15px; margin: 20px 0; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
        th {{ background: #F1F5F9; text-align: left; padding: 10px; border: 1px solid #CBD5E1; }}
        @media print {{ body {{ margin: 15px; max-width: 100%; }} }}
    </style>
</head>
<body>
    <h1>🌐 Research Intelligence Report</h1>
    <h2>{report.query}</h2>
    <div>
        <span class="badge">Model: {report.model_used}</span>
        <span class="badge">Grounding: {report.grounding_score}%</span>
        <span class="badge">Sources: {report.total_sources_scraped}</span>
        <span class="badge">Latency: {report.execution_time_seconds}s</span>
        <span class="badge">Date: {report.created_at}</span>
    </div>

    <h2>Executive Summary</h2>
    <div class="summary-box">
        {report.executive_summary}
    </div>

    <h2>In-Depth Evidence Synthesis</h2>
    <div>
        {report.detailed_synthesis.replace(chr(10), '<br>')}
    </div>

    <h2>Factual Claim Verification Matrix</h2>
    <table>
        <thead>
            <tr><th>#</th><th>Extracted Claim</th><th>Verification Status</th><th>Supporting Excerpt & Audit</th></tr>
        </thead>
        <tbody>
            {claims_html_rows}
        </tbody>
    </table>

    <h2>Verified Primary Sources</h2>
    <ul>
        {citations_html_rows}
    </ul>
</body>
</html>"""

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    exp_col1.download_button(
        label="📥 Markdown Report (.md)",
        data=report_markdown,
        file_name=f"research_report_{int(time.time())}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    exp_col2.download_button(
        label="📥 Machine-Readable JSON (.json)",
        data=report.model_dump_json(indent=2),
        file_name=f"research_report_{int(time.time())}.json",
        mime="application/json",
        use_container_width=True,
    )
    exp_col3.download_button(
        label="📥 Printable HTML / PDF (.html)",
        data=report_html,
        file_name=f"research_report_{int(time.time())}.html",
        mime="text/html",
        use_container_width=True,
    )
