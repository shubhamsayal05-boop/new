"""AI Advisor page and shared diagnostic hooks."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.layout import render_ai_insight_compact, render_diagnostic_issues, render_page_header, section_container
from vehicle_plotter.ai_advisor import AdvisorReport, collect_diagnostics, enhance_with_llm


def build_advisor_context(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def run_advisor(context: dict[str, Any]) -> AdvisorReport:
    return collect_diagnostics(context)


def render_ai_advisor_page(context: dict[str, Any]) -> None:
    render_page_header(
        "AI Advisor",
        "Automated issue detection and actionable fix suggestions for your measurement session.",
    )

    col_scan, col_ai = st.columns([1, 1])
    with col_scan:
        scan = st.button("Scan session", type="primary", use_container_width=True)
    with col_ai:
        use_llm = st.toggle("Enhance with OpenAI", value=False, help="Requires OPENAI_API_KEY in Streamlit secrets.")

    api_key = ""
    if use_llm:
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=_secret_api_key(),
            help="Stored only for this session unless set in .streamlit/secrets.toml as OPENAI_API_KEY.",
        )

    if scan or "dl_advisor_report" not in st.session_state:
        report = run_advisor(context)
        if use_llm and api_key.strip():
            try:
                report.ai_narrative = enhance_with_llm(report, context, api_key.strip())
            except Exception as exc:
                st.warning(f"LLM enhancement unavailable: {exc}")
        st.session_state["dl_advisor_report"] = report
    else:
        report = st.session_state.get("dl_advisor_report") or run_advisor(context)

    st.metric("Session health", f"{report.health_score}/100")
    render_ai_insight_compact(report.summary, report.issues[0].title if report.issues else None)

    if report.ai_narrative:
        with section_container("AI action plan", "LLM"):
            st.markdown(report.ai_narrative)

    with section_container("Detected issues", f"{len(report.issues)} total"):
        render_diagnostic_issues(report.issues)

    with section_container("Quick fixes by category"):
        categories: dict[str, list] = {}
        for issue in report.issues:
            categories.setdefault(issue.category, []).append(issue)
        for category, items in categories.items():
            st.markdown(f"**{category}** ({len(items)})")
            for issue in items[:5]:
                st.markdown(f"- **{issue.title}**: {issue.suggestions[0] if issue.suggestions else issue.description}")


def render_ai_sidebar_hint(context: dict[str, Any]) -> None:
    report = run_advisor(context)
    if report.issues:
        top = report.issues[0]
        st.markdown("---")
        st.markdown("**AI Advisor**")
        st.caption(report.summary)
        if st.button("Open AI Advisor", use_container_width=True):
            st.session_state["dl_nav_page"] = "AI Advisor"
            st.rerun()


def _secret_api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""
