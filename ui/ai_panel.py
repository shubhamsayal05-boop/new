"""AI Advisor page and shared diagnostic hooks."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.layout import render_ai_insight_compact, render_diagnostic_issues, render_page_header, section_container
from vehicle_plotter.ai_advisor import AdvisorReport, collect_diagnostics, enhance_with_copilot
from vehicle_plotter.copilot_client import CopilotConfig, load_copilot_config


def build_advisor_context(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def run_advisor(context: dict[str, Any]) -> AdvisorReport:
    return collect_diagnostics(context)


def render_ai_advisor_page(context: dict[str, Any]) -> None:
    render_page_header(
        "AI Advisor",
        "Automated issue detection and actionable fix suggestions for your measurement session.",
    )

    config = load_copilot_config()

    col_scan, col_ai = st.columns([1, 1])
    with col_scan:
        scan = st.button("Scan session", type="primary", use_container_width=True)
    with col_ai:
        use_copilot = st.toggle(
            "Enhance with Microsoft Copilot",
            value=False,
            help="Uses your company's Azure OpenAI / Copilot deployment configured in secrets.",
        )

    if use_copilot:
        with st.expander("Microsoft Copilot connection", expanded=not config.is_configured):
            st.caption(
                "Configure once in `.streamlit/secrets.toml` under `[copilot]` — values below override secrets for this session."
            )
            endpoint = st.text_input(
                "Azure OpenAI endpoint",
                value=config.azure_endpoint,
                placeholder="https://your-resource.openai.azure.com/",
            )
            deployment = st.text_input(
                "Deployment name",
                value=config.deployment,
                placeholder="gpt-4o / copilot-deployment",
            )
            api_key = st.text_input("API key", type="password", value=config.api_key if config.api_key else "")
            api_version = st.text_input("API version", value=config.api_version)
            config = CopilotConfig(
                azure_endpoint=endpoint.strip(),
                api_key=api_key.strip(),
                deployment=deployment.strip(),
                api_version=api_version.strip() or "2024-08-01-preview",
            )
            if not config.is_configured:
                st.info(
                    "Example `secrets.toml`:\n\n"
                    "```toml\n[copilot]\n"
                    'azure_endpoint = "https://YOUR-RESOURCE.openai.azure.com/"\n'
                    'api_key = "YOUR-KEY"\n'
                    'deployment = "YOUR-DEPLOYMENT"\n'
                    'api_version = "2024-08-01-preview"\n```'
                )

    if scan or "dl_advisor_report" not in st.session_state:
        report = run_advisor(context)
        if use_copilot:
            try:
                report.ai_narrative = enhance_with_copilot(report, context, config)
            except Exception as exc:
                st.warning(f"Microsoft Copilot enhancement unavailable: {exc}")
        st.session_state["dl_advisor_report"] = report
    else:
        report = st.session_state.get("dl_advisor_report") or run_advisor(context)

    st.metric("Session health", f"{report.health_score}/100")
    render_ai_insight_compact(report.summary, report.issues[0].title if report.issues else None)

    if report.ai_narrative:
        with section_container("Copilot action plan", "Microsoft Copilot"):
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
