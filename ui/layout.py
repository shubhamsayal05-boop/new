"""DriveLab Pro layout components."""

from __future__ import annotations

import html

import streamlit as st

from ui.theme import APP_NAME, APP_TAGLINE, APP_VERSION, NAV_PAGES, THEME_CSS, WORKFLOW_BY_PAGE, WORKFLOW_STEPS


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_topbar(health_score: int = 100) -> None:
    if health_score >= 80:
        dot_class, label = "ok", "Healthy"
    elif health_score >= 50:
        dot_class, label = "warn", "Needs review"
    else:
        dot_class, label = "bad", "Action required"

    st.markdown(
        f"""
<div class="dl-topbar">
  <div class="dl-brand-block">
    <div class="dl-logo">DL</div>
    <div>
      <div class="dl-brand-name">{APP_NAME}</div>
      <div class="dl-brand-sub">{APP_TAGLINE} · v{APP_VERSION}</div>
    </div>
  </div>
  <div class="dl-health">
    <span class="dl-health-dot {dot_class}"></span>
    Session {label} · {health_score}/100
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str = "") -> None:
    desc_html = f'<p class="dl-page-desc">{html.escape(description)}</p>' if description else ""
    st.markdown(
        f'<div class="dl-page-head"><h1 class="dl-page-title">{html.escape(title)}</h1>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def render_workflow_strip(active_page: str) -> None:
    active_step = WORKFLOW_BY_PAGE.get(active_page, 1)
    chips = []
    for index, label in enumerate(WORKFLOW_STEPS, start=1):
        cls = "dl-wf-step active" if index == active_step else "dl-wf-step"
        chips.append(f'<span class="{cls}"><span class="dl-wf-num">{index}</span>{html.escape(label)}</span>')
    st.markdown(f'<div class="dl-workflow">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_metric_grid(cards: list[tuple[str, str, str]]) -> None:
    """Render metric cards: (label, value, hint)."""

    items = []
    for label, value, hint in cards:
        hint_html = f'<div class="dl-card-hint">{html.escape(hint)}</div>' if hint else ""
        items.append(
            f'<div class="dl-card"><div class="dl-card-label">{html.escape(label)}</div>'
            f'<div class="dl-card-value">{html.escape(value)}</div>{hint_html}</div>'
        )
    st.markdown(f'<div class="dl-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def render_section(title: str, badge: str = "", *, open_body: bool = True) -> None:
    badge_html = f"<span>{html.escape(badge)}</span>" if badge else ""
    st.markdown(
        f'<div class="dl-section"><div class="dl-section-head">{html.escape(title)}{badge_html}</div></div>',
        unsafe_allow_html=True,
    )


def section_container(title: str, badge: str = ""):
    """Context manager wrapper using Streamlit container + section header."""

    class _Section:
        def __enter__(self):
            self._container = st.container()
            self._container.__enter__()
            badge_html = f"<span>{html.escape(badge)}</span>" if badge else ""
            st.markdown(
                f'<div class="dl-section-head">{html.escape(title)}{badge_html}</div>',
                unsafe_allow_html=True,
            )
            return self

        def __exit__(self, *args):
            return self._container.__exit__(*args)

    return _Section()


def render_ai_insight_compact(summary: str, top_issue_title: str | None = None) -> None:
    extra = ""
    if top_issue_title:
        extra = f'<div class="dl-issue-body">Top issue: {html.escape(top_issue_title)}</div>'
    st.markdown(
        f"""
<div class="dl-ai-panel">
  <div class="dl-ai-title"><span class="dl-ai-badge">AI ADVISOR</span> Session insight</div>
  <div class="dl-issue-body">{html.escape(summary)}</div>
  {extra}
</div>
""",
        unsafe_allow_html=True,
    )


def render_diagnostic_issues(issues: list, max_items: int | None = None) -> None:
    from vehicle_plotter.ai_advisor import DiagnosticIssue

    shown = issues[:max_items] if max_items else issues
    if not shown:
        st.markdown(
            '<div class="dl-ai-panel"><div class="dl-ai-title">'
            '<span class="dl-ai-badge">AI ADVISOR</span> No issues detected</div>'
            '<div class="dl-issue-body">Your session configuration looks good.</div></div>',
            unsafe_allow_html=True,
        )
        return

    blocks = []
    for issue in shown:
        if not isinstance(issue, DiagnosticIssue):
            continue
        fixes = "".join(f"<li>{html.escape(s)}</li>" for s in issue.suggestions[:4])
        blocks.append(
            f"""
<div class="dl-issue {html.escape(issue.severity)}">
  <div class="dl-issue-title">[{html.escape(issue.severity.upper())}] {html.escape(issue.title)}</div>
  <div class="dl-issue-body">{html.escape(issue.description)}</div>
  <ul class="dl-issue-fix">{fixes}</ul>
</div>"""
        )
    st.markdown(
        f'<div class="dl-ai-panel"><div class="dl-ai-title">'
        f'<span class="dl-ai-badge">AI ADVISOR</span> {len(issues)} finding(s)</div>'
        f'{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_nav() -> str:
    labels = [p[0] for p in NAV_PAGES]
    descriptions = {p[0]: p[1] for p in NAV_PAGES}

    st.markdown("##### Workspace")
    page = st.radio(
        "Navigation",
        labels,
        key="dl_nav_page",
        label_visibility="collapsed",
        format_func=lambda x: x,
    )
    st.caption(descriptions.get(page, ""))
    return page
