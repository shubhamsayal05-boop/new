"""DriveLab Pro design system — distinct from the original Vehicle Plotter UI."""

from __future__ import annotations

APP_NAME = "DriveLab Pro"
APP_TAGLINE = "INCA measurement intelligence platform"
APP_VERSION = "2.0"

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --dl-bg: #0b0f14;
    --dl-surface: #121820;
    --dl-surface-2: #1a2230;
    --dl-border: #2a3544;
    --dl-ink: #e8edf4;
    --dl-muted: #8b9cb3;
    --dl-accent: #14b8a6;
    --dl-accent-dim: rgba(20, 184, 166, 0.15);
    --dl-warn: #f59e0b;
    --dl-danger: #ef4444;
    --dl-ok: #22c55e;
    --dl-radius: 12px;
    --dl-font: 'DM Sans', system-ui, sans-serif;
    --dl-mono: 'JetBrains Mono', ui-monospace, monospace;
}

html, body, [class*="css"], .stApp {
    font-family: var(--dl-font);
}
.stApp {
    background: var(--dl-bg);
    color: var(--dl-ink);
}
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(180deg, #0b0f14 0%, #0e1319 100%);
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1480px;
}

/* Hide default header/footer chrome */
header[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }

/* Sidebar — command rail */
section[data-testid="stSidebar"] {
    background: #080b10;
    border-right: 1px solid var(--dl-border);
}
section[data-testid="stSidebar"] * { color: var(--dl-ink); }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: var(--dl-muted);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: var(--dl-surface) !important;
    color: var(--dl-ink) !important;
    border-color: var(--dl-border) !important;
}

/* Top bar */
.dl-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 14px 20px;
    margin-bottom: 20px;
    background: var(--dl-surface);
    border: 1px solid var(--dl-border);
    border-radius: var(--dl-radius);
}
.dl-brand-block { display: flex; align-items: center; gap: 14px; }
.dl-logo {
    width: 44px; height: 44px; border-radius: 10px;
    background: linear-gradient(135deg, #0d9488, #14b8a6);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1.1rem; color: #042f2e;
    box-shadow: 0 0 24px rgba(20,184,166,0.35);
}
.dl-brand-name { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
.dl-brand-sub { font-size: 0.82rem; color: var(--dl-muted); margin-top: 2px; }
.dl-health {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 14px; border-radius: 999px;
    background: var(--dl-surface-2); border: 1px solid var(--dl-border);
    font-size: 0.85rem; font-weight: 600;
}
.dl-health-dot { width: 8px; height: 8px; border-radius: 50%; }
.dl-health-dot.ok { background: var(--dl-ok); box-shadow: 0 0 8px var(--dl-ok); }
.dl-health-dot.warn { background: var(--dl-warn); }
.dl-health-dot.bad { background: var(--dl-danger); }

/* Page header */
.dl-page-head {
    margin: 0 0 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--dl-border);
}
.dl-page-title { font-size: 1.55rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }
.dl-page-desc { color: var(--dl-muted); font-size: 0.92rem; margin: 6px 0 0; max-width: 720px; }

/* Workflow steps */
.dl-workflow {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px;
}
.dl-wf-step {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 12px; border-radius: 8px; font-size: 0.78rem; font-weight: 600;
    background: var(--dl-surface); border: 1px solid var(--dl-border); color: var(--dl-muted);
}
.dl-wf-step.active {
    background: var(--dl-accent-dim); border-color: rgba(20,184,166,0.45); color: var(--dl-accent);
}
.dl-wf-num {
    width: 18px; height: 18px; border-radius: 5px;
    background: var(--dl-surface-2); display: inline-flex;
    align-items: center; justify-content: center; font-size: 0.68rem;
}

/* Cards & metrics */
.dl-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 18px; }
.dl-card {
    background: var(--dl-surface);
    border: 1px solid var(--dl-border);
    border-radius: var(--dl-radius);
    padding: 16px 18px;
}
.dl-card-label { font-size: 0.72rem; font-weight: 600; color: var(--dl-muted); text-transform: uppercase; letter-spacing: 0.07em; }
.dl-card-value { font-size: 1.65rem; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
.dl-card-hint { font-size: 0.76rem; color: var(--dl-muted); margin-top: 4px; }

/* AI insight panel */
.dl-ai-panel {
    background: linear-gradient(135deg, rgba(20,184,166,0.08), rgba(59,130,246,0.06));
    border: 1px solid rgba(20,184,166,0.28);
    border-radius: var(--dl-radius);
    padding: 16px 18px;
    margin-bottom: 18px;
}
.dl-ai-title {
    display: flex; align-items: center; gap: 10px;
    font-weight: 700; font-size: 0.95rem; margin-bottom: 8px;
}
.dl-ai-badge {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 3px 8px; border-radius: 999px;
    background: rgba(20,184,166,0.2); color: var(--dl-accent);
}
.dl-issue {
    padding: 12px 14px; margin-top: 10px;
    background: var(--dl-surface-2); border-radius: 10px;
    border-left: 3px solid var(--dl-border);
}
.dl-issue.critical { border-left-color: var(--dl-danger); }
.dl-issue.warning { border-left-color: var(--dl-warn); }
.dl-issue.info { border-left-color: var(--dl-accent); }
.dl-issue-title { font-weight: 600; font-size: 0.88rem; }
.dl-issue-body { color: var(--dl-muted); font-size: 0.82rem; margin-top: 4px; }
.dl-issue-fix { margin-top: 8px; font-size: 0.8rem; }
.dl-issue-fix li { margin: 2px 0; }

/* Section blocks */
.dl-section {
    background: var(--dl-surface);
    border: 1px solid var(--dl-border);
    border-radius: var(--dl-radius);
    padding: 18px 20px;
    margin-bottom: 16px;
}
.dl-section-head {
    font-size: 1rem; font-weight: 700; margin-bottom: 12px;
    display: flex; align-items: center; gap: 10px;
}
.dl-section-head span {
    font-size: 0.7rem; font-weight: 600; color: var(--dl-muted);
    background: var(--dl-surface-2); padding: 2px 8px; border-radius: 6px;
}

/* Plotly & dataframes in dark shell */
[data-testid="stPlotlyChart"] {
    background: var(--dl-surface);
    border: 1px solid var(--dl-border);
    border-radius: var(--dl-radius);
    padding: 6px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: var(--dl-radius);
    border: 1px solid var(--dl-border);
}
[data-testid="stExpander"] {
    background: var(--dl-surface) !important;
    border: 1px solid var(--dl-border) !important;
    border-radius: var(--dl-radius) !important;
}
[data-testid="stAlert"] { border-radius: 10px; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0d9488, #14b8a6) !important;
    border: none !important;
    color: #042f2e !important;
    font-weight: 700 !important;
}
</style>
"""

NAV_PAGES = [
    ("Dashboard", "Overview & session health"),
    ("Data Source", "Load files & map signals"),
    ("Speed Analysis", "Launch / decel curves"),
    ("OPD Analysis", "One-pedal deceleration"),
    ("Reference Data", "Targets & scatter plots"),
    ("AI Advisor", "Diagnostics & fix suggestions"),
]

WORKFLOW_BY_PAGE = {
    "Dashboard": 1,
    "Data Source": 2,
    "Speed Analysis": 3,
    "OPD Analysis": 4,
    "Reference Data": 5,
    "AI Advisor": 6,
}

WORKFLOW_STEPS = [
    "Connect data",
    "Map signals",
    "Analyze curves",
    "Run OPD",
    "Add references",
    "AI review",
]
