from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vehicle_plotter.exporting import export_plot_workbook
from vehicle_plotter.metrics import (
    DEFAULT_SPEED_BANDS,
    average_band_long,
    band_key,
    band_label,
    calculate_metrics,
    export_metrics_workbook,
    format_metrics_table,
    make_average_band_figure,
    make_metric_figure,
)
from vehicle_plotter.mdf_io import (
    MdfDependencyError,
    discover_channel_names,
    read_channel_unit,
    read_measurement_frame,
)
from vehicle_plotter.plotting import (
    MODE_DASHES,
    PEDAL_COLORS,
    RANDOM_SCATTER_COLORS,
    curve_style_key,
    legend_label,
    make_random_scatter_figure,
    make_speed_accel_figure,
    mode_dash_map,
    mode_key,
    random_scatter_style_key,
    scatter_dash_map,
    scatter_source_key,
    short_group_label,
)
from vehicle_plotter.processing import (
    DECEL_TARGETS,
    LAUNCH_TARGETS,
    ProcessingSettings,
    average_file_curves,
    normalize_measurement_frame,
    process_measurement_frame,
)
from vehicle_plotter.random_scatter import (
    ParsedRandomScatterData,
    SCATTER_LAYOUTS,
    average_selected_series,
    export_random_scatter_workbook,
    offset_selected_series,
    parse_pasted_scatter_text,
    parse_scatter_file,
)
from vehicle_plotter.target_data import (
    ParsedTargetData,
    TARGET_LAYOUTS,
    convert_target_curves,
    parse_pasted_target_text,
    parse_target_file,
)
from vehicle_plotter.trendlines import (
    TRENDLINE_DEGREES,
    add_trendline_traces,
    build_trendline_data,
    export_trendline_workbook,
    make_trendline_figure,
    trendline_equations,
)
from vehicle_plotter.units import (
    ACCELERATION_UNITS,
    SPEED_UNITS,
    convert_measurement_units,
    convert_speed_value,
    normalize_acceleration_unit,
    normalize_speed_unit,
)
from vehicle_plotter.colors import normalize_color_value


APP_TITLE = "Vehicle Speed vs Acceleration Plotter"
TEST_TYPES = ["Launch", "Deceleration", "Ignore"]
TARGET_OPTIONS = ["Auto", "All targets in file"] + [
    str(value) for value in sorted(set(LAUNCH_TARGETS + DECEL_TARGETS))
]
RESULT_KEYS = ["averaged_curves", "audit_table", "result_signature", "load_errors"]
PLOT_FILTER_KEYS = ["plot_test_types", "plot_scenarios", "plot_targets"]
CLEAR_NONCE_KEY = "clear_all_nonce"
CLEAR_RERUN_KEY = "clear_all_force_rerun"


# --------------------------------------------------------------------------- #
# Design system: global styling, hero header, section headers, status cards.
# --------------------------------------------------------------------------- #

_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --vp-bg: #f5f7fb;
    --vp-surface: #ffffff;
    --vp-ink: #0f1b2d;
    --vp-muted: #5b6b82;
    --vp-border: #e4e9f2;
    --vp-primary: #2563eb;
    --vp-primary-dark: #1d4ed8;
    --vp-accent: #7c3aed;
    --vp-radius: 14px;
    --vp-shadow: 0 1px 2px rgba(15, 27, 45, 0.04), 0 8px 24px rgba(15, 27, 45, 0.06);
}

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.stApp { background: var(--vp-bg); color: var(--vp-ink); }

/* Constrain and pad the main content column */
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1360px; }

h1, h2, h3, h4 { color: var(--vp-ink); font-weight: 700; letter-spacing: -0.01em; }

/* ------- Hero ------- */
.vp-hero {
    position: relative;
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 22px;
    color: #fff;
    background:
        radial-gradient(1200px 300px at 90% -40%, rgba(124, 58, 237, 0.55), transparent 60%),
        linear-gradient(120deg, #1e3a8a 0%, #2563eb 45%, #4f46e5 100%);
    box-shadow: 0 18px 40px rgba(37, 99, 235, 0.28);
    overflow: hidden;
}
.vp-hero::after {
    content: "";
    position: absolute; inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.10) 1px, transparent 1px);
    background-size: 18px 18px;
    opacity: 0.4; pointer-events: none;
}
.vp-hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase;
    color: rgba(255,255,255,0.85);
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 5px 12px; border-radius: 999px;
}
.vp-hero-title { font-size: 2.15rem; font-weight: 800; margin: 14px 0 6px; line-height: 1.1; }
.vp-hero-sub { font-size: 1.02rem; color: rgba(255,255,255,0.90); max-width: 760px; margin: 0; }
.vp-steps { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; position: relative; z-index: 1; }
.vp-step {
    display: inline-flex; align-items: center; gap: 9px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    color: #fff; font-size: 0.82rem; font-weight: 500;
    padding: 8px 13px; border-radius: 10px; backdrop-filter: blur(4px);
}
.vp-step b { font-weight: 700; }
.vp-step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 6px;
    background: rgba(255,255,255,0.22); font-size: 0.72rem; font-weight: 700;
}

/* ------- Section headers ------- */
.vp-section {
    display: flex; align-items: center; gap: 13px;
    margin: 30px 0 8px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--vp-border);
}
.vp-section-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; flex: 0 0 40px;
    border-radius: 11px;
    background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(124,58,237,0.12));
    color: var(--vp-primary);
    border: 1px solid rgba(37,99,235,0.18);
}
.vp-section-icon svg { width: 20px; height: 20px; }
.vp-section-title { font-size: 1.22rem; font-weight: 700; line-height: 1.15; }
.vp-section-sub { font-size: 0.86rem; color: var(--vp-muted); margin-top: 2px; }

/* ------- Status cards ------- */
.vp-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 6px 0 4px; }
.vp-stat {
    background: var(--vp-surface);
    border: 1px solid var(--vp-border);
    border-radius: var(--vp-radius);
    padding: 16px 18px;
    box-shadow: var(--vp-shadow);
    position: relative; overflow: hidden;
}
.vp-stat::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--vp-primary), var(--vp-accent));
}
.vp-stat-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--vp-muted); }
.vp-stat-value { font-size: 1.7rem; font-weight: 800; color: var(--vp-ink); line-height: 1.1; margin-top: 4px; }
.vp-stat-hint { font-size: 0.78rem; color: var(--vp-muted); margin-top: 3px; }
.vp-badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 2px 9px; border-radius: 999px; margin-top: 6px;
}
.vp-badge.ok { background: rgba(16,185,129,0.12); color: #047857; }
.vp-badge.idle { background: rgba(100,116,139,0.14); color: #475569; }

/* ------- Sidebar ------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(185deg, #101a2e 0%, #16233d 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #dbe4f3; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #aebbd1; font-weight: 500; }
/* Keep values typed into sidebar inputs dark-on-white for contrast */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div[value],
section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #0f1b2d; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #0d1626; border: 1px dashed rgba(255,255,255,0.28); border-radius: 12px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #c8d4e8; }
.vp-brand { display: flex; align-items: center; gap: 12px; padding: 4px 2px 14px; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.vp-brand-logo {
    width: 42px; height: 42px; border-radius: 12px; flex: 0 0 42px;
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    box-shadow: 0 6px 16px rgba(37,99,235,0.4);
}
.vp-brand-logo svg { width: 22px; height: 22px; color: #fff; }
.vp-brand-name { font-size: 1.02rem; font-weight: 700; color: #fff; line-height: 1.15; }
.vp-brand-tag { font-size: 0.72rem; color: #9fb0cc; }

/* ------- Controls ------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 10px; font-weight: 600; border: 1px solid var(--vp-border);
    transition: transform .05s ease, box-shadow .15s ease, background .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    box-shadow: var(--vp-shadow); transform: translateY(-1px);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--vp-primary), var(--vp-primary-dark));
    border: none; color: #fff;
}
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.18); color: #fff;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.14); }

/* Inputs */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-baseweb="select"] > div, [data-testid="stTextArea"] textarea {
    border-radius: 10px !important;
}

/* Expanders as cards */
[data-testid="stExpander"] {
    border: 1px solid var(--vp-border) !important;
    border-radius: var(--vp-radius) !important;
    background: var(--vp-surface);
    box-shadow: var(--vp-shadow);
    overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* Bordered containers -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--vp-radius) !important;
}

/* Dataframes */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: var(--vp-radius); overflow: hidden; border: 1px solid var(--vp-border);
}

/* Alerts */
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--vp-border); }

/* Plotly chart card */
[data-testid="stPlotlyChart"] {
    background: var(--vp-surface);
    border: 1px solid var(--vp-border);
    border-radius: var(--vp-radius);
    padding: 8px 6px;
    box-shadow: var(--vp-shadow);
}

/* Tabs */
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] { border-radius: 10px 10px 0 0; }

footer, #MainMenu { visibility: hidden; }
</style>
"""

_ICONS = {
    "database": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
    "ruler": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8l5-5 13 13-5 5z"/><path d="M8 6l2 2M11 9l2 2M14 12l2 2"/></svg>',
    "signal": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 4 5-8 4 6"/></svg>',
    "target": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/></svg>',
    "scatter": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><circle cx="8" cy="15" r="1.4"/><circle cx="12" cy="9" r="1.4"/><circle cx="16" cy="13" r="1.4"/><circle cx="19" cy="7" r="1.4"/></svg>',
    "tag": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12l-8 8-9-9V3h8z"/><circle cx="7.5" cy="7.5" r="1.4"/></svg>',
    "audit": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3 8-8"/><path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l3-4 3 2 5-7"/></svg>',
    "metric": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21a9 9 0 1 1 9-9"/><path d="M12 12l4-3"/></svg>',
    "car": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 13l2-5a2 2 0 0 1 2-1h10a2 2 0 0 1 2 1l2 5"/><path d="M3 13h18v4H3z"/><circle cx="7" cy="17" r="1.6"/><circle cx="17" cy="17" r="1.6"/></svg>',
}


def inject_global_styles() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    steps = [
        ("1", "Load files"),
        ("2", "Configure signals & units"),
        ("3", "Import targets / scatter"),
        ("4", "Label tests"),
        ("5", "Plot & style"),
        ("6", "Metrics & export"),
    ]
    chips = "".join(
        f'<span class="vp-step"><span class="vp-step-num">{num}</span>{label}</span>'
        for num, label in steps
    )
    st.markdown(
        f"""
<div class="vp-hero">
    <span class="vp-hero-eyebrow">{_ICONS['car']}&nbsp; Vehicle Performance Studio</span>
    <div class="vp-hero-title">{APP_TITLE}</div>
    <p class="vp-hero-sub">{APP_TAGLINE}. Load ETAS INCA MDF/MF4/DAT files, average
    repeated runs, overlay target curves, and export publication-ready plots and metrics.</p>
    <div class="vp-steps">{chips}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", icon: str = "chart") -> None:
    svg = _ICONS.get(icon, _ICONS["chart"])
    sub_html = f'<div class="vp-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
<div class="vp-section">
    <div class="vp-section-icon">{svg}</div>
    <div>
        <div class="vp-section-title">{title}</div>
        {sub_html}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _stat_card(label: str, value: str, hint: str = "", badge: str = "", badge_kind: str = "idle") -> str:
    hint_html = f'<div class="vp-stat-hint">{hint}</div>' if hint else ""
    badge_html = f'<span class="vp-badge {badge_kind}">{badge}</span>' if badge else ""
    return (
        f'<div class="vp-stat"><div class="vp-stat-label">{label}</div>'
        f'<div class="vp-stat-value">{value}</div>{hint_html}{badge_html}</div>'
    )


def render_status_strip(cards: list[str]) -> None:
    st.markdown(
        f'<div class="vp-stats">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.markdown(
        f"""
<div class="vp-brand">
    <span class="vp-brand-logo">{_ICONS['car']}</span>
    <div>
        <div class="vp-brand-name">Vehicle Plotter</div>
        <div class="vp-brand-tag">Speed vs Acceleration Studio</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


APP_TAGLINE = "Measurement analytics for launch & deceleration performance"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="\U0001F697",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            f"### {APP_TITLE}\n"
            "Analyze ETAS INCA measurement files, average repeated runs, and "
            "export professional Speed vs Acceleration plots and metrics."
        )
    },
)

inject_global_styles()

if st.session_state.pop(CLEAR_RERUN_KEY, False):
    st.rerun()


@st.cache_data(show_spinner=False)
def cached_discover_uploaded_channels(file_name: str, signature: str, file_bytes: bytes) -> list[str]:
    del file_name, signature
    return discover_channel_names(file_bytes)


@st.cache_data(show_spinner=False)
def cached_discover_local_channels(file_path: str, signature: str) -> list[str]:
    del signature
    return discover_channel_names(file_path)


@st.cache_data(show_spinner=False)
def cached_uploaded_channel_unit(
    file_name: str,
    signature: str,
    file_bytes: bytes,
    channel: str,
) -> str:
    del file_name, signature
    return read_channel_unit(file_bytes, channel)


@st.cache_data(show_spinner=False)
def cached_local_channel_unit(file_path: str, signature: str, channel: str) -> str:
    del signature
    return read_channel_unit(file_path, channel)


def content_hash(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()


def clear_results() -> None:
    for key in RESULT_KEYS + PLOT_FILTER_KEYS:
        st.session_state.pop(key, None)


def clear_all() -> None:
    st.cache_data.clear()
    next_nonce = int(st.session_state.get(CLEAR_NONCE_KEY, 0)) + 1
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state[CLEAR_NONCE_KEY] = next_nonce
    st.session_state[CLEAR_RERUN_KEY] = True


def resettable_key(name: str) -> str:
    return f"{name}_{st.session_state.get(CLEAR_NONCE_KEY, 0)}"


def default_metadata(uploaded_file_names: list[str]) -> pd.DataFrame:
    rows = []
    for file_name in uploaded_file_names:
        lowered = file_name.casefold()
        test_type = "Deceleration" if any(token in lowered for token in ["decel", "tip", "coast"]) else "Launch"
        rows.append(
            {
                "file_name": file_name,
                "test_type": test_type,
                "mode_label": "",
                "regen_label": "",
                "target_pedal": "Auto",
            }
        )
    return pd.DataFrame(rows)


def metadata_for_current_uploads(
    uploaded_file_names: list[str],
    source_signature: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    upload_signature = (tuple(uploaded_file_names), source_signature)
    if st.session_state.get("upload_signature") == upload_signature:
        return st.session_state["file_metadata"].copy()

    existing = st.session_state.get("file_metadata")
    if existing is None or existing.empty:
        metadata = default_metadata(uploaded_file_names)
    else:
        metadata = existing[existing["file_name"].isin(uploaded_file_names)].copy()
        known = set(metadata["file_name"])
        missing = [name for name in uploaded_file_names if name not in known]
        if missing:
            metadata = pd.concat([metadata, default_metadata(missing)], ignore_index=True)

    st.session_state["upload_signature"] = upload_signature
    st.session_state["file_metadata"] = metadata.reset_index(drop=True)
    st.session_state["metadata_editor_version"] = st.session_state.get("metadata_editor_version", 0) + 1
    clear_results()
    return st.session_state["file_metadata"].copy()


def default_target_metadata(target_source_names: list[str]) -> pd.DataFrame:
    rows = []
    for source_name in target_source_names:
        label = Path(source_name).stem if source_name != "Pasted target data" else "Pasted Target"
        rows.append({"source_name": source_name, "series_label": label or "Target"})
    return pd.DataFrame(rows)


def metadata_for_current_targets(
    target_source_names: list[str],
    source_signature: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    target_signature = (tuple(target_source_names), source_signature)
    if st.session_state.get("target_signature") == target_signature:
        return st.session_state["target_metadata"].copy()

    existing = st.session_state.get("target_metadata")
    if existing is None or existing.empty:
        metadata = default_target_metadata(target_source_names)
    else:
        metadata = existing[existing["source_name"].isin(target_source_names)].copy()
        known = set(metadata["source_name"])
        missing = [name for name in target_source_names if name not in known]
        if missing:
            metadata = pd.concat([metadata, default_target_metadata(missing)], ignore_index=True)

    st.session_state["target_signature"] = target_signature
    st.session_state["target_metadata"] = metadata.reset_index(drop=True)
    st.session_state["target_metadata_editor_version"] = (
        st.session_state.get("target_metadata_editor_version", 0) + 1
    )
    for key in PLOT_FILTER_KEYS:
        st.session_state.pop(key, None)
    return st.session_state["target_metadata"].copy()


def default_scatter_metadata(scatter_source_names: list[str]) -> pd.DataFrame:
    rows = []
    for source_name in scatter_source_names:
        label = Path(source_name).stem if source_name != "Pasted random scatter data" else "Pasted Scatter"
        rows.append({"source_name": source_name, "series_label": label or "Scatter"})
    return pd.DataFrame(rows)


def metadata_for_current_scatter(
    scatter_source_names: list[str],
    source_signature: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    scatter_signature = (tuple(scatter_source_names), source_signature)
    if st.session_state.get("scatter_signature") == scatter_signature:
        return st.session_state["scatter_metadata"].copy()

    existing = st.session_state.get("scatter_metadata")
    if existing is None or existing.empty:
        metadata = default_scatter_metadata(scatter_source_names)
    else:
        metadata = existing[existing["source_name"].isin(scatter_source_names)].copy()
        known = set(metadata["source_name"])
        missing = [name for name in scatter_source_names if name not in known]
        if missing:
            metadata = pd.concat([metadata, default_scatter_metadata(missing)], ignore_index=True)

    st.session_state["scatter_signature"] = scatter_signature
    st.session_state["scatter_metadata"] = metadata.reset_index(drop=True)
    st.session_state["scatter_metadata_editor_version"] = (
        st.session_state.get("scatter_metadata_editor_version", 0) + 1
    )
    return st.session_state["scatter_metadata"].copy()


def project_config_json(
    signal_map: dict[str, str],
    settings: ProcessingSettings,
    file_metadata: pd.DataFrame,
    units: dict[str, str] | None = None,
) -> str:
    payload = {
        "signal_map": signal_map,
        "settings": settings.__dict__,
        "file_metadata": file_metadata.to_dict(orient="records"),
        "units": units or {},
    }
    return json.dumps(payload, indent=2)


def processing_signature(
    file_signatures: dict[str, str],
    signal_map: dict[str, str],
    settings: ProcessingSettings,
    metadata: pd.DataFrame,
    units: dict[str, str],
) -> str:
    payload = {
        "files": file_signatures,
        "signals": signal_map,
        "settings": settings.__dict__,
        "labels": metadata.to_dict(orient="records"),
        "units": units,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def signal_selectbox(label: str, key: str, options: list[str]) -> str:
    if st.session_state.get(key, "") not in options:
        st.session_state[key] = ""
    return st.selectbox(
        label,
        options,
        key=key,
        format_func=lambda option: "Select signal..." if option == "" else option,
    )


def unit_from_sources(
    channel: str,
    sources: dict[str, bytes | str],
    signatures: dict[str, str],
) -> str:
    if not channel:
        return ""

    for display_name, source in sources.items():
        try:
            if isinstance(source, bytes):
                return cached_uploaded_channel_unit(
                    display_name,
                    signatures[display_name],
                    source,
                    channel,
                )
            return cached_local_channel_unit(source, signatures[display_name], channel)
        except Exception:
            continue
    return ""


LINE_STYLE_OPTIONS = ["solid", "dot", "dash", "dashdot", "longdash", "longdashdot"]
COLOR_COLUMN_HELP = "Examples: red, blue, green, orange, #D32F2F, rgb(255,0,0)"


def default_style_table(plot_data: pd.DataFrame) -> pd.DataFrame:
    if plot_data.empty:
        return pd.DataFrame(
            columns=["style_key", "legend_order", "curve", "color", "thickness", "line_style"]
        )

    dash_by_mode = mode_dash_map(plot_data)
    rows = []
    grouped = plot_data.groupby(["test_type", "group_label", "target_pedal"], sort=True)
    for (test_type, group_label, target_pedal), _curve in grouped:
        target = int(target_pedal)
        group_text = str(group_label)
        test_text = str(test_type)
        rows.append(
            {
                "style_key": curve_style_key(test_text, group_text, target),
                "legend_order": len(rows) + 1,
                "curve": f"{test_text} | {legend_label(group_text, target)}",
                "color": PEDAL_COLORS.get(target, "#616161"),
                "thickness": 2.25,
                "line_style": dash_by_mode.get(mode_key(group_text), "solid"),
            }
        )
    return pd.DataFrame(rows)


def default_scatter_style_table(scatter_data: pd.DataFrame) -> pd.DataFrame:
    if scatter_data.empty:
        return pd.DataFrame(
            columns=["style_key", "legend_order", "curve", "color", "thickness", "line_style"]
        )

    rows = []
    series_names = list(dict.fromkeys(scatter_data["series_name"].dropna().astype(str)))
    dash_by_source = scatter_dash_map(scatter_data)
    for index, series_name in enumerate(series_names):
        series_data = scatter_data[scatter_data["series_name"].astype(str) == series_name]
        rows.append(
            {
                "style_key": random_scatter_style_key(series_name),
                "legend_order": index + 1,
                "curve": series_name,
                "color": RANDOM_SCATTER_COLORS[index % len(RANDOM_SCATTER_COLORS)],
                "thickness": 2.25,
                "line_style": dash_by_source.get(scatter_source_key(series_data), "solid"),
            }
        )
    return pd.DataFrame(rows)


def style_overrides_from_table(style_table: pd.DataFrame) -> dict[str, dict[str, object]]:
    overrides: dict[str, dict[str, object]] = {}
    if style_table.empty:
        return overrides

    for _, row in style_table.iterrows():
        style_key = str(row.get("style_key") or "")
        if not style_key:
            continue
        overrides[style_key] = {
            "color": normalize_color_value(row.get("color"), "#616161"),
            "width": float(row.get("thickness") or 2.25),
            "dash": str(row.get("line_style") or "solid"),
        }
    return overrides


def sync_multiselect_options(key: str, options: list[object]) -> None:
    signature_key = f"{key}_options_signature"
    signature = tuple(str(option) for option in options)
    if st.session_state.get(signature_key) != signature:
        st.session_state[key] = options
        st.session_state[signature_key] = signature
        return

    selected = st.session_state.get(key)
    if selected is None:
        return
    valid_options = set(options)
    cleaned = [value for value in selected if value in valid_options]
    if cleaned != selected:
        st.session_state[key] = cleaned


def legend_order_from_table(style_table: pd.DataFrame) -> list[str]:
    if style_table.empty or "legend_order" not in style_table.columns:
        return []
    ordered = style_table.copy()
    ordered["legend_order"] = pd.to_numeric(ordered["legend_order"], errors="coerce")
    default_order = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
    ordered["legend_order"] = ordered["legend_order"].where(
        ordered["legend_order"].notna(),
        default_order,
    )
    return ordered.sort_values(["legend_order", "curve"])["style_key"].astype(str).tolist()


def average_band_trend_source(average_bands: pd.DataFrame) -> pd.DataFrame:
    if average_bands.empty:
        return pd.DataFrame()
    source = average_bands.copy()
    scenario_count = source["scenario"].dropna().nunique()
    if scenario_count > 1:
        source["trend_group"] = source.apply(
            lambda row: (
                f"{row['speed_band']}_"
                f"{short_group_label(str(row['scenario']).split('|', maxsplit=1)[-1])}"
            ),
            axis=1,
        )
    else:
        source["trend_group"] = source["speed_band"].astype(str)
    return source


def selected_metric_trend_source(metrics: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    if metrics.empty or metric_column not in metrics.columns:
        return pd.DataFrame()
    source = metrics[["scenario", "target_pedal", metric_column]].copy()
    source = source.dropna(subset=[metric_column])
    if source.empty:
        return source
    source["trend_group"] = source["scenario"].map(
        lambda scenario: short_group_label(str(scenario).split("|", maxsplit=1)[-1])
    )
    source = source.rename(columns={metric_column: "metric_value"})
    return source


def tag_metric_trendlines(trendlines: pd.DataFrame, metric_graph: str) -> pd.DataFrame:
    if trendlines.empty:
        return trendlines
    tagged = trendlines.copy()
    tagged["metric_graph"] = metric_graph
    tagged["trendline_name"] = (
        tagged["metric_graph"].astype(str) + " | " + tagged["trendline_name"].astype(str)
    )
    return tagged


def render_help_section() -> None:
    with st.expander("Help / Documentation", expanded=False):
        st.markdown(
            """
### What this tool does

This tool reads ETAS INCA measurement files, extracts usable launch and
deceleration data, averages repeated runs, and plots **Speed vs Acceleration**
curves. It can also overlay imported target curves from Excel, CSV, text, or
data pasted from Excel. The optional random scatter section can plot any
Excel-style X/Y data when you need a quick comparison that is not speed vs
acceleration.

### Basic workflow

1. Load vehicle measurement files using **Read from local folder** for large
   MDF/MF4/DAT files, or **Upload in browser** for smaller files.
2. Or skip measurement files and import target data only.
3. Select the four required MDF signals when measurement files are loaded:
   Speed, Acceleration, Brake, and Accelerator Pedal.
4. Confirm the detected MDF units and choose graph/export units.
5. Import or paste target Speed vs Acceleration data if reference curves are
   needed.
6. Set **Series Labels** for target files, such as Normal, Eco, or Sport.
7. Label measurement files as Launch, Deceleration, or Ignore.
8. Apply mode and regen labels, then click **Generate plot** for measured data.
9. Use plot controls to choose test types, including Target, mode/regen
   combinations, pedal targets, and optional curve styling.
10. Use **Metrics (optional)** to calculate peak acceleration, speed-band
    average acceleration, road-load speed, and slope between two speed points.
11. Export the graph PNG, Excel workbook, metrics workbook, audit CSV, or
    project JSON.

### Terminology

- **Launch**: Acceleration maneuver at a selected accelerator pedal
  percentage.
- **0% creep launch**: Low-speed creep behavior. The tool starts this test
  when brake drops below the configured creep brake level and ends when brake
  is reapplied.
- **Deceleration**: Tip-out maneuver where the driver releases to a target
  pedal percentage after reaching the test speed.
- **Mode label**: Vehicle mode such as Normal, Sport, or Eco.
- **Regen / extra label**: Extra deceleration label such as Regen Level 3 or
  Max Regen.
- **Target pedal**: Pedal percentage used to identify and group a curve.
- **Pedal tolerance**: Allowed band around the target pedal. For example,
  10% with a 2% tolerance accepts 8% to 12%.
- **Brake applied threshold**: Brake values above this threshold are treated
  as brake applied.
- **Minimum usable points**: Minimum number of valid samples required before
  the tool plots a curve.
- **Speed interval**: Speed-axis spacing used to align and average curves.

### Importing target data

Target data can come from `.xlsx`, `.xlsm`, `.xls`, `.csv`, or `.txt` files,
or it can be pasted directly from Excel.

Supported layouts:

- Column A: Speed values for the x-axis.
- Columns B, C, D, etc.: Acceleration values for pedal levels.
- Or paired columns such as x1/y1, x2/y2, x3/y3.
- Column headers should identify pedal levels, for example `5%`, `10%`, or
  `20% pedal`.

If the data starts somewhere else on the sheet, the tool drops empty rows and
columns and tries to find the table. The default target layout is **One speed
column, multiple acceleration columns**. Use **Auto detect** or **Paired
speed/acceleration columns** when needed.

The tool tries to detect units from headers such as `Speed (KPH)` or
`Acceleration (g)`. If units are not detected, select the target input units
manually. Target curves are converted into the same graph/export units chosen
for the measured MDF data.

### Random scatter plots

Use **Random Scatter plots (optional)** for arbitrary Excel-style plots. This
section supports files or pasted data with either one X column and multiple Y
columns, or paired X/Y columns. It is independent of the main Speed vs
Acceleration plot, so it can be used without MDF or target data. All series
from the first imported source default to solid lines, the second source
defaults to dotted lines, and later sources use the next dash styles.

The random scatter section can also create derived series. Use average series
to average selected curves over their shared X range, or offset series to shift
a selected curve in X or Y.

### Trendlines

Random scatter and metric graphs can show trendlines. Choose Linear or a 2nd
through 5th order polynomial, optionally show equations, create a trendline-only
graph, and export the fitted trendline data to Excel. Trendlines can use either
smooth generated X-values or the exact existing X-values from the source curve.

### Metrics

Metrics are calculated from the curves currently selected in **Plot Controls**.
Speed-band averages use the current graph speed unit. Road load is the speed
where acceleration crosses zero; if the curve does not cross zero, the tool
shows `N/A`. Slope uses linear interpolation at the two speed points entered by
the user, and shows `N/A` if either speed point is outside a curve. Average
acceleration speed bands are plotted together and also available as separate
band plots. If enabled, road load can also be estimated from a linear trendline
extension when the measured curve does not cross acceleration = 0.

### Interpreting results

Measured curves and target curves can be shown on the same graph. Curves with
the same pedal percentage use the same color. Mode comparisons use line style,
for example Normal is solid, Sport is dotted, and Eco is dash-dot. Imported
target curves use a long-dash reference style.

Hovering over a curve shows only the nearest point coordinates, like Excel.
Use **Curve Style and Legend Order (optional)** to adjust individual curve
color, thickness, line style, and legend order. Color accepts common names
like `red`, `blue`, and `green`, or codes like `#D32F2F`.

For any questions, please contact Shubham Ketkale.
"""
        )


render_hero()
render_help_section()


with st.sidebar:
    sidebar_brand()
    st.header("1. Load files")
    load_method = st.radio(
        "File loading method",
        ["Read from local folder", "Upload in browser"],
        help="Reading from a local folder is recommended for MDF files larger than 200 MB.",
        key=resettable_key("load_method"),
    )
    uploaded_files = []
    folder_text = ""
    include_subfolders = False
    if load_method == "Read from local folder":
        folder_text = st.text_input(
            "Folder containing measurement files",
            placeholder=r"C:\Users\YourName\Desktop\Vehicle_Data",
            key=resettable_key("measurement_folder"),
        )
        include_subfolders = st.checkbox(
            "Include subfolders",
            value=False,
            key=resettable_key("include_subfolders"),
        )
        st.caption("Recommended for large files. Data stays on this laptop and is read directly from disk.")
    else:
        try:
            uploaded_files = st.file_uploader(
                "Select .mf4, .dat, or .mdf files",
                type=["mf4", "dat", "mdf"],
                accept_multiple_files=True,
                key=resettable_key("measurement_files"),
                max_upload_size=1024,
            )
        except MemoryError:
            st.error(
                "The browser upload ran out of memory. Use `Read from local folder` "
                "for large measurement files."
            )
            st.stop()
        st.caption("Uploads up to 1 GB are enabled, but local-folder loading uses much less memory.")
    st.button(
        "Clear all and start over",
        width="stretch",
        on_click=clear_all,
    )

    st.header("2. Processing settings (optional)")
    pedal_tolerance = st.number_input("Pedal tolerance (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    brake_threshold = st.number_input("Brake applied threshold", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    speed_bin = st.number_input(
        "Speed interval for plot / Excel export (graph unit)",
        min_value=0.05,
        max_value=20.0,
        value=1.0,
        step=0.05,
        help=(
            "The default interval is 1.0. Choose 0.5, 0.1, or 0.05 for finer plot "
            "and Excel data resolution."
        ),
    )
    min_points = st.number_input("Minimum usable points per file/target", min_value=1, max_value=10000, value=10, step=1)

    st.subheader("0% creep launch detection")
    creep_release_percent = st.number_input(
        "Creep start brake level (% of maximum)",
        min_value=1.0,
        max_value=99.0,
        value=33.0,
        step=1.0,
    )

    st.subheader("Deceleration detection")
    use_speed_gate = st.checkbox("Require decel start near target speed", value=True)
    decel_start_speed = st.number_input("Decel start speed target (KPH reference)", min_value=0.0, max_value=300.0, value=150.0, step=5.0)
    decel_start_speed_tolerance = st.number_input("Decel start speed tolerance (KPH reference)", min_value=0.0, max_value=100.0, value=10.0, step=5.0)

    raster_text = st.text_input(
        "Optional MDF resample step in seconds",
        value="",
        help="Leave blank to use the file's native time rasters. Example: 0.01 for 10 ms.",
    )

settings = ProcessingSettings(
    pedal_tolerance=float(pedal_tolerance),
    brake_threshold=float(brake_threshold),
    creep_brake_release_fraction=float(creep_release_percent) / 100.0,
    decel_start_speed=float(decel_start_speed),
    decel_start_speed_tolerance=float(decel_start_speed_tolerance),
    speed_bin=float(speed_bin),
    min_points=int(min_points),
    use_decel_speed_gate=bool(use_speed_gate),
)

raster_step_s = None
if raster_text.strip():
    try:
        raster_step_s = float(raster_text)
        if raster_step_s <= 0:
            st.sidebar.error("Resample step must be greater than 0.")
            raster_step_s = None
    except ValueError:
        st.sidebar.error("Resample step must be a number, for example 0.01.")

measurement_sources: dict[str, bytes | str] = {}
file_signatures: dict[str, str] = {}
if load_method == "Read from local folder":
    if not folder_text.strip():
        st.info("Enter an INCA measurement folder, or import target data below for target-only plotting.")
    else:
        measurement_folder = Path(folder_text.strip().strip('"'))
        if not measurement_folder.is_dir():
            st.warning("The specified measurement folder was not found. You can still import target data below.")
        else:
            iterator = measurement_folder.rglob("*") if include_subfolders else measurement_folder.glob("*")
            local_files = sorted(
                [
                    path for path in iterator
                    if path.is_file() and path.suffix.casefold() in {".mf4", ".mdf", ".dat"}
                ],
                key=lambda path: str(path).casefold(),
            )
            if not local_files:
                st.info("No `.mf4`, `.mdf`, or `.dat` files were found. Target-only plotting is still available.")
            for path in local_files:
                display_name = str(path.relative_to(measurement_folder))
                file_stat = path.stat()
                measurement_sources[display_name] = str(path)
                file_signatures[display_name] = f"{file_stat.st_size}:{file_stat.st_mtime_ns}"
else:
    if not uploaded_files:
        st.info("Select INCA measurement files, or import target data below for target-only plotting.")
    else:
        seen_file_names: dict[str, int] = {}
        try:
            for uploaded_file in uploaded_files:
                original_name = uploaded_file.name
                seen_file_names[original_name] = seen_file_names.get(original_name, 0) + 1
                if seen_file_names[original_name] == 1:
                    display_name = original_name
                else:
                    path = Path(original_name)
                    display_name = f"{path.stem} ({seen_file_names[original_name]}){path.suffix}"
                file_bytes = uploaded_file.getvalue()
                measurement_sources[display_name] = file_bytes
                file_signatures[display_name] = content_hash(file_bytes)
        except MemoryError:
            st.error("Files are too large to keep in browser memory. Switch to `Read from local folder`.")
            st.stop()

file_names = list(measurement_sources.keys())
has_measurements = bool(measurement_sources)
metadata = metadata_for_current_uploads(file_names, tuple(file_signatures.items())) if has_measurements else pd.DataFrame()

_mode_short = "Local folder" if load_method == "Read from local folder" else "Browser upload"
_files_badge = ("Loaded", "ok") if has_measurements else ("Awaiting files", "idle")
_raster_hint = "Native rasters" if raster_step_s is None else f"{raster_step_s:g} s step"
render_status_strip(
    [
        _stat_card("Loading mode", _mode_short, hint="Source of measurement data"),
        _stat_card(
            "Measurement files",
            str(len(file_names)),
            hint="Ready for signal mapping" if has_measurements else "Local folder or browser upload",
            badge=_files_badge[0],
            badge_kind=_files_badge[1],
        ),
        _stat_card("Resample step", _raster_hint, hint="Optional MDF time raster"),
        _stat_card(
            "Session",
            "Active" if has_measurements else "Target-only ready",
            hint="Import targets/scatter without MDF data",
            badge="Live" if has_measurements else "Idle",
            badge_kind="ok" if has_measurements else "idle",
        ),
    ]
)

speed_signal = ""
accel_signal = ""
brake_signal = ""
pedal_signal = ""
detected_speed_unit = ""
detected_accel_unit = ""
speed_input_unit = "KPH"
acceleration_input_unit = "m/s^2"

section_header(
    "Units & Conversion",
    "Confirm detected units and choose the graph/export units",
    icon="ruler",
)
if has_measurements:
    section_header(
        "Signal Selection",
        "Map the four required MDF channels",
        icon="signal",
    )
    channel_errors: list[str] = []
    all_channels: set[str] = set()

    with st.spinner("Reading channel lists from measurement files..."):
        for file_name, source in measurement_sources.items():
            try:
                if isinstance(source, bytes):
                    all_channels.update(
                        cached_discover_uploaded_channels(file_name, file_signatures[file_name], source)
                    )
                else:
                    all_channels.update(cached_discover_local_channels(source, file_signatures[file_name]))
            except MdfDependencyError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                channel_errors.append(f"{file_name}: {exc}")

    if channel_errors:
        st.warning("Some files could not be inspected:\n\n" + "\n".join(f"- {err}" for err in channel_errors))

    channel_options = sorted(all_channels, key=str.casefold)
    if not channel_options:
        st.warning("No MDF channel names were found. You can still plot imported target data.")
        signal_options = [""]
    else:
        signal_options = [""] + channel_options

    signal_cols = st.columns(4)
    with signal_cols[0]:
        speed_signal = signal_selectbox("Speed signal", "speed_signal", signal_options)
    with signal_cols[1]:
        accel_signal = signal_selectbox("Acceleration signal", "accel_signal", signal_options)
    with signal_cols[2]:
        brake_signal = signal_selectbox("Brake signal", "brake_signal", signal_options)
    with signal_cols[3]:
        pedal_signal = signal_selectbox("Accelerator pedal signal", "pedal_signal", signal_options)

    detected_speed_unit = unit_from_sources(speed_signal, measurement_sources, file_signatures)
    detected_accel_unit = unit_from_sources(accel_signal, measurement_sources, file_signatures)
    normalized_speed_unit = normalize_speed_unit(detected_speed_unit)
    normalized_accel_unit = normalize_acceleration_unit(detected_accel_unit)
    st.caption(
        "Detected from selected signals: "
        f"Speed = `{detected_speed_unit or 'not available'}`, "
        f"Acceleration = `{detected_accel_unit or 'not available'}`. "
        "Confirm the input unit if the MDF unit is missing or not recognized."
    )

    unit_cols = st.columns(4)
    with unit_cols[0]:
        speed_input_unit = st.selectbox(
            "Speed input unit",
            SPEED_UNITS,
            index=SPEED_UNITS.index(normalized_speed_unit) if normalized_speed_unit in SPEED_UNITS else 0,
            key=f"speed_input_{speed_signal}_{detected_speed_unit}",
        )
    with unit_cols[1]:
        speed_output_unit = st.selectbox(
            "Speed graph/export unit",
            ["KPH", "MPH"],
            index=0,
            key=f"speed_output_{speed_signal}",
        )
    with unit_cols[2]:
        acceleration_input_unit = st.selectbox(
            "Acceleration input unit",
            ACCELERATION_UNITS,
            index=ACCELERATION_UNITS.index(normalized_accel_unit) if normalized_accel_unit in ACCELERATION_UNITS else 0,
            key=f"accel_input_{accel_signal}_{detected_accel_unit}",
        )
    with unit_cols[3]:
        acceleration_output_unit = st.selectbox(
            "Acceleration graph/export unit",
            ACCELERATION_UNITS,
            index=0,
            key=f"accel_output_{accel_signal}",
        )
else:
    st.caption("No INCA/MDF files loaded. Select output units for target-only plotting.")
    unit_cols = st.columns(2)
    with unit_cols[0]:
        speed_output_unit = st.selectbox(
            "Speed graph/export unit",
            ["KPH", "MPH"],
            index=0,
            key=resettable_key("target_only_speed_output"),
        )
    with unit_cols[1]:
        acceleration_output_unit = st.selectbox(
            "Acceleration graph/export unit",
            ACCELERATION_UNITS,
            index=0,
            key=resettable_key("target_only_acceleration_output"),
        )
    speed_input_unit = speed_output_unit
    acceleration_input_unit = acceleration_output_unit

signal_map = {
    "speed": speed_signal,
    "acceleration": accel_signal,
    "brake": brake_signal,
    "pedal": pedal_signal,
}
selected_channels = tuple(channel for channel in dict.fromkeys(signal_map.values()) if channel)
display_units = {
    "detected_speed": detected_speed_unit,
    "detected_acceleration": detected_accel_unit,
    "speed_input": speed_input_unit,
    "speed_output": speed_output_unit,
    "acceleration_input": acceleration_input_unit,
    "acceleration_output": acceleration_output_unit,
}
settings = replace(
    settings,
    decel_start_speed=convert_speed_value(
        float(decel_start_speed),
        "KPH",
        speed_output_unit,
    ),
    decel_start_speed_tolerance=convert_speed_value(
        float(decel_start_speed_tolerance),
        "KPH",
        speed_output_unit,
    ),
)

target_curves = pd.DataFrame()
target_parse_errors: list[str] = []

section_header(
    "Target Data",
    "Overlay reference Speed vs Acceleration curves (optional)",
    icon="target",
)
with st.expander("Import or paste target Speed vs Acceleration data", expanded=False):
    st.write(
        "Use this when target/reference curves are available in Excel, CSV, "
        "text files, or copied directly from Excel."
    )
    target_layout = st.selectbox(
        "Target data layout",
        TARGET_LAYOUTS,
        index=1,
        help=(
            "The default is one Speed column with multiple Acceleration columns. "
            "Choose paired layout when the sheet is x1/y1, x2/y2, x3/y3, etc."
        ),
        key=resettable_key("target_data_layout"),
    )
    target_files = st.file_uploader(
        "Target files (.xlsx, .xlsm, .xls, .csv, .txt)",
        type=["xlsx", "xlsm", "xls", "csv", "txt"],
        accept_multiple_files=True,
        key=resettable_key("target_data_files"),
    )
    pasted_target_text = st.text_area(
        "Paste target data from Excel",
        height=140,
        placeholder=(
            "Speed (KPH)\t5%\t10%\t15%\n"
            "0\t0.00\t0.00\t0.00\n"
            "10\t0.12\t0.20\t0.28"
        ),
        key=resettable_key("target_data_paste"),
    )

    target_source_payloads: dict[str, bytes | str] = {}
    target_source_signatures: dict[str, str] = {}
    seen_target_names: dict[str, int] = {}
    for target_file in target_files or []:
        original_name = target_file.name
        seen_target_names[original_name] = seen_target_names.get(original_name, 0) + 1
        if seen_target_names[original_name] == 1:
            display_name = original_name
        else:
            path = Path(original_name)
            display_name = f"{path.stem} ({seen_target_names[original_name]}){path.suffix}"
        file_bytes = target_file.getvalue()
        target_source_payloads[display_name] = file_bytes
        target_source_signatures[display_name] = content_hash(file_bytes)

    if pasted_target_text.strip():
        target_source_payloads["Pasted target data"] = pasted_target_text
        target_source_signatures["Pasted target data"] = content_hash(
            pasted_target_text.encode("utf-8")
        )

    parsed_targets: list[ParsedTargetData] = []
    target_sources_present = bool(target_source_payloads)
    if target_sources_present:
        target_metadata = metadata_for_current_targets(
            list(target_source_payloads.keys()),
            tuple(target_source_signatures.items()),
        )
        st.write("Series Labels")
        edited_target_metadata = st.data_editor(
            target_metadata,
            key=f"target_metadata_editor_{st.session_state.get('target_metadata_editor_version', 0)}",
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "source_name": st.column_config.TextColumn("Source", disabled=True),
                "series_label": st.column_config.TextColumn("Series label"),
            },
        )
        if not edited_target_metadata.equals(target_metadata):
            for key in PLOT_FILTER_KEYS:
                st.session_state.pop(key, None)
        st.session_state["target_metadata"] = edited_target_metadata.copy()
        target_labels_by_source = {
            str(row["source_name"]): str(row.get("series_label") or "Target")
            for _, row in edited_target_metadata.iterrows()
        }

        for source_name, payload in target_source_payloads.items():
            series_label = target_labels_by_source.get(source_name, "Target")
            try:
                if isinstance(payload, bytes):
                    parsed_targets.append(
                        parse_target_file(source_name, payload, series_label, target_layout)
                    )
                else:
                    parsed_targets.append(
                        parse_pasted_target_text(payload, series_label, target_layout)
                    )
            except Exception as exc:
                target_parse_errors.append(f"{source_name}: {exc}")

        detected_target_speed_unit = next(
            (target.detected_speed_unit for target in parsed_targets if target.detected_speed_unit),
            None,
        )
        detected_target_accel_unit = next(
            (
                target.detected_acceleration_unit
                for target in parsed_targets
                if target.detected_acceleration_unit
            ),
            None,
        )

        st.caption(
            "Detected target units: "
            f"Speed = `{detected_target_speed_unit or 'not available'}`, "
            f"Acceleration = `{detected_target_accel_unit or 'not available'}`."
        )
        target_unit_cols = st.columns(2)
        with target_unit_cols[0]:
            target_speed_input_unit = st.selectbox(
                "Target speed input unit",
                SPEED_UNITS,
                index=SPEED_UNITS.index(detected_target_speed_unit or speed_output_unit),
                key=resettable_key("target_speed_input_unit"),
            )
        with target_unit_cols[1]:
            target_acceleration_input_unit = st.selectbox(
                "Target acceleration input unit",
                ACCELERATION_UNITS,
                index=ACCELERATION_UNITS.index(
                    detected_target_accel_unit or acceleration_output_unit
                ),
                key=resettable_key("target_acceleration_input_unit"),
            )

        if parsed_targets:
            raw_target_curves = pd.concat(
                [target.curves for target in parsed_targets],
                ignore_index=True,
            )
            target_curves = convert_target_curves(
                raw_target_curves,
                target_speed_input_unit,
                speed_output_unit,
                target_acceleration_input_unit,
                acceleration_output_unit,
            )
            target_summary = (
                target_curves.groupby(["group_label", "target_pedal"], as_index=False)
                .agg(
                    points=("speed", "count"),
                    speed_min=("speed", "min"),
                    speed_max=("speed", "max"),
                )
                .sort_values(["group_label", "target_pedal"])
            )
            st.success(f"Loaded {len(target_summary)} target curve(s).")
            st.dataframe(target_summary, width="stretch", hide_index=True)

        if target_parse_errors:
            st.warning(
                "Some target data could not be loaded:\n\n"
                + "\n".join(f"- {err}" for err in target_parse_errors)
            )
    else:
        st.caption(
            "Expected layouts: one Speed column with multiple Acceleration columns, "
            "or paired columns like x1/y1, x2/y2, x3/y3."
        )

scatter_curves = pd.DataFrame()
scatter_parse_errors: list[str] = []

section_header(
    "Random Scatter Plots",
    "Plot arbitrary Excel-style X/Y data, independent of MDF (optional)",
    icon="scatter",
)
with st.expander("Import or paste arbitrary X/Y scatter data", expanded=False):
    st.write(
        "Use this for Excel-style scatter plots that are not necessarily Speed vs Acceleration."
    )
    scatter_layout = st.selectbox(
        "Random scatter data layout",
        SCATTER_LAYOUTS,
        index=1,
        help=(
            "The default is one X column with multiple Y columns. "
            "Choose paired layout when the sheet is x1/y1, x2/y2, x3/y3, etc."
        ),
        key=resettable_key("random_scatter_layout"),
    )
    axis_cols = st.columns(2)
    with axis_cols[0]:
        scatter_x_axis_label = st.text_input(
            "Random scatter X-axis label",
            value="X",
            key=resettable_key("random_scatter_x_axis_label"),
        )
    with axis_cols[1]:
        scatter_y_axis_label = st.text_input(
            "Random scatter Y-axis label",
            value="Y",
            key=resettable_key("random_scatter_y_axis_label"),
        )

    scatter_files = st.file_uploader(
        "Random scatter files (.xlsx, .xlsm, .xls, .csv, .txt)",
        type=["xlsx", "xlsm", "xls", "csv", "txt"],
        accept_multiple_files=True,
        key=resettable_key("random_scatter_files"),
    )
    pasted_scatter_text = st.text_area(
        "Paste random scatter data from Excel",
        height=140,
        placeholder=(
            "X\tSeries 1\tSeries 2\n"
            "0\t1.2\t2.4\n"
            "10\t1.8\t2.9"
        ),
        key=resettable_key("random_scatter_paste"),
    )

    scatter_source_payloads: dict[str, bytes | str] = {}
    scatter_source_signatures: dict[str, str] = {}
    seen_scatter_names: dict[str, int] = {}
    for scatter_file in scatter_files or []:
        original_name = scatter_file.name
        seen_scatter_names[original_name] = seen_scatter_names.get(original_name, 0) + 1
        if seen_scatter_names[original_name] == 1:
            display_name = original_name
        else:
            path = Path(original_name)
            display_name = f"{path.stem} ({seen_scatter_names[original_name]}){path.suffix}"
        file_bytes = scatter_file.getvalue()
        scatter_source_payloads[display_name] = file_bytes
        scatter_source_signatures[display_name] = content_hash(file_bytes)

    if pasted_scatter_text.strip():
        scatter_source_payloads["Pasted random scatter data"] = pasted_scatter_text
        scatter_source_signatures["Pasted random scatter data"] = content_hash(
            pasted_scatter_text.encode("utf-8")
        )

    parsed_scatters: list[ParsedRandomScatterData] = []
    if scatter_source_payloads:
        scatter_metadata = metadata_for_current_scatter(
            list(scatter_source_payloads.keys()),
            tuple(scatter_source_signatures.items()),
        )
        st.write("Series Labels")
        edited_scatter_metadata = st.data_editor(
            scatter_metadata,
            key=f"scatter_metadata_editor_{st.session_state.get('scatter_metadata_editor_version', 0)}",
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "source_name": st.column_config.TextColumn("Source", disabled=True),
                "series_label": st.column_config.TextColumn("Series label"),
            },
        )
        st.session_state["scatter_metadata"] = edited_scatter_metadata.copy()
        scatter_labels_by_source = {
            str(row["source_name"]): str(row.get("series_label") or "Scatter")
            for _, row in edited_scatter_metadata.iterrows()
        }

        for source_order, (source_name, payload) in enumerate(scatter_source_payloads.items()):
            series_label = scatter_labels_by_source.get(source_name, "Scatter")
            try:
                if isinstance(payload, bytes):
                    parsed_scatter = parse_scatter_file(source_name, payload, series_label, scatter_layout)
                else:
                    parsed_scatter = parse_pasted_scatter_text(payload, series_label, scatter_layout)
                parsed_curves = parsed_scatter.curves.copy()
                parsed_curves["source_name"] = source_name
                parsed_curves["source_order"] = source_order
                parsed_scatters.append(ParsedRandomScatterData(parsed_curves))
            except Exception as exc:
                scatter_parse_errors.append(f"{source_name}: {exc}")

        if parsed_scatters:
            scatter_curves = pd.concat(
                [scatter.curves for scatter in parsed_scatters],
                ignore_index=True,
            )
            scatter_summary = (
                scatter_curves.groupby("series_name", as_index=False)
                .agg(points=("x", "count"), x_min=("x", "min"), x_max=("x", "max"))
                .sort_values("series_name")
            )
            st.success(f"Loaded {len(scatter_summary)} random scatter series.")
            st.dataframe(scatter_summary, width="stretch", hide_index=True)

            scatter_series_options = list(dict.fromkeys(scatter_curves["series_name"].dropna().astype(str)))
            sync_multiselect_options("random_scatter_series", scatter_series_options)
            selected_scatter_series = st.multiselect(
                "Random scatter series to show",
                scatter_series_options,
                default=scatter_series_options,
                key="random_scatter_series",
            )
            scatter_plot_data = scatter_curves[
                scatter_curves["series_name"].isin(selected_scatter_series)
            ].copy()

            if scatter_plot_data.empty:
                st.info("No random scatter series match the selected filters.")
            else:
                derived_scatter_data = pd.DataFrame()
                with st.expander("Average / Offset Random Scatter Series (optional)", expanded=False):
                    derived_rows = []
                    st.write("Create extra curves from the imported random scatter series.")
                    avg_cols = st.columns([2, 1])
                    with avg_cols[0]:
                        average_series = st.multiselect(
                            "Series to average",
                            scatter_series_options,
                            key=resettable_key("random_scatter_average_series"),
                        )
                    with avg_cols[1]:
                        average_name = st.text_input(
                            "Average series name",
                            value="Average_Selected",
                            key=resettable_key("random_scatter_average_name"),
                        )
                    create_average_series = st.checkbox(
                        "Create average series",
                        value=False,
                        key=resettable_key("random_scatter_create_average"),
                    )
                    if create_average_series:
                        if len(average_series) < 2:
                            st.warning("Select at least two series to create an average.")
                        else:
                            average_curve = average_selected_series(
                                scatter_curves,
                                average_series,
                                average_name,
                            )
                            if average_curve.empty:
                                st.warning(
                                    "Average could not be created because the selected series "
                                    "do not share an overlapping X range."
                                )
                            else:
                                derived_rows.append(average_curve)

                    offset_cols = st.columns([2, 1, 1])
                    with offset_cols[0]:
                        offset_source = st.selectbox(
                            "Series to offset",
                            scatter_series_options,
                            key=resettable_key("random_scatter_offset_source"),
                        )
                    with offset_cols[1]:
                        offset_axis = st.selectbox(
                            "Offset axis",
                            ["Y", "X"],
                            key=resettable_key("random_scatter_offset_axis"),
                        )
                    with offset_cols[2]:
                        offset_amount = st.number_input(
                            "Offset amount",
                            value=0.0,
                            step=0.1,
                            key=resettable_key("random_scatter_offset_amount"),
                        )
                    offset_name = st.text_input(
                        "Offset series name",
                        value=f"{offset_source}_offset" if offset_source else "Offset_Selected",
                        key=resettable_key("random_scatter_offset_name"),
                    )
                    create_offset_series = st.checkbox(
                        "Create offset series",
                        value=False,
                        key=resettable_key("random_scatter_create_offset"),
                    )
                    if create_offset_series:
                        offset_curve = offset_selected_series(
                            scatter_curves,
                            offset_source,
                            offset_name,
                            float(offset_amount),
                            offset_axis,
                        )
                        if offset_curve.empty:
                            st.warning("Offset curve could not be created from the selected series.")
                        else:
                            derived_rows.append(offset_curve)

                    if derived_rows:
                        derived_scatter_data = pd.concat(derived_rows, ignore_index=True)
                        st.success(f"Created {derived_scatter_data['series_name'].nunique()} derived series.")
                        st.dataframe(
                            derived_scatter_data[
                                ["series_name", "derived_type", "derived_from", "x", "y"]
                            ],
                            width="stretch",
                            hide_index=True,
                        )
                        st.download_button(
                            "Download derived random scatter Excel",
                            data=export_random_scatter_workbook(derived_scatter_data),
                            file_name="random_scatter_derived_series.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                if not derived_scatter_data.empty:
                    scatter_plot_data = pd.concat(
                        [scatter_plot_data, derived_scatter_data],
                        ignore_index=True,
                    )

                scatter_style_defaults = default_scatter_style_table(scatter_plot_data)
                scatter_style_overrides = {}
                scatter_trace_order: list[str] = []
                scatter_show_markers = False
                with st.expander("Random Scatter Curve Style and Legend Order (optional)", expanded=False):
                    scatter_show_markers = st.checkbox(
                        "Smooth lines with markers",
                        value=False,
                        key=resettable_key("random_scatter_smooth_markers"),
                    )
                    st.caption(f"Color accepts a name or code. {COLOR_COLUMN_HELP}")
                    edited_scatter_style_table = st.data_editor(
                        scatter_style_defaults,
                        key=(
                            "random_scatter_style_editor_"
                            + hashlib.sha1(
                                "|".join(scatter_style_defaults["style_key"].astype(str)).encode("utf-8")
                            ).hexdigest()
                        ),
                        width="stretch",
                        hide_index=True,
                        num_rows="fixed",
                        column_config={
                            "style_key": st.column_config.TextColumn("Style key", disabled=True),
                            "legend_order": st.column_config.NumberColumn("Legend order", min_value=1, step=1),
                            "curve": st.column_config.TextColumn("Curve", disabled=True),
                            "color": st.column_config.TextColumn("Color name/code", help=COLOR_COLUMN_HELP),
                            "thickness": st.column_config.NumberColumn(
                                "Thickness",
                                min_value=0.5,
                                max_value=10.0,
                                step=0.25,
                            ),
                            "line_style": st.column_config.SelectboxColumn(
                                "Line style",
                                options=LINE_STYLE_OPTIONS,
                            ),
                        },
                    )
                    scatter_style_overrides = style_overrides_from_table(edited_scatter_style_table)
                    scatter_trace_order = legend_order_from_table(edited_scatter_style_table)

                scatter_trendline_data = pd.DataFrame()
                scatter_show_trendline_only = False
                scatter_show_equations = False
                with st.expander("Random Scatter Trendlines (optional)", expanded=False):
                    show_scatter_trendlines = st.checkbox(
                        "Show trendlines on random scatter plot",
                        value=False,
                        key=resettable_key("random_scatter_show_trendlines"),
                    )
                    if show_scatter_trendlines:
                        trendline_series_options = list(
                            dict.fromkeys(scatter_plot_data["series_name"].dropna().astype(str))
                        )
                        selected_trendline_series = st.multiselect(
                            "Series for trendlines",
                            trendline_series_options,
                            default=trendline_series_options,
                            key=resettable_key("random_scatter_trendline_series"),
                        )
                        trend_cols = st.columns(3)
                        with trend_cols[0]:
                            scatter_trendline_choice = st.selectbox(
                                "Trendline type",
                                list(TRENDLINE_DEGREES.keys()),
                                key=resettable_key("random_scatter_trendline_degree"),
                            )
                        with trend_cols[1]:
                            scatter_trendline_x_mode = st.selectbox(
                                "Trendline X-values",
                                ["Smooth generated X-values", "Trendlines using existing X-values"],
                                key=resettable_key("random_scatter_trendline_x_mode"),
                            )
                        with trend_cols[2]:
                            scatter_show_equations = st.checkbox(
                                "Show equations",
                                value=True,
                                key=resettable_key("random_scatter_show_equations"),
                            )
                        scatter_trendline_points = 200
                        if scatter_trendline_x_mode == "Smooth generated X-values":
                            scatter_trendline_points = st.number_input(
                                "Trendline points",
                                min_value=10,
                                max_value=5000,
                                value=200,
                                step=10,
                                key=resettable_key("random_scatter_trendline_points"),
                            )
                        scatter_show_trendline_only = st.checkbox(
                            "Show trendline-only graph",
                            value=False,
                            key=resettable_key("random_scatter_trendline_only"),
                        )
                        scatter_trendline_data = build_trendline_data(
                            scatter_plot_data,
                            x_col="x",
                            y_col="y",
                            group_col="series_name",
                            selected_groups=selected_trendline_series,
                            degree=TRENDLINE_DEGREES[scatter_trendline_choice],
                            points=int(scatter_trendline_points),
                            use_existing_x=(
                                scatter_trendline_x_mode == "Trendlines using existing X-values"
                            ),
                        )
                        if scatter_trendline_data.empty:
                            st.warning(
                                "No trendlines were created. Each selected series needs at least "
                                "degree + 1 unique X values."
                            )
                        elif scatter_show_equations:
                            st.dataframe(
                                trendline_equations(scatter_trendline_data),
                                width="stretch",
                                hide_index=True,
                            )

                scatter_fig = make_random_scatter_figure(
                    scatter_plot_data,
                    title="Random Scatter Plot",
                    x_axis_title=scatter_x_axis_label,
                    y_axis_title=scatter_y_axis_label,
                    style_overrides=scatter_style_overrides,
                    trace_order=scatter_trace_order,
                    show_markers=scatter_show_markers,
                    smooth_lines=True,
                )
                add_trendline_traces(
                    scatter_fig,
                    scatter_trendline_data,
                    show_equations=scatter_show_equations,
                )
                st.plotly_chart(scatter_fig, width="stretch")

                if scatter_show_trendline_only and not scatter_trendline_data.empty:
                    scatter_trendline_fig = make_trendline_figure(
                        scatter_trendline_data,
                        title="Random Scatter Trendline Graph",
                        x_axis_title=scatter_x_axis_label,
                        y_axis_title=scatter_y_axis_label,
                        show_equations=scatter_show_equations,
                    )
                    st.plotly_chart(scatter_trendline_fig, width="stretch")

                scatter_export_cols = st.columns(3)
                with scatter_export_cols[0]:
                    try:
                        st.download_button(
                            "Download random scatter Excel",
                            data=export_random_scatter_workbook(scatter_plot_data),
                            file_name="random_scatter_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    except ImportError:
                        st.warning("Install the updated requirements to enable Excel export.")
                with scatter_export_cols[1]:
                    if not scatter_trendline_data.empty:
                        st.download_button(
                            "Download trendline Excel",
                            data=export_trendline_workbook(scatter_trendline_data),
                            file_name="random_scatter_trendline_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                with scatter_export_cols[2]:
                    try:
                        scatter_png = scatter_fig.to_image(format="png", width=1600, height=900, scale=2)
                        st.download_button(
                            "Download random scatter PNG",
                            data=scatter_png,
                            file_name="random_scatter_plot.png",
                            mime="image/png",
                        )
                    except Exception as exc:
                        st.warning(f"PNG export is unavailable until Kaleido is installed correctly: {exc}")

        if scatter_parse_errors:
            st.warning(
                "Some random scatter data could not be loaded:\n\n"
                + "\n".join(f"- {err}" for err in scatter_parse_errors)
            )
    else:
        st.caption(
            "Expected layouts: one X column with multiple Y columns, "
            "or paired columns like x1/y1, x2/y2, x3/y3."
        )

save_labels = False
run_clicked = False
if has_measurements:
    section_header(
        "File Labels",
        "Classify each run and assign mode, regen, and target pedal",
        icon="tag",
    )
    st.write(
        "For single-pedal maneuvers use `Auto`, or select the known pedal value. "
        "For a 0% creep launch, select target `0` explicitly so the brake-release logic is used."
    )

    with st.expander("Apply the same labels to multiple files", expanded=True):
        with st.form("bulk_label_form", clear_on_submit=False):
            bulk_files = st.multiselect("Files to label", file_names)
            bulk_cols = st.columns(4)
            with bulk_cols[0]:
                bulk_type = st.selectbox("Test type", TEST_TYPES)
            with bulk_cols[1]:
                bulk_mode = st.text_input("Mode label", placeholder="Normal Mode")
            with bulk_cols[2]:
                bulk_regen = st.text_input("Regen / extra label", placeholder="Regen Level 1")
            with bulk_cols[3]:
                bulk_target = st.selectbox("Target pedal", TARGET_OPTIONS)
            apply_labels = st.form_submit_button("Apply labels to selected files")

        if apply_labels:
            if not bulk_files:
                st.warning("Select one or more files before applying bulk labels.")
            else:
                metadata = st.session_state["file_metadata"].copy()
                selected_rows = metadata["file_name"].isin(bulk_files)
                metadata.loc[selected_rows, "test_type"] = bulk_type
                metadata.loc[selected_rows, "mode_label"] = bulk_mode
                metadata.loc[selected_rows, "regen_label"] = bulk_regen
                metadata.loc[selected_rows, "target_pedal"] = bulk_target
                st.session_state["file_metadata"] = metadata
                st.session_state["metadata_editor_version"] += 1
                clear_results()
                st.rerun()

    editor_key = f"file_metadata_editor_{st.session_state.get('metadata_editor_version', 0)}"
    with st.form("file_metadata_form"):
        edited_metadata = st.data_editor(
            st.session_state["file_metadata"],
            key=editor_key,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "file_name": st.column_config.TextColumn("File", disabled=True),
                "test_type": st.column_config.SelectboxColumn("Test type", options=TEST_TYPES, required=True),
                "mode_label": st.column_config.TextColumn("Mode label"),
                "regen_label": st.column_config.TextColumn("Regen / extra label"),
                "target_pedal": st.column_config.SelectboxColumn("Target pedal", options=TARGET_OPTIONS, required=True),
            },
        )
        label_cols = st.columns([1, 1, 4])
        with label_cols[0]:
            save_labels = st.form_submit_button("Save labels")
        with label_cols[1]:
            run_clicked = st.form_submit_button("Generate plot", type="primary")

    if save_labels or run_clicked:
        st.session_state["file_metadata"] = edited_metadata.copy()
        metadata = edited_metadata.copy()

    if save_labels and not run_clicked:
        clear_results()
        st.success("Labels saved. Click Generate plot when you are ready.")

config_metadata = st.session_state.get("file_metadata", pd.DataFrame())
config = project_config_json(signal_map, settings, config_metadata, display_units)
st.download_button(
    "Download current setup as JSON",
    data=config,
    file_name="vehicle_plotter_project_config.json",
    mime="application/json",
)

current_signature = processing_signature(
    file_signatures,
    signal_map,
    settings,
    config_metadata,
    display_units,
)

if run_clicked and has_measurements:
    missing_signals = [name for name, value in signal_map.items() if not value]
    if missing_signals:
        st.error("Select all four signals before generating a plot.")
        st.stop()

    curve_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    load_errors: list[str] = []
    with st.spinner("Loading selected signals and applying usable-data logic..."):
        metadata_by_file = {
            str(row["file_name"]): row for _, row in st.session_state["file_metadata"].iterrows()
        }
        for file_name, source in measurement_sources.items():
            if str(metadata_by_file[file_name].get("test_type") or "") == "Ignore":
                continue
            try:
                raw_frame = read_measurement_frame(source, selected_channels, raster_step_s)
                measurement_frame = normalize_measurement_frame(raw_frame, signal_map)
                measurement_frame = convert_measurement_units(
                    measurement_frame,
                    speed_input_unit,
                    speed_output_unit,
                    acceleration_input_unit,
                    acceleration_output_unit,
                )
                file_curves, file_audit = process_measurement_frame(
                    file_name,
                    measurement_frame,
                    metadata_by_file[file_name],
                    settings,
                )
                curve_rows.extend(file_curves)
                audit_rows.extend(file_audit)
                del raw_frame, measurement_frame
            except MemoryError:
                load_errors.append(
                    f"{file_name}: insufficient memory. Try a larger optional MDF resample step, "
                    "for example 0.01 or 0.1 seconds."
                )
            except Exception as exc:
                load_errors.append(f"{file_name}: {exc}")

    averaged_curves, audit_table = average_file_curves(curve_rows, audit_rows)
    st.session_state["averaged_curves"] = averaged_curves
    st.session_state["audit_table"] = audit_table
    st.session_state["load_errors"] = load_errors
    st.session_state["result_signature"] = current_signature
    for key in PLOT_FILTER_KEYS:
        st.session_state.pop(key, None)

has_generated_measurements = (
    "averaged_curves" in st.session_state
    and st.session_state.get("result_signature") == current_signature
)
if has_measurements and "averaged_curves" in st.session_state and not has_generated_measurements:
    st.warning(
        "Signals, labels, or processing settings have changed. Click Generate plot "
        "to refresh measured results. Imported target curves can still be plotted."
    )

load_errors = st.session_state.get("load_errors", [])
if load_errors:
    st.warning("Some files could not be processed:\n\n" + "\n".join(f"- {err}" for err in load_errors))

averaged_curves = (
    st.session_state["averaged_curves"]
    if has_generated_measurements
    else pd.DataFrame()
)
audit_table = (
    st.session_state.get("audit_table", pd.DataFrame())
    if has_generated_measurements
    else pd.DataFrame()
)

if has_generated_measurements:
    section_header(
        "Usable Data Audit",
        "How each file and target was classified during processing",
        icon="audit",
    )
    if audit_table.empty:
        st.info("No usable data was found. Check signal selections, labels, target pedals, and tolerances.")
    else:
        st.dataframe(audit_table, width="stretch", hide_index=True)
        st.download_button(
            "Download audit CSV",
            data=audit_table.to_csv(index=False),
            file_name="usable_data_audit.csv",
            mime="text/csv",
        )
elif has_measurements:
    st.info("Click Generate plot to process the loaded measurement files.")

combined_available = pd.concat([averaged_curves, target_curves], ignore_index=True)
if combined_available.empty:
    if scatter_curves.empty:
        st.info("Load measurement files and click Generate plot, import target data, or import random scatter data to create a plot.")
    else:
        st.info("Random scatter data is shown above. Load measurement files or target Speed vs Acceleration data to use Plot Controls and Metrics.")
    st.stop()

section_header(
    "Plot Controls",
    "Filter curves, style them, and export the Speed vs Acceleration graph",
    icon="chart",
)
available_test_types = [
    test_type for test_type in ["Launch", "Deceleration", "Target"]
    if test_type in set(combined_available["test_type"])
]
sync_multiselect_options("plot_test_types", available_test_types)
selected_test_types = st.multiselect(
    "Test types to show",
    available_test_types,
    default=available_test_types,
    key="plot_test_types",
)

available = combined_available[combined_available["test_type"].isin(selected_test_types)].copy()
available["scenario"] = available["test_type"] + " | " + available["group_label"]
scenario_options = sorted(available["scenario"].dropna().unique().tolist())
target_options = sorted(available["target_pedal"].dropna().astype(int).unique().tolist())
sync_multiselect_options("plot_scenarios", scenario_options)
sync_multiselect_options("plot_targets", target_options)

control_cols = st.columns(2)
with control_cols[0]:
    selected_scenarios = st.multiselect(
        "Test / mode / regen combinations to show",
        scenario_options,
        default=scenario_options,
        key="plot_scenarios",
    )
with control_cols[1]:
    selected_targets = st.multiselect(
        "Pedal targets to show",
        target_options,
        default=target_options,
        key="plot_targets",
    )

plot_data = available[
    available["scenario"].isin(selected_scenarios)
    & available["target_pedal"].isin(selected_targets)
].copy()

if plot_data.empty:
    st.info("No curves match the selected plot filters.")
    st.stop()

style_defaults = default_style_table(plot_data)
style_overrides = {}
with st.expander("Curve Style and Legend Order (optional)", expanded=False):
    st.caption(f"Color accepts a name or code. {COLOR_COLUMN_HELP}")
    edited_style_table = st.data_editor(
        style_defaults,
        key=f"curve_style_editor_{hashlib.sha1('|'.join(style_defaults['style_key'].astype(str)).encode('utf-8')).hexdigest()}",
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        column_config={
            "style_key": st.column_config.TextColumn("Style key", disabled=True),
            "legend_order": st.column_config.NumberColumn("Legend order", min_value=1, step=1),
            "curve": st.column_config.TextColumn("Curve", disabled=True),
            "color": st.column_config.TextColumn("Color name/code", help=COLOR_COLUMN_HELP),
            "thickness": st.column_config.NumberColumn("Thickness", min_value=0.5, max_value=10.0, step=0.25),
            "line_style": st.column_config.SelectboxColumn("Line style", options=LINE_STYLE_OPTIONS),
        },
    )
    style_overrides = style_overrides_from_table(edited_style_table)
    trace_order = legend_order_from_table(edited_style_table)

fig = make_speed_accel_figure(
    plot_data,
    title="Speed vs Acceleration",
    speed_unit=speed_output_unit,
    acceleration_unit=acceleration_output_unit,
    style_overrides=style_overrides,
    trace_order=trace_order,
)
st.plotly_chart(fig, width="stretch")

export_cols = st.columns(3)
with export_cols[0]:
    try:
        workbook_bytes = export_plot_workbook(
            plot_data,
            speed_unit=speed_output_unit,
            acceleration_unit=acceleration_output_unit,
        )
        st.download_button(
            "Download data Excel",
            data=workbook_bytes,
            file_name="speed_vs_acceleration_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.warning("Install the updated requirements to enable Excel export.")
with export_cols[1]:
    st.download_button(
        "Download project JSON",
        data=project_config_json(
            signal_map,
            settings,
            config_metadata,
            display_units,
        ),
        file_name="vehicle_plotter_project_config.json",
        mime="application/json",
    )
with export_cols[2]:
    try:
        png_bytes = fig.to_image(format="png", width=1600, height=900, scale=2)
        st.download_button(
            "Download graph PNG",
            data=png_bytes,
            file_name="speed_vs_acceleration.png",
            mime="image/png",
        )
    except Exception as exc:
        st.warning(f"PNG export is unavailable until Kaleido is installed correctly: {exc}")

section_header(
    "Metrics",
    "Peak acceleration, road load, slope, and speed-band averages (optional)",
    icon="metric",
)
with st.expander("Calculate Speed vs Acceleration metrics", expanded=False):
    st.caption(
        "Metrics use the curves currently selected in Plot Controls. "
        f"Speed bands use the current graph speed unit: {speed_output_unit}."
    )
    slope_cols = st.columns(2)
    with slope_cols[0]:
        slope_start_speed = st.number_input(
            f"Slope start speed ({speed_output_unit})",
            value=80.0,
            step=5.0,
            key=resettable_key("metrics_slope_start_speed"),
        )
    with slope_cols[1]:
        slope_end_speed = st.number_input(
            f"Slope end speed ({speed_output_unit})",
            value=120.0,
            step=5.0,
            key=resettable_key("metrics_slope_end_speed"),
        )

    if slope_start_speed == slope_end_speed:
        st.warning("Choose two different speed points to calculate slope.")

    extend_road_load = st.checkbox(
        "Calculate road load using linear trendline extension when curve does not cross zero",
        value=False,
        help=(
            "When enabled, curves that do not intersect the acceleration = 0 axis "
            "will use a linear fit extension to estimate road-load speed."
        ),
        key=resettable_key("metrics_extend_road_load"),
    )

    metrics_table = calculate_metrics(
        plot_data,
        speed_bands=DEFAULT_SPEED_BANDS,
        slope_start=float(slope_start_speed),
        slope_end=float(slope_end_speed),
        extend_road_load=extend_road_load,
    )
    metrics_display = format_metrics_table(
        metrics_table,
        speed_unit=speed_output_unit,
        acceleration_unit=acceleration_output_unit,
        slope_start=float(slope_start_speed),
        slope_end=float(slope_end_speed),
        speed_bands=DEFAULT_SPEED_BANDS,
    )

    if metrics_display.empty:
        st.info("No metrics are available for the selected curves.")
    else:
        st.dataframe(metrics_display, width="stretch", hide_index=True)

        metric_plot_options = {
            "Select Plot": (None, None),
            f"Peak acceleration ({acceleration_output_unit})": (
                "peak_acceleration",
                f"Peak acceleration ({acceleration_output_unit})",
            ),
            f"Road load speed ({speed_output_unit})": (
                "road_load_speed",
                f"Road load speed ({speed_output_unit})",
            ),
            (
                f"Slope {float(slope_start_speed):g}-{float(slope_end_speed):g} "
                f"({acceleration_output_unit}/{speed_output_unit})"
            ): (
                "slope",
                f"Slope ({acceleration_output_unit}/{speed_output_unit})",
            ),
        }
        combined_average_graph_label = "Combined average acceleration bands"
        additional_metric_graph_label = "Additional metric plot"
        separate_band_graph_labels = [
            f"Separate {band_label(low, high, speed_output_unit)}"
            for low, high in DEFAULT_SPEED_BANDS
        ]

        metric_trendline_enabled = False
        metric_show_equations = False
        metric_show_trendline_only = False
        metric_trendline_choice = "Linear"
        metric_trendline_points = 200
        metric_trendline_x_mode = "Smooth generated X-values"
        selected_metric_trendline_graphs: list[str] = []
        with st.expander("Metric Trendlines (optional)", expanded=False):
            metric_trendline_enabled = st.checkbox(
                "Show trendlines on metric graphs",
                value=False,
                key=resettable_key("metric_show_trendlines"),
            )
            if metric_trendline_enabled:
                metric_trend_cols = st.columns(3)
                with metric_trend_cols[0]:
                    metric_trendline_choice = st.selectbox(
                        "Trendline type",
                        list(TRENDLINE_DEGREES.keys()),
                        key=resettable_key("metric_trendline_degree"),
                    )
                with metric_trend_cols[1]:
                    metric_trendline_x_mode = st.selectbox(
                        "Trendline X-values",
                        ["Smooth generated X-values", "Trendlines using existing X-values"],
                        key=resettable_key("metric_trendline_x_mode"),
                    )
                with metric_trend_cols[2]:
                    metric_show_equations = st.checkbox(
                        "Show equations",
                        value=True,
                        key=resettable_key("metric_show_equations"),
                    )
                if metric_trendline_x_mode == "Smooth generated X-values":
                    metric_trendline_points = st.number_input(
                        "Trendline points",
                        min_value=10,
                        max_value=5000,
                        value=200,
                        step=10,
                        key=resettable_key("metric_trendline_points"),
                    )
                selected_metric_trendline_graphs = st.multiselect(
                    "Metric graphs to add trendlines to",
                    [combined_average_graph_label, additional_metric_graph_label]
                    + separate_band_graph_labels,
                    default=[],
                    key=resettable_key("metric_trendline_graphs"),
                )
                metric_show_trendline_only = st.checkbox(
                    "Show trendline-only metric graphs",
                    value=False,
                    key=resettable_key("metric_trendline_only"),
                )

        average_bands = average_band_long(
            metrics_table,
            speed_unit=speed_output_unit,
            acceleration_unit=acceleration_output_unit,
            speed_bands=DEFAULT_SPEED_BANDS,
        )
        avg_band_fig = make_average_band_figure(
            metrics_table,
            speed_unit=speed_output_unit,
            acceleration_unit=acceleration_output_unit,
            speed_bands=DEFAULT_SPEED_BANDS,
        )
        metric_trendline_exports: list[pd.DataFrame] = []
        average_band_trendlines = pd.DataFrame()
        if (
            metric_trendline_enabled
            and combined_average_graph_label in selected_metric_trendline_graphs
        ):
            average_band_source = average_band_trend_source(average_bands)
            average_band_trendlines = build_trendline_data(
                average_band_source,
                x_col="target_pedal",
                y_col="average_acceleration",
                group_col="trend_group",
                degree=TRENDLINE_DEGREES[metric_trendline_choice],
                points=int(metric_trendline_points),
                trend_suffix="trend",
                use_existing_x=(
                    metric_trendline_x_mode == "Trendlines using existing X-values"
                ),
            )
            if not average_band_trendlines.empty:
                metric_trendline_exports.append(
                    tag_metric_trendlines(average_band_trendlines, "Average acceleration bands")
                )
                add_trendline_traces(
                    avg_band_fig,
                    average_band_trendlines,
                    show_equations=metric_show_equations,
                )
        st.plotly_chart(avg_band_fig, width="stretch")

        selected_metric_plot = st.selectbox(
            "Additional metric plot",
            list(metric_plot_options.keys()),
            key=resettable_key("selected_metric_plot"),
        )
        metric_column, metric_axis = metric_plot_options[selected_metric_plot]
        metric_fig = None
        selected_metric_trendlines = pd.DataFrame()
        if metric_column is not None:
            metric_fig = make_metric_figure(
                metrics_table,
                metric_column=metric_column,
                metric_label=selected_metric_plot,
                y_axis_title=metric_axis,
            )
        else:
            st.info("Select a metric in Additional metric plot to show that graph.")
        if (
            metric_trendline_enabled
            and metric_column is not None
            and additional_metric_graph_label in selected_metric_trendline_graphs
        ):
            selected_metric_source = selected_metric_trend_source(metrics_table, metric_column)
            selected_metric_trendlines = build_trendline_data(
                selected_metric_source,
                x_col="target_pedal",
                y_col="metric_value",
                group_col="trend_group",
                degree=TRENDLINE_DEGREES[metric_trendline_choice],
                points=int(metric_trendline_points),
                trend_suffix="trend",
                use_existing_x=(
                    metric_trendline_x_mode == "Trendlines using existing X-values"
                ),
            )
            if not selected_metric_trendlines.empty:
                metric_trendline_exports.append(
                    tag_metric_trendlines(selected_metric_trendlines, selected_metric_plot)
                )
                add_trendline_traces(
                    metric_fig,
                    selected_metric_trendlines,
                    show_equations=metric_show_equations,
                )
        if metric_fig is not None:
            st.plotly_chart(metric_fig, width="stretch")

        with st.expander("Separate Average Acceleration Band Plots", expanded=False):
            for low, high in DEFAULT_SPEED_BANDS:
                key = band_key(low, high)
                label = band_label(low, high, speed_output_unit)
                separate_band_fig = make_metric_figure(
                    metrics_table,
                    metric_column=key,
                    metric_label=label,
                    y_axis_title=f"Average acceleration ({acceleration_output_unit})",
                )
                separate_graph_label = f"Separate {label}"
                if (
                    metric_trendline_enabled
                    and separate_graph_label in selected_metric_trendline_graphs
                ):
                    separate_band_source = selected_metric_trend_source(metrics_table, key)
                    separate_band_trendlines = build_trendline_data(
                        separate_band_source,
                        x_col="target_pedal",
                        y_col="metric_value",
                        group_col="trend_group",
                        degree=TRENDLINE_DEGREES[metric_trendline_choice],
                        points=int(metric_trendline_points),
                        trend_suffix="trend",
                        use_existing_x=(
                            metric_trendline_x_mode == "Trendlines using existing X-values"
                        ),
                    )
                    if not separate_band_trendlines.empty:
                        metric_trendline_exports.append(
                            tag_metric_trendlines(separate_band_trendlines, label)
                        )
                        add_trendline_traces(
                            separate_band_fig,
                            separate_band_trendlines,
                            show_equations=metric_show_equations,
                        )
                st.plotly_chart(separate_band_fig, width="stretch")

        metric_trendline_data = (
            pd.concat(metric_trendline_exports, ignore_index=True)
            if metric_trendline_exports
            else pd.DataFrame()
        )
        if metric_trendline_enabled:
            if metric_trendline_data.empty:
                st.warning(
                    "No metric trendlines were created. Each metric series needs at least "
                    "degree + 1 points."
                )
            elif metric_show_equations:
                st.dataframe(
                    trendline_equations(metric_trendline_data),
                    width="stretch",
                    hide_index=True,
                )

        if metric_show_trendline_only and not metric_trendline_data.empty:
            metric_trendline_fig = make_trendline_figure(
                metric_trendline_data,
                title="Metric Trendline Graph",
                x_axis_title="Pedal (%)",
                y_axis_title="Metric value",
                show_equations=metric_show_equations,
            )
            st.plotly_chart(metric_trendline_fig, width="stretch")

        try:
            metric_export_cols = st.columns(2)
            with metric_export_cols[0]:
                st.download_button(
                    "Download metrics Excel",
                    data=export_metrics_workbook(metrics_display, average_bands),
                    file_name="speed_vs_acceleration_metrics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with metric_export_cols[1]:
                if not metric_trendline_data.empty:
                    st.download_button(
                        "Download metric trendline Excel",
                        data=export_trendline_workbook(metric_trendline_data),
                        file_name="metric_trendline_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
        except ImportError:
            st.warning("Install the updated requirements to enable metrics Excel export.")
