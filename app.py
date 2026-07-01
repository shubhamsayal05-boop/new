"""DriveLab Pro — INCA measurement intelligence platform."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.ai_panel import build_advisor_context, render_ai_advisor_page, render_ai_sidebar_hint, run_advisor
from ui.app_helpers import *
from ui.layout import (
    inject_theme,
    render_diagnostic_issues,
    render_metric_grid,
    render_page_header,
    render_sidebar_nav,
    render_topbar,
    render_workflow_strip,
    section_container,
)
from ui.theme import APP_NAME, APP_TAGLINE, APP_VERSION

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


def section_header(title: str, subtitle: str = "", icon: str = "chart") -> None:
    st.markdown(f"#### {title}")
    if subtitle:
        st.caption(subtitle)


def render_status_strip(cards: list[str]) -> None:
    import re

    parsed: list[tuple[str, str, str]] = []
    for card in cards:
        label = re.search(r"vp-stat-label\">([^<]+)", card)
        value = re.search(r"vp-stat-value\">([^<]+)", card)
        hint = re.search(r"vp-stat-hint\">([^<]+)", card)
        parsed.append(
            (
                label.group(1) if label else "Metric",
                value.group(1) if value else "—",
                hint.group(1) if hint else "",
            )
        )
    render_metric_grid(parsed)


def render_help_section() -> None:
    with st.expander("Documentation", expanded=False):
        st.markdown(
            f"**{APP_NAME} v{APP_VERSION}** analyzes INCA `.mf4` / `.dat` files for launch/deceleration "
            "curves, one-pedal regen analysis, reference overlays, and AI-guided troubleshooting."
        )


st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            f"### {APP_NAME}\n{APP_TAGLINE}\n\n"
            "Professional INCA measurement analytics with AI-assisted diagnostics."
        )
    },
)

inject_theme()

if st.session_state.pop(CLEAR_RERUN_KEY, False):
    st.rerun()

with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption(APP_TAGLINE)
    active_page = render_sidebar_nav()
    st.divider()
    st.markdown("**Data ingestion**")
    load_method = st.radio(
        "Source",
        ["Read from local folder", "Upload in browser"],
        help="Local folder is recommended for files larger than 200 MB.",
        key=resettable_key("load_method"),
    )
    uploaded_files = []
    folder_text = ""
    include_subfolders = False
    if load_method == "Read from local folder":
        folder_text = st.text_input(
            "Measurement folder",
            placeholder=r"C:\Data\INCA_Export",
            key=resettable_key("measurement_folder"),
        )
        include_subfolders = st.checkbox(
            "Include subfolders",
            value=False,
            key=resettable_key("include_subfolders"),
        )
    else:
        try:
            uploaded_files = st.file_uploader(
                "`.mf4` / `.dat` files",
                type=["mf4", "dat"],
                accept_multiple_files=True,
                key=resettable_key("measurement_files"),
            )
        except MemoryError:
            st.error("Browser upload ran out of memory. Use local folder loading.")
            st.stop()
    st.divider()
    with st.expander("Processing parameters", expanded=False):
        pedal_tolerance = st.number_input("Pedal tolerance (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        brake_threshold = st.number_input("Brake threshold", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        speed_bin = st.number_input("Speed bin (graph unit)", min_value=0.05, max_value=20.0, value=1.0, step=0.05)
        min_points = st.number_input("Min points per curve", min_value=1, max_value=10000, value=10, step=1)
        creep_release_percent = st.number_input("Creep brake release (%)", min_value=1.0, max_value=99.0, value=33.0, step=1.0)
        use_speed_gate = st.checkbox("Decel start speed gate", value=True)
        decel_start_speed = st.number_input("Decel start speed (KPH)", min_value=0.0, max_value=300.0, value=150.0, step=5.0)
        decel_start_speed_tolerance = st.number_input("Start speed tolerance (KPH)", min_value=0.0, max_value=100.0, value=10.0, step=5.0)
        raster_text = st.text_input("Resample step (s)", value="", help="Blank = native raster. Example: 0.01")
    st.button("Reset session", use_container_width=True, on_click=clear_all)

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
            st.sidebar.error("Resample step must be > 0.")
            raster_step_s = None
    except ValueError:
        st.sidebar.error("Resample step must be numeric.")

measurement_sources: dict[str, bytes | str] = {}
file_signatures: dict[str, str] = {}
if load_method == "Read from local folder":
    if folder_text.strip():
        measurement_folder = Path(folder_text.strip().strip('"'))
        if measurement_folder.is_dir():
            iterator = measurement_folder.rglob("*") if include_subfolders else measurement_folder.glob("*")
            local_files = sorted(
                [p for p in iterator if p.is_file() and p.suffix.casefold() in {".mf4", ".dat"}],
                key=lambda p: str(p).casefold(),
            )
            for path in local_files:
                display_name = str(path.relative_to(measurement_folder))
                stat = path.stat()
                measurement_sources[display_name] = str(path)
                file_signatures[display_name] = f"{stat.st_size}:{stat.st_mtime_ns}"
else:
    if uploaded_files:
        seen: dict[str, int] = {}
        for uploaded_file in uploaded_files:
            original_name = uploaded_file.name
            seen[original_name] = seen.get(original_name, 0) + 1
            if seen[original_name] == 1:
                display_name = original_name
            else:
                path = Path(original_name)
                display_name = f"{path.stem} ({seen[original_name]}){path.suffix}"
            file_bytes = uploaded_file.getvalue()
            measurement_sources[display_name] = file_bytes
            file_signatures[display_name] = content_hash(file_bytes)

file_names = list(measurement_sources.keys())
has_measurements = bool(measurement_sources)
metadata = metadata_for_current_uploads(file_names, tuple(file_signatures.items())) if has_measurements else pd.DataFrame()

# Build advisor context (refreshed after page render)
def _advisor_context(extra: dict | None = None) -> dict:
    ctx = build_advisor_context(
        has_measurements=has_measurements,
        file_count=len(file_names),
        load_method=load_method,
        active_page=active_page,
        signal_map=st.session_state.get("dl_signal_map", {}),
        load_errors=st.session_state.get("load_errors", []),
        opd_load_errors=st.session_state.get("opd_load_errors", []),
        audit_table=st.session_state.get("audit_table", pd.DataFrame()),
        opd_summary=st.session_state.get("opd_summary", pd.DataFrame()),
        has_generated_measurements="averaged_curves" in st.session_state,
        has_opd_results="opd_decel_curves" in st.session_state,
        results_stale=bool(st.session_state.get("dl_results_stale", False)),
        opd_stale=bool(st.session_state.get("dl_opd_stale", False)),
        mdf_dependency_missing=False,
    )
    if extra:
        ctx.update(extra)
    return ctx


advisor_report = run_advisor(_advisor_context())

render_topbar(advisor_report.health_score)
render_workflow_strip(active_page)

if active_page == "Dashboard":
    render_page_header("Command Center", "Session overview, health status, and recommended next steps.")
    render_metric_grid(
        [
            ("Files loaded", str(len(file_names)), "MF4 / DAT measurements"),
            ("Health score", f"{advisor_report.health_score}/100", "AI advisor scan"),
            ("Active workspace", active_page, "Use sidebar to switch modules"),
            ("Formats", "MF4 · DAT", "INCA exports only"),
        ]
    )
    render_diagnostic_issues(advisor_report.issues, max_items=5)
    if advisor_report.issues:
        st.info("Open **AI Advisor** for the full action plan and optional OpenAI enhancement.")
    render_help_section()

elif active_page == "AI Advisor":
    render_ai_advisor_page(_advisor_context())

elif active_page == "OPD Analysis":
    from opd_tab import render_one_pedal_tab

    render_page_header(
        "One-Pedal Analysis",
        "Lift-off regen events, jerk traces, and UN R13-H compliance from MF4/DAT files.",
    )
    render_one_pedal_tab(
        measurement_sources=measurement_sources,
        file_signatures=file_signatures,
        raster_step_s=raster_step_s,
        section_header=section_header,
        resettable_key=resettable_key,
    )

else:
    from ui.pages import workflow

    render_page_header(
        active_page,
        {
            "Data Source": "Connect INCA files and map speed, acceleration, brake, and pedal channels.",
            "Speed Analysis": "Label tests, extract usable segments, plot curves, and compute metrics.",
            "Reference Data": "Import target curves and arbitrary scatter datasets.",
        }.get(active_page, ""),
    )
    workflow_vars = workflow.render_analysis_sections(
        active_page,
        measurement_sources=measurement_sources,
        file_signatures=file_signatures,
        file_names=file_names,
        has_measurements=has_measurements,
        metadata=metadata,
        settings=settings,
        raster_step_s=raster_step_s,
        decel_start_speed=float(decel_start_speed),
        decel_start_speed_tolerance=float(decel_start_speed_tolerance),
        section_header=section_header,
    )
    if workflow_vars.get("signal_map"):
        st.session_state["dl_signal_map"] = workflow_vars["signal_map"]

advisor_report = run_advisor(_advisor_context())
render_ai_sidebar_hint(_advisor_context())
