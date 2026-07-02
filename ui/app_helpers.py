"""Shared Streamlit helpers for DriveLab Pro."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from vehicle_plotter.colors import normalize_color_value
from vehicle_plotter.mdf_io import discover_channel_names, read_channel_unit
from vehicle_plotter.plotting import (
    PEDAL_COLORS,
    RANDOM_SCATTER_COLORS,
    curve_style_key,
    legend_label,
    mode_dash_map,
    mode_key,
    random_scatter_style_key,
    scatter_dash_map,
    scatter_source_key,
    short_group_label,
)
from vehicle_plotter.processing import DECEL_TARGETS, LAUNCH_TARGETS, ProcessingSettings

TARGET_OPTIONS = ["Auto", "All targets in file"] + [
    str(value) for value in sorted(set(LAUNCH_TARGETS + DECEL_TARGETS))
]

APP_TITLE = "DriveLab Pro"
TEST_TYPES = ["Launch", "Deceleration", "Ignore"]
RESULT_KEYS = ["averaged_curves", "audit_table", "result_signature", "load_errors"]
PLOT_FILTER_KEYS = ["plot_test_types", "plot_scenarios", "plot_targets"]
CLEAR_NONCE_KEY = "clear_all_nonce"
CLEAR_RERUN_KEY = "clear_all_force_rerun"
LINE_STYLE_OPTIONS = ["solid", "dot", "dash", "dashdot", "longdash", "longdashdot"]
COLOR_COLUMN_HELP = "Examples: red, blue, green, orange, #D32F2F, rgb(255,0,0)"

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


