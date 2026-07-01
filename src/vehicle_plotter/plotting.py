"""Plotly figure construction and styling helpers.

Style keys are the contract between :func:`app.default_style_table` (which
seeds the editable style table) and the figure builders here: given a curve's
identity, both sides derive the same key so per-curve colour/width/dash
overrides can be looked up reliably.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go

# Curves that share a pedal percentage share a colour.
PEDAL_COLORS: dict[int, str] = {
    0: "#616161",
    5: "#1E88E5",
    10: "#43A047",
    15: "#FB8C00",
    20: "#E53935",
    25: "#8E24AA",
    30: "#00ACC1",
    40: "#F4511E",
    50: "#3949AB",
    60: "#7CB342",
    80: "#6D4C41",
    100: "#D81B60",
}

# Line-dash styles cycled through for different modes / sources.
MODE_DASHES = ["solid", "dot", "dashdot", "dash", "longdash", "longdashdot"]

# Reference style applied to imported target curves.
TARGET_DASH = "longdash"

RANDOM_SCATTER_COLORS = [
    "#1E88E5", "#E53935", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1",
    "#F4511E", "#3949AB", "#7CB342", "#D81B60", "#6D4C41", "#546E7A",
]

_DEFAULT_COLOR = "#616161"
_DEFAULT_WIDTH = 2.25


def short_group_label(text: object) -> str:
    """Return a compact, whitespace-normalised group label."""

    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    return cleaned or "Default"


def mode_key(group_label: object) -> str:
    """Return the mode identity used to pick a dash style for a curve."""

    return short_group_label(group_label).casefold()


def legend_label(group_label: object, target: int) -> str:
    """Human-friendly legend label for a curve."""

    return f"{short_group_label(group_label)} {int(target)}%"


def curve_style_key(test_type: str, group_label: str, target: int) -> str:
    """Stable identity for a measured/target Speed vs Acceleration curve."""

    return f"{test_type}|{short_group_label(group_label)}|{int(target)}"


def random_scatter_style_key(series_name: object) -> str:
    """Stable identity for a random-scatter series."""

    return f"scatter|{str(series_name)}"


def mode_dash_map(plot_data: pd.DataFrame) -> dict[str, str]:
    """Assign a dash style to each non-target mode in ``plot_data``."""

    if plot_data.empty or "group_label" not in plot_data.columns:
        return {}
    measured = plot_data
    if "test_type" in plot_data.columns:
        measured = plot_data[plot_data["test_type"].astype(str) != "Target"]
    modes = list(dict.fromkeys(mode_key(label) for label in measured["group_label"]))
    return {mode: MODE_DASHES[index % len(MODE_DASHES)] for index, mode in enumerate(modes)}


def scatter_source_key(series_data: pd.DataFrame) -> str:
    """Return the source identity for a random-scatter series."""

    if "source_order" in series_data.columns:
        orders = series_data["source_order"].dropna()
        if not orders.empty:
            return f"order:{int(orders.iloc[0])}"
    if "source_name" in series_data.columns:
        names = series_data["source_name"].dropna()
        if not names.empty:
            return f"name:{names.iloc[0]}"
    return ""


def scatter_dash_map(scatter_data: pd.DataFrame) -> dict[str, str]:
    """Assign a dash style per random-scatter source (first solid, then dot...)."""

    mapping: dict[str, str] = {}
    if scatter_data.empty:
        return mapping
    if "source_order" in scatter_data.columns and not scatter_data["source_order"].dropna().empty:
        orders = sorted(int(value) for value in scatter_data["source_order"].dropna().unique())
        for position, order in enumerate(orders):
            mapping[f"order:{order}"] = MODE_DASHES[position % len(MODE_DASHES)]
    if "source_name" in scatter_data.columns:
        names = list(dict.fromkeys(scatter_data["source_name"].dropna().astype(str)))
        for position, name in enumerate(names):
            mapping.setdefault(f"name:{name}", MODE_DASHES[position % len(MODE_DASHES)])
    return mapping


def _resolve_style(
    key: str,
    style_overrides: dict[str, dict[str, object]],
    default_color: str,
    default_dash: str,
) -> tuple[str, float, str]:
    override = style_overrides.get(key, {}) if style_overrides else {}
    color = str(override.get("color") or default_color)
    try:
        width = float(override.get("width", _DEFAULT_WIDTH))
    except (TypeError, ValueError):
        width = _DEFAULT_WIDTH
    dash = str(override.get("dash") or default_dash)
    return color, width, dash


def _apply_common_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> None:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="closest",
        template="plotly_white",
        legend=dict(traceorder="normal"),
        margin=dict(l=70, r=30, t=60, b=60),
    )


def make_speed_accel_figure(
    plot_data: pd.DataFrame,
    title: str = "Speed vs Acceleration",
    speed_unit: str = "KPH",
    acceleration_unit: str = "m/s^2",
    style_overrides: dict[str, dict[str, object]] | None = None,
    trace_order: list[str] | None = None,
) -> go.Figure:
    """Build the main Speed vs Acceleration line figure."""

    fig = go.Figure()
    style_overrides = style_overrides or {}
    order_index = {key: index for index, key in enumerate(trace_order or [])}
    dash_by_mode = mode_dash_map(plot_data)

    if plot_data.empty:
        _apply_common_layout(
            fig, title, f"Speed ({speed_unit})", f"Acceleration ({acceleration_unit})"
        )
        return fig

    grouped = plot_data.groupby(["test_type", "group_label", "target_pedal"], sort=True)
    pending: list[tuple[int, go.Scatter]] = []
    for (test_type, group_label, target_pedal), curve in grouped:
        target = int(target_pedal)
        test_text = str(test_type)
        group_text = str(group_label)
        key = curve_style_key(test_text, group_text, target)
        default_dash = TARGET_DASH if test_text == "Target" else dash_by_mode.get(
            mode_key(group_text), "solid"
        )
        color, width, dash = _resolve_style(
            key, style_overrides, PEDAL_COLORS.get(target, _DEFAULT_COLOR), default_dash
        )
        ordered_curve = curve.sort_values("speed")
        rank = order_index.get(key, len(order_index) + len(pending))
        trace = go.Scatter(
            x=ordered_curve["speed"],
            y=ordered_curve["acceleration"],
            mode="lines",
            name=f"{test_text} | {legend_label(group_text, target)}",
            line=dict(color=color, width=width, dash=dash),
            legendrank=rank,
            hovertemplate=(
                f"Speed: %{{x:.3f}} {speed_unit}<br>"
                f"Acceleration: %{{y:.4f}} {acceleration_unit}<extra></extra>"
            ),
        )
        pending.append((rank, trace))

    for _, trace in sorted(pending, key=lambda item: item[0]):
        fig.add_trace(trace)

    _apply_common_layout(
        fig, title, f"Speed ({speed_unit})", f"Acceleration ({acceleration_unit})"
    )
    return fig


def make_random_scatter_figure(
    scatter_plot_data: pd.DataFrame,
    title: str = "Random Scatter Plot",
    x_axis_title: str = "X",
    y_axis_title: str = "Y",
    style_overrides: dict[str, dict[str, object]] | None = None,
    trace_order: list[str] | None = None,
    show_markers: bool = False,
    smooth_lines: bool = True,
) -> go.Figure:
    """Build an arbitrary X/Y scatter/line figure for the random scatter section."""

    fig = go.Figure()
    style_overrides = style_overrides or {}
    order_index = {key: index for index, key in enumerate(trace_order or [])}

    if scatter_plot_data.empty:
        _apply_common_layout(fig, title, x_axis_title, y_axis_title)
        return fig

    dash_by_source = scatter_dash_map(scatter_plot_data)
    series_names = list(
        dict.fromkeys(scatter_plot_data["series_name"].dropna().astype(str))
    )
    mode = "lines+markers" if show_markers else "lines"
    line_shape = "spline" if smooth_lines else "linear"

    pending: list[tuple[int, go.Scatter]] = []
    for position, series_name in enumerate(series_names):
        series_data = scatter_plot_data[
            scatter_plot_data["series_name"].astype(str) == series_name
        ].sort_values("x")
        key = random_scatter_style_key(series_name)
        default_color = RANDOM_SCATTER_COLORS[position % len(RANDOM_SCATTER_COLORS)]
        default_dash = dash_by_source.get(scatter_source_key(series_data), "solid")
        color, width, dash = _resolve_style(
            key, style_overrides, default_color, default_dash
        )
        rank = order_index.get(key, len(order_index) + len(pending))
        trace = go.Scatter(
            x=series_data["x"],
            y=series_data["y"],
            mode=mode,
            name=series_name,
            line=dict(color=color, width=width, dash=dash, shape=line_shape),
            marker=dict(color=color, size=6),
            legendrank=rank,
            hovertemplate=(
                f"{x_axis_title}: %{{x:.4f}}<br>"
                f"{y_axis_title}: %{{y:.4f}}<extra></extra>"
            ),
        )
        pending.append((rank, trace))

    for _, trace in sorted(pending, key=lambda item: item[0]):
        fig.add_trace(trace)

    _apply_common_layout(fig, title, x_axis_title, y_axis_title)
    return fig
