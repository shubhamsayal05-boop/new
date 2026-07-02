"""Speed vs Acceleration metrics: peak, road load, slope, and speed bands.

Metrics are computed per curve, where a curve is identified by its scenario
(``test_type | group_label``) and target pedal. All speed inputs/outputs are in
the current graph speed unit; this module does not perform unit conversion.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Speed bands expressed in the current graph speed unit.
DEFAULT_SPEED_BANDS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150)]

_GROUP_COLUMNS = ["scenario", "test_type", "group_label", "target_pedal"]


def band_key(low: float, high: float) -> str:
    """Column-safe key identifying a speed band."""

    return f"band_{low:g}_{high:g}"


def band_label(low: float, high: float, speed_unit: str) -> str:
    """Human-readable label for a speed band including its unit."""

    return f"{low:g}-{high:g} {speed_unit}"


def _ensure_scenario(plot_data: pd.DataFrame) -> pd.DataFrame:
    frame = plot_data.copy()
    if "scenario" not in frame.columns:
        frame["scenario"] = (
            frame["test_type"].astype(str) + " | " + frame["group_label"].astype(str)
        )
    return frame


def _peak_acceleration(curve: pd.DataFrame) -> float:
    accel = pd.to_numeric(curve["acceleration"], errors="coerce").dropna()
    return float(accel.max()) if not accel.empty else np.nan


def _road_load_speed(curve: pd.DataFrame, extend: bool) -> float:
    ordered = (
        curve[["speed", "acceleration"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values("speed")
    )
    if len(ordered) < 2:
        return np.nan
    speed = ordered["speed"].to_numpy()
    accel = ordered["acceleration"].to_numpy()

    for index in range(1, len(accel)):
        a0, a1 = accel[index - 1], accel[index]
        if a0 == 0:
            return float(speed[index - 1])
        if (a0 > 0 > a1) or (a0 < 0 < a1):
            fraction = a0 / (a0 - a1)
            return float(speed[index - 1] + fraction * (speed[index] - speed[index - 1]))
    if accel[-1] == 0:
        return float(speed[-1])

    if extend:
        try:
            slope, intercept = np.polyfit(speed, accel, 1)
        except (np.linalg.LinAlgError, ValueError):
            return np.nan
        if slope == 0:
            return np.nan
        return float(-intercept / slope)
    return np.nan


def _slope_between(curve: pd.DataFrame, slope_start: float, slope_end: float) -> float:
    if slope_start == slope_end:
        return np.nan
    ordered = (
        curve[["speed", "acceleration"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values("speed")
    )
    if len(ordered) < 2:
        return np.nan
    speed = ordered["speed"].to_numpy()
    accel = ordered["acceleration"].to_numpy()
    low, high = float(speed.min()), float(speed.max())
    if not (low <= slope_start <= high and low <= slope_end <= high):
        return np.nan
    accel_start = float(np.interp(slope_start, speed, accel))
    accel_end = float(np.interp(slope_end, speed, accel))
    return (accel_end - accel_start) / (slope_end - slope_start)


def _band_average(curve: pd.DataFrame, low: float, high: float) -> float:
    ordered = curve[["speed", "acceleration"]].apply(pd.to_numeric, errors="coerce").dropna()
    within = ordered[(ordered["speed"] >= low) & (ordered["speed"] <= high)]
    accel = within["acceleration"]
    return float(accel.mean()) if not accel.empty else np.nan


def calculate_metrics(
    plot_data: pd.DataFrame,
    speed_bands: list[tuple[float, float]] = DEFAULT_SPEED_BANDS,
    slope_start: float = 80.0,
    slope_end: float = 120.0,
    extend_road_load: bool = False,
) -> pd.DataFrame:
    """Compute per-curve metrics for the supplied plot data."""

    columns = _GROUP_COLUMNS + ["peak_acceleration", "road_load_speed", "slope"] + [
        band_key(low, high) for low, high in speed_bands
    ]
    if plot_data is None or plot_data.empty:
        return pd.DataFrame(columns=columns)

    frame = _ensure_scenario(plot_data)
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(_GROUP_COLUMNS, sort=True)
    for (scenario, test_type, group_label, target_pedal), curve in grouped:
        row: dict[str, object] = {
            "scenario": scenario,
            "test_type": test_type,
            "group_label": group_label,
            "target_pedal": int(target_pedal),
            "peak_acceleration": _peak_acceleration(curve),
            "road_load_speed": _road_load_speed(curve, extend_road_load),
            "slope": _slope_between(curve, slope_start, slope_end),
        }
        for low, high in speed_bands:
            row[band_key(low, high)] = _band_average(curve, low, high)
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def _format_value(value: object, digits: int = 4) -> object:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(number):
        return "N/A"
    return round(number, digits)


def format_metrics_table(
    metrics_table: pd.DataFrame,
    speed_unit: str,
    acceleration_unit: str,
    slope_start: float,
    slope_end: float,
    speed_bands: list[tuple[float, float]] = DEFAULT_SPEED_BANDS,
) -> pd.DataFrame:
    """Return a display-friendly metrics table with units and ``N/A`` handling."""

    if metrics_table is None or metrics_table.empty:
        return pd.DataFrame()

    display = pd.DataFrame()
    display["Scenario"] = metrics_table["scenario"].astype(str)
    display["Target pedal (%)"] = metrics_table["target_pedal"].astype(int)
    display[f"Peak acceleration ({acceleration_unit})"] = metrics_table[
        "peak_acceleration"
    ].map(_format_value)
    display[f"Road load speed ({speed_unit})"] = metrics_table["road_load_speed"].map(
        _format_value
    )
    display[
        f"Slope {slope_start:g}-{slope_end:g} ({acceleration_unit}/{speed_unit})"
    ] = metrics_table["slope"].map(lambda value: _format_value(value, 6))
    for low, high in speed_bands:
        key = band_key(low, high)
        if key in metrics_table.columns:
            display[f"Avg accel {band_label(low, high, speed_unit)} ({acceleration_unit})"] = (
                metrics_table[key].map(_format_value)
            )
    return display


def average_band_long(
    metrics_table: pd.DataFrame,
    speed_unit: str,
    acceleration_unit: str,
    speed_bands: list[tuple[float, float]] = DEFAULT_SPEED_BANDS,
) -> pd.DataFrame:
    """Melt the per-band average columns into a long, tidy frame."""

    columns = ["scenario", "target_pedal", "speed_band", "average_acceleration"]
    if metrics_table is None or metrics_table.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for _, record in metrics_table.iterrows():
        for low, high in speed_bands:
            key = band_key(low, high)
            if key not in metrics_table.columns:
                continue
            value = record[key]
            rows.append(
                {
                    "scenario": record["scenario"],
                    "target_pedal": int(record["target_pedal"]),
                    "speed_band": band_label(low, high, speed_unit),
                    "average_acceleration": value,
                }
            )
    long_frame = pd.DataFrame(rows, columns=columns)
    return long_frame.dropna(subset=["average_acceleration"]).reset_index(drop=True)


def _scenario_short(scenario: object) -> str:
    text = str(scenario)
    return text.split("|", maxsplit=1)[-1].strip() if "|" in text else text


def make_average_band_figure(
    metrics_table: pd.DataFrame,
    speed_unit: str,
    acceleration_unit: str,
    speed_bands: list[tuple[float, float]] = DEFAULT_SPEED_BANDS,
) -> go.Figure:
    """Plot average band acceleration versus target pedal for each band/scenario."""

    fig = go.Figure()
    long_frame = average_band_long(metrics_table, speed_unit, acceleration_unit, speed_bands)
    if long_frame.empty:
        fig.update_layout(
            title="Average acceleration by speed band",
            xaxis_title="Pedal (%)",
            yaxis_title=f"Average acceleration ({acceleration_unit})",
            template="plotly_white",
            hovermode="closest",
        )
        return fig

    multiple_scenarios = long_frame["scenario"].nunique() > 1
    for (scenario, band), group in long_frame.groupby(["scenario", "speed_band"], sort=True):
        ordered = group.sort_values("target_pedal")
        name = f"{_scenario_short(scenario)} · {band}" if multiple_scenarios else str(band)
        fig.add_trace(
            go.Scatter(
                x=ordered["target_pedal"],
                y=ordered["average_acceleration"],
                mode="lines+markers",
                name=name,
                hovertemplate=(
                    "Pedal: %{x}%<br>"
                    f"Avg accel: %{{y:.4f}} {acceleration_unit}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Average acceleration by speed band",
        xaxis_title="Pedal (%)",
        yaxis_title=f"Average acceleration ({acceleration_unit})",
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=70, r=30, t=60, b=60),
    )
    return fig


def make_metric_figure(
    metrics_table: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    y_axis_title: str,
) -> go.Figure:
    """Plot a single metric column versus target pedal, one trace per scenario."""

    fig = go.Figure()
    if (
        metrics_table is None
        or metrics_table.empty
        or metric_column not in metrics_table.columns
    ):
        fig.update_layout(
            title=metric_label,
            xaxis_title="Pedal (%)",
            yaxis_title=y_axis_title,
            template="plotly_white",
            hovermode="closest",
        )
        return fig

    frame = metrics_table[["scenario", "target_pedal", metric_column]].copy()
    frame[metric_column] = pd.to_numeric(frame[metric_column], errors="coerce")
    frame = frame.dropna(subset=[metric_column])
    for scenario, group in frame.groupby("scenario", sort=True):
        ordered = group.sort_values("target_pedal")
        fig.add_trace(
            go.Scatter(
                x=ordered["target_pedal"],
                y=ordered[metric_column],
                mode="lines+markers",
                name=_scenario_short(scenario),
                hovertemplate="Pedal: %{x}%<br>%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=metric_label,
        xaxis_title="Pedal (%)",
        yaxis_title=y_axis_title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=70, r=30, t=60, b=60),
    )
    return fig


def export_metrics_workbook(
    metrics_display: pd.DataFrame,
    average_bands: pd.DataFrame,
) -> bytes:
    """Export the metrics table and long band data to an Excel workbook."""

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        display = metrics_display if metrics_display is not None else pd.DataFrame()
        display.to_excel(writer, sheet_name="Metrics", index=False)
        bands = average_bands if average_bands is not None else pd.DataFrame()
        bands.to_excel(writer, sheet_name="Average bands", index=False)
    return buffer.getvalue()
