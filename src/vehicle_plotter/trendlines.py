"""Polynomial trendline fitting, plotting, and export.

Trendlines are fitted with :func:`numpy.polyfit`. For each group a polynomial
of the requested degree is fitted to the source points, then evaluated either
on a smooth generated X grid or on the exact existing X values.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TRENDLINE_DEGREES = {
    "Linear": 1,
    "2nd order polynomial": 2,
    "3rd order polynomial": 3,
    "4th order polynomial": 4,
    "5th order polynomial": 5,
}

_TREND_COLUMNS = ["trendline_name", "group", "degree", "equation", "r_squared", "x", "y"]

_SUPERSCRIPTS = {2: "\u00b2", 3: "\u00b3", 4: "\u2074", 5: "\u2075"}


def _format_equation(coefficients: np.ndarray) -> str:
    """Return a human-readable ``y = ...`` string for polyfit coefficients."""

    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        value = float(coefficient)
        if power == 0:
            term = f"{value:.4g}"
        elif power == 1:
            term = f"{value:.4g}x"
        else:
            term = f"{value:.4g}x{_SUPERSCRIPTS.get(power, '^' + str(power))}"
        if not terms:
            terms.append(term)
        else:
            sign = "-" if value < 0 else "+"
            magnitude = term.lstrip("-") if value < 0 else term
            terms.append(f"{sign} {magnitude}")
    return "y = " + " ".join(terms)


def _r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    ss_res = float(np.sum((y_actual - y_predicted) ** 2))
    ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def build_trendline_data(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    selected_groups: list | None = None,
    degree: int = 1,
    points: int = 200,
    trend_suffix: str = "trend",
    use_existing_x: bool = False,
) -> pd.DataFrame:
    """Fit a polynomial trendline per group and return the fitted samples.

    Groups with fewer than ``degree + 1`` unique X values are skipped.
    """

    if data is None or data.empty:
        return pd.DataFrame(columns=_TREND_COLUMNS)
    for column in (x_col, y_col, group_col):
        if column not in data.columns:
            return pd.DataFrame(columns=_TREND_COLUMNS)

    degree = int(degree)
    allowed = None
    if selected_groups is not None:
        allowed = {str(group) for group in selected_groups}

    frames: list[pd.DataFrame] = []
    for group_value, group_frame in data.groupby(group_col, sort=True):
        if allowed is not None and str(group_value) not in allowed:
            continue
        numeric = pd.DataFrame(
            {
                "x": pd.to_numeric(group_frame[x_col], errors="coerce"),
                "y": pd.to_numeric(group_frame[y_col], errors="coerce"),
            }
        ).dropna()
        if numeric.empty:
            continue
        unique_x = np.unique(numeric["x"].to_numpy())
        if unique_x.size < degree + 1:
            continue

        x_fit = numeric["x"].to_numpy(dtype="float64")
        y_fit = numeric["y"].to_numpy(dtype="float64")
        try:
            coefficients = np.polyfit(x_fit, y_fit, degree)
        except (np.linalg.LinAlgError, ValueError):
            continue
        polynomial = np.poly1d(coefficients)

        if use_existing_x:
            x_output = np.sort(unique_x)
        else:
            x_output = np.linspace(
                float(unique_x.min()), float(unique_x.max()), num=max(int(points), 2)
            )
        y_output = polynomial(x_output)

        equation = _format_equation(coefficients)
        r_squared = _r_squared(y_fit, polynomial(x_fit))
        name = f"{group_value} {trend_suffix}".strip()

        frames.append(
            pd.DataFrame(
                {
                    "trendline_name": name,
                    "group": str(group_value),
                    "degree": degree,
                    "equation": equation,
                    "r_squared": r_squared,
                    "x": x_output,
                    "y": y_output,
                }
            )[_TREND_COLUMNS]
        )

    if not frames:
        return pd.DataFrame(columns=_TREND_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def trendline_equations(trendline_data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per trendline with its equation and fit quality."""

    if trendline_data is None or trendline_data.empty:
        return pd.DataFrame(columns=["trendline_name", "equation", "r_squared", "degree"])
    summary = (
        trendline_data[["trendline_name", "equation", "r_squared", "degree"]]
        .drop_duplicates(subset=["trendline_name"])
        .reset_index(drop=True)
    )
    summary["r_squared"] = summary["r_squared"].astype(float).round(4)
    return summary


def add_trendline_traces(
    fig: go.Figure,
    trendline_data: pd.DataFrame,
    show_equations: bool = False,
) -> go.Figure:
    """Overlay dashed trendline traces on an existing figure."""

    if fig is None or trendline_data is None or trendline_data.empty:
        return fig
    for name, group in trendline_data.groupby("trendline_name", sort=True):
        ordered = group.sort_values("x")
        label = str(name)
        if show_equations and "equation" in ordered.columns and not ordered.empty:
            label = f"{name} ({ordered['equation'].iloc[0]})"
        fig.add_trace(
            go.Scatter(
                x=ordered["x"],
                y=ordered["y"],
                mode="lines",
                name=label,
                line=dict(dash="dot", width=1.75),
                hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
            )
        )
    return fig


def make_trendline_figure(
    trendline_data: pd.DataFrame,
    title: str = "Trendline Graph",
    x_axis_title: str = "X",
    y_axis_title: str = "Y",
    show_equations: bool = False,
) -> go.Figure:
    """Build a standalone figure containing only the trendlines."""

    fig = go.Figure()
    add_trendline_traces(fig, trendline_data, show_equations=show_equations)
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        hovermode="closest",
        template="plotly_white",
        margin=dict(l=70, r=30, t=60, b=60),
    )
    return fig


def export_trendline_workbook(trendline_data: pd.DataFrame) -> bytes:
    """Export fitted trendline samples and equations to an Excel workbook."""

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if trendline_data is None or trendline_data.empty:
            pd.DataFrame(columns=_TREND_COLUMNS).to_excel(
                writer, sheet_name="Trendlines", index=False
            )
        else:
            trendline_data.to_excel(writer, sheet_name="Trendlines", index=False)
            trendline_equations(trendline_data).to_excel(
                writer, sheet_name="Equations", index=False
            )
    return buffer.getvalue()
