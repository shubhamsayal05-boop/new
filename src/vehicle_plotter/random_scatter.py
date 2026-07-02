"""Import and derive arbitrary X/Y scatter series.

Random scatter data is normalised to a tidy frame with ``series_name``, ``x``
and ``y`` columns. Derived series additionally carry ``derived_type`` and
``derived_from`` so the UI can describe how they were produced.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import _tabular

SCATTER_LAYOUTS = [
    "Auto detect",
    "One X column, multiple Y columns",
    "Paired X/Y columns",
]

_CURVE_COLUMNS = ["series_name", "x", "y"]
_DERIVED_COLUMNS = ["series_name", "x", "y", "derived_type", "derived_from"]


@dataclass
class ParsedRandomScatterData:
    """Container for parsed random-scatter curves."""

    curves: pd.DataFrame


def _empty_curves() -> pd.DataFrame:
    return pd.DataFrame(columns=_CURVE_COLUMNS)


def _normalize_layout(layout: str) -> str:
    text = (layout or "").casefold()
    if "pair" in text:
        return "paired"
    if "auto" in text:
        return "auto"
    return "single"


def _series_name(header: str, position: int, series_label: str, multiple: bool) -> str:
    header_text = str(header or "").strip()
    is_synthetic = not header_text or header_text.startswith("col")
    if is_synthetic:
        return f"{series_label} {position}" if multiple else series_label
    if multiple:
        return f"{series_label} - {header_text}"
    return header_text or series_label


def _decide_layout(headers: list[str], numeric: pd.DataFrame) -> str:
    if numeric.shape[1] % 2 == 0:
        joined = " ".join(str(header).casefold() for header in headers)
        if "x" in joined and "y" in joined:
            return "paired"
    return "single"


def _build_single(headers: list[str], numeric: pd.DataFrame, series_label: str) -> pd.DataFrame:
    if numeric.shape[1] < 2:
        return _empty_curves()
    x_values = numeric.iloc[:, 0]
    multiple = numeric.shape[1] > 2
    frames: list[pd.DataFrame] = []
    for position in range(1, numeric.shape[1]):
        header = headers[position] if position < len(headers) else ""
        name = _series_name(header, position, series_label, multiple)
        block = pd.DataFrame({"x": x_values, "y": numeric.iloc[:, position]}).dropna()
        if block.empty:
            continue
        block["series_name"] = name
        frames.append(block[_CURVE_COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else _empty_curves()


def _build_paired(headers: list[str], numeric: pd.DataFrame, series_label: str) -> pd.DataFrame:
    if numeric.shape[1] < 2:
        return _empty_curves()
    pair_count = numeric.shape[1] // 2
    multiple = pair_count > 1
    frames: list[pd.DataFrame] = []
    pair_index = 0
    for left in range(0, numeric.shape[1] - 1, 2):
        right = left + 1
        header = headers[right] if right < len(headers) else ""
        name = _series_name(header, pair_index + 1, series_label, multiple)
        block = pd.DataFrame(
            {"x": numeric.iloc[:, left], "y": numeric.iloc[:, right]}
        ).dropna()
        pair_index += 1
        if block.empty:
            continue
        block["series_name"] = name
        frames.append(block[_CURVE_COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else _empty_curves()


def _parse_table(frame: pd.DataFrame, series_label: str, layout: str) -> ParsedRandomScatterData:
    headers, numeric = _tabular.split_header(frame)
    if numeric.empty:
        return ParsedRandomScatterData(_empty_curves())

    mode = _normalize_layout(layout)
    if mode == "auto":
        mode = _decide_layout(headers, numeric)

    if mode == "paired":
        curves = _build_paired(headers, numeric, series_label)
    else:
        curves = _build_single(headers, numeric, series_label)
    return ParsedRandomScatterData(curves)


def parse_scatter_file(
    source_name: str,
    payload: bytes,
    series_label: str,
    layout: str,
) -> ParsedRandomScatterData:
    """Parse an uploaded random-scatter file."""

    raw = _tabular.read_raw_bytes(source_name, payload)
    return _parse_table(raw, series_label or "Scatter", layout)


def parse_pasted_scatter_text(
    text: str,
    series_label: str,
    layout: str,
) -> ParsedRandomScatterData:
    """Parse random-scatter data pasted from Excel."""

    raw = _tabular.read_raw_text(text)
    return _parse_table(raw, series_label or "Scatter", layout)


def _series_xy(scatter_curves: pd.DataFrame, series_name: str) -> pd.DataFrame:
    subset = scatter_curves[scatter_curves["series_name"].astype(str) == str(series_name)]
    subset = subset[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
    return subset.sort_values("x").reset_index(drop=True)


def average_selected_series(
    scatter_curves: pd.DataFrame,
    series_names: list[str],
    average_name: str,
) -> pd.DataFrame:
    """Average selected series over their shared X range.

    Returns an empty frame when fewer than two valid series are supplied or the
    series do not share an overlapping X range.
    """

    prepared = []
    for name in series_names:
        xy = _series_xy(scatter_curves, name)
        if len(xy) >= 2:
            prepared.append(xy)
    if len(prepared) < 2:
        return pd.DataFrame(columns=_DERIVED_COLUMNS)

    lower = max(float(xy["x"].min()) for xy in prepared)
    upper = min(float(xy["x"].max()) for xy in prepared)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return pd.DataFrame(columns=_DERIVED_COLUMNS)

    grid_points: set[float] = set()
    for xy in prepared:
        in_range = xy["x"][(xy["x"] >= lower) & (xy["x"] <= upper)]
        grid_points.update(in_range.tolist())
    grid_points.update([lower, upper])
    grid = np.array(sorted(grid_points), dtype="float64")
    if grid.size < 2:
        grid = np.linspace(lower, upper, num=50)

    stacked = np.vstack(
        [np.interp(grid, xy["x"].to_numpy(), xy["y"].to_numpy()) for xy in prepared]
    )
    averaged_y = stacked.mean(axis=0)

    return pd.DataFrame(
        {
            "series_name": average_name or "Average_Selected",
            "x": grid,
            "y": averaged_y,
            "derived_type": "average",
            "derived_from": ", ".join(str(name) for name in series_names),
        }
    )[_DERIVED_COLUMNS]


def offset_selected_series(
    scatter_curves: pd.DataFrame,
    source_series: str,
    offset_name: str,
    offset_amount: float,
    offset_axis: str,
) -> pd.DataFrame:
    """Shift a single series along the X or Y axis by ``offset_amount``."""

    xy = _series_xy(scatter_curves, source_series)
    if xy.empty:
        return pd.DataFrame(columns=_DERIVED_COLUMNS)

    x_values = xy["x"].to_numpy(dtype="float64")
    y_values = xy["y"].to_numpy(dtype="float64")
    if str(offset_axis).upper() == "X":
        x_values = x_values + float(offset_amount)
    else:
        y_values = y_values + float(offset_amount)

    return pd.DataFrame(
        {
            "series_name": offset_name or f"{source_series}_offset",
            "x": x_values,
            "y": y_values,
            "derived_type": "offset",
            "derived_from": str(source_series),
        }
    )[_DERIVED_COLUMNS]


def _wide_series_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Lay out each series as adjacent X/Y columns for a spreadsheet export."""

    blocks: list[pd.DataFrame] = []
    for name in dict.fromkeys(data["series_name"].astype(str)):
        subset = data[data["series_name"].astype(str) == name].reset_index(drop=True)
        block = pd.DataFrame(
            {
                f"{name} X": pd.to_numeric(subset["x"], errors="coerce").reset_index(drop=True),
                f"{name} Y": pd.to_numeric(subset["y"], errors="coerce").reset_index(drop=True),
            }
        )
        blocks.append(block)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, axis=1)


def export_random_scatter_workbook(data: pd.DataFrame) -> bytes:
    """Export random-scatter series to an in-memory Excel workbook (bytes)."""

    buffer = io.BytesIO()
    long_columns = [
        column
        for column in ["series_name", "derived_type", "derived_from", "x", "y"]
        if column in data.columns
    ]
    long_frame = data[long_columns] if long_columns else data
    wide_frame = _wide_series_frame(data)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if not wide_frame.empty:
            wide_frame.to_excel(writer, sheet_name="Series", index=False)
        long_frame.to_excel(writer, sheet_name="Long", index=False)
    return buffer.getvalue()
