"""Import target/reference Speed vs Acceleration curves.

Target data is normalised into the same tidy shape as measured curves so both
can be concatenated and plotted together:

``test_type`` ("Target"), ``group_label`` (the user series label),
``target_pedal`` (int), ``speed``, ``acceleration``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import _tabular
from .units import convert_acceleration_value, convert_speed_value

TARGET_LAYOUTS = [
    "Auto detect",
    "One speed column, multiple acceleration columns",
    "Paired speed/acceleration columns",
]

_CURVE_COLUMNS = ["test_type", "group_label", "target_pedal", "speed", "acceleration"]


@dataclass
class ParsedTargetData:
    """Container for parsed target curves plus any detected input units."""

    curves: pd.DataFrame
    detected_speed_unit: str = ""
    detected_acceleration_unit: str = ""


def _normalize_layout(layout: str) -> str:
    text = (layout or "").casefold()
    if "pair" in text:
        return "paired"
    if "auto" in text:
        return "auto"
    return "single"


def _empty_curves() -> pd.DataFrame:
    return pd.DataFrame(columns=_CURVE_COLUMNS)


def _pedal_for_column(header: str, position: int) -> int:
    number = _tabular.extract_number(header)
    if number is not None:
        return int(round(number))
    # Fall back to a deterministic, non-colliding pedal id based on position.
    return int(position * 5)


def _decide_layout(headers: list[str], numeric: pd.DataFrame) -> str:
    column_count = numeric.shape[1]
    if column_count % 2 == 0:
        joined = " ".join(str(header).casefold() for header in headers)
        if "x" in joined and "y" in joined:
            return "paired"
    return "single"


def _build_single(
    headers: list[str], numeric: pd.DataFrame, series_label: str
) -> tuple[pd.DataFrame, str, str]:
    if numeric.shape[1] < 2:
        return _empty_curves(), "", ""

    speed_unit = _tabular.extract_unit(headers[0]) if headers else ""
    accel_unit = ""
    frames: list[pd.DataFrame] = []
    speed = numeric.iloc[:, 0]
    for position in range(1, numeric.shape[1]):
        header = headers[position] if position < len(headers) else ""
        if not accel_unit:
            accel_unit = _tabular.extract_unit(header)
        pedal = _pedal_for_column(header, position)
        block = pd.DataFrame(
            {
                "speed": speed,
                "acceleration": numeric.iloc[:, position],
            }
        ).dropna()
        if block.empty:
            continue
        block["test_type"] = "Target"
        block["group_label"] = series_label
        block["target_pedal"] = pedal
        frames.append(block[_CURVE_COLUMNS])

    if not frames:
        return _empty_curves(), speed_unit, accel_unit
    return pd.concat(frames, ignore_index=True), speed_unit, accel_unit


def _build_paired(
    headers: list[str], numeric: pd.DataFrame, series_label: str
) -> tuple[pd.DataFrame, str, str]:
    if numeric.shape[1] < 2:
        return _empty_curves(), "", ""

    speed_unit = _tabular.extract_unit(headers[0]) if headers else ""
    accel_unit = ""
    frames: list[pd.DataFrame] = []
    pair_index = 0
    for left in range(0, numeric.shape[1] - 1, 2):
        right = left + 1
        y_header = headers[right] if right < len(headers) else ""
        if not accel_unit:
            accel_unit = _tabular.extract_unit(y_header)
        pedal = _pedal_for_column(y_header, pair_index + 1)
        block = pd.DataFrame(
            {
                "speed": numeric.iloc[:, left],
                "acceleration": numeric.iloc[:, right],
            }
        ).dropna()
        pair_index += 1
        if block.empty:
            continue
        block["test_type"] = "Target"
        block["group_label"] = series_label
        block["target_pedal"] = pedal
        frames.append(block[_CURVE_COLUMNS])

    if not frames:
        return _empty_curves(), speed_unit, accel_unit
    return pd.concat(frames, ignore_index=True), speed_unit, accel_unit


def _parse_table(frame: pd.DataFrame, series_label: str, layout: str) -> ParsedTargetData:
    headers, numeric = _tabular.split_header(frame)
    if numeric.empty:
        return ParsedTargetData(_empty_curves())

    mode = _normalize_layout(layout)
    if mode == "auto":
        mode = _decide_layout(headers, numeric)

    if mode == "paired":
        curves, speed_unit, accel_unit = _build_paired(headers, numeric, series_label)
    else:
        curves, speed_unit, accel_unit = _build_single(headers, numeric, series_label)

    return ParsedTargetData(curves, speed_unit, accel_unit)


def parse_target_file(
    source_name: str,
    payload: bytes,
    series_label: str,
    layout: str,
) -> ParsedTargetData:
    """Parse an uploaded target file into :class:`ParsedTargetData`."""

    raw = _tabular.read_raw_bytes(source_name, payload)
    return _parse_table(raw, series_label or "Target", layout)


def parse_pasted_target_text(
    text: str,
    series_label: str,
    layout: str,
) -> ParsedTargetData:
    """Parse target data pasted from Excel into :class:`ParsedTargetData`."""

    raw = _tabular.read_raw_text(text)
    return _parse_table(raw, series_label or "Target", layout)


def convert_target_curves(
    raw_curves: pd.DataFrame,
    speed_input_unit: str,
    speed_output_unit: str,
    acceleration_input_unit: str,
    acceleration_output_unit: str,
) -> pd.DataFrame:
    """Convert target curves into the chosen graph/export units."""

    if raw_curves is None or raw_curves.empty:
        return _empty_curves()

    curves = raw_curves.copy()
    speed_factor = convert_speed_value(1.0, speed_input_unit, speed_output_unit)
    accel_factor = convert_acceleration_value(
        1.0, acceleration_input_unit, acceleration_output_unit
    )
    curves["speed"] = pd.to_numeric(curves["speed"], errors="coerce") * speed_factor
    curves["acceleration"] = (
        pd.to_numeric(curves["acceleration"], errors="coerce") * accel_factor
    )
    curves = curves.dropna(subset=["speed", "acceleration"])
    curves["target_pedal"] = curves["target_pedal"].astype(int)
    return curves.reset_index(drop=True)
