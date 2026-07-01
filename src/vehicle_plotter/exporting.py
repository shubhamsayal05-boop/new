"""Excel export for the main Speed vs Acceleration plot data."""

from __future__ import annotations

import io

import pandas as pd

from .plotting import legend_label


def _curve_name(test_type: object, group_label: object, target_pedal: object) -> str:
    return f"{test_type} | {legend_label(group_label, int(target_pedal))}"


def _wide_frame(plot_data: pd.DataFrame, speed_unit: str, acceleration_unit: str) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    grouped = plot_data.groupby(["test_type", "group_label", "target_pedal"], sort=True)
    for (test_type, group_label, target_pedal), curve in grouped:
        ordered = curve.sort_values("speed").reset_index(drop=True)
        name = _curve_name(test_type, group_label, target_pedal)
        block = pd.DataFrame(
            {
                f"{name} Speed ({speed_unit})": pd.to_numeric(
                    ordered["speed"], errors="coerce"
                ).reset_index(drop=True),
                f"{name} Accel ({acceleration_unit})": pd.to_numeric(
                    ordered["acceleration"], errors="coerce"
                ).reset_index(drop=True),
            }
        )
        blocks.append(block)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, axis=1)


def export_plot_workbook(
    plot_data: pd.DataFrame,
    speed_unit: str = "KPH",
    acceleration_unit: str = "m/s^2",
) -> bytes:
    """Export the plotted curves to an in-memory Excel workbook (bytes).

    The workbook contains a ``Curves`` sheet with one Speed/Acceleration column
    pair per curve, and a ``Long`` sheet with the tidy underlying rows.
    """

    buffer = io.BytesIO()
    if plot_data is None or plot_data.empty:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame(columns=["speed", "acceleration"]).to_excel(
                writer, sheet_name="Curves", index=False
            )
        return buffer.getvalue()

    wide = _wide_frame(plot_data, speed_unit, acceleration_unit)
    long_columns = [
        column
        for column in ["test_type", "group_label", "target_pedal", "scenario", "speed", "acceleration"]
        if column in plot_data.columns
    ]
    long_frame = plot_data[long_columns].copy()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if not wide.empty:
            wide.to_excel(writer, sheet_name="Curves", index=False)
        long_frame.to_excel(writer, sheet_name="Long", index=False)
    return buffer.getvalue()
