"""Turn raw measurement frames into averaged Speed vs Acceleration curves.

The processing pipeline is intentionally tolerant of noisy, real-world INCA
data. Given a normalised measurement frame (time, speed, acceleration, brake,
pedal) it isolates *usable* launch or deceleration segments for each target
pedal, bins them onto a uniform speed grid, and averages repeated runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Standard accelerator-pedal targets offered to the user. These also seed the
# "All targets in file" auto-extraction mode.
LAUNCH_TARGETS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
DECEL_TARGETS = [0, 5, 10, 15, 20, 25, 30]

_SIGNAL_ROLES = ("speed", "acceleration", "brake", "pedal")


@dataclass
class ProcessingSettings:
    """User-tunable parameters controlling usable-data extraction."""

    pedal_tolerance: float = 2.0
    brake_threshold: float = 0.0
    creep_brake_release_fraction: float = 0.33
    decel_start_speed: float = 150.0
    decel_start_speed_tolerance: float = 10.0
    speed_bin: float = 1.0
    min_points: int = 10
    use_decel_speed_gate: bool = True
    extra: dict = field(default_factory=dict)


def normalize_measurement_frame(
    raw_frame: pd.DataFrame,
    signal_map: dict[str, str],
) -> pd.DataFrame:
    """Rename the selected channels to canonical role columns.

    ``signal_map`` maps roles (``speed``/``acceleration``/``brake``/``pedal``)
    to the channel name chosen by the user. The output frame keeps ``time`` and
    exposes one numeric column per role.
    """

    columns: dict[str, Any] = {}
    if "time" in raw_frame.columns:
        columns["time"] = pd.to_numeric(raw_frame["time"], errors="coerce")
    else:
        columns["time"] = pd.Series(np.arange(len(raw_frame)), dtype="float64")

    for role in _SIGNAL_ROLES:
        channel = signal_map.get(role, "")
        if channel and channel in raw_frame.columns:
            columns[role] = pd.to_numeric(raw_frame[channel], errors="coerce")
        else:
            columns[role] = pd.Series([np.nan] * len(raw_frame), dtype="float64")

    frame = pd.DataFrame(columns)
    frame = frame.dropna(subset=["speed", "acceleration"], how="all").reset_index(drop=True)
    return frame


def _clean_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "<na>"}:
        return ""
    return text


def build_group_label(mode_label: Any, regen_label: Any) -> str:
    """Combine the mode and regen labels into a single grouping label."""

    parts = [part for part in (_clean_label(mode_label), _clean_label(regen_label)) if part]
    return " ".join(parts) if parts else "Default"


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs for runs of ``True`` in ``mask``."""

    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _bin_segment(segment: pd.DataFrame, speed_bin: float) -> pd.DataFrame:
    """Average acceleration onto a uniform speed grid for a single segment."""

    usable = segment.dropna(subset=["speed", "acceleration"])
    if usable.empty or speed_bin <= 0:
        return pd.DataFrame(columns=["speed", "acceleration"])
    grid = (usable["speed"] / speed_bin).round() * speed_bin
    binned = (
        usable.assign(_grid=grid)
        .groupby("_grid", as_index=False)["acceleration"]
        .mean()
        .rename(columns={"_grid": "speed"})
        .sort_values("speed")
        .reset_index(drop=True)
    )
    return binned[["speed", "acceleration"]]


def _detect_dominant_pedal(frame: pd.DataFrame, settings: ProcessingSettings, is_decel: bool) -> list[int]:
    pedal = frame["pedal"].dropna()
    if pedal.empty:
        return []
    if is_decel:
        active = pedal[pedal <= 40]
    else:
        active = pedal[pedal > max(settings.pedal_tolerance, 1.0)]
    if active.empty:
        active = pedal
    rounded = (active / 5.0).round() * 5.0
    if rounded.empty:
        return []
    dominant = int(rounded.mode().iloc[0])
    return [dominant]


def _resolve_targets(
    metadata: pd.Series,
    frame: pd.DataFrame,
    settings: ProcessingSettings,
    is_decel: bool,
) -> list[int]:
    raw = _clean_label(metadata.get("target_pedal"))
    if raw == "" or raw.casefold() == "auto":
        return _detect_dominant_pedal(frame, settings, is_decel)
    if raw.casefold() == "all targets in file":
        return list(DECEL_TARGETS if is_decel else LAUNCH_TARGETS)
    try:
        return [int(float(raw))]
    except ValueError:
        return _detect_dominant_pedal(frame, settings, is_decel)


def _launch_mask(frame: pd.DataFrame, target: int, settings: ProcessingSettings) -> np.ndarray:
    pedal = frame["pedal"].to_numpy(dtype="float64")
    brake = frame["brake"].to_numpy(dtype="float64")
    brake_released = np.nan_to_num(brake, nan=0.0) <= settings.brake_threshold

    if target <= 0:
        # 0% creep launch: use brake-release logic instead of a pedal band.
        finite_brake = brake[np.isfinite(brake)]
        if finite_brake.size == 0:
            return np.zeros(len(frame), dtype=bool)
        creep_level = float(np.nanmax(finite_brake)) * settings.creep_brake_release_fraction
        low_pedal = np.nan_to_num(pedal, nan=0.0) <= max(settings.pedal_tolerance, 1.0)
        return (np.nan_to_num(brake, nan=0.0) < creep_level) & low_pedal

    tol = settings.pedal_tolerance
    pedal_band = np.abs(np.nan_to_num(pedal, nan=-999.0) - target) <= tol
    return pedal_band & brake_released


def _decel_mask(frame: pd.DataFrame, target: int, settings: ProcessingSettings) -> np.ndarray:
    pedal = frame["pedal"].to_numpy(dtype="float64")
    tol = settings.pedal_tolerance
    return np.abs(np.nan_to_num(pedal, nan=-999.0) - target) <= tol


def _segment_is_valid_decel(segment: pd.DataFrame, settings: ProcessingSettings) -> bool:
    if not settings.use_decel_speed_gate:
        return True
    speeds = segment["speed"].dropna()
    if speeds.empty:
        return False
    start_speed = float(speeds.iloc[0])
    reference = settings.decel_start_speed
    tolerance = max(settings.decel_start_speed_tolerance, 0.0)
    # The maneuver should begin at (or above) the target start speed.
    return start_speed >= reference - tolerance


def process_measurement_frame(
    file_name: str,
    frame: pd.DataFrame,
    metadata: pd.Series,
    settings: ProcessingSettings,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Extract usable, speed-binned curves for a single measurement file.

    Returns a tuple ``(curves, audit_rows)`` where ``curves`` is a list of
    per-target DataFrames (columns ``file_name``, ``test_type``,
    ``group_label``, ``target_pedal``, ``speed``, ``acceleration``) and
    ``audit_rows`` is a list of dictionaries describing what was found.
    """

    test_type = _clean_label(metadata.get("test_type")) or "Launch"
    if test_type == "Ignore":
        return [], []

    is_decel = test_type.casefold().startswith("decel")
    group_label = build_group_label(metadata.get("mode_label"), metadata.get("regen_label"))

    curves: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    if frame.empty or "speed" not in frame.columns:
        audit_rows.append(
            {
                "file_name": file_name,
                "test_type": test_type,
                "group_label": group_label,
                "target_pedal": None,
                "segments": 0,
                "points": 0,
                "speed_min": None,
                "speed_max": None,
                "status": "No usable speed/acceleration data",
            }
        )
        return curves, audit_rows

    targets = _resolve_targets(metadata, frame, settings, is_decel)
    if not targets:
        audit_rows.append(
            {
                "file_name": file_name,
                "test_type": test_type,
                "group_label": group_label,
                "target_pedal": None,
                "segments": 0,
                "points": 0,
                "speed_min": None,
                "speed_max": None,
                "status": "Could not determine a target pedal",
            }
        )
        return curves, audit_rows

    for target in sorted(set(int(value) for value in targets)):
        if is_decel:
            mask = _decel_mask(frame, target, settings)
        else:
            mask = _launch_mask(frame, target, settings)

        segments = _contiguous_segments(mask)
        binned_segments: list[pd.DataFrame] = []
        kept_segments = 0
        for start, end in segments:
            segment = frame.iloc[start:end]
            if len(segment) < 2:
                continue
            if is_decel and not _segment_is_valid_decel(segment, settings):
                continue
            binned = _bin_segment(segment, settings.speed_bin)
            if binned.empty:
                continue
            binned_segments.append(binned)
            kept_segments += 1

        if not binned_segments:
            audit_rows.append(
                {
                    "file_name": file_name,
                    "test_type": test_type,
                    "group_label": group_label,
                    "target_pedal": target,
                    "segments": 0,
                    "points": 0,
                    "speed_min": None,
                    "speed_max": None,
                    "status": "No usable segments for this target",
                }
            )
            continue

        combined = pd.concat(binned_segments, ignore_index=True)
        averaged = (
            combined.groupby("speed", as_index=False)["acceleration"].mean().sort_values("speed")
        )
        point_count = len(averaged)
        if point_count < settings.min_points:
            audit_rows.append(
                {
                    "file_name": file_name,
                    "test_type": test_type,
                    "group_label": group_label,
                    "target_pedal": target,
                    "segments": kept_segments,
                    "points": point_count,
                    "speed_min": float(averaged["speed"].min()),
                    "speed_max": float(averaged["speed"].max()),
                    "status": f"Below minimum usable points ({settings.min_points})",
                }
            )
            continue

        curve = averaged.copy()
        curve.insert(0, "file_name", file_name)
        curve.insert(1, "test_type", "Deceleration" if is_decel else "Launch")
        curve.insert(2, "group_label", group_label)
        curve.insert(3, "target_pedal", target)
        curves.append(curve.reset_index(drop=True))

        audit_rows.append(
            {
                "file_name": file_name,
                "test_type": "Deceleration" if is_decel else "Launch",
                "group_label": group_label,
                "target_pedal": target,
                "segments": kept_segments,
                "points": point_count,
                "speed_min": float(averaged["speed"].min()),
                "speed_max": float(averaged["speed"].max()),
                "status": "Usable",
            }
        )

    return curves, audit_rows


def average_file_curves(
    curve_rows: list[pd.DataFrame],
    audit_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average per-file curves that share a scenario and build the audit table.

    Curves are combined on the shared speed grid, averaging repeated runs
    across files for each ``(test_type, group_label, target_pedal)``.
    """

    audit_table = pd.DataFrame(audit_rows)
    if not audit_table.empty:
        audit_table = audit_table.sort_values(
            ["file_name", "test_type", "group_label", "target_pedal"],
            na_position="last",
        ).reset_index(drop=True)

    if not curve_rows:
        empty = pd.DataFrame(
            columns=["test_type", "group_label", "target_pedal", "speed", "acceleration"]
        )
        return empty, audit_table

    combined = pd.concat(curve_rows, ignore_index=True)
    averaged = (
        combined.groupby(
            ["test_type", "group_label", "target_pedal", "speed"], as_index=False
        )["acceleration"]
        .mean()
        .sort_values(["test_type", "group_label", "target_pedal", "speed"])
        .reset_index(drop=True)
    )
    averaged["target_pedal"] = averaged["target_pedal"].astype(int)
    return averaged, audit_table
