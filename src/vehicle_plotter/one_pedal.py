"""One-pedal (OPD) deceleration analytics for MDF/MF4/DAT measurement files.

Detects lift-off deceleration events from time-series speed, acceleration,
accelerator pedal, and brake signals. Produces deceleration-vs-speed curves,
jerk traces, coast (zero-acceleration) points, UN R13-H brake-lamp flags, and
optional energy-recovery estimates when motor/battery channels are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .processing import _contiguous_segments, _detect_dominant_pedal

R13H_DECEL_THRESHOLD_MS2 = 1.3

_OPTIONAL_ROLES = (
    "motor_torque",
    "motor_torque_2",
    "batt_current",
    "batt_voltage",
    "batt_soc",
    "road_gradient",
)


@dataclass
class OnePedalSettings:
    """Parameters controlling OPD event detection and curve binning."""

    pedal_tolerance: float = 2.0
    brake_threshold: float = 0.0
    decel_start_speed: float = 150.0
    decel_start_speed_tolerance: float = 10.0
    use_decel_speed_gate: bool = True
    speed_bin: float = 1.0
    min_event_duration_s: float = 3.0
    min_points: int = 10
    target_pedal: str = "Auto"
    r13h_threshold_ms2: float = R13H_DECEL_THRESHOLD_MS2
    vehicle_mass_kg: float = 2781.4
    extra: dict = field(default_factory=dict)


def normalize_opd_frame(
    raw_frame: pd.DataFrame,
    signal_map: dict[str, str],
) -> pd.DataFrame:
    """Build a canonical OPD frame from a raw MDF channel table."""

    columns: dict[str, Any] = {}
    if "time" in raw_frame.columns:
        columns["time"] = pd.to_numeric(raw_frame["time"], errors="coerce")
    else:
        columns["time"] = pd.Series(np.arange(len(raw_frame)), dtype="float64")

    core_roles = ("speed", "acceleration", "brake", "pedal")
    for role in core_roles + _OPTIONAL_ROLES:
        channel = signal_map.get(role, "")
        if channel and channel in raw_frame.columns:
            columns[role] = pd.to_numeric(raw_frame[channel], errors="coerce")
        elif role in core_roles:
            columns[role] = pd.Series([np.nan] * len(raw_frame), dtype="float64")

    frame = pd.DataFrame(columns)
    frame = frame.dropna(subset=["speed", "acceleration"], how="all").reset_index(drop=True)
    return frame


def _resolve_target_pedal(frame: pd.DataFrame, settings: OnePedalSettings) -> list[int]:
    raw = str(settings.target_pedal or "Auto").strip()
    if raw == "" or raw.casefold() == "auto":
        return _detect_dominant_pedal(frame, _PedalProxySettings(settings), is_decel=True)
    try:
        return [int(float(raw))]
    except ValueError:
        return _detect_dominant_pedal(frame, _PedalProxySettings(settings), is_decel=True)


@dataclass
class _PedalProxySettings:
    pedal_tolerance: float

    def __init__(self, settings: OnePedalSettings) -> None:
        self.pedal_tolerance = settings.pedal_tolerance


def _event_mask(frame: pd.DataFrame, target: int, settings: OnePedalSettings) -> np.ndarray:
    pedal = frame["pedal"].to_numpy(dtype="float64")
    brake = frame["brake"].to_numpy(dtype="float64")
    speed = frame["speed"].to_numpy(dtype="float64")

    tol = settings.pedal_tolerance
    pedal_band = np.abs(np.nan_to_num(pedal, nan=-999.0) - target) <= tol
    brake_released = np.nan_to_num(brake, nan=0.0) <= settings.brake_threshold

    speed_diff = np.diff(speed, prepend=speed[0])
    decreasing = np.isfinite(speed) & (speed_diff <= 0.05)

    return pedal_band & brake_released & decreasing


def _segment_passes_gates(segment: pd.DataFrame, settings: OnePedalSettings) -> bool:
    if segment.empty or len(segment) < 2:
        return False

    times = segment["time"].dropna()
    if len(times) >= 2:
        duration = float(times.iloc[-1] - times.iloc[0])
        if duration < settings.min_event_duration_s:
            return False

    if settings.use_decel_speed_gate:
        speeds = segment["speed"].dropna()
        if speeds.empty:
            return False
        start_speed = float(speeds.iloc[0])
        if start_speed < settings.decel_start_speed - settings.decel_start_speed_tolerance:
            return False

    return True


def _bin_decel_curve(segment: pd.DataFrame, speed_bin: float) -> pd.DataFrame:
    usable = segment.dropna(subset=["speed", "acceleration"])
    if usable.empty or speed_bin <= 0:
        return pd.DataFrame(columns=["speed", "acceleration"])
    grid = (usable["speed"] / speed_bin).round() * speed_bin
    binned = (
        usable.assign(_grid=grid)
        .groupby("_grid", as_index=False)["acceleration"]
        .mean()
        .rename(columns={"_grid": "speed"})
        .sort_values("speed", ascending=False)
        .reset_index(drop=True)
    )
    return binned[["speed", "acceleration"]]


def _compute_jerk(segment: pd.DataFrame) -> pd.Series:
    times = pd.to_numeric(segment["time"], errors="coerce")
    accel = pd.to_numeric(segment["acceleration"], errors="coerce")
    if len(times) < 2:
        return pd.Series([np.nan] * len(segment), index=segment.index)
    dt = times.diff().replace(0, np.nan)
    jerk = accel.diff() / dt
    return jerk


def _find_coast_speed(curve: pd.DataFrame) -> float | None:
    if curve.empty or len(curve) < 2:
        return None
    speeds = curve["speed"].to_numpy(dtype="float64")
    accels = curve["acceleration"].to_numpy(dtype="float64")
    for index in range(len(curve) - 1):
        a0, a1 = accels[index], accels[index + 1]
        if not (np.isfinite(a0) and np.isfinite(a1)):
            continue
        if a0 >= 0 >= a1 or a0 <= 0 <= a1:
            if a1 == a0:
                return float(speeds[index])
            fraction = a0 / (a0 - a1)
            return float(speeds[index] + fraction * (speeds[index + 1] - speeds[index]))
    return None


def _estimate_recovery_kwh(segment: pd.DataFrame, settings: OnePedalSettings) -> float | None:
    """Rough regen energy from motor torque (Nm) and vehicle mass if torque unavailable."""

    times = pd.to_numeric(segment["time"], errors="coerce").to_numpy(dtype="float64")
    speed_ms = pd.to_numeric(segment["speed"], errors="coerce").to_numpy(dtype="float64") / 3.6
    if len(times) < 2:
        return None

    power_w: np.ndarray | None = None
    if "motor_torque" in segment.columns:
        torque = pd.to_numeric(segment["motor_torque"], errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(torque).any():
            power_w = torque * np.nan_to_num(speed_ms, nan=0.0)

    if power_w is None and {"batt_current", "batt_voltage"}.issubset(segment.columns):
        current = pd.to_numeric(segment["batt_current"], errors="coerce").to_numpy(dtype="float64")
        voltage = pd.to_numeric(segment["batt_voltage"], errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(current).any() and np.isfinite(voltage).any():
            power_w = current * voltage

    if power_w is None:
        return None

    regen = np.where(power_w < 0, -power_w, 0.0)
    dt = np.diff(times, prepend=times[0])
    dt = np.where(dt > 0, dt, np.nan)
    energy_j = np.nansum(regen * dt)
    if not np.isfinite(energy_j) or energy_j <= 0:
        return None
    return float(energy_j / 3.6e6)


def _summarize_event(
    file_name: str,
    event_id: int,
    target: int,
    segment: pd.DataFrame,
    curve: pd.DataFrame,
    settings: OnePedalSettings,
    group_label: str = "",
) -> dict[str, object]:
    times = segment["time"].dropna()
    speeds = segment["speed"].dropna()
    accels = segment["acceleration"].dropna()
    jerk = _compute_jerk(segment)
    peak_jerk = float(jerk.min()) if jerk.notna().any() else None

    min_accel = float(accels.min()) if not accels.empty else None
    mean_accel = float(accels.mean()) if not accels.empty else None
    r13h_violation = bool(min_accel is not None and min_accel <= -settings.r13h_threshold_ms2)

    brake = segment["brake"].dropna()
    brake_applied = bool((brake > settings.brake_threshold).any()) if not brake.empty else False

    coast_speed = _find_coast_speed(curve)
    recovery_kwh = _estimate_recovery_kwh(segment, settings)

    pedal_vals = segment["pedal"].dropna()
    pedal_mean = float(pedal_vals.mean()) if not pedal_vals.empty else float(target)

    return {
        "file_name": file_name,
        "event_id": event_id,
        "group_label": group_label,
        "target_pedal": target,
        "pedal_mean_pct": round(pedal_mean, 2),
        "start_time_s": float(times.iloc[0]) if len(times) else None,
        "end_time_s": float(times.iloc[-1]) if len(times) else None,
        "duration_s": float(times.iloc[-1] - times.iloc[0]) if len(times) >= 2 else None,
        "speed_start": float(speeds.iloc[0]) if len(speeds) else None,
        "speed_end": float(speeds.iloc[-1]) if len(speeds) else None,
        "ax_min": min_accel,
        "ax_mean": mean_accel,
        "peak_jerk": peak_jerk,
        "coast_speed": coast_speed,
        "r13h_violation": r13h_violation,
        "brake_blending": brake_applied,
        "recovery_kwh": recovery_kwh,
        "curve_points": len(curve),
        "status": "Usable",
    }


def analyze_one_pedal_file(
    file_name: str,
    frame: pd.DataFrame,
    settings: OnePedalSettings,
    group_label: str = "",
) -> tuple[list[pd.DataFrame], list[dict[str, object]], pd.DataFrame]:
    """Detect OPD events in one normalised, unit-converted measurement frame.

    Returns ``(decel_curves, event_summaries, jerk_traces)`` where each decel
    curve has columns ``file_name``, ``event_id``, ``target_pedal``,
    ``group_label``, ``speed``, ``acceleration``.
    """

    curves: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    jerk_rows: list[pd.DataFrame] = []

    if frame.empty:
        summaries.append(
            {
                "file_name": file_name,
                "event_id": None,
                "group_label": group_label,
                "target_pedal": None,
                "status": "No usable speed/acceleration data",
            }
        )
        return curves, summaries, pd.DataFrame()

    targets = _resolve_target_pedal(frame, settings)
    if not targets:
        summaries.append(
            {
                "file_name": file_name,
                "event_id": None,
                "group_label": group_label,
                "target_pedal": None,
                "status": "Could not determine target pedal",
            }
        )
        return curves, summaries, pd.DataFrame()

    for target in sorted(set(int(value) for value in targets)):
        mask = _event_mask(frame, target, settings)
        segments = _contiguous_segments(mask)
        event_counter = 0

        for start, end in segments:
            segment = frame.iloc[start:end].copy()
            if not _segment_passes_gates(segment, settings):
                continue

            curve = _bin_decel_curve(segment, settings.speed_bin)
            if len(curve) < settings.min_points:
                continue

            event_counter += 1
            summary = _summarize_event(
                file_name, event_counter, target, segment, curve, settings, group_label
            )
            summaries.append(summary)

            tagged = curve.copy()
            tagged.insert(0, "file_name", file_name)
            tagged.insert(1, "event_id", event_counter)
            tagged.insert(2, "target_pedal", target)
            tagged.insert(3, "group_label", group_label)
            curves.append(tagged.reset_index(drop=True))

            jerk = _compute_jerk(segment)
            jerk_rows.append(
                pd.DataFrame(
                    {
                        "file_name": file_name,
                        "event_id": event_counter,
                        "target_pedal": target,
                        "group_label": group_label,
                        "time": segment["time"].values,
                        "speed": segment["speed"].values,
                        "acceleration": segment["acceleration"].values,
                        "jerk": jerk.values,
                    }
                )
            )

        if event_counter == 0:
            summaries.append(
                {
                    "file_name": file_name,
                    "event_id": None,
                    "group_label": group_label,
                    "target_pedal": target,
                    "status": "No usable OPD events for this target",
                }
            )

    jerk_traces = pd.concat(jerk_rows, ignore_index=True) if jerk_rows else pd.DataFrame()
    return curves, summaries, jerk_traces


def aggregate_opd_results(
    curve_rows: list[pd.DataFrame],
    summary_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine per-event curves and build the event summary table."""

    summary_table = pd.DataFrame(summary_rows)
    if not summary_table.empty:
        sort_cols = [col for col in ("file_name", "target_pedal", "event_id") if col in summary_table.columns]
        summary_table = summary_table.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    if not curve_rows:
        empty = pd.DataFrame(
            columns=["file_name", "event_id", "target_pedal", "group_label", "speed", "acceleration"]
        )
        return empty, summary_table

    combined = pd.concat(curve_rows, ignore_index=True)
    return combined, summary_table


def opd_kpi_cards(summary_table: pd.DataFrame) -> dict[str, object]:
    """Derive headline KPIs from usable event rows."""

    usable = summary_table[summary_table.get("status", pd.Series(dtype=str)) == "Usable"]
    if usable.empty:
        return {
            "event_count": 0,
            "max_decel": None,
            "mean_decel": None,
            "r13h_count": 0,
            "brake_blend_count": 0,
        }

    ax_min = pd.to_numeric(usable["ax_min"], errors="coerce")
    ax_mean = pd.to_numeric(usable["ax_mean"], errors="coerce")
    return {
        "event_count": len(usable),
        "max_decel": float(ax_min.min()) if ax_min.notna().any() else None,
        "mean_decel": float(ax_mean.mean()) if ax_mean.notna().any() else None,
        "r13h_count": int(usable["r13h_violation"].fillna(False).sum()),
        "brake_blend_count": int(usable["brake_blending"].fillna(False).sum()),
    }
