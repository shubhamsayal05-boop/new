"""Extended one-pedal mode (OPM/OPD) metrics — OEM / benchmarking practice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

G_MS2 = 9.80665
JERK_COMFORT_CEILING_MS3 = 3.0
GB21670_BRAKE_LAMP_MS2 = 1.3


@dataclass
class SensitivityAdjustments:
    """What-if tuning knobs (sliders now; optimizer can target these later)."""

    accel_scale: float = 1.0
    accel_offset_ms2: float = 0.0
    speed_scale: float = 1.0
    speed_offset_kph: float = 0.0
    pedal_offset_pct: float = 0.0
    jerk_smooth_samples: int = 1
    pedal_tolerance_delta: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accel_scale": self.accel_scale,
            "accel_offset_ms2": self.accel_offset_ms2,
            "speed_scale": self.speed_scale,
            "speed_offset_kph": self.speed_offset_kph,
            "pedal_offset_pct": self.pedal_offset_pct,
            "jerk_smooth_samples": self.jerk_smooth_samples,
            "pedal_tolerance_delta": self.pedal_tolerance_delta,
        }


@dataclass
class OpmMetricSettings:
    """Thresholds aligned with benchmarking / compliance practice."""

    jerk_comfort_limit_ms3: float = JERK_COMFORT_CEILING_MS3
    decel_band_02g_ms2: float = -0.2 * G_MS2
    decel_band_05g_ms2: float = -0.5 * G_MS2
    tip_out_target_decel_ms2: float = -1.0
    stop_speed_threshold_kph: float = 3.0
    gb21670_brake_lamp_ms2: float = GB21670_BRAKE_LAMP_MS2
    vehicle_mass_kg: float = 2781.4
    speed_interval_kph: float = 10.0
    extra: dict = field(default_factory=dict)


def apply_sensitivity(frame: pd.DataFrame, adj: SensitivityAdjustments) -> pd.DataFrame:
    """Apply what-if scaling to speed, acceleration, and pedal traces."""

    out = frame.copy()
    if "speed" in out.columns:
        out["speed"] = pd.to_numeric(out["speed"], errors="coerce") * adj.speed_scale + adj.speed_offset_kph
    if "acceleration" in out.columns:
        accel = pd.to_numeric(out["acceleration"], errors="coerce")
        out["acceleration"] = accel * adj.accel_scale + adj.accel_offset_ms2
    if "pedal" in out.columns:
        out["pedal"] = pd.to_numeric(out["pedal"], errors="coerce") + adj.pedal_offset_pct
    return out


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window=window, center=True, min_periods=1).mean()


def compute_jerk_trace(segment: pd.DataFrame, smooth_samples: int = 1) -> pd.Series:
    """First-class jerk channel da/dt (m/s³)."""

    times = pd.to_numeric(segment["time"], errors="coerce")
    accel = smooth_series(pd.to_numeric(segment["acceleration"], errors="coerce"), smooth_samples)
    if len(times) < 2:
        return pd.Series([np.nan] * len(segment), index=segment.index)
    dt = times.diff().replace(0, np.nan)
    return accel.diff() / dt


def _crossing_time(times: np.ndarray, values: np.ndarray, threshold: float, direction: str = "below") -> float | None:
    """Return time of first threshold crossing."""

    for i in range(1, len(values)):
        v0, v1 = values[i - 1], values[i]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        if direction == "below" and v0 > threshold >= v1:
            frac = (v0 - threshold) / (v0 - v1) if v1 != v0 else 0.0
            return float(times[i - 1] + frac * (times[i] - times[i - 1]))
        if direction == "above" and v0 < threshold <= v1:
            frac = (threshold - v0) / (v1 - v0) if v1 != v0 else 0.0
            return float(times[i - 1] + frac * (times[i] - times[i - 1]))
    return None


def _find_zero_accel_speed(speeds: np.ndarray, accels: np.ndarray) -> float | None:
    for i in range(len(speeds) - 1):
        a0, a1 = accels[i], accels[i + 1]
        if not (np.isfinite(a0) and np.isfinite(a1)):
            continue
        if a0 >= 0 >= a1 or a0 <= 0 <= a1:
            if a1 == a0:
                return float(speeds[i])
            frac = a0 / (a0 - a1)
            return float(speeds[i] + frac * (speeds[i + 1] - speeds[i]))
    return None


def classify_ramp_shape(pedal: np.ndarray, decel: np.ndarray) -> str:
    """Classify post-lift regen ramp: flat, linear, or progressive."""

    mask = np.isfinite(pedal) & np.isfinite(decel) & (decel < -0.05)
    if mask.sum() < 5:
        return "insufficient_data"
    p = pedal[mask]
    d = np.abs(decel[mask])
    if np.nanstd(d) < 0.08:
        return "flat"
    # Linear vs progressive via quadratic term on normalized pedal
    p_n = (p - p.min()) / max(p.max() - p.min(), 1e-6)
    try:
        lin = np.polyfit(p_n, d, 1)
        lin_pred = np.polyval(lin, p_n)
        quad = np.polyfit(p_n, d, 2)
        quad_pred = np.polyval(quad, p_n)
        ss_res_lin = np.sum((d - lin_pred) ** 2)
        ss_res_quad = np.sum((d - quad_pred) ** 2)
        if ss_res_quad < ss_res_lin * 0.85 and quad[0] > 0.05:
            return "progressive"
        return "linear"
    except Exception:
        return "linear"


def compute_mfdd(times: np.ndarray, speeds_kph: np.ndarray, v1_kph: float, v2_kph: float) -> float | None:
    """Mean fully developed deceleration between two speeds (ISO-style convention)."""

    if len(times) < 3:
        return None
    v_ms = speeds_kph / 3.6
    mask = (speeds_kph <= v1_kph) & (speeds_kph >= v2_kph)
    if mask.sum() < 2:
        return None
    t = times[mask]
    v = v_ms[mask]
    dt = t[-1] - t[0]
    if dt <= 0:
        return None
    return float((v[0] ** 2 - v[-1] ** 2) / (2 * dt))


def kinetic_energy_j(mass_kg: float, speed_kph: float) -> float:
    v_ms = speed_kph / 3.6
    return 0.5 * mass_kg * v_ms * v_ms


def estimate_regen_energy_j(segment: pd.DataFrame) -> float | None:
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
        c = pd.to_numeric(segment["batt_current"], errors="coerce").to_numpy(dtype="float64")
        v = pd.to_numeric(segment["batt_voltage"], errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(c).any():
            power_w = c * v
    if power_w is None:
        return None
    regen = np.where(power_w < 0, -power_w, 0.0)
    dt = np.diff(times, prepend=times[0])
    dt = np.where(dt > 0, dt, np.nan)
    energy = np.nansum(regen * dt)
    return float(energy) if np.isfinite(energy) and energy > 0 else None


def decel_by_speed_interval(segment: pd.DataFrame, interval_kph: float) -> pd.DataFrame:
    usable = segment.dropna(subset=["speed", "acceleration"])
    if usable.empty or interval_kph <= 0:
        return pd.DataFrame(columns=["speed_band_low", "speed_band_high", "ax_mean", "points"])
    grid = (usable["speed"] / interval_kph).astype(int) * interval_kph
    rows = []
    for band in sorted(grid.unique(), reverse=True):
        band_data = usable[grid == band]
        if band_data.empty:
            continue
        rows.append(
            {
                "speed_band_low": float(band),
                "speed_band_high": float(band + interval_kph),
                "ax_mean": float(band_data["acceleration"].mean()),
                "points": len(band_data),
            }
        )
    return pd.DataFrame(rows)


def pedal_decel_map(segment: pd.DataFrame, pedal_bin_pct: float = 2.0) -> pd.DataFrame:
    usable = segment.dropna(subset=["pedal", "acceleration"])
    if usable.empty:
        return pd.DataFrame(columns=["pedal_pct", "ax_mean"])
    grid = (usable["pedal"] / pedal_bin_pct).round() * pedal_bin_pct
    return (
        usable.assign(_grid=grid)
        .groupby("_grid", as_index=False)["acceleration"]
        .mean()
        .rename(columns={"_grid": "pedal_pct", "acceleration": "ax_mean"})
        .sort_values("pedal_pct")
    )


def analyze_opm_event(
    segment: pd.DataFrame,
    metric_settings: OpmMetricSettings,
    adjustments: SensitivityAdjustments | None = None,
    brake_threshold: float = 0.0,
) -> dict[str, Any]:
    """Compute full OPM/OPD metric bucket for one event segment."""

    adj = adjustments or SensitivityAdjustments()
    seg = apply_sensitivity(segment, adj)
    times = pd.to_numeric(seg["time"], errors="coerce").to_numpy(dtype="float64")
    speeds = pd.to_numeric(seg["speed"], errors="coerce").to_numpy(dtype="float64")
    accels = pd.to_numeric(seg["acceleration"], errors="coerce").to_numpy(dtype="float64")
    pedals = pd.to_numeric(seg["pedal"], errors="coerce").to_numpy(dtype="float64") if "pedal" in seg.columns else np.full(len(seg), np.nan)
    brakes = pd.to_numeric(seg["brake"], errors="coerce").to_numpy(dtype="float64") if "brake" in seg.columns else np.zeros(len(seg))

    jerk = compute_jerk_trace(seg, adj.jerk_smooth_samples).to_numpy(dtype="float64")
    finite_jerk = jerk[np.isfinite(jerk)]
    peak_jerk = float(np.min(finite_jerk)) if finite_jerk.size else None  # most negative = hardest braking jerk
    peak_jerk_abs = float(np.max(np.abs(finite_jerk))) if finite_jerk.size else None
    jerk_above_comfort = int(np.sum(np.abs(finite_jerk) > metric_settings.jerk_comfort_limit_ms3)) if finite_jerk.size else 0

    finite_accel = accels[np.isfinite(accels)]
    peak_decel = float(np.min(finite_accel)) if finite_accel.size else None
    mean_decel = float(np.mean(finite_accel)) if finite_accel.size else None

    # Tip-out transient
    t0 = times[0] if len(times) else 0.0
    rise_80 = None
    if peak_decel is not None:
        target_80 = 0.8 * peak_decel
        rise_80 = _crossing_time(times, accels, target_80, direction="below")
        if rise_80 is not None:
            rise_80 = rise_80 - t0
    rise_target = _crossing_time(times, accels, metric_settings.tip_out_target_decel_ms2, direction="below")
    if rise_target is not None:
        rise_target = rise_target - t0

    early_mask = times <= (t0 + 3.0)
    early_jerk = jerk[early_mask & np.isfinite(jerk)]
    tip_in_peak_jerk = float(np.min(early_jerk)) if early_jerk.size else None

    # Coast / zero-torque point
    zero_torque_speed = _find_zero_accel_speed(speeds, accels)
    ramp_shape = classify_ramp_shape(pedals, accels)

    # Blending
    brake_active = brakes > brake_threshold
    blend_fraction = float(brake_active.sum() / len(brake_active)) if len(brake_active) else 0.0
    first_blend_t = None
    if brake_active.any():
        idx = int(np.argmax(brake_active))
        first_blend_t = float(times[idx] - t0) if len(times) > idx else None
    max_brake = float(np.nanmax(brakes)) if brakes.size else 0.0

    # Stop quality / GB 21670
    speed_end = float(speeds[-1]) if len(speeds) else None
    speed_start = float(speeds[0]) if len(speeds) else None
    stopped = speed_end is not None and speed_end <= metric_settings.stop_speed_threshold_kph
    regen_only_stop = bool(stopped and not brake_active.any())
    gb21670_lamp = bool(peak_decel is not None and peak_decel <= -metric_settings.gb21670_brake_lamp_ms2)

    # Energy recovery
    ke_start = kinetic_energy_j(metric_settings.vehicle_mass_kg, speed_start) if speed_start else None
    ke_end = kinetic_energy_j(metric_settings.vehicle_mass_kg, speed_end) if speed_end else None
    ke_delta = (ke_start - ke_end) if ke_start is not None and ke_end is not None else None
    regen_j = estimate_regen_energy_j(seg)
    recovery_pct = float(100.0 * regen_j / ke_delta) if regen_j and ke_delta and ke_delta > 0 else None

    # Benchmark bands
    band_02 = accels <= metric_settings.decel_band_02g_ms2
    band_05 = accels <= metric_settings.decel_band_05g_ms2
    dt = np.diff(times, prepend=times[0])
    dt = np.where(dt > 0, dt, 0.0)
    time_at_02g = float(np.sum(dt[band_02 & np.isfinite(accels)]))
    time_at_05g = float(np.sum(dt[band_05 & np.isfinite(accels)]))
    dist_m = np.nansum((speeds / 3.6) * dt)
    mfdd_100_50 = compute_mfdd(times, speeds, 100.0, 50.0) if speed_start and speed_start >= 50 else None
    mfdd_80_20 = compute_mfdd(times, speeds, 80.0, 20.0) if speed_start and speed_start >= 20 else None

    speed_intervals = decel_by_speed_interval(seg, metric_settings.speed_interval_kph)
    pedal_map = pedal_decel_map(seg)

    return {
        "peak_decel_ms2": peak_decel,
        "mean_decel_ms2": mean_decel,
        "peak_jerk_ms3": peak_jerk,
        "peak_jerk_abs_ms3": peak_jerk_abs,
        "jerk_comfort_violations": jerk_above_comfort,
        "jerk_comfort_limit_ms3": metric_settings.jerk_comfort_limit_ms3,
        "tip_out_rise_80pct_s": rise_80,
        "tip_out_rise_to_target_s": rise_target,
        "tip_out_target_decel_ms2": metric_settings.tip_out_target_decel_ms2,
        "tip_in_peak_jerk_ms3": tip_in_peak_jerk,
        "zero_torque_speed_kph": zero_torque_speed,
        "ramp_shape": ramp_shape,
        "brake_blend_fraction": blend_fraction,
        "first_brake_blend_s": first_blend_t,
        "max_brake_pct": max_brake,
        "speed_start_kph": speed_start,
        "speed_end_kph": speed_end,
        "stopped_on_regen": regen_only_stop,
        "gb21670_brake_lamp_required": gb21670_lamp,
        "gb21670_threshold_ms2": metric_settings.gb21670_brake_lamp_ms2,
        "recovery_pct": recovery_pct,
        "regen_energy_kwh": (regen_j / 3.6e6) if regen_j else None,
        "kinetic_energy_delta_kj": (ke_delta / 1000.0) if ke_delta else None,
        "time_at_02g_s": time_at_02g,
        "time_at_05g_s": time_at_05g,
        "distance_m": dist_m,
        "mfdd_100_50_ms2": mfdd_100_50,
        "mfdd_80_20_ms2": mfdd_80_20,
        "speed_intervals": speed_intervals,
        "pedal_decel_map": pedal_map,
        "adjustments": adj.as_dict(),
    }


def build_time_series_record(
    file_name: str,
    event_id: int,
    target_pedal: int,
    group_label: str,
    segment: pd.DataFrame,
    adjustments: SensitivityAdjustments | None = None,
) -> pd.DataFrame:
    """Event time-series with first-class jerk for plotting and what-if replay."""

    adj = adjustments or SensitivityAdjustments()
    seg = apply_sensitivity(segment, adj)
    jerk = compute_jerk_trace(seg, adj.jerk_smooth_samples)
    return pd.DataFrame(
        {
            "file_name": file_name,
            "event_id": event_id,
            "target_pedal": target_pedal,
            "group_label": group_label,
            "time": seg["time"].values,
            "speed": seg["speed"].values,
            "acceleration": seg["acceleration"].values,
            "pedal": seg["pedal"].values if "pedal" in seg.columns else np.nan,
            "brake": seg["brake"].values if "brake" in seg.columns else 0.0,
            "jerk": jerk.values,
        }
    )


def recompute_summary_with_adjustments(
    raw_segments: dict[tuple[str, int], pd.DataFrame],
    summary_rows: list[dict[str, object]],
    metric_settings: OpmMetricSettings,
    adjustments: SensitivityAdjustments,
    brake_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-run metrics on stored raw segments when what-if sliders change."""

    updated_rows: list[dict[str, object]] = []
    time_parts: list[pd.DataFrame] = []
    interval_parts: list[pd.DataFrame] = []
    pedal_parts: list[pd.DataFrame] = []

    for row in summary_rows:
        if row.get("status") != "Usable":
            updated_rows.append(row)
            continue
        key = (str(row["file_name"]), int(row["event_id"]))
        segment = raw_segments.get(key)
        if segment is None or segment.empty:
            updated_rows.append(row)
            continue
        metrics = analyze_opm_event(segment, metric_settings, adjustments, brake_threshold)
        merged = {**row, **{k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}}
        updated_rows.append(merged)
        time_parts.append(
            build_time_series_record(
                str(row["file_name"]),
                int(row["event_id"]),
                int(row.get("target_pedal") or 0),
                str(row.get("group_label") or ""),
                segment,
                adjustments,
            )
        )
        si = metrics["speed_intervals"].copy()
        si["file_name"] = row["file_name"]
        si["event_id"] = row["event_id"]
        interval_parts.append(si)
        pm = metrics["pedal_decel_map"].copy()
        pm["file_name"] = row["file_name"]
        pm["event_id"] = row["event_id"]
        pedal_parts.append(pm)

    summary = pd.DataFrame(updated_rows)
    time_series = pd.concat(time_parts, ignore_index=True) if time_parts else pd.DataFrame()
    intervals = pd.concat(interval_parts, ignore_index=True) if interval_parts else pd.DataFrame()
    pedal_maps = pd.concat(pedal_parts, ignore_index=True) if pedal_parts else pd.DataFrame()
    return summary, time_series, intervals, pedal_maps
