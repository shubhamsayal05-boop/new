"""Tests for extended OPM metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_plotter.opd_metrics import OpmMetricSettings, SensitivityAdjustments, analyze_opm_event, apply_sensitivity
from vehicle_plotter.one_pedal import OnePedalSettings, analyze_one_pedal_file


def _segment() -> pd.DataFrame:
    t = np.linspace(0, 20, 200)
    speed = np.clip(100 - 4 * t, 2, None)
    accel = np.gradient(speed / 3.6, t)
    return pd.DataFrame({"time": t, "speed": speed, "acceleration": accel, "pedal": 10.0, "brake": 0.0})


def test_jerk_computed_first_class():
    seg = _segment()
    m = analyze_opm_event(seg, OpmMetricSettings())
    assert m["peak_jerk_abs_ms3"] is not None
    assert m["peak_jerk_ms3"] is not None


def test_sensitivity_scales_accel():
    seg = _segment()
    adj = SensitivityAdjustments(accel_scale=1.2)
    scaled = apply_sensitivity(seg, adj)
    assert abs(float(scaled["acceleration"].mean())) > abs(float(seg["acceleration"].mean())) * 1.05


def test_analyze_returns_extended_metrics():
    frame = _segment()
    settings = OnePedalSettings(target_pedal="10", use_decel_speed_gate=False, min_event_duration_s=2.0)
    curves, summaries, jerk, ts, raw, intervals, pedals = analyze_one_pedal_file("t.mf4", frame, settings)
    usable = [s for s in summaries if s.get("status") == "Usable"]
    assert usable
    assert "ramp_shape" in usable[0]
    assert not ts.empty
    assert "jerk" in ts.columns
