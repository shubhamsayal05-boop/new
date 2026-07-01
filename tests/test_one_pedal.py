"""Tests for one-pedal deceleration analytics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_plotter.one_pedal import OnePedalSettings, analyze_one_pedal_file, opd_kpi_cards


def _synthetic_opd_frame(duration_s: float = 30.0, sample_hz: float = 10.0) -> pd.DataFrame:
    times = np.arange(0, duration_s, 1.0 / sample_hz)
    speed = np.clip(120.0 - 3.5 * times, 5.0, None)
    speed_ms = speed / 3.6
    accel = np.gradient(speed_ms, times)
    return pd.DataFrame(
        {
            "time": times,
            "speed": speed,
            "acceleration": accel,
            "pedal": 10.0,
            "brake": 0.0,
        }
    )


def test_detects_opd_event():
    frame = _synthetic_opd_frame()
    settings = OnePedalSettings(target_pedal="10", min_event_duration_s=2.0, use_decel_speed_gate=False)
    curves, summaries, jerk = analyze_one_pedal_file("synthetic.mf4", frame, settings)
    usable = [row for row in summaries if row.get("status") == "Usable"]
    assert len(usable) >= 1
    assert curves
    assert not jerk.empty


def test_r13h_flag_when_strong_decel():
    frame = _synthetic_opd_frame()
    frame["acceleration"] = -1.5
    settings = OnePedalSettings(
        target_pedal="10",
        min_event_duration_s=2.0,
        use_decel_speed_gate=False,
        r13h_threshold_ms2=1.3,
    )
    _, summaries, _ = analyze_one_pedal_file("strong.mf4", frame, settings)
    usable = [row for row in summaries if row.get("status") == "Usable"]
    assert usable and usable[0]["r13h_violation"] is True


def test_kpi_cards_empty():
    kpis = opd_kpi_cards(pd.DataFrame())
    assert kpis["event_count"] == 0
