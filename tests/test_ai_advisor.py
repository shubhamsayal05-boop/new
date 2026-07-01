"""Tests for AI advisor diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_plotter.ai_advisor import collect_diagnostics


def test_no_files_warning():
    report = collect_diagnostics({"has_measurements": False, "file_count": 0})
    assert report.health_score < 100
    assert any("No measurement files" in i.title for i in report.issues)


def test_missing_signal_critical():
    report = collect_diagnostics(
        {
            "has_measurements": True,
            "file_count": 1,
            "signal_map": {"speed": "VehicleSpeed", "acceleration": "", "brake": "Brake", "pedal": "Pedal"},
        }
    )
    assert any(i.severity == "critical" for i in report.issues)


def test_audit_failure_suggestions():
    audit = pd.DataFrame(
        [{"file_name": "a.mf4", "status": "No usable segments for this target", "target_pedal": 10}]
    )
    report = collect_diagnostics({"has_measurements": True, "file_count": 1, "audit_table": audit, "has_generated_measurements": True})
    assert any(i.category == "Audit" for i in report.issues)
