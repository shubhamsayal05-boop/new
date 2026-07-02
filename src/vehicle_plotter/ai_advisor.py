"""Diagnostic engine and optional LLM enhancement for measurement workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class DiagnosticIssue:
    """A single detected workflow or data-quality issue."""

    severity: str
    category: str
    title: str
    description: str
    suggestions: list[str] = field(default_factory=list)
    related_page: str = "Dashboard"


@dataclass
class AdvisorReport:
    """Structured output from the advisor engine."""

    issues: list[DiagnosticIssue] = field(default_factory=list)
    summary: str = ""
    health_score: int = 100
    ai_narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "health_score": self.health_score,
            "ai_narrative": self.ai_narrative,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _issue(
    severity: str,
    category: str,
    title: str,
    description: str,
    suggestions: list[str],
    page: str = "Dashboard",
) -> DiagnosticIssue:
    return DiagnosticIssue(
        severity=severity,
        category=category,
        title=title,
        description=description,
        suggestions=suggestions,
        related_page=page,
    )


def collect_diagnostics(context: dict[str, Any]) -> AdvisorReport:
    """Inspect session context and return ranked diagnostic issues."""

    issues: list[DiagnosticIssue] = []

    has_measurements = bool(context.get("has_measurements"))
    file_count = int(context.get("file_count") or 0)
    signal_map: dict[str, str] = context.get("signal_map") or {}
    load_errors: list[str] = list(context.get("load_errors") or [])
    opd_load_errors: list[str] = list(context.get("opd_load_errors") or [])
    audit_raw = context.get("audit_table")
    audit_table: pd.DataFrame = audit_raw if isinstance(audit_raw, pd.DataFrame) else pd.DataFrame()
    opd_raw = context.get("opd_summary")
    opd_summary: pd.DataFrame = opd_raw if isinstance(opd_raw, pd.DataFrame) else pd.DataFrame()
    has_generated = bool(context.get("has_generated_measurements"))
    has_opd_results = bool(context.get("has_opd_results"))
    results_stale = bool(context.get("results_stale"))
    opd_stale = bool(context.get("opd_stale"))
    mdf_dependency_missing = bool(context.get("mdf_dependency_missing"))
    active_page = str(context.get("active_page") or "")

    if mdf_dependency_missing:
        issues.append(
            _issue(
                "critical",
                "Dependencies",
                "MDF reader not installed",
                "The asammdf package is required to read .mf4 and .dat INCA files.",
                [
                    "Run: pip install -r requirements.txt",
                    "Restart the Streamlit app after installing dependencies.",
                ],
                "Data Source",
            )
        )

    if not has_measurements:
        issues.append(
            _issue(
                "warning",
                "Data",
                "No measurement files loaded",
                "Speed vs Acceleration and OPD workflows need at least one .mf4 or .dat file.",
                [
                    "Use **Read from local folder** for files larger than 200 MB.",
                    "Point the folder path to your INCA export directory.",
                    "You can still import target reference curves without measurement files.",
                ],
                "Data Source",
            )
        )
    elif file_count == 1:
        issues.append(
            _issue(
                "info",
                "Data",
                "Single file loaded",
                "Only one measurement file is in the session.",
                [
                    "Load additional runs to compare repeatability across files.",
                    "Use bulk labeling to apply consistent test type and mode labels.",
                ],
                "Data Source",
            )
        )

    if has_measurements:
        for role, label in (
            ("speed", "Speed"),
            ("acceleration", "Acceleration"),
            ("brake", "Brake"),
            ("pedal", "Accelerator pedal"),
        ):
            if not signal_map.get(role):
                issues.append(
                    _issue(
                        "critical",
                        "Signals",
                        f"{label} signal not mapped",
                        f"The {label.lower()} channel is required before processing measured data.",
                        [
                            f"Open **Data Source** and select the INCA channel for {label}.",
                            "Common names: VehicleSpeed, AccelerationChassis, BrakePosition, AcceleratorPedal.",
                            "Run **AI Advisor → Scan session** after mapping to confirm readiness.",
                        ],
                        "Data Source",
                    )
                )

    for err in load_errors:
        lowered = err.casefold()
        if "memory" in lowered:
            issues.append(
                _issue(
                    "critical",
                    "Processing",
                    "Memory error during file processing",
                    err,
                    [
                        "Set an MDF resample step (e.g. 0.01 or 0.1 s) in Processing settings.",
                        "Switch from browser upload to **Read from local folder**.",
                        "Process fewer files per batch.",
                    ],
                    "Speed Analysis",
                )
            )
        elif "channel" in lowered or "found" in lowered:
            issues.append(
                _issue(
                    "warning",
                    "Signals",
                    "Channel read failure",
                    err,
                    [
                        "Verify the selected signal exists in this file's channel list.",
                        "Re-map signals on the Data Source page.",
                    ],
                    "Data Source",
                )
            )
        else:
            issues.append(
                _issue(
                    "warning",
                    "Processing",
                    "File processing error",
                    err,
                    ["Check file integrity.", "Confirm the file is a valid INCA .mf4 or .dat export."],
                    "Speed Analysis",
                )
            )

    if results_stale and has_measurements:
        issues.append(
            _issue(
                "warning",
                "Workflow",
                "Speed analysis results are out of date",
                "Signals, labels, or processing settings changed since the last plot generation.",
                [
                    "Go to **Speed Analysis** and click **Generate plot**.",
                    "Review file labels and pedal targets before regenerating.",
                ],
                "Speed Analysis",
            )
        )

    if has_measurements and not has_generated and not results_stale:
        missing = [r for r in ("speed", "acceleration", "brake", "pedal") if not signal_map.get(r)]
        if not missing:
            issues.append(
                _issue(
                    "info",
                    "Workflow",
                    "Ready to generate Speed vs Acceleration plot",
                    "Files and signals are configured but no plot has been generated yet.",
                    [
                        "Label files as Launch or Deceleration on the Speed Analysis page.",
                        "Click **Generate plot** to extract usable segments.",
                    ],
                    "Speed Analysis",
                )
            )

    if not audit_table.empty and "status" in audit_table.columns:
        failed = audit_table[audit_table["status"].astype(str) != "Usable"]
        for _, row in failed.iterrows():
            status = str(row.get("status") or "")
            file_name = str(row.get("file_name") or "Unknown file")
            target = row.get("target_pedal")
            suggestions = _audit_suggestions(status)
            issues.append(
                _issue(
                    "warning" if "minimum" in status.casefold() else "warning",
                    "Audit",
                    f"No usable data: {file_name}",
                    f"Target {target}: {status}" if pd.notna(target) else status,
                    suggestions,
                    "Speed Analysis",
                )
            )

    if has_generated and not audit_table.empty:
        usable = audit_table[audit_table.get("status", pd.Series(dtype=str)) == "Usable"]
        if usable.empty:
            issues.append(
                _issue(
                    "critical",
                    "Audit",
                    "Zero usable curves extracted",
                    "Processing completed but no curves passed the usable-data gates.",
                    [
                        "Increase pedal tolerance (e.g. 2–3%) if pedal trace is noisy.",
                        "For decel tests, verify decel start speed gate matches your test (default 150 KPH).",
                        "Confirm test type labels (Launch vs Deceleration) match the maneuver.",
                        "For 0% creep launch, set target pedal to 0 explicitly.",
                    ],
                    "Speed Analysis",
                )
            )

    for err in opd_load_errors:
        issues.append(
            _issue(
                "warning",
                "OPD",
                "OPD file processing error",
                err,
                ["Check OPD signal mapping.", "Try a coarser resample step for large files."],
                "OPD Analysis",
            )
        )

    if opd_stale:
        issues.append(
            _issue(
                "warning",
                "Workflow",
                "OPD results are out of date",
                "OPD settings or file selection changed since the last OPD run.",
                ["Open **OPD Analysis** and click **Run OPD analysis**."],
                "OPD Analysis",
            )
        )

    if not opd_summary.empty and "status" in opd_summary.columns:
        opd_failed = opd_summary[opd_summary["status"].astype(str) != "Usable"]
        for _, row in opd_failed.iterrows():
            status = str(row.get("status") or "")
            file_name = str(row.get("file_name") or "Unknown file")
            issues.append(
                _issue(
                    "warning",
                    "OPD",
                    f"OPD issue: {file_name}",
                    status,
                    _opd_suggestions(status),
                    "OPD Analysis",
                )
            )

    if not opd_summary.empty and has_opd_results:
        usable_opd = opd_summary[opd_summary.get("status", pd.Series(dtype=str)) == "Usable"]
        if not usable_opd.empty:
            r13h = usable_opd.get("r13h_violation")
            if r13h is not None and bool(r13h.fillna(False).any()):
                count = int(r13h.fillna(False).sum())
                issues.append(
                    _issue(
                        "warning",
                        "Regulatory",
                        f"UN R13-H brake-lamp threshold exceeded ({count} event(s))",
                        "Regenerative deceleration exceeded 1.3 m/s² during at least one OPD event.",
                        [
                            "Review decel-vs-speed curves where acceleration drops below −1.3 m/s².",
                            "Coordinate with controls team on regen torque ramp limits.",
                            "Verify brake-lamp illumination logic in the vehicle calibration.",
                        ],
                        "OPD Analysis",
                    )
                )
            blend = usable_opd.get("brake_blending")
            if blend is not None and bool(blend.fillna(False).any()):
                issues.append(
                    _issue(
                        "info",
                        "OPD",
                        "Brake blending detected during OPD events",
                        "Brake position rose above threshold while pedal was held in the OPD band.",
                        [
                            "Check if friction brake was applied unintentionally.",
                            "Increase brake released threshold slightly if sensor noise is present.",
                        ],
                        "OPD Analysis",
                    )
                )

    issues.sort(key=lambda item: (SEVERITY_ORDER.get(item.severity, 9), item.category, item.title))
    health = _health_score(issues)
    summary = _build_summary(issues, health, active_page)
    return AdvisorReport(issues=issues, summary=summary, health_score=health)


def _audit_suggestions(status: str) -> list[str]:
    text = status.casefold()
    if "minimum" in text or "below minimum" in text:
        return [
            "Lower **Minimum usable points** in processing settings.",
            "Widen pedal tolerance or verify the pedal held steady during the maneuver.",
            "Check that speed bin is not too coarse for short segments.",
        ]
    if "no usable segments" in text:
        return [
            "Verify target pedal matches the recorded pedal trace.",
            "For decel, confirm the maneuver starts near the configured start speed.",
            "Ensure brake stays below the applied threshold during the segment.",
        ]
    if "target pedal" in text:
        return ["Set target pedal explicitly instead of Auto.", "Inspect the accelerator pedal channel scaling."]
    if "no usable speed" in text:
        return ["Re-map speed and acceleration channels.", "Confirm unit conversion settings are correct."]
    return ["Review signal mapping, labels, and processing tolerances on the Speed Analysis page."]


def _opd_suggestions(status: str) -> list[str]:
    text = status.casefold()
    if "no usable opd" in text:
        return [
            "Set target pedal to the recorded lift-off level (e.g. 10%).",
            "Ensure speed is decreasing and brake is released during the event.",
            "Reduce minimum event duration if the decel window is short.",
        ]
    if "target pedal" in text:
        return ["Use Auto or match the dominant pedal level in the trace."]
    return ["Review OPD signal mapping and detection settings."]


def _health_score(issues: list[DiagnosticIssue]) -> int:
    score = 100
    for issue in issues:
        if issue.severity == "critical":
            score -= 25
        elif issue.severity == "warning":
            score -= 10
        else:
            score -= 3
    return max(0, min(100, score))


def _build_summary(issues: list[DiagnosticIssue], health: int, active_page: str) -> str:
    if not issues:
        return "Session looks healthy. All configured workflows are ready."
    critical = sum(1 for i in issues if i.severity == "critical")
    warnings = sum(1 for i in issues if i.severity == "warning")
    parts = [f"Health score {health}/100."]
    if critical:
        parts.append(f"{critical} critical issue(s) need attention.")
    if warnings:
        parts.append(f"{warnings} warning(s) detected.")
    if active_page:
        page_issues = [i for i in issues if i.related_page == active_page]
        if page_issues:
            parts.append(f"Current page ({active_page}): {page_issues[0].title}.")
    return " ".join(parts)


def enhance_with_copilot(
    report: AdvisorReport,
    context: dict[str, Any],
    config: "CopilotConfig | None" = None,
) -> str:
    """Optional Microsoft Copilot (Azure OpenAI) enhancement; returns narrative text."""

    from vehicle_plotter.copilot_client import CopilotConfig, complete_with_copilot, load_copilot_config

    cfg = config or load_copilot_config()
    payload = {
        "health_score": report.health_score,
        "summary": report.summary,
        "issues": [asdict(i) for i in report.issues[:12]],
        "context": {
            k: v
            for k, v in context.items()
            if k
            in {
                "file_count",
                "has_measurements",
                "has_generated_measurements",
                "has_opd_results",
                "active_page",
                "load_method",
                "signal_map",
            }
        },
    }
    system_prompt = (
        "You are Microsoft Copilot assisting vehicle performance engineers with INCA MF4/DAT "
        "measurement analysis in DriveLab Pro."
    )
    user_prompt = (
        "Given the diagnostic JSON below, write a concise action plan (max 250 words) with numbered steps. "
        "Prioritize critical issues first. Be specific about signal names, tolerances, and which app "
        "workspace to open (Data Source, Speed Analysis, OPD Analysis). Do not invent data not in the JSON.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    return complete_with_copilot(system_prompt, user_prompt, cfg)


def enhance_with_llm(
    report: AdvisorReport,
    context: dict[str, Any],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Deprecated: use :func:`enhance_with_copilot` with Azure OpenAI configuration."""

    del api_key, model
    return enhance_with_copilot(report, context)
