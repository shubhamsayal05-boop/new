"""Streamlit UI for the One-Pedal (OPD) Analysis tab."""

from __future__ import annotations

import hashlib
from typing import Callable

import pandas as pd
import streamlit as st

from vehicle_plotter.exporting import export_opd_workbook
from vehicle_plotter.mdf_io import MdfDependencyError, discover_channel_names, read_channel_unit, read_measurement_frame
from vehicle_plotter.one_pedal import (
    OnePedalSettings,
    aggregate_opd_results,
    analyze_one_pedal_file,
    normalize_opd_frame,
    opd_kpi_cards,
)
from vehicle_plotter.opd_metrics import SensitivityAdjustments, recompute_summary_with_adjustments
from vehicle_plotter.plotting import (
    make_opm_blending_figure,
    make_opm_pedal_decel_figure,
    make_opm_speed_interval_figure,
    make_opm_trace_figure,
    make_opd_decel_speed_figure,
    make_opd_jerk_figure,
)
from vehicle_plotter.processing import DECEL_TARGETS, build_group_label
from vehicle_plotter.units import (
    ACCELERATION_UNITS,
    SPEED_UNITS,
    convert_measurement_units,
    convert_speed_value,
    normalize_acceleration_unit,
    normalize_speed_unit,
)

OPD_RESULT_KEYS = [
    "opd_decel_curves",
    "opd_summary",
    "opd_jerk",
    "opd_time_series",
    "opd_speed_intervals",
    "opd_pedal_maps",
    "opd_raw_segments",
    "opd_signature",
    "opd_load_errors",
    "opd_summary_base",
]
OPD_TARGET_OPTIONS = ["Auto"] + [str(value) for value in DECEL_TARGETS]

# Common INCA channel name hints for auto-selection.
_CHANNEL_HINTS: dict[str, tuple[str, ...]] = {
    "speed": ("vehiclespeed", "vehicle_speed", "v_veh"),
    "acceleration": ("accelerationchassis", "acceleration_chassis", "ax"),
    "brake": ("brakeposition", "brake_position", "brake"),
    "pedal": ("acceleratorpedal", "accelerator_pedal", "app"),
    "motor_torque": ("emtorque", "motor_torque", "em_torque"),
    "motor_torque_2": ("em2torque", "em2_torque"),
    "batt_current": ("battcurrent", "battery_current"),
    "batt_voltage": ("battvoltage", "battery_voltage"),
    "batt_soc": ("battsoc", "soc"),
    "road_gradient": ("roadgradient", "gradient"),
}


@st.cache_data(show_spinner=False)
def _cached_discover_uploaded(file_name: str, signature: str, file_bytes: bytes) -> list[str]:
    del file_name, signature
    return discover_channel_names(file_bytes)


@st.cache_data(show_spinner=False)
def _cached_discover_local(file_path: str, signature: str) -> list[str]:
    del signature
    return discover_channel_names(file_path)


@st.cache_data(show_spinner=False)
def _cached_uploaded_unit(file_name: str, signature: str, file_bytes: bytes, channel: str) -> str:
    del file_name, signature
    return read_channel_unit(file_bytes, channel)


@st.cache_data(show_spinner=False)
def _cached_local_unit(file_path: str, signature: str, channel: str) -> str:
    del signature
    return read_channel_unit(file_path, channel)


def _guess_channel(role: str, options: list[str]) -> str:
    hints = _CHANNEL_HINTS.get(role, ())
    lowered = {opt: opt.casefold().replace("_", "").replace(" ", "") for opt in options if opt}
    for hint in hints:
        for option, key in lowered.items():
            if hint in key:
                return option
    return ""


def _signal_selectbox(label: str, key: str, options: list[str], default: str = "") -> str:
    if st.session_state.get(key, "") not in options:
        st.session_state[key] = default if default in options else ""
    return st.selectbox(
        label,
        options,
        key=key,
        format_func=lambda option: "Select signal..." if option == "" else option,
    )


def _unit_from_sources(
    channel: str,
    sources: dict[str, bytes | str],
    signatures: dict[str, str],
) -> str:
    if not channel:
        return ""
    for display_name, source in sources.items():
        try:
            if isinstance(source, bytes):
                return _cached_uploaded_unit(display_name, signatures[display_name], source, channel)
            return _cached_local_unit(source, signatures[display_name], channel)
        except Exception:
            continue
    return ""


def _opd_signature(
    file_signatures: dict[str, str],
    signal_map: dict[str, str],
    settings: OnePedalSettings,
    speed_input: str,
    speed_output: str,
    accel_input: str,
    accel_output: str,
    selected_files: list[str],
) -> str:
    payload = {
        "files": {name: file_signatures[name] for name in selected_files},
        "signals": signal_map,
        "settings": settings.__dict__,
        "units": {
            "speed_in": speed_input,
            "speed_out": speed_output,
            "accel_in": accel_input,
            "accel_out": accel_output,
        },
    }
    import json

    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _serialize_segments(raw: dict[tuple[str, int], pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {f"{name}|{eid}": df for (name, eid), df in raw.items()}


def _deserialize_segments(stored: dict[str, pd.DataFrame]) -> dict[tuple[str, int], pd.DataFrame]:
    out: dict[tuple[str, int], pd.DataFrame] = {}
    for key, df in stored.items():
        name, eid = key.rsplit("|", 1)
        out[(name, int(eid))] = df
    return out


def _apply_what_if(
    summary_base: pd.DataFrame,
    raw_segments: dict[tuple[str, int], pd.DataFrame],
    settings: OnePedalSettings,
    adjustments: SensitivityAdjustments,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_rows = summary_base.to_dict(orient="records")
    summary, time_series, intervals, pedal_maps = recompute_summary_with_adjustments(
        raw_segments,
        base_rows,
        settings.to_metric_settings(),
        adjustments,
        settings.brake_threshold,
    )
    jerk = time_series[["file_name", "event_id", "target_pedal", "group_label", "time", "speed", "acceleration", "jerk"]].copy() if not time_series.empty else pd.DataFrame()
    return summary, time_series, jerk, intervals, pedal_maps


def _render_metric_bucket(title: str, metrics: dict[str, object], keys: list[tuple[str, str]]) -> None:
    st.markdown(f"**{title}**")
    cols = st.columns(min(len(keys), 4))
    for idx, (key, label) in enumerate(keys):
        val = metrics.get(key)
        display = f"{val:.3g}" if isinstance(val, (int, float)) and val is not None else (str(val) if val is not None else "N/A")
        cols[idx % len(cols)].metric(label, display)


    for key in OPD_RESULT_KEYS:
        st.session_state.pop(key, None)


def render_one_pedal_tab(
    measurement_sources: dict[str, bytes | str],
    file_signatures: dict[str, str],
    raster_step_s: float | None,
    section_header: Callable[..., None],
    resettable_key: Callable[[str], str],
) -> None:
    """Render the One-Pedal Analysis workspace (`.mf4` / `.dat` measurement files only)."""

    file_names = list(measurement_sources.keys())
    has_measurements = bool(measurement_sources)

    if not has_measurements:
        st.info(
            "Load `.mf4` or `.dat` measurement files using the sidebar "
            "(local folder or browser upload), then map signals below."
        )
        return

    st.caption(
        "OPD analysis reads the same measurement files loaded in the sidebar. "
        "Only `.mf4` and `.dat` formats are supported for this workflow."
    )

    all_channels: set[str] = set()
    channel_errors: list[str] = []
    with st.spinner("Reading channel lists from measurement files..."):
        for file_name, source in measurement_sources.items():
            try:
                if isinstance(source, bytes):
                    all_channels.update(
                        _cached_discover_uploaded(file_name, file_signatures[file_name], source)
                    )
                else:
                    all_channels.update(_cached_discover_local(source, file_signatures[file_name]))
            except MdfDependencyError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                channel_errors.append(f"{file_name}: {exc}")

    if channel_errors:
        st.warning("Some files could not be inspected:\n\n" + "\n".join(f"- {e}" for e in channel_errors))

    channel_options = sorted(all_channels, key=str.casefold)
    signal_options = [""] + channel_options if channel_options else [""]

    from ui.shared_signals import (
        ensure_shared_defaults,
        persist_shared_core_signals,
        sync_widgets_from_shared,
    )

    defaults = {role: _guess_channel(role, channel_options) for role in ("speed", "acceleration", "brake", "pedal")}
    valid = set(signal_options)
    ensure_shared_defaults(defaults, valid)
    sync_widgets_from_shared("opd", resettable_key, valid)

    section_header("OPD Signal Mapping", "Required channels for event detection", icon="signal")
    core_cols = st.columns(4)
    with core_cols[0]:
        speed_signal = _signal_selectbox("Speed signal", resettable_key("opd_speed_signal"), signal_options, defaults["speed"])
    with core_cols[1]:
        accel_signal = _signal_selectbox(
            "Acceleration signal", resettable_key("opd_accel_signal"), signal_options, defaults["acceleration"]
        )
    with core_cols[2]:
        brake_signal = _signal_selectbox("Brake signal", resettable_key("opd_brake_signal"), signal_options, defaults["brake"])
    with core_cols[3]:
        pedal_signal = _signal_selectbox(
            "Accelerator pedal signal", resettable_key("opd_pedal_signal"), signal_options, defaults["pedal"]
        )

    persist_shared_core_signals(
        "opd",
        resettable_key,
        {
            "speed": speed_signal,
            "acceleration": accel_signal,
            "brake": brake_signal,
            "pedal": pedal_signal,
        },
    )
    if any((speed_signal, accel_signal, brake_signal, pedal_signal)):
        st.caption("Core signals are shared with **Data Source / Speed Analysis** — selections stay in sync.")

    with st.expander("Optional EV signals (energy recovery)", expanded=False):
        opt_cols = st.columns(3)
        with opt_cols[0]:
            motor_signal = _signal_selectbox(
                "Motor torque (optional)", resettable_key("opd_motor_signal"), signal_options, defaults.get("motor_torque", "")
            )
        with opt_cols[1]:
            motor2_signal = _signal_selectbox(
                "Motor 2 torque (optional)", resettable_key("opd_motor2_signal"), signal_options, defaults.get("motor_torque_2", "")
            )
        with opt_cols[2]:
            gradient_signal = _signal_selectbox(
                "Road gradient (optional)", resettable_key("opd_gradient_signal"), signal_options, defaults.get("road_gradient", "")
            )
        batt_cols = st.columns(2)
        with batt_cols[0]:
            batt_current_signal = _signal_selectbox(
                "Battery current (optional)", resettable_key("opd_batt_i_signal"), signal_options, defaults.get("batt_current", "")
            )
        with batt_cols[1]:
            batt_voltage_signal = _signal_selectbox(
                "Battery voltage (optional)", resettable_key("opd_batt_v_signal"), signal_options, defaults.get("batt_voltage", "")
            )

    detected_speed = _unit_from_sources(speed_signal, measurement_sources, file_signatures)
    detected_accel = _unit_from_sources(accel_signal, measurement_sources, file_signatures)
    norm_speed = normalize_speed_unit(detected_speed)
    norm_accel = normalize_acceleration_unit(detected_accel)

    unit_cols = st.columns(4)
    with unit_cols[0]:
        speed_input_unit = st.selectbox(
            "Speed input unit",
            SPEED_UNITS,
            index=SPEED_UNITS.index(norm_speed) if norm_speed in SPEED_UNITS else 0,
            key=resettable_key("opd_speed_input"),
        )
    with unit_cols[1]:
        speed_output_unit = st.selectbox("Speed display unit", ["KPH", "MPH"], index=0, key=resettable_key("opd_speed_output"))
    with unit_cols[2]:
        accel_input_unit = st.selectbox(
            "Acceleration input unit",
            ACCELERATION_UNITS,
            index=ACCELERATION_UNITS.index(norm_accel) if norm_accel in ACCELERATION_UNITS else 0,
            key=resettable_key("opd_accel_input"),
        )
    with unit_cols[3]:
        accel_output_unit = st.selectbox(
            "Acceleration display unit",
            ACCELERATION_UNITS,
            index=0,
            key=resettable_key("opd_accel_output"),
        )

    signal_map = {
        "speed": speed_signal,
        "acceleration": accel_signal,
        "brake": brake_signal,
        "pedal": pedal_signal,
        "motor_torque": motor_signal,
        "motor_torque_2": motor2_signal,
        "batt_current": batt_current_signal,
        "batt_voltage": batt_voltage_signal,
        "road_gradient": gradient_signal,
    }
    selected_channels = tuple(ch for ch in dict.fromkeys(signal_map.values()) if ch)

    section_header("OPD Detection Settings", "Tune event gates and regulatory thresholds", icon="ruler")
    set_cols = st.columns(4)
    with set_cols[0]:
        pedal_tolerance = st.number_input("Pedal tolerance (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5, key=resettable_key("opd_pedal_tol"))
    with set_cols[1]:
        brake_threshold = st.number_input("Brake released threshold", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key=resettable_key("opd_brake_thr"))
    with set_cols[2]:
        speed_bin = st.number_input("Speed bin (display unit)", min_value=0.05, max_value=20.0, value=1.0, step=0.05, key=resettable_key("opd_speed_bin"))
    with set_cols[3]:
        min_duration = st.number_input("Min event duration (s)", min_value=0.5, max_value=120.0, value=3.0, step=0.5, key=resettable_key("opd_min_dur"))

    gate_cols = st.columns(4)
    with gate_cols[0]:
        use_speed_gate = st.checkbox("Require decel start speed", value=True, key=resettable_key("opd_speed_gate"))
    with gate_cols[1]:
        decel_start = st.number_input("Decel start speed (KPH ref)", min_value=0.0, max_value=300.0, value=150.0, step=5.0, key=resettable_key("opd_decel_start"))
    with gate_cols[2]:
        decel_tol = st.number_input("Start speed tolerance (KPH ref)", min_value=0.0, max_value=100.0, value=10.0, step=5.0, key=resettable_key("opd_decel_tol"))
    with gate_cols[3]:
        target_pedal = st.selectbox("Target pedal", OPD_TARGET_OPTIONS, index=0, key=resettable_key("opd_target_pedal"))

    extra_cols = st.columns(3)
    with extra_cols[0]:
        vehicle_mass = st.number_input("Vehicle mass (kg)", min_value=500.0, max_value=10000.0, value=2781.4, step=10.0, key=resettable_key("opd_mass"))
    with extra_cols[1]:
        r13h_threshold = st.number_input("UN R13-H threshold (m/s²)", min_value=0.5, max_value=5.0, value=1.3, step=0.1, key=resettable_key("opd_r13h"))
    with extra_cols[2]:
        min_points = st.number_input("Min points per event", min_value=3, max_value=500, value=10, step=1, key=resettable_key("opd_min_pts"))

    comfort_cols = st.columns(2)
    with comfort_cols[0]:
        jerk_comfort = st.number_input("Jerk comfort ceiling (m/s³)", min_value=0.5, max_value=10.0, value=3.0, step=0.1, key=resettable_key("opd_jerk_comfort"))
    with comfort_cols[1]:
        speed_interval = st.number_input("Speed interval for benchmarking (KPH)", min_value=5.0, max_value=25.0, value=10.0, step=1.0, key=resettable_key("opd_speed_int"))

    settings = OnePedalSettings(
        pedal_tolerance=float(pedal_tolerance),
        brake_threshold=float(brake_threshold),
        decel_start_speed=convert_speed_value(float(decel_start), "KPH", speed_output_unit),
        decel_start_speed_tolerance=convert_speed_value(float(decel_tol), "KPH", speed_output_unit),
        use_decel_speed_gate=bool(use_speed_gate),
        speed_bin=float(speed_bin),
        min_event_duration_s=float(min_duration),
        min_points=int(min_points),
        target_pedal=str(target_pedal),
        r13h_threshold_ms2=float(r13h_threshold),
        vehicle_mass_kg=float(vehicle_mass),
        jerk_comfort_limit_ms3=float(jerk_comfort),
        speed_interval_kph=float(speed_interval),
    )

    selected_files = st.multiselect(
        "Files to analyze",
        file_names,
        default=file_names,
        key=resettable_key("opd_selected_files"),
    )

    run_opd = st.button("Run OPD analysis", type="primary", key=resettable_key("opd_run"))

    current_sig = _opd_signature(
        file_signatures,
        signal_map,
        settings,
        speed_input_unit,
        speed_output_unit,
        accel_input_unit,
        accel_output_unit,
        selected_files,
    )

    if run_opd:
        required = ["speed", "acceleration", "brake", "pedal"]
        missing = [role for role in required if not signal_map.get(role)]
        if missing:
            st.error(f"Select all required signals: {', '.join(missing)}")
            return
        if not selected_files:
            st.warning("Select at least one measurement file.")
            return

        curve_rows: list[pd.DataFrame] = []
        summary_rows: list[dict[str, object]] = []
        jerk_parts: list[pd.DataFrame] = []
        time_parts: list[pd.DataFrame] = []
        interval_parts: list[pd.DataFrame] = []
        pedal_parts: list[pd.DataFrame] = []
        raw_all: dict[tuple[str, int], pd.DataFrame] = {}
        load_errors: list[str] = []

        with st.spinner("Running full OPM/OPD analysis (signatures, jerk, transients, blending, energy)..."):
            for file_name in selected_files:
                source = measurement_sources.get(file_name)
                if source is None:
                    continue
                group_label = Path_stem_label(file_name)
                try:
                    raw = read_measurement_frame(source, selected_channels, raster_step_s)
                    frame = normalize_opd_frame(raw, signal_map)
                    frame = convert_measurement_units(
                        frame,
                        speed_input_unit,
                        speed_output_unit,
                        accel_input_unit,
                        accel_output_unit,
                    )
                    curves, summaries, jerk, time_series, raw_segments, intervals, pedal_maps = analyze_one_pedal_file(
                        file_name, frame, settings, group_label=group_label
                    )
                    curve_rows.extend(curves)
                    summary_rows.extend(summaries)
                    raw_all.update(raw_segments)
                    if not jerk.empty:
                        jerk_parts.append(jerk)
                    if not time_series.empty:
                        time_parts.append(time_series)
                    if not intervals.empty:
                        interval_parts.append(intervals)
                    if not pedal_maps.empty:
                        pedal_parts.append(pedal_maps)
                except MemoryError:
                    load_errors.append(f"{file_name}: insufficient memory — try a larger resample step.")
                except Exception as exc:
                    load_errors.append(f"{file_name}: {exc}")

        decel_curves, summary_table = aggregate_opd_results(curve_rows, summary_rows)
        jerk_traces = pd.concat(jerk_parts, ignore_index=True) if jerk_parts else pd.DataFrame()
        time_series_all = pd.concat(time_parts, ignore_index=True) if time_parts else pd.DataFrame()
        speed_intervals = pd.concat(interval_parts, ignore_index=True) if interval_parts else pd.DataFrame()
        pedal_maps_all = pd.concat(pedal_parts, ignore_index=True) if pedal_parts else pd.DataFrame()

        st.session_state["opd_decel_curves"] = decel_curves
        st.session_state["opd_summary_base"] = summary_table.copy()
        st.session_state["opd_summary"] = summary_table
        st.session_state["opd_jerk"] = jerk_traces
        st.session_state["opd_time_series"] = time_series_all
        st.session_state["opd_speed_intervals"] = speed_intervals
        st.session_state["opd_pedal_maps"] = pedal_maps_all
        st.session_state["opd_raw_segments"] = _serialize_segments(raw_all)
        st.session_state["opd_signature"] = current_sig
        st.session_state["opd_load_errors"] = load_errors
        st.session_state["opd_settings_snapshot"] = settings

    has_results = st.session_state.get("opd_signature") == current_sig and "opd_decel_curves" in st.session_state
    if run_opd and not has_results:
        return

    if not has_results:
        if st.session_state.get("opd_decel_curves") is not None:
            st.warning("Settings or file selection changed. Click **Run OPD analysis** to refresh.")
        else:
            st.info("Configure signals and click **Run OPD analysis**.")
        return

    decel_curves = st.session_state.get("opd_decel_curves", pd.DataFrame())
    summary_table = st.session_state.get("opd_summary", pd.DataFrame())
    jerk_traces = st.session_state.get("opd_jerk", pd.DataFrame())
    time_series = st.session_state.get("opd_time_series", pd.DataFrame())
    speed_intervals = st.session_state.get("opd_speed_intervals", pd.DataFrame())
    pedal_maps = st.session_state.get("opd_pedal_maps", pd.DataFrame())
    load_errors = st.session_state.get("opd_load_errors", [])
    settings_snapshot: OnePedalSettings = st.session_state.get("opd_settings_snapshot", settings)
    raw_stored = st.session_state.get("opd_raw_segments", {})

    section_header("What-if tuning", "Adjust traces live — optimizer can target these parameters later", icon="ruler")
    with st.expander("Sensitivity sliders (preview without re-loading MF4)", expanded=False):
        w_cols = st.columns(4)
        with w_cols[0]:
            accel_scale = st.slider("Acceleration scale", 0.5, 1.5, 1.0, 0.02, key=resettable_key("opd_adj_accel_scale"))
        with w_cols[1]:
            accel_offset = st.slider("Acceleration offset (m/s²)", -1.0, 1.0, 0.0, 0.05, key=resettable_key("opd_adj_accel_off"))
        with w_cols[2]:
            speed_scale = st.slider("Speed scale", 0.9, 1.1, 1.0, 0.01, key=resettable_key("opd_adj_speed_scale"))
        with w_cols[3]:
            pedal_offset = st.slider("Pedal offset (%)", -5.0, 5.0, 0.0, 0.5, key=resettable_key("opd_adj_pedal_off"))
        w_cols2 = st.columns(3)
        with w_cols2[0]:
            jerk_smooth = st.slider("Jerk smoothing (samples)", 1, 15, 1, key=resettable_key("opd_adj_jerk_smooth"))
        with w_cols2[1]:
            speed_off = st.slider("Speed offset (KPH)", -10.0, 10.0, 0.0, 1.0, key=resettable_key("opd_adj_speed_off"))
        with w_cols2[2]:
            st.caption("Changes recompute jerk, transients, and compliance flags instantly.")

        adjustments = SensitivityAdjustments(
            accel_scale=float(accel_scale),
            accel_offset_ms2=float(accel_offset),
            speed_scale=float(speed_scale),
            speed_offset_kph=float(speed_off),
            pedal_offset_pct=float(pedal_offset),
            jerk_smooth_samples=int(jerk_smooth),
        )
        if raw_stored and not st.session_state.get("opd_summary_base", pd.DataFrame()).empty:
            summary_table, time_series, jerk_traces, speed_intervals, pedal_maps = _apply_what_if(
                st.session_state["opd_summary_base"],
                _deserialize_segments(raw_stored),
                settings_snapshot,
                adjustments,
            )
            st.session_state["opd_summary"] = summary_table
            st.session_state["opd_time_series"] = time_series
            st.session_state["opd_jerk"] = jerk_traces
            st.session_state["opd_speed_intervals"] = speed_intervals
            st.session_state["opd_pedal_maps"] = pedal_maps

    if load_errors:
        st.warning("Some files could not be processed:\n\n" + "\n".join(f"- {e}" for e in load_errors))

    usable = summary_table[summary_table.get("status", pd.Series(dtype=str)) == "Usable"] if not summary_table.empty else pd.DataFrame()
    if usable.empty:
        st.warning("No usable OPM events detected. Check pedal target, tolerances, and signal mapping.")
        if not summary_table.empty:
            st.dataframe(summary_table, width="stretch", hide_index=True)
        return

    kpis = opd_kpi_cards(summary_table)
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Events", kpis["event_count"])
    kpi_cols[1].metric("Max decel", f"{kpis['max_decel']:.2f}" if kpis["max_decel"] is not None else "N/A")
    kpi_cols[2].metric("Mean decel", f"{kpis['mean_decel']:.2f}" if kpis["mean_decel"] is not None else "N/A")
    kpi_cols[3].metric("GB21670 / R13-H", kpis["r13h_count"])
    kpi_cols[4].metric("Brake blend", kpis["brake_blend_count"])
    jerk_viol = int(pd.to_numeric(usable.get("jerk_comfort_violations", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not usable.empty else 0
    kpi_cols[5].metric("Jerk > 3 m/s³ samples", jerk_viol)

    tab_sig, tab_jerk, tab_pedal, tab_tip, tab_blend, tab_bench, tab_audit = st.tabs(
        [
            "Decel signature",
            "Jerk & comfort",
            "Pedal mapping",
            "Tip-out transient",
            "Blending & stop",
            "Energy & benchmark",
            "Full audit",
        ]
    )

    with tab_sig:
        st.plotly_chart(
            make_opd_decel_speed_figure(decel_curves, speed_unit=speed_output_unit, acceleration_unit=accel_output_unit, r13h_threshold_ms2=settings.r13h_threshold_ms2),
            width="stretch",
        )
        if not time_series.empty:
            st.plotly_chart(
                make_opm_trace_figure(time_series, "acceleration", "Acceleration vs Time", f"Acceleration ({accel_output_unit})"),
                width="stretch",
            )
        if not speed_intervals.empty:
            st.plotly_chart(
                make_opm_speed_interval_figure(speed_intervals, speed_unit=speed_output_unit, acceleration_unit=accel_output_unit),
                width="stretch",
            )

    with tab_jerk:
        st.caption("Literature comfort ceiling ≈ ±3 m/s³. Engine-brake-like OPD targets ~0.5 m/s³ peak jerk.")
        if not jerk_traces.empty:
            st.plotly_chart(make_opd_jerk_figure(jerk_traces, acceleration_unit=accel_output_unit), width="stretch")
            st.plotly_chart(
                make_opm_trace_figure(time_series, "jerk", "Jerk vs Time (first-class channel)", "Jerk (m/s³)"),
                width="stretch",
            )
        for _, row in usable.iterrows():
            _render_metric_bucket(
                f"Event {row.get('event_id')} — {row.get('file_name')}",
                row.to_dict(),
                [
                    ("peak_jerk_abs", "Peak |jerk| (m/s³)"),
                    ("jerk_comfort_violations", "Samples > comfort"),
                    ("tip_in_peak_jerk_ms3", "Tip-in peak jerk"),
                ],
            )

    with tab_pedal:
        if not pedal_maps.empty:
            st.plotly_chart(make_opm_pedal_decel_figure(pedal_maps, acceleration_unit=accel_output_unit), width="stretch")
        if not time_series.empty:
            st.plotly_chart(
                make_opm_trace_figure(time_series, "acceleration", "Accel vs Time (pedal overlay context)", f"Ax ({accel_output_unit})"),
                width="stretch",
            )
        for _, row in usable.iterrows():
            _render_metric_bucket(
                f"Event {row.get('event_id')}",
                row.to_dict(),
                [
                    ("zero_torque_speed_kph", "Zero-torque speed (KPH)"),
                    ("ramp_shape", "Ramp shape"),
                    ("coast_speed", "Coast speed (KPH)"),
                ],
            )

    with tab_tip:
        for _, row in usable.iterrows():
            _render_metric_bucket(
                f"Event {row.get('event_id')} — tip-out",
                row.to_dict(),
                [
                    ("tip_out_rise_80pct_s", "Rise to 80% peak (s)"),
                    ("tip_out_rise_to_target_s", "Rise to −1 m/s² (s)"),
                    ("ax_min", "Peak decel (m/s²)"),
                    ("tip_in_peak_jerk_ms3", "Tip-in jerk (m/s³)"),
                ],
            )

    with tab_blend:
        if not time_series.empty:
            st.plotly_chart(make_opm_blending_figure(time_series), width="stretch")
        for _, row in usable.iterrows():
            _render_metric_bucket(
                f"Event {row.get('event_id')} — blending / stop",
                row.to_dict(),
                [
                    ("brake_blend_fraction", "Brake active fraction"),
                    ("first_brake_blend_s", "First blend (s)"),
                    ("stopped_on_regen", "Regen-only stop"),
                    ("gb21670_brake_lamp_required", "GB21670 lamp req."),
                ],
            )

    with tab_bench:
        for _, row in usable.iterrows():
            _render_metric_bucket(
                f"Event {row.get('event_id')} — benchmark",
                row.to_dict(),
                [
                    ("time_at_02g_s", "Time at 0.2g (s)"),
                    ("time_at_05g_s", "Time at 0.5g (s)"),
                    ("mfdd_100_50_ms2", "MFDD 100→50"),
                    ("mfdd_80_20_ms2", "MFDD 80→20"),
                    ("recovery_pct", "Recovery (%)"),
                    ("distance_m", "Distance (m)"),
                ],
            )

    with tab_audit:
        st.dataframe(summary_table, width="stretch", hide_index=True)

    export_cols = st.columns(2)
    with export_cols[0]:
        try:
            st.download_button(
                "Download OPM Excel workbook",
                data=export_opd_workbook(
                    decel_curves,
                    summary_table,
                    jerk_traces,
                    time_series,
                    speed_intervals,
                    pedal_maps,
                    speed_unit=speed_output_unit,
                    acceleration_unit=accel_output_unit,
                ),
                file_name="opm_opd_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.warning("Install openpyxl to enable Excel export.")
    with export_cols[1]:
        st.download_button(
            "Download full metrics CSV",
            data=summary_table.to_csv(index=False),
            file_name="opm_event_metrics.csv",
            mime="text/csv",
        )


def Path_stem_label(file_name: str) -> str:
    from pathlib import Path

    return build_group_label("", Path(file_name).stem)
