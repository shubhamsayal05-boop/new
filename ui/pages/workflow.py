"""Analysis workflow sections split by DriveLab Pro navigation page."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from ui.app_helpers import *
from ui.shared_signals import get_shared_signals
from vehicle_plotter.exporting import export_plot_workbook
from vehicle_plotter.metrics import (
    DEFAULT_SPEED_BANDS,
    average_band_long,
    band_key,
    band_label,
    calculate_metrics,
    export_metrics_workbook,
    format_metrics_table,
    make_average_band_figure,
    make_metric_figure,
)
from vehicle_plotter.mdf_io import MdfDependencyError, read_measurement_frame
from vehicle_plotter.plotting import make_random_scatter_figure, make_speed_accel_figure
from vehicle_plotter.processing import (
    ProcessingSettings,
    average_file_curves,
    normalize_measurement_frame,
    process_measurement_frame,
)
from vehicle_plotter.random_scatter import (
    ParsedRandomScatterData,
    SCATTER_LAYOUTS,
    average_selected_series,
    export_random_scatter_workbook,
    offset_selected_series,
    parse_pasted_scatter_text,
    parse_scatter_file,
)
from vehicle_plotter.target_data import (
    ParsedTargetData,
    TARGET_LAYOUTS,
    convert_target_curves,
    parse_pasted_target_text,
    parse_target_file,
)
from vehicle_plotter.trendlines import (
    TRENDLINE_DEGREES,
    add_trendline_traces,
    build_trendline_data,
    export_trendline_workbook,
    make_trendline_figure,
    trendline_equations,
)
from vehicle_plotter.units import (
    ACCELERATION_UNITS,
    SPEED_UNITS,
    convert_measurement_units,
    convert_speed_value,
    normalize_acceleration_unit,
    normalize_speed_unit,
)


def render_analysis_sections(
    active_page: str,
    *,
    measurement_sources: dict,
    file_signatures: dict,
    file_names: list,
    has_measurements: bool,
    metadata: pd.DataFrame,
    settings: ProcessingSettings,
    raster_step_s: float | None,
    decel_start_speed: float,
    decel_start_speed_tolerance: float,
    section_header,
    resettable_key,
) -> dict:
    speed_signal = accel_signal = brake_signal = pedal_signal = ""
    detected_speed_unit = detected_accel_unit = ""
    speed_input_unit = "KPH"
    acceleration_input_unit = "m/s^2"
    speed_output_unit = "KPH"
    acceleration_output_unit = "m/s^2"
    signal_map: dict = {}
    selected_channels: tuple = ()
    display_units: dict = {}
    target_curves = pd.DataFrame()
    scatter_curves = pd.DataFrame()
    target_parse_errors: list = []
    scatter_parse_errors: list = []
    run_clicked = False
    save_labels = False

    if active_page in ("Data Source", "Speed Analysis", "Reference Data"):
        speed_signal = ""
        accel_signal = ""
        brake_signal = ""
        pedal_signal = ""
        detected_speed_unit = ""
        detected_accel_unit = ""
        speed_input_unit = "KPH"
        acceleration_input_unit = "m/s^2"

        section_header(
            "Units & Conversion",
            "Confirm detected units and choose the graph/export units",
            icon="ruler",
        )
        if has_measurements:
            section_header(
                "Signal Selection",
                "Map the four required measurement channels",
                icon="signal",
            )
            channel_errors: list[str] = []
            all_channels: set[str] = set()

            with st.spinner("Reading channel lists from measurement files..."):
                for file_name, source in measurement_sources.items():
                    try:
                        if isinstance(source, bytes):
                            all_channels.update(
                                cached_discover_uploaded_channels(file_name, file_signatures[file_name], source)
                            )
                        else:
                            all_channels.update(cached_discover_local_channels(source, file_signatures[file_name]))
                    except MdfDependencyError as exc:
                        st.error(str(exc))
                        st.stop()
                    except Exception as exc:
                        channel_errors.append(f"{file_name}: {exc}")

            if channel_errors:
                st.warning("Some files could not be inspected:\n\n" + "\n".join(f"- {err}" for err in channel_errors))

            channel_options = sorted(all_channels, key=str.casefold)
            if not channel_options:
                st.warning("No channel names were found. You can still plot imported target data.")
                signal_options = [""]
            else:
                signal_options = [""] + channel_options

            from ui.shared_signals import (
                ensure_shared_defaults,
                persist_shared_core_signals,
                sync_widgets_from_shared,
            )

            valid = set(signal_options)
            auto_defaults = {
                "speed": next((o for o in channel_options if "vehiclespeed" in o.casefold().replace("_", "")), ""),
                "acceleration": next(
                    (o for o in channel_options if "accelerationchassis" in o.casefold().replace("_", "")), ""
                ),
                "brake": next((o for o in channel_options if "brakeposition" in o.casefold().replace("_", "")), ""),
                "pedal": next((o for o in channel_options if "acceleratorpedal" in o.casefold().replace("_", "")), ""),
            }
            ensure_shared_defaults(auto_defaults, valid)
            sync_widgets_from_shared("speed", resettable_key, valid)

            signal_cols = st.columns(4)
            with signal_cols[0]:
                speed_signal = signal_selectbox("Speed signal", "speed_signal", signal_options)
            with signal_cols[1]:
                accel_signal = signal_selectbox("Acceleration signal", "accel_signal", signal_options)
            with signal_cols[2]:
                brake_signal = signal_selectbox("Brake signal", "brake_signal", signal_options)
            with signal_cols[3]:
                pedal_signal = signal_selectbox("Accelerator pedal signal", "pedal_signal", signal_options)

            persist_shared_core_signals(
                "speed",
                resettable_key,
                {
                    "speed": speed_signal,
                    "acceleration": accel_signal,
                    "brake": brake_signal,
                    "pedal": pedal_signal,
                },
            )
            if any((speed_signal, accel_signal, brake_signal, pedal_signal)):
                st.caption("Core signals are shared with **OPD Analysis** — no need to re-select there.")

            detected_speed_unit = unit_from_sources(speed_signal, measurement_sources, file_signatures)
            detected_accel_unit = unit_from_sources(accel_signal, measurement_sources, file_signatures)
            normalized_speed_unit = normalize_speed_unit(detected_speed_unit)
            normalized_accel_unit = normalize_acceleration_unit(detected_accel_unit)
            st.caption(
                "Detected from selected signals: "
                f"Speed = `{detected_speed_unit or 'not available'}`, "
                f"Acceleration = `{detected_accel_unit or 'not available'}`. "
                "Confirm the input unit if the MDF unit is missing or not recognized."
            )

            unit_cols = st.columns(4)
            with unit_cols[0]:
                speed_input_unit = st.selectbox(
                    "Speed input unit",
                    SPEED_UNITS,
                    index=SPEED_UNITS.index(normalized_speed_unit) if normalized_speed_unit in SPEED_UNITS else 0,
                    key=f"speed_input_{speed_signal}_{detected_speed_unit}",
                )
            with unit_cols[1]:
                speed_output_unit = st.selectbox(
                    "Speed graph/export unit",
                    ["KPH", "MPH"],
                    index=0,
                    key=f"speed_output_{speed_signal}",
                )
            with unit_cols[2]:
                acceleration_input_unit = st.selectbox(
                    "Acceleration input unit",
                    ACCELERATION_UNITS,
                    index=ACCELERATION_UNITS.index(normalized_accel_unit) if normalized_accel_unit in ACCELERATION_UNITS else 0,
                    key=f"accel_input_{accel_signal}_{detected_accel_unit}",
                )
            with unit_cols[3]:
                acceleration_output_unit = st.selectbox(
                    "Acceleration graph/export unit",
                    ACCELERATION_UNITS,
                    index=0,
                    key=f"accel_output_{accel_signal}",
                )
        else:
            st.caption("No INCA/MDF files loaded. Select output units for target-only plotting.")
            unit_cols = st.columns(2)
            with unit_cols[0]:
                speed_output_unit = st.selectbox(
                    "Speed graph/export unit",
                    ["KPH", "MPH"],
                    index=0,
                    key=resettable_key("target_only_speed_output"),
                )
            with unit_cols[1]:
                acceleration_output_unit = st.selectbox(
                    "Acceleration graph/export unit",
                    ACCELERATION_UNITS,
                    index=0,
                    key=resettable_key("target_only_acceleration_output"),
                )
            speed_input_unit = speed_output_unit
            acceleration_input_unit = acceleration_output_unit

        signal_map = {
            "speed": speed_signal,
            "acceleration": accel_signal,
            "brake": brake_signal,
            "pedal": pedal_signal,
        }
        selected_channels = tuple(channel for channel in dict.fromkeys(signal_map.values()) if channel)
        display_units = {
            "detected_speed": detected_speed_unit,
            "detected_acceleration": detected_accel_unit,
            "speed_input": speed_input_unit,
            "speed_output": speed_output_unit,
            "acceleration_input": acceleration_input_unit,
            "acceleration_output": acceleration_output_unit,
        }
        settings = replace(
            settings,
            decel_start_speed=convert_speed_value(
                float(decel_start_speed),
                "KPH",
                speed_output_unit,
            ),
            decel_start_speed_tolerance=convert_speed_value(
                float(decel_start_speed_tolerance),
                "KPH",
                speed_output_unit,
            ),
        )

        target_curves = pd.DataFrame()
        target_parse_errors: list[str] = []


    if active_page == "Reference Data":
        section_header(
            "Target Data",
            "Overlay reference Speed vs Acceleration curves (optional)",
            icon="target",
        )
        with st.expander("Import or paste target Speed vs Acceleration data", expanded=False):
            st.write(
                "Use this when target/reference curves are available in Excel, CSV, "
                "text files, or copied directly from Excel."
            )
            target_layout = st.selectbox(
                "Target data layout",
                TARGET_LAYOUTS,
                index=1,
                help=(
                    "The default is one Speed column with multiple Acceleration columns. "
                    "Choose paired layout when the sheet is x1/y1, x2/y2, x3/y3, etc."
                ),
                key=resettable_key("target_data_layout"),
            )
            target_files = st.file_uploader(
                "Target files (.xlsx, .xlsm, .xls, .csv, .txt)",
                type=["xlsx", "xlsm", "xls", "csv", "txt"],
                accept_multiple_files=True,
                key=resettable_key("target_data_files"),
            )
            pasted_target_text = st.text_area(
                "Paste target data from Excel",
                height=140,
                placeholder=(
                    "Speed (KPH)\t5%\t10%\t15%\n"
                    "0\t0.00\t0.00\t0.00\n"
                    "10\t0.12\t0.20\t0.28"
                ),
                key=resettable_key("target_data_paste"),
            )

            target_source_payloads: dict[str, bytes | str] = {}
            target_source_signatures: dict[str, str] = {}
            seen_target_names: dict[str, int] = {}
            for target_file in target_files or []:
                original_name = target_file.name
                seen_target_names[original_name] = seen_target_names.get(original_name, 0) + 1
                if seen_target_names[original_name] == 1:
                    display_name = original_name
                else:
                    path = Path(original_name)
                    display_name = f"{path.stem} ({seen_target_names[original_name]}){path.suffix}"
                file_bytes = target_file.getvalue()
                target_source_payloads[display_name] = file_bytes
                target_source_signatures[display_name] = content_hash(file_bytes)

            if pasted_target_text.strip():
                target_source_payloads["Pasted target data"] = pasted_target_text
                target_source_signatures["Pasted target data"] = content_hash(
                    pasted_target_text.encode("utf-8")
                )

            parsed_targets: list[ParsedTargetData] = []
            target_sources_present = bool(target_source_payloads)
            if target_sources_present:
                target_metadata = metadata_for_current_targets(
                    list(target_source_payloads.keys()),
                    tuple(target_source_signatures.items()),
                )
                st.write("Series Labels")
                edited_target_metadata = st.data_editor(
                    target_metadata,
                    key=f"target_metadata_editor_{st.session_state.get('target_metadata_editor_version', 0)}",
                    width="stretch",
                    hide_index=True,
                    num_rows="fixed",
                    column_config={
                        "source_name": st.column_config.TextColumn("Source", disabled=True),
                        "series_label": st.column_config.TextColumn("Series label"),
                    },
                )
                if not edited_target_metadata.equals(target_metadata):
                    for key in PLOT_FILTER_KEYS:
                        st.session_state.pop(key, None)
                st.session_state["target_metadata"] = edited_target_metadata.copy()
                target_labels_by_source = {
                    str(row["source_name"]): str(row.get("series_label") or "Target")
                    for _, row in edited_target_metadata.iterrows()
                }

                for source_name, payload in target_source_payloads.items():
                    series_label = target_labels_by_source.get(source_name, "Target")
                    try:
                        if isinstance(payload, bytes):
                            parsed_targets.append(
                                parse_target_file(source_name, payload, series_label, target_layout)
                            )
                        else:
                            parsed_targets.append(
                                parse_pasted_target_text(payload, series_label, target_layout)
                            )
                    except Exception as exc:
                        target_parse_errors.append(f"{source_name}: {exc}")

                detected_target_speed_unit = next(
                    (target.detected_speed_unit for target in parsed_targets if target.detected_speed_unit),
                    None,
                )
                detected_target_accel_unit = next(
                    (
                        target.detected_acceleration_unit
                        for target in parsed_targets
                        if target.detected_acceleration_unit
                    ),
                    None,
                )

                st.caption(
                    "Detected target units: "
                    f"Speed = `{detected_target_speed_unit or 'not available'}`, "
                    f"Acceleration = `{detected_target_accel_unit or 'not available'}`."
                )
                target_unit_cols = st.columns(2)
                with target_unit_cols[0]:
                    target_speed_input_unit = st.selectbox(
                        "Target speed input unit",
                        SPEED_UNITS,
                        index=SPEED_UNITS.index(detected_target_speed_unit or speed_output_unit),
                        key=resettable_key("target_speed_input_unit"),
                    )
                with target_unit_cols[1]:
                    target_acceleration_input_unit = st.selectbox(
                        "Target acceleration input unit",
                        ACCELERATION_UNITS,
                        index=ACCELERATION_UNITS.index(
                            detected_target_accel_unit or acceleration_output_unit
                        ),
                        key=resettable_key("target_acceleration_input_unit"),
                    )

                if parsed_targets:
                    raw_target_curves = pd.concat(
                        [target.curves for target in parsed_targets],
                        ignore_index=True,
                    )
                    target_curves = convert_target_curves(
                        raw_target_curves,
                        target_speed_input_unit,
                        speed_output_unit,
                        target_acceleration_input_unit,
                        acceleration_output_unit,
                    )
                    target_summary = (
                        target_curves.groupby(["group_label", "target_pedal"], as_index=False)
                        .agg(
                            points=("speed", "count"),
                            speed_min=("speed", "min"),
                            speed_max=("speed", "max"),
                        )
                        .sort_values(["group_label", "target_pedal"])
                    )
                    st.success(f"Loaded {len(target_summary)} target curve(s).")
                    st.dataframe(target_summary, width="stretch", hide_index=True)

                if target_parse_errors:
                    st.warning(
                        "Some target data could not be loaded:\n\n"
                        + "\n".join(f"- {err}" for err in target_parse_errors)
                    )
            else:
                st.caption(
                    "Expected layouts: one Speed column with multiple Acceleration columns, "
                    "or paired columns like x1/y1, x2/y2, x3/y3."
                )

        scatter_curves = pd.DataFrame()
        scatter_parse_errors: list[str] = []

        section_header(
            "Random Scatter Plots",
            "Plot arbitrary Excel-style X/Y data, independent of MDF (optional)",
            icon="scatter",
        )
        with st.expander("Import or paste arbitrary X/Y scatter data", expanded=False):
            st.write(
                "Use this for Excel-style scatter plots that are not necessarily Speed vs Acceleration."
            )
            scatter_layout = st.selectbox(
                "Random scatter data layout",
                SCATTER_LAYOUTS,
                index=1,
                help=(
                    "The default is one X column with multiple Y columns. "
                    "Choose paired layout when the sheet is x1/y1, x2/y2, x3/y3, etc."
                ),
                key=resettable_key("random_scatter_layout"),
            )
            axis_cols = st.columns(2)
            with axis_cols[0]:
                scatter_x_axis_label = st.text_input(
                    "Random scatter X-axis label",
                    value="X",
                    key=resettable_key("random_scatter_x_axis_label"),
                )
            with axis_cols[1]:
                scatter_y_axis_label = st.text_input(
                    "Random scatter Y-axis label",
                    value="Y",
                    key=resettable_key("random_scatter_y_axis_label"),
                )

            scatter_files = st.file_uploader(
                "Random scatter files (.xlsx, .xlsm, .xls, .csv, .txt)",
                type=["xlsx", "xlsm", "xls", "csv", "txt"],
                accept_multiple_files=True,
                key=resettable_key("random_scatter_files"),
            )
            pasted_scatter_text = st.text_area(
                "Paste random scatter data from Excel",
                height=140,
                placeholder=(
                    "X\tSeries 1\tSeries 2\n"
                    "0\t1.2\t2.4\n"
                    "10\t1.8\t2.9"
                ),
                key=resettable_key("random_scatter_paste"),
            )

            scatter_source_payloads: dict[str, bytes | str] = {}
            scatter_source_signatures: dict[str, str] = {}
            seen_scatter_names: dict[str, int] = {}
            for scatter_file in scatter_files or []:
                original_name = scatter_file.name
                seen_scatter_names[original_name] = seen_scatter_names.get(original_name, 0) + 1
                if seen_scatter_names[original_name] == 1:
                    display_name = original_name
                else:
                    path = Path(original_name)
                    display_name = f"{path.stem} ({seen_scatter_names[original_name]}){path.suffix}"
                file_bytes = scatter_file.getvalue()
                scatter_source_payloads[display_name] = file_bytes
                scatter_source_signatures[display_name] = content_hash(file_bytes)

            if pasted_scatter_text.strip():
                scatter_source_payloads["Pasted random scatter data"] = pasted_scatter_text
                scatter_source_signatures["Pasted random scatter data"] = content_hash(
                    pasted_scatter_text.encode("utf-8")
                )

            parsed_scatters: list[ParsedRandomScatterData] = []
            if scatter_source_payloads:
                scatter_metadata = metadata_for_current_scatter(
                    list(scatter_source_payloads.keys()),
                    tuple(scatter_source_signatures.items()),
                )
                st.write("Series Labels")
                edited_scatter_metadata = st.data_editor(
                    scatter_metadata,
                    key=f"scatter_metadata_editor_{st.session_state.get('scatter_metadata_editor_version', 0)}",
                    width="stretch",
                    hide_index=True,
                    num_rows="fixed",
                    column_config={
                        "source_name": st.column_config.TextColumn("Source", disabled=True),
                        "series_label": st.column_config.TextColumn("Series label"),
                    },
                )
                st.session_state["scatter_metadata"] = edited_scatter_metadata.copy()
                scatter_labels_by_source = {
                    str(row["source_name"]): str(row.get("series_label") or "Scatter")
                    for _, row in edited_scatter_metadata.iterrows()
                }

                for source_order, (source_name, payload) in enumerate(scatter_source_payloads.items()):
                    series_label = scatter_labels_by_source.get(source_name, "Scatter")
                    try:
                        if isinstance(payload, bytes):
                            parsed_scatter = parse_scatter_file(source_name, payload, series_label, scatter_layout)
                        else:
                            parsed_scatter = parse_pasted_scatter_text(payload, series_label, scatter_layout)
                        parsed_curves = parsed_scatter.curves.copy()
                        parsed_curves["source_name"] = source_name
                        parsed_curves["source_order"] = source_order
                        parsed_scatters.append(ParsedRandomScatterData(parsed_curves))
                    except Exception as exc:
                        scatter_parse_errors.append(f"{source_name}: {exc}")

                if parsed_scatters:
                    scatter_curves = pd.concat(
                        [scatter.curves for scatter in parsed_scatters],
                        ignore_index=True,
                    )
                    scatter_summary = (
                        scatter_curves.groupby("series_name", as_index=False)
                        .agg(points=("x", "count"), x_min=("x", "min"), x_max=("x", "max"))
                        .sort_values("series_name")
                    )
                    st.success(f"Loaded {len(scatter_summary)} random scatter series.")
                    st.dataframe(scatter_summary, width="stretch", hide_index=True)

                    scatter_series_options = list(dict.fromkeys(scatter_curves["series_name"].dropna().astype(str)))
                    sync_multiselect_options("random_scatter_series", scatter_series_options)
                    selected_scatter_series = st.multiselect(
                        "Random scatter series to show",
                        scatter_series_options,
                        default=scatter_series_options,
                        key="random_scatter_series",
                    )
                    scatter_plot_data = scatter_curves[
                        scatter_curves["series_name"].isin(selected_scatter_series)
                    ].copy()

                    if scatter_plot_data.empty:
                        st.info("No random scatter series match the selected filters.")
                    else:
                        derived_scatter_data = pd.DataFrame()
                        with st.expander("Average / Offset Random Scatter Series (optional)", expanded=False):
                            derived_rows = []
                            st.write("Create extra curves from the imported random scatter series.")
                            avg_cols = st.columns([2, 1])
                            with avg_cols[0]:
                                average_series = st.multiselect(
                                    "Series to average",
                                    scatter_series_options,
                                    key=resettable_key("random_scatter_average_series"),
                                )
                            with avg_cols[1]:
                                average_name = st.text_input(
                                    "Average series name",
                                    value="Average_Selected",
                                    key=resettable_key("random_scatter_average_name"),
                                )
                            create_average_series = st.checkbox(
                                "Create average series",
                                value=False,
                                key=resettable_key("random_scatter_create_average"),
                            )
                            if create_average_series:
                                if len(average_series) < 2:
                                    st.warning("Select at least two series to create an average.")
                                else:
                                    average_curve = average_selected_series(
                                        scatter_curves,
                                        average_series,
                                        average_name,
                                    )
                                    if average_curve.empty:
                                        st.warning(
                                            "Average could not be created because the selected series "
                                            "do not share an overlapping X range."
                                        )
                                    else:
                                        derived_rows.append(average_curve)

                            offset_cols = st.columns([2, 1, 1])
                            with offset_cols[0]:
                                offset_source = st.selectbox(
                                    "Series to offset",
                                    scatter_series_options,
                                    key=resettable_key("random_scatter_offset_source"),
                                )
                            with offset_cols[1]:
                                offset_axis = st.selectbox(
                                    "Offset axis",
                                    ["Y", "X"],
                                    key=resettable_key("random_scatter_offset_axis"),
                                )
                            with offset_cols[2]:
                                offset_amount = st.number_input(
                                    "Offset amount",
                                    value=0.0,
                                    step=0.1,
                                    key=resettable_key("random_scatter_offset_amount"),
                                )
                            offset_name = st.text_input(
                                "Offset series name",
                                value=f"{offset_source}_offset" if offset_source else "Offset_Selected",
                                key=resettable_key("random_scatter_offset_name"),
                            )
                            create_offset_series = st.checkbox(
                                "Create offset series",
                                value=False,
                                key=resettable_key("random_scatter_create_offset"),
                            )
                            if create_offset_series:
                                offset_curve = offset_selected_series(
                                    scatter_curves,
                                    offset_source,
                                    offset_name,
                                    float(offset_amount),
                                    offset_axis,
                                )
                                if offset_curve.empty:
                                    st.warning("Offset curve could not be created from the selected series.")
                                else:
                                    derived_rows.append(offset_curve)

                            if derived_rows:
                                derived_scatter_data = pd.concat(derived_rows, ignore_index=True)
                                st.success(f"Created {derived_scatter_data['series_name'].nunique()} derived series.")
                                st.dataframe(
                                    derived_scatter_data[
                                        ["series_name", "derived_type", "derived_from", "x", "y"]
                                    ],
                                    width="stretch",
                                    hide_index=True,
                                )
                                st.download_button(
                                    "Download derived random scatter Excel",
                                    data=export_random_scatter_workbook(derived_scatter_data),
                                    file_name="random_scatter_derived_series.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )

                        if not derived_scatter_data.empty:
                            scatter_plot_data = pd.concat(
                                [scatter_plot_data, derived_scatter_data],
                                ignore_index=True,
                            )

                        scatter_style_defaults = default_scatter_style_table(scatter_plot_data)
                        scatter_style_overrides = {}
                        scatter_trace_order: list[str] = []
                        scatter_show_markers = False
                        with st.expander("Random Scatter Curve Style and Legend Order (optional)", expanded=False):
                            scatter_show_markers = st.checkbox(
                                "Smooth lines with markers",
                                value=False,
                                key=resettable_key("random_scatter_smooth_markers"),
                            )
                            st.caption(f"Color accepts a name or code. {COLOR_COLUMN_HELP}")
                            edited_scatter_style_table = st.data_editor(
                                scatter_style_defaults,
                                key=(
                                    "random_scatter_style_editor_"
                                    + hashlib.sha1(
                                        "|".join(scatter_style_defaults["style_key"].astype(str)).encode("utf-8")
                                    ).hexdigest()
                                ),
                                width="stretch",
                                hide_index=True,
                                num_rows="fixed",
                                column_config={
                                    "style_key": st.column_config.TextColumn("Style key", disabled=True),
                                    "legend_order": st.column_config.NumberColumn("Legend order", min_value=1, step=1),
                                    "curve": st.column_config.TextColumn("Curve", disabled=True),
                                    "color": st.column_config.TextColumn("Color name/code", help=COLOR_COLUMN_HELP),
                                    "thickness": st.column_config.NumberColumn(
                                        "Thickness",
                                        min_value=0.5,
                                        max_value=10.0,
                                        step=0.25,
                                    ),
                                    "line_style": st.column_config.SelectboxColumn(
                                        "Line style",
                                        options=LINE_STYLE_OPTIONS,
                                    ),
                                },
                            )
                            scatter_style_overrides = style_overrides_from_table(edited_scatter_style_table)
                            scatter_trace_order = legend_order_from_table(edited_scatter_style_table)

                        scatter_trendline_data = pd.DataFrame()
                        scatter_show_trendline_only = False
                        scatter_show_equations = False
                        with st.expander("Random Scatter Trendlines (optional)", expanded=False):
                            show_scatter_trendlines = st.checkbox(
                                "Show trendlines on random scatter plot",
                                value=False,
                                key=resettable_key("random_scatter_show_trendlines"),
                            )
                            if show_scatter_trendlines:
                                trendline_series_options = list(
                                    dict.fromkeys(scatter_plot_data["series_name"].dropna().astype(str))
                                )
                                selected_trendline_series = st.multiselect(
                                    "Series for trendlines",
                                    trendline_series_options,
                                    default=trendline_series_options,
                                    key=resettable_key("random_scatter_trendline_series"),
                                )
                                trend_cols = st.columns(3)
                                with trend_cols[0]:
                                    scatter_trendline_choice = st.selectbox(
                                        "Trendline type",
                                        list(TRENDLINE_DEGREES.keys()),
                                        key=resettable_key("random_scatter_trendline_degree"),
                                    )
                                with trend_cols[1]:
                                    scatter_trendline_x_mode = st.selectbox(
                                        "Trendline X-values",
                                        ["Smooth generated X-values", "Trendlines using existing X-values"],
                                        key=resettable_key("random_scatter_trendline_x_mode"),
                                    )
                                with trend_cols[2]:
                                    scatter_show_equations = st.checkbox(
                                        "Show equations",
                                        value=True,
                                        key=resettable_key("random_scatter_show_equations"),
                                    )
                                scatter_trendline_points = 200
                                if scatter_trendline_x_mode == "Smooth generated X-values":
                                    scatter_trendline_points = st.number_input(
                                        "Trendline points",
                                        min_value=10,
                                        max_value=5000,
                                        value=200,
                                        step=10,
                                        key=resettable_key("random_scatter_trendline_points"),
                                    )
                                scatter_show_trendline_only = st.checkbox(
                                    "Show trendline-only graph",
                                    value=False,
                                    key=resettable_key("random_scatter_trendline_only"),
                                )
                                scatter_trendline_data = build_trendline_data(
                                    scatter_plot_data,
                                    x_col="x",
                                    y_col="y",
                                    group_col="series_name",
                                    selected_groups=selected_trendline_series,
                                    degree=TRENDLINE_DEGREES[scatter_trendline_choice],
                                    points=int(scatter_trendline_points),
                                    use_existing_x=(
                                        scatter_trendline_x_mode == "Trendlines using existing X-values"
                                    ),
                                )
                                if scatter_trendline_data.empty:
                                    st.warning(
                                        "No trendlines were created. Each selected series needs at least "
                                        "degree + 1 unique X values."
                                    )
                                elif scatter_show_equations:
                                    st.dataframe(
                                        trendline_equations(scatter_trendline_data),
                                        width="stretch",
                                        hide_index=True,
                                    )

                        scatter_fig = make_random_scatter_figure(
                            scatter_plot_data,
                            title="Random Scatter Plot",
                            x_axis_title=scatter_x_axis_label,
                            y_axis_title=scatter_y_axis_label,
                            style_overrides=scatter_style_overrides,
                            trace_order=scatter_trace_order,
                            show_markers=scatter_show_markers,
                            smooth_lines=True,
                        )
                        add_trendline_traces(
                            scatter_fig,
                            scatter_trendline_data,
                            show_equations=scatter_show_equations,
                        )
                        st.plotly_chart(scatter_fig, width="stretch")

                        if scatter_show_trendline_only and not scatter_trendline_data.empty:
                            scatter_trendline_fig = make_trendline_figure(
                                scatter_trendline_data,
                                title="Random Scatter Trendline Graph",
                                x_axis_title=scatter_x_axis_label,
                                y_axis_title=scatter_y_axis_label,
                                show_equations=scatter_show_equations,
                            )
                            st.plotly_chart(scatter_trendline_fig, width="stretch")

                        scatter_export_cols = st.columns(3)
                        with scatter_export_cols[0]:
                            try:
                                st.download_button(
                                    "Download random scatter Excel",
                                    data=export_random_scatter_workbook(scatter_plot_data),
                                    file_name="random_scatter_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )
                            except ImportError:
                                st.warning("Install the updated requirements to enable Excel export.")
                        with scatter_export_cols[1]:
                            if not scatter_trendline_data.empty:
                                st.download_button(
                                    "Download trendline Excel",
                                    data=export_trendline_workbook(scatter_trendline_data),
                                    file_name="random_scatter_trendline_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )
                        with scatter_export_cols[2]:
                            try:
                                scatter_png = scatter_fig.to_image(format="png", width=1600, height=900, scale=2)
                                st.download_button(
                                    "Download random scatter PNG",
                                    data=scatter_png,
                                    file_name="random_scatter_plot.png",
                                    mime="image/png",
                                )
                            except Exception as exc:
                                st.warning(f"PNG export is unavailable until Kaleido is installed correctly: {exc}")

                if scatter_parse_errors:
                    st.warning(
                        "Some random scatter data could not be loaded:\n\n"
                        + "\n".join(f"- {err}" for err in scatter_parse_errors)
                    )
            else:
                st.caption(
                    "Expected layouts: one X column with multiple Y columns, "
                    "or paired columns like x1/y1, x2/y2, x3/y3."
                )


    if active_page == "Speed Analysis":
        save_labels = False
        run_clicked = False
        if has_measurements:
            section_header(
                "File Labels",
                "Classify each run and assign mode, regen, and target pedal",
                icon="tag",
            )
            st.write(
                "For single-pedal maneuvers use `Auto`, or select the known pedal value. "
                "For a 0% creep launch, select target `0` explicitly so the brake-release logic is used."
            )

            with st.expander("Apply the same labels to multiple files", expanded=True):
                with st.form("bulk_label_form", clear_on_submit=False):
                    bulk_files = st.multiselect("Files to label", file_names)
                    bulk_cols = st.columns(4)
                    with bulk_cols[0]:
                        bulk_type = st.selectbox("Test type", TEST_TYPES)
                    with bulk_cols[1]:
                        bulk_mode = st.text_input("Mode label", placeholder="Normal Mode")
                    with bulk_cols[2]:
                        bulk_regen = st.text_input("Regen / extra label", placeholder="Regen Level 1")
                    with bulk_cols[3]:
                        bulk_target = st.selectbox("Target pedal", TARGET_OPTIONS)
                    apply_labels = st.form_submit_button("Apply labels to selected files")

                if apply_labels:
                    if not bulk_files:
                        st.warning("Select one or more files before applying bulk labels.")
                    else:
                        metadata = st.session_state["file_metadata"].copy()
                        selected_rows = metadata["file_name"].isin(bulk_files)
                        metadata.loc[selected_rows, "test_type"] = bulk_type
                        metadata.loc[selected_rows, "mode_label"] = bulk_mode
                        metadata.loc[selected_rows, "regen_label"] = bulk_regen
                        metadata.loc[selected_rows, "target_pedal"] = bulk_target
                        st.session_state["file_metadata"] = metadata
                        st.session_state["metadata_editor_version"] += 1
                        clear_results()
                        st.rerun()

            editor_key = f"file_metadata_editor_{st.session_state.get('metadata_editor_version', 0)}"
            with st.form("file_metadata_form"):
                edited_metadata = st.data_editor(
                    st.session_state["file_metadata"],
                    key=editor_key,
                    width="stretch",
                    hide_index=True,
                    num_rows="fixed",
                    column_config={
                        "file_name": st.column_config.TextColumn("File", disabled=True),
                        "test_type": st.column_config.SelectboxColumn("Test type", options=TEST_TYPES, required=True),
                        "mode_label": st.column_config.TextColumn("Mode label"),
                        "regen_label": st.column_config.TextColumn("Regen / extra label"),
                        "target_pedal": st.column_config.SelectboxColumn("Target pedal", options=TARGET_OPTIONS, required=True),
                    },
                )
                label_cols = st.columns([1, 1, 4])
                with label_cols[0]:
                    save_labels = st.form_submit_button("Save labels")
                with label_cols[1]:
                    run_clicked = st.form_submit_button("Generate plot", type="primary")

            if save_labels or run_clicked:
                st.session_state["file_metadata"] = edited_metadata.copy()
                metadata = edited_metadata.copy()

            if save_labels and not run_clicked:
                clear_results()
                st.success("Labels saved. Click Generate plot when you are ready.")

        config_metadata = st.session_state.get("file_metadata", pd.DataFrame())
        config = project_config_json(signal_map, settings, config_metadata, display_units)
        st.download_button(
            "Download current setup as JSON",
            data=config,
            file_name="vehicle_plotter_project_config.json",
            mime="application/json",
        )

        current_signature = processing_signature(
            file_signatures,
            signal_map,
            settings,
            config_metadata,
            display_units,
        )

        if run_clicked and has_measurements:
            missing_signals = [name for name, value in signal_map.items() if not value]
            if missing_signals:
                st.error("Select all four signals before generating a plot.")
                st.stop()

            curve_rows: list[pd.DataFrame] = []
            audit_rows: list[dict[str, object]] = []
            load_errors: list[str] = []
            with st.spinner("Loading selected signals and applying usable-data logic..."):
                metadata_by_file = {
                    str(row["file_name"]): row for _, row in st.session_state["file_metadata"].iterrows()
                }
                for file_name, source in measurement_sources.items():
                    if str(metadata_by_file[file_name].get("test_type") or "") == "Ignore":
                        continue
                    try:
                        raw_frame = read_measurement_frame(source, selected_channels, raster_step_s)
                        measurement_frame = normalize_measurement_frame(raw_frame, signal_map)
                        measurement_frame = convert_measurement_units(
                            measurement_frame,
                            speed_input_unit,
                            speed_output_unit,
                            acceleration_input_unit,
                            acceleration_output_unit,
                        )
                        file_curves, file_audit = process_measurement_frame(
                            file_name,
                            measurement_frame,
                            metadata_by_file[file_name],
                            settings,
                        )
                        curve_rows.extend(file_curves)
                        audit_rows.extend(file_audit)
                        del raw_frame, measurement_frame
                    except MemoryError:
                        load_errors.append(
                            f"{file_name}: insufficient memory. Try a larger optional MDF resample step, "
                            "for example 0.01 or 0.1 seconds."
                        )
                    except Exception as exc:
                        load_errors.append(f"{file_name}: {exc}")

            averaged_curves, audit_table = average_file_curves(curve_rows, audit_rows)
            st.session_state["averaged_curves"] = averaged_curves
            st.session_state["audit_table"] = audit_table
            st.session_state["load_errors"] = load_errors
            st.session_state["result_signature"] = current_signature
            for key in PLOT_FILTER_KEYS:
                st.session_state.pop(key, None)

        has_generated_measurements = (
            "averaged_curves" in st.session_state
            and st.session_state.get("result_signature") == current_signature
        )
        if has_measurements and "averaged_curves" in st.session_state and not has_generated_measurements:
            st.session_state["dl_results_stale"] = True
            st.warning(
                "Signals, labels, or processing settings have changed. Click Generate plot "
                "to refresh measured results. Imported target curves can still be plotted."
            )
        else:
            st.session_state["dl_results_stale"] = False

        load_errors = st.session_state.get("load_errors", [])
        if load_errors:
            st.warning("Some files could not be processed:\n\n" + "\n".join(f"- {err}" for err in load_errors))

        averaged_curves = (
            st.session_state["averaged_curves"]
            if has_generated_measurements
            else pd.DataFrame()
        )
        audit_table = (
            st.session_state.get("audit_table", pd.DataFrame())
            if has_generated_measurements
            else pd.DataFrame()
        )

        if has_generated_measurements:
            section_header(
                "Usable Data Audit",
                "How each file and target was classified during processing",
                icon="audit",
            )
            if audit_table.empty:
                st.info("No usable data was found. Check signal selections, labels, target pedals, and tolerances.")
            else:
                st.dataframe(audit_table, width="stretch", hide_index=True)
                st.download_button(
                    "Download audit CSV",
                    data=audit_table.to_csv(index=False),
                    file_name="usable_data_audit.csv",
                    mime="text/csv",
                )
        elif has_measurements:
            st.info("Click Generate plot to process the loaded measurement files.")

        combined_available = pd.concat([averaged_curves, target_curves], ignore_index=True)
        if combined_available.empty:
            if scatter_curves.empty:
                st.info("Load measurement files and click Generate plot, import target data, or import random scatter data to create a plot.")
            else:
                st.info("Random scatter data is shown above. Load measurement files or target Speed vs Acceleration data to use Plot Controls and Metrics.")
            st.stop()

        section_header(
            "Plot Controls",
            "Filter curves, style them, and export the Speed vs Acceleration graph",
            icon="chart",
        )
        available_test_types = [
            test_type for test_type in ["Launch", "Deceleration", "Target"]
            if test_type in set(combined_available["test_type"])
        ]
        sync_multiselect_options("plot_test_types", available_test_types)
        selected_test_types = st.multiselect(
            "Test types to show",
            available_test_types,
            default=available_test_types,
            key="plot_test_types",
        )

        available = combined_available[combined_available["test_type"].isin(selected_test_types)].copy()
        available["scenario"] = available["test_type"] + " | " + available["group_label"]
        scenario_options = sorted(available["scenario"].dropna().unique().tolist())
        target_options = sorted(available["target_pedal"].dropna().astype(int).unique().tolist())
        sync_multiselect_options("plot_scenarios", scenario_options)
        sync_multiselect_options("plot_targets", target_options)

        control_cols = st.columns(2)
        with control_cols[0]:
            selected_scenarios = st.multiselect(
                "Test / mode / regen combinations to show",
                scenario_options,
                default=scenario_options,
                key="plot_scenarios",
            )
        with control_cols[1]:
            selected_targets = st.multiselect(
                "Pedal targets to show",
                target_options,
                default=target_options,
                key="plot_targets",
            )

        plot_data = available[
            available["scenario"].isin(selected_scenarios)
            & available["target_pedal"].isin(selected_targets)
        ].copy()

        if plot_data.empty:
            st.info("No curves match the selected plot filters.")
            st.stop()

        style_defaults = default_style_table(plot_data)
        style_overrides = {}
        with st.expander("Curve Style and Legend Order (optional)", expanded=False):
            st.caption(f"Color accepts a name or code. {COLOR_COLUMN_HELP}")
            edited_style_table = st.data_editor(
                style_defaults,
                key=f"curve_style_editor_{hashlib.sha1('|'.join(style_defaults['style_key'].astype(str)).encode('utf-8')).hexdigest()}",
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "style_key": st.column_config.TextColumn("Style key", disabled=True),
                    "legend_order": st.column_config.NumberColumn("Legend order", min_value=1, step=1),
                    "curve": st.column_config.TextColumn("Curve", disabled=True),
                    "color": st.column_config.TextColumn("Color name/code", help=COLOR_COLUMN_HELP),
                    "thickness": st.column_config.NumberColumn("Thickness", min_value=0.5, max_value=10.0, step=0.25),
                    "line_style": st.column_config.SelectboxColumn("Line style", options=LINE_STYLE_OPTIONS),
                },
            )
            style_overrides = style_overrides_from_table(edited_style_table)
            trace_order = legend_order_from_table(edited_style_table)

        fig = make_speed_accel_figure(
            plot_data,
            title="Speed vs Acceleration",
            speed_unit=speed_output_unit,
            acceleration_unit=acceleration_output_unit,
            style_overrides=style_overrides,
            trace_order=trace_order,
        )
        st.plotly_chart(fig, width="stretch")

        export_cols = st.columns(3)
        with export_cols[0]:
            try:
                workbook_bytes = export_plot_workbook(
                    plot_data,
                    speed_unit=speed_output_unit,
                    acceleration_unit=acceleration_output_unit,
                )
                st.download_button(
                    "Download data Excel",
                    data=workbook_bytes,
                    file_name="speed_vs_acceleration_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except ImportError:
                st.warning("Install the updated requirements to enable Excel export.")
        with export_cols[1]:
            st.download_button(
                "Download project JSON",
                data=project_config_json(
                    signal_map,
                    settings,
                    config_metadata,
                    display_units,
                ),
                file_name="vehicle_plotter_project_config.json",
                mime="application/json",
            )
        with export_cols[2]:
            try:
                png_bytes = fig.to_image(format="png", width=1600, height=900, scale=2)
                st.download_button(
                    "Download graph PNG",
                    data=png_bytes,
                    file_name="speed_vs_acceleration.png",
                    mime="image/png",
                )
            except Exception as exc:
                st.warning(f"PNG export is unavailable until Kaleido is installed correctly: {exc}")

        section_header(
            "Metrics",
            "Peak acceleration, road load, slope, and speed-band averages (optional)",
            icon="metric",
        )
        with st.expander("Calculate Speed vs Acceleration metrics", expanded=False):
            st.caption(
                "Metrics use the curves currently selected in Plot Controls. "
                f"Speed bands use the current graph speed unit: {speed_output_unit}."
            )
            slope_cols = st.columns(2)
            with slope_cols[0]:
                slope_start_speed = st.number_input(
                    f"Slope start speed ({speed_output_unit})",
                    value=80.0,
                    step=5.0,
                    key=resettable_key("metrics_slope_start_speed"),
                )
            with slope_cols[1]:
                slope_end_speed = st.number_input(
                    f"Slope end speed ({speed_output_unit})",
                    value=120.0,
                    step=5.0,
                    key=resettable_key("metrics_slope_end_speed"),
                )

            if slope_start_speed == slope_end_speed:
                st.warning("Choose two different speed points to calculate slope.")

            extend_road_load = st.checkbox(
                "Calculate road load using linear trendline extension when curve does not cross zero",
                value=False,
                help=(
                    "When enabled, curves that do not intersect the acceleration = 0 axis "
                    "will use a linear fit extension to estimate road-load speed."
                ),
                key=resettable_key("metrics_extend_road_load"),
            )

            metrics_table = calculate_metrics(
                plot_data,
                speed_bands=DEFAULT_SPEED_BANDS,
                slope_start=float(slope_start_speed),
                slope_end=float(slope_end_speed),
                extend_road_load=extend_road_load,
            )
            metrics_display = format_metrics_table(
                metrics_table,
                speed_unit=speed_output_unit,
                acceleration_unit=acceleration_output_unit,
                slope_start=float(slope_start_speed),
                slope_end=float(slope_end_speed),
                speed_bands=DEFAULT_SPEED_BANDS,
            )

            if metrics_display.empty:
                st.info("No metrics are available for the selected curves.")
            else:
                st.dataframe(metrics_display, width="stretch", hide_index=True)

                metric_plot_options = {
                    "Select Plot": (None, None),
                    f"Peak acceleration ({acceleration_output_unit})": (
                        "peak_acceleration",
                        f"Peak acceleration ({acceleration_output_unit})",
                    ),
                    f"Road load speed ({speed_output_unit})": (
                        "road_load_speed",
                        f"Road load speed ({speed_output_unit})",
                    ),
                    (
                        f"Slope {float(slope_start_speed):g}-{float(slope_end_speed):g} "
                        f"({acceleration_output_unit}/{speed_output_unit})"
                    ): (
                        "slope",
                        f"Slope ({acceleration_output_unit}/{speed_output_unit})",
                    ),
                }
                combined_average_graph_label = "Combined average acceleration bands"
                additional_metric_graph_label = "Additional metric plot"
                separate_band_graph_labels = [
                    f"Separate {band_label(low, high, speed_output_unit)}"
                    for low, high in DEFAULT_SPEED_BANDS
                ]

                metric_trendline_enabled = False
                metric_show_equations = False
                metric_show_trendline_only = False
                metric_trendline_choice = "Linear"
                metric_trendline_points = 200
                metric_trendline_x_mode = "Smooth generated X-values"
                selected_metric_trendline_graphs: list[str] = []
                with st.expander("Metric Trendlines (optional)", expanded=False):
                    metric_trendline_enabled = st.checkbox(
                        "Show trendlines on metric graphs",
                        value=False,
                        key=resettable_key("metric_show_trendlines"),
                    )
                    if metric_trendline_enabled:
                        metric_trend_cols = st.columns(3)
                        with metric_trend_cols[0]:
                            metric_trendline_choice = st.selectbox(
                                "Trendline type",
                                list(TRENDLINE_DEGREES.keys()),
                                key=resettable_key("metric_trendline_degree"),
                            )
                        with metric_trend_cols[1]:
                            metric_trendline_x_mode = st.selectbox(
                                "Trendline X-values",
                                ["Smooth generated X-values", "Trendlines using existing X-values"],
                                key=resettable_key("metric_trendline_x_mode"),
                            )
                        with metric_trend_cols[2]:
                            metric_show_equations = st.checkbox(
                                "Show equations",
                                value=True,
                                key=resettable_key("metric_show_equations"),
                            )
                        if metric_trendline_x_mode == "Smooth generated X-values":
                            metric_trendline_points = st.number_input(
                                "Trendline points",
                                min_value=10,
                                max_value=5000,
                                value=200,
                                step=10,
                                key=resettable_key("metric_trendline_points"),
                            )
                        selected_metric_trendline_graphs = st.multiselect(
                            "Metric graphs to add trendlines to",
                            [combined_average_graph_label, additional_metric_graph_label]
                            + separate_band_graph_labels,
                            default=[],
                            key=resettable_key("metric_trendline_graphs"),
                        )
                        metric_show_trendline_only = st.checkbox(
                            "Show trendline-only metric graphs",
                            value=False,
                            key=resettable_key("metric_trendline_only"),
                        )

                average_bands = average_band_long(
                    metrics_table,
                    speed_unit=speed_output_unit,
                    acceleration_unit=acceleration_output_unit,
                    speed_bands=DEFAULT_SPEED_BANDS,
                )
                avg_band_fig = make_average_band_figure(
                    metrics_table,
                    speed_unit=speed_output_unit,
                    acceleration_unit=acceleration_output_unit,
                    speed_bands=DEFAULT_SPEED_BANDS,
                )
                metric_trendline_exports: list[pd.DataFrame] = []
                average_band_trendlines = pd.DataFrame()
                if (
                    metric_trendline_enabled
                    and combined_average_graph_label in selected_metric_trendline_graphs
                ):
                    average_band_source = average_band_trend_source(average_bands)
                    average_band_trendlines = build_trendline_data(
                        average_band_source,
                        x_col="target_pedal",
                        y_col="average_acceleration",
                        group_col="trend_group",
                        degree=TRENDLINE_DEGREES[metric_trendline_choice],
                        points=int(metric_trendline_points),
                        trend_suffix="trend",
                        use_existing_x=(
                            metric_trendline_x_mode == "Trendlines using existing X-values"
                        ),
                    )
                    if not average_band_trendlines.empty:
                        metric_trendline_exports.append(
                            tag_metric_trendlines(average_band_trendlines, "Average acceleration bands")
                        )
                        add_trendline_traces(
                            avg_band_fig,
                            average_band_trendlines,
                            show_equations=metric_show_equations,
                        )
                st.plotly_chart(avg_band_fig, width="stretch")

                selected_metric_plot = st.selectbox(
                    "Additional metric plot",
                    list(metric_plot_options.keys()),
                    key=resettable_key("selected_metric_plot"),
                )
                metric_column, metric_axis = metric_plot_options[selected_metric_plot]
                metric_fig = None
                selected_metric_trendlines = pd.DataFrame()
                if metric_column is not None:
                    metric_fig = make_metric_figure(
                        metrics_table,
                        metric_column=metric_column,
                        metric_label=selected_metric_plot,
                        y_axis_title=metric_axis,
                    )
                else:
                    st.info("Select a metric in Additional metric plot to show that graph.")
                if (
                    metric_trendline_enabled
                    and metric_column is not None
                    and additional_metric_graph_label in selected_metric_trendline_graphs
                ):
                    selected_metric_source = selected_metric_trend_source(metrics_table, metric_column)
                    selected_metric_trendlines = build_trendline_data(
                        selected_metric_source,
                        x_col="target_pedal",
                        y_col="metric_value",
                        group_col="trend_group",
                        degree=TRENDLINE_DEGREES[metric_trendline_choice],
                        points=int(metric_trendline_points),
                        trend_suffix="trend",
                        use_existing_x=(
                            metric_trendline_x_mode == "Trendlines using existing X-values"
                        ),
                    )
                    if not selected_metric_trendlines.empty:
                        metric_trendline_exports.append(
                            tag_metric_trendlines(selected_metric_trendlines, selected_metric_plot)
                        )
                        add_trendline_traces(
                            metric_fig,
                            selected_metric_trendlines,
                            show_equations=metric_show_equations,
                        )
                if metric_fig is not None:
                    st.plotly_chart(metric_fig, width="stretch")

                with st.expander("Separate Average Acceleration Band Plots", expanded=False):
                    for low, high in DEFAULT_SPEED_BANDS:
                        key = band_key(low, high)
                        label = band_label(low, high, speed_output_unit)
                        separate_band_fig = make_metric_figure(
                            metrics_table,
                            metric_column=key,
                            metric_label=label,
                            y_axis_title=f"Average acceleration ({acceleration_output_unit})",
                        )
                        separate_graph_label = f"Separate {label}"
                        if (
                            metric_trendline_enabled
                            and separate_graph_label in selected_metric_trendline_graphs
                        ):
                            separate_band_source = selected_metric_trend_source(metrics_table, key)
                            separate_band_trendlines = build_trendline_data(
                                separate_band_source,
                                x_col="target_pedal",
                                y_col="metric_value",
                                group_col="trend_group",
                                degree=TRENDLINE_DEGREES[metric_trendline_choice],
                                points=int(metric_trendline_points),
                                trend_suffix="trend",
                                use_existing_x=(
                                    metric_trendline_x_mode == "Trendlines using existing X-values"
                                ),
                            )
                            if not separate_band_trendlines.empty:
                                metric_trendline_exports.append(
                                    tag_metric_trendlines(separate_band_trendlines, label)
                                )
                                add_trendline_traces(
                                    separate_band_fig,
                                    separate_band_trendlines,
                                    show_equations=metric_show_equations,
                                )
                        st.plotly_chart(separate_band_fig, width="stretch")

                metric_trendline_data = (
                    pd.concat(metric_trendline_exports, ignore_index=True)
                    if metric_trendline_exports
                    else pd.DataFrame()
                )
                if metric_trendline_enabled:
                    if metric_trendline_data.empty:
                        st.warning(
                            "No metric trendlines were created. Each metric series needs at least "
                            "degree + 1 points."
                        )
                    elif metric_show_equations:
                        st.dataframe(
                            trendline_equations(metric_trendline_data),
                            width="stretch",
                            hide_index=True,
                        )

                if metric_show_trendline_only and not metric_trendline_data.empty:
                    metric_trendline_fig = make_trendline_figure(
                        metric_trendline_data,
                        title="Metric Trendline Graph",
                        x_axis_title="Pedal (%)",
                        y_axis_title="Metric value",
                        show_equations=metric_show_equations,
                    )
                    st.plotly_chart(metric_trendline_fig, width="stretch")

                try:
                    metric_export_cols = st.columns(2)
                    with metric_export_cols[0]:
                        st.download_button(
                            "Download metrics Excel",
                            data=export_metrics_workbook(metrics_display, average_bands),
                            file_name="speed_vs_acceleration_metrics.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    with metric_export_cols[1]:
                        if not metric_trendline_data.empty:
                            st.download_button(
                                "Download metric trendline Excel",
                                data=export_trendline_workbook(metric_trendline_data),
                                file_name="metric_trendline_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                except ImportError:
                    st.warning("Install the updated requirements to enable metrics Excel export.")

    return {
        "signal_map": {**signal_map, **get_shared_signals()} if signal_map else get_shared_signals(),
        "speed_output_unit": speed_output_unit,
        "acceleration_output_unit": acceleration_output_unit,
        "target_curves": target_curves,
        "scatter_curves": scatter_curves,
    }
