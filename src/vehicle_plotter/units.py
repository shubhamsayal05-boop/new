"""Unit definitions and conversions for speed and acceleration data.

All conversions are performed by first converting a value to a canonical SI
base unit (metres per second for speed, metres per second squared for
acceleration) and then to the requested target unit.
"""

from __future__ import annotations

import re

import pandas as pd

SPEED_UNITS = ["KPH", "MPH", "m/s"]
ACCELERATION_UNITS = ["m/s^2", "g", "ft/s^2"]

_MS_PER_UNIT_SPEED = {
    "KPH": 1000.0 / 3600.0,
    "MPH": 1609.344 / 3600.0,
    "m/s": 1.0,
}

_MS2_PER_UNIT_ACCEL = {
    "m/s^2": 1.0,
    "g": 9.80665,
    "ft/s^2": 0.3048,
}

_SPEED_ALIASES = {
    "kph": "KPH",
    "km/h": "KPH",
    "kmh": "KPH",
    "kmph": "KPH",
    "kilometersperhour": "KPH",
    "kilometreperhour": "KPH",
    "mph": "MPH",
    "mi/h": "MPH",
    "milesperhour": "MPH",
    "m/s": "m/s",
    "mps": "m/s",
    "meterspersecond": "m/s",
    "metrespersecond": "m/s",
}

_ACCEL_ALIASES = {
    "m/s^2": "m/s^2",
    "m/s2": "m/s^2",
    "m/s²": "m/s^2",
    "ms^-2": "m/s^2",
    "ms-2": "m/s^2",
    "meterspersecondsquared": "m/s^2",
    "g": "g",
    "gs": "g",
    "g's": "g",
    "gravity": "g",
    "grav": "g",
    "ft/s^2": "ft/s^2",
    "ft/s2": "ft/s^2",
    "fps2": "ft/s^2",
    "feetpersecondsquared": "ft/s^2",
}


def _clean(raw: object) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    text = text.strip("[](){}")
    return text.strip()


def _key(raw: object) -> str:
    text = _clean(raw).casefold()
    return re.sub(r"\s+", "", text)


def normalize_speed_unit(detected: object) -> str:
    """Map a detected/free-form speed unit onto a canonical ``SPEED_UNITS`` value.

    Returns an empty string when the unit cannot be recognised.
    """

    key = _key(detected)
    if not key:
        return ""
    if key in _SPEED_ALIASES:
        return _SPEED_ALIASES[key]
    for alias, canonical in _SPEED_ALIASES.items():
        if alias in key:
            return canonical
    return ""


def normalize_acceleration_unit(detected: object) -> str:
    """Map a detected/free-form acceleration unit onto ``ACCELERATION_UNITS``.

    Returns an empty string when the unit cannot be recognised.
    """

    key = _key(detected)
    if not key:
        return ""
    if key in _ACCEL_ALIASES:
        return _ACCEL_ALIASES[key]
    # Longest aliases first so that "m/s^2" is preferred over the bare "g".
    for alias in sorted(_ACCEL_ALIASES, key=len, reverse=True):
        if alias in key:
            return _ACCEL_ALIASES[alias]
    return ""


def convert_speed_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a single speed value between supported speed units."""

    source = normalize_speed_unit(from_unit) or (from_unit if from_unit in _MS_PER_UNIT_SPEED else "KPH")
    target = normalize_speed_unit(to_unit) or (to_unit if to_unit in _MS_PER_UNIT_SPEED else "KPH")
    metres_per_second = float(value) * _MS_PER_UNIT_SPEED[source]
    return metres_per_second / _MS_PER_UNIT_SPEED[target]


def convert_acceleration_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a single acceleration value between supported acceleration units."""

    source = normalize_acceleration_unit(from_unit) or (
        from_unit if from_unit in _MS2_PER_UNIT_ACCEL else "m/s^2"
    )
    target = normalize_acceleration_unit(to_unit) or (
        to_unit if to_unit in _MS2_PER_UNIT_ACCEL else "m/s^2"
    )
    metres_per_second_squared = float(value) * _MS2_PER_UNIT_ACCEL[source]
    return metres_per_second_squared / _MS2_PER_UNIT_ACCEL[target]


def _speed_factor(from_unit: str, to_unit: str) -> float:
    source = normalize_speed_unit(from_unit) or (from_unit if from_unit in _MS_PER_UNIT_SPEED else "KPH")
    target = normalize_speed_unit(to_unit) or (to_unit if to_unit in _MS_PER_UNIT_SPEED else "KPH")
    return _MS_PER_UNIT_SPEED[source] / _MS_PER_UNIT_SPEED[target]


def _accel_factor(from_unit: str, to_unit: str) -> float:
    source = normalize_acceleration_unit(from_unit) or (
        from_unit if from_unit in _MS2_PER_UNIT_ACCEL else "m/s^2"
    )
    target = normalize_acceleration_unit(to_unit) or (
        to_unit if to_unit in _MS2_PER_UNIT_ACCEL else "m/s^2"
    )
    return _MS2_PER_UNIT_ACCEL[source] / _MS2_PER_UNIT_ACCEL[target]


def convert_measurement_units(
    frame: pd.DataFrame,
    speed_input_unit: str,
    speed_output_unit: str,
    acceleration_input_unit: str,
    acceleration_output_unit: str,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``speed`` and ``acceleration`` converted.

    The frame is expected to already be normalised (see
    :func:`vehicle_plotter.processing.normalize_measurement_frame`) so that it
    exposes ``speed`` and ``acceleration`` columns.
    """

    converted = frame.copy()
    if "speed" in converted.columns:
        converted["speed"] = pd.to_numeric(converted["speed"], errors="coerce") * _speed_factor(
            speed_input_unit, speed_output_unit
        )
    if "acceleration" in converted.columns:
        converted["acceleration"] = pd.to_numeric(
            converted["acceleration"], errors="coerce"
        ) * _accel_factor(acceleration_input_unit, acceleration_output_unit)
    return converted
