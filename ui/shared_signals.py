"""Shared INCA signal selections synced across Speed Analysis and OPD workspaces."""

from __future__ import annotations

from typing import Callable

import streamlit as st

SHARED_CORE_KEY = "dl_shared_signals"
CORE_ROLES = ("speed", "acceleration", "brake", "pedal")

SPEED_WIDGET_KEYS: dict[str, str] = {
    "speed": "speed_signal",
    "acceleration": "accel_signal",
    "brake": "brake_signal",
    "pedal": "pedal_signal",
}


def opd_widget_keys(resettable_key: Callable[[str], str]) -> dict[str, str]:
    return {
        "speed": resettable_key("opd_speed_signal"),
        "acceleration": resettable_key("opd_accel_signal"),
        "brake": resettable_key("opd_brake_signal"),
        "pedal": resettable_key("opd_pedal_signal"),
    }


def get_shared_signals() -> dict[str, str]:
    return dict(st.session_state.get(SHARED_CORE_KEY, {}))


def ensure_shared_defaults(defaults: dict[str, str], valid_options: set[str]) -> None:
    """Seed shared core signals from auto-detect when nothing is selected yet."""

    shared = st.session_state.setdefault(SHARED_CORE_KEY, {})
    changed = False
    for role in CORE_ROLES:
        if shared.get(role):
            continue
        guess = defaults.get(role, "")
        if guess and guess in valid_options:
            shared[role] = guess
            changed = True
    if changed:
        st.session_state[SHARED_CORE_KEY] = shared


def sync_widgets_from_shared(
    workspace: str,
    resettable_key: Callable[[str], str],
    valid_options: set[str],
) -> None:
    """Apply shared selections to the active workspace widget keys before rendering."""

    shared = get_shared_signals()
    keys = SPEED_WIDGET_KEYS if workspace == "speed" else opd_widget_keys(resettable_key)
    for role, widget_key in keys.items():
        value = str(shared.get(role, "") or "")
        if value and value not in valid_options:
            value = ""
        if value or role in shared:
            st.session_state[widget_key] = value


def persist_shared_core_signals(
    workspace: str,
    resettable_key: Callable[[str], str],
    values: dict[str, str],
) -> None:
    """Save core signal picks and mirror them to the other workspace's widget keys."""

    shared = st.session_state.setdefault(SHARED_CORE_KEY, {})
    for role in CORE_ROLES:
        if role in values:
            shared[role] = str(values.get(role) or "")
    st.session_state[SHARED_CORE_KEY] = shared
    st.session_state["dl_signal_map"] = {**st.session_state.get("dl_signal_map", {}), **shared}

    if workspace == "speed":
        mirror_keys = opd_widget_keys(resettable_key)
    else:
        mirror_keys = SPEED_WIDGET_KEYS

    for role, widget_key in mirror_keys.items():
        st.session_state[widget_key] = shared.get(role, "")
