"""Reading ETAS INCA measurement files (MDF/MF4/DAT) with :mod:`asammdf`.

All functions accept either a filesystem path (``str``) or the raw file
contents (``bytes``). :mod:`asammdf` is an optional dependency; if it is not
installed a :class:`MdfDependencyError` is raised with actionable guidance so
the Streamlit layer can present a friendly message.
"""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd


class MdfDependencyError(RuntimeError):
    """Raised when the optional MDF reader dependency is unavailable."""


_DEPENDENCY_MESSAGE = (
    "Reading MDF/MF4/DAT files requires the 'asammdf' package. Install the "
    "project requirements (pip install -r requirements.txt) to enable "
    "measurement-file loading. Target-only plotting still works without it."
)


def _load_mdf(source: bytes | str):
    try:
        from asammdf import MDF
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise MdfDependencyError(_DEPENDENCY_MESSAGE) from exc

    if isinstance(source, (bytes, bytearray)):
        return MDF(io.BytesIO(bytes(source)))
    return MDF(source)


def discover_channel_names(source: bytes | str) -> list[str]:
    """Return the sorted, de-duplicated list of channel names in a file."""

    mdf = _load_mdf(source)
    try:
        names: set[str] = set()
        for name in mdf.channels_db:
            if name and name != "time":
                names.add(str(name))
        return sorted(names, key=str.casefold)
    finally:
        _safe_close(mdf)


def read_channel_unit(source: bytes | str, channel: str) -> str:
    """Return the engineering unit string reported for ``channel``."""

    if not channel:
        return ""
    mdf = _load_mdf(source)
    try:
        try:
            signal = mdf.get(channel)
        except Exception:
            return ""
        return str(getattr(signal, "unit", "") or "")
    finally:
        _safe_close(mdf)


def read_measurement_frame(
    source: bytes | str,
    channels: Iterable[str],
    raster_step_s: float | None = None,
) -> pd.DataFrame:
    """Read the requested channels into a single time-aligned DataFrame.

    The returned frame contains one column per requested channel plus a
    ``time`` column with the shared time base (in seconds). When
    ``raster_step_s`` is provided the signals are resampled onto a uniform time
    raster of that step, which greatly reduces memory usage for large files.
    """

    requested = [str(channel) for channel in channels if str(channel)]
    if not requested:
        return pd.DataFrame(columns=["time"])

    mdf = _load_mdf(source)
    try:
        available = [name for name in requested if name in mdf.channels_db]
        missing = [name for name in requested if name not in mdf.channels_db]
        if not available:
            raise ValueError(
                "None of the selected channels were found in this file: "
                + ", ".join(requested)
            )

        working = mdf
        resampled = None
        if raster_step_s and raster_step_s > 0:
            resampled = mdf.resample(raster=float(raster_step_s))
            working = resampled

        try:
            frame = working.to_dataframe(
                channels=available,
                time_as_date=False,
                use_display_names=False,
            )
        finally:
            if resampled is not None:
                _safe_close(resampled)

        frame = frame.reset_index()
        # asammdf names the index "timestamps"; standardise to "time".
        rename = {}
        for candidate in ("timestamps", "index", "time"):
            if candidate in frame.columns:
                rename[candidate] = "time"
                break
        frame = frame.rename(columns=rename)
        if "time" not in frame.columns:
            frame.insert(0, "time", range(len(frame)))

        for name in missing:
            frame[name] = pd.NA

        ordered = ["time"] + [name for name in requested if name in frame.columns]
        return frame[ordered]
    finally:
        _safe_close(mdf)


def _safe_close(mdf) -> None:
    close = getattr(mdf, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
