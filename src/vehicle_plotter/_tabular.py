"""Shared helpers for reading loosely-structured tabular imports.

Both the target-data and random-scatter importers accept Excel, CSV, tab
separated text, or data pasted straight out of Excel. This module centralises
the "read a messy table, drop the blanks, find the header row, pull out units"
logic used by both.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

_UNIT_RE = re.compile(r"[\(\[]\s*([^)\]]+?)\s*[\)\]]")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def read_raw_bytes(source_name: str, payload: bytes) -> pd.DataFrame:
    """Read an uploaded file's bytes into a header-less DataFrame."""

    suffix = Path(source_name).suffix.casefold()
    buffer = io.BytesIO(payload)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(buffer, header=None, dtype=object)
    text = payload.decode("utf-8-sig", errors="replace")
    return read_raw_text(text)


def read_raw_text(text: str) -> pd.DataFrame:
    """Read pasted/plain text into a header-less DataFrame, guessing the delimiter."""

    stripped = text.strip("\n")
    if not stripped.strip():
        return pd.DataFrame()
    if "\t" in stripped:
        sep = "\t"
    elif "," in stripped:
        sep = ","
    else:
        sep = r"\s+"
    return pd.read_csv(
        io.StringIO(stripped),
        header=None,
        dtype=object,
        sep=sep,
        engine="python",
        skip_blank_lines=True,
    )


def clean_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-empty rows and columns and reset the index."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    cleaned = frame.replace(r"^\s*$", pd.NA, regex=True)
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


def _looks_numeric(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text.replace(",", ""))
        return True
    except ValueError:
        return False


def split_header(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Return ``(headers, numeric_data)`` from a cleaned table.

    The first row is treated as a header when it contains any non-numeric cell;
    otherwise synthetic ``col0``, ``col1`` ... names are generated.
    """

    cleaned = clean_table(frame)
    if cleaned.empty:
        return [], pd.DataFrame()

    first_row = cleaned.iloc[0].tolist()
    has_header = any(not _looks_numeric(cell) for cell in first_row)

    if has_header:
        headers = [
            str(cell).strip() if cell is not None and str(cell).strip() else f"col{index}"
            for index, cell in enumerate(first_row)
        ]
        data = cleaned.iloc[1:].reset_index(drop=True)
    else:
        headers = [f"col{index}" for index in range(cleaned.shape[1])]
        data = cleaned

    numeric = data.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    numeric.columns = range(numeric.shape[1])
    return headers, numeric


def extract_unit(header: str) -> str:
    """Return the unit found inside parentheses/brackets in a header, if any."""

    if not header:
        return ""
    match = _UNIT_RE.search(str(header))
    return match.group(1).strip() if match else ""


def extract_number(header: str) -> float | None:
    """Return the first number embedded in a header string, or ``None``."""

    if header is None:
        return None
    match = _NUMBER_RE.search(str(header))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None
