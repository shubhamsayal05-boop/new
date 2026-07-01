# Vehicle Speed vs Acceleration Plotter

A Streamlit application for analysing ETAS INCA measurement files (MDF / MF4 /
DAT). It extracts usable launch and deceleration data, averages repeated runs,
and plots **Speed vs Acceleration** curves. It can also overlay imported target
curves, plot arbitrary Excel-style X/Y scatter data, fit trendlines, and
calculate Speed vs Acceleration metrics.

## Features

- Read measurement files from a local folder (recommended for large files) or
  upload them in the browser.
- Select the four required MDF signals (Speed, Acceleration, Brake, Accelerator
  Pedal) with automatic unit detection and configurable graph/export units.
- Label files as Launch, Deceleration, or Ignore, apply mode/regen labels, and
  bulk-apply labels to multiple files.
- Usable-data extraction with configurable pedal tolerance, brake threshold,
  0% creep-launch detection, deceleration speed gating, speed-interval binning,
  and minimum usable points.
- Import or paste target Speed vs Acceleration curves from `.xlsx`, `.xlsm`,
  `.xls`, `.csv`, `.txt`, or directly from Excel.
- Optional random scatter plots for arbitrary X/Y data, including derived
  average and offset series.
- Metrics: peak acceleration, speed-band average acceleration, road-load speed
  (with optional linear-extension estimate), and slope between two speed points.
- Trendlines (linear through 5th-order polynomial) on random scatter and metric
  graphs, with equations and Excel export.
- Export the graph PNG, data Excel workbooks, metrics workbook, audit CSV, and
  project configuration JSON.

## Project layout

```
app.py                     # Streamlit entry point (UI and orchestration)
src/vehicle_plotter/       # Application logic package
  colors.py                # Colour parsing/validation
  exporting.py             # Speed vs Acceleration Excel export
  mdf_io.py                # MDF/MF4/DAT reading (via asammdf)
  metrics.py               # Metrics + metric figures
  plotting.py              # Plotly figures and styling helpers
  processing.py            # Usable-data extraction and averaging
  random_scatter.py        # Arbitrary X/Y import + derived series
  target_data.py           # Target curve import + unit conversion
  trendlines.py            # Polynomial trendline fitting/export
  units.py                 # Speed/acceleration unit definitions + conversions
```

## Getting started

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

## Notes

- `asammdf` is required to read MDF/MF4/DAT files. Target-only and random
  scatter plotting still work if it is not installed.
- PNG export requires `kaleido`.
- Excel export requires `openpyxl`.
