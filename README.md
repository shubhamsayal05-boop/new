# Vehicle Speed vs Acceleration Plotter

Streamlit analytics tool for ETAS INCA **`.mf4`** and **`.dat`** measurement files.

## Features

- **Speed vs Acceleration** — launch and deceleration curve extraction, target overlays, metrics, Excel/PNG export
- **One-Pedal (OPD) Analysis** — lift-off regen event detection, deceleration-vs-speed curves, jerk traces, UN R13-H flags, optional energy recovery KPIs

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Load measurement files via **Read from local folder** (recommended for large files) or **Upload in browser** (`.mf4` / `.dat` only).

## Project layout

- `app.py` — main Streamlit application
- `opd_tab.py` — One-Pedal Analysis tab UI
- `src/vehicle_plotter/` — processing, plotting, MDF I/O, OPD analytics

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```
