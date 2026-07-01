# DriveLab Pro

**DriveLab Pro** is a professional INCA measurement intelligence platform for `.mf4` and `.dat` files. It replaces the original single-page plotter with a modular workspace, dark engineering UI, and built-in **AI Advisor** for issue detection and fix suggestions.

## Workspaces

| Module | Purpose |
|--------|---------|
| **Dashboard** | Session health, KPIs, top diagnostics |
| **Data Source** | File loading, channel mapping, units |
| **Speed Analysis** | Launch/decel extraction, plots, metrics |
| **OPD Analysis** | One-pedal regen events, jerk, R13-H flags |
| **Reference Data** | Target curves & scatter imports |
| **AI Advisor** | Rule-based diagnostics + optional OpenAI action plan |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use **Read from local folder** for large MF4 files. Map INCA channels on **Data Source**, then run analysis on **Speed Analysis** or **OPD Analysis**.

## AI Advisor

The AI Advisor scans your session automatically and surfaces:

- Missing signal mappings
- Processing / memory errors
- Audit failures (no usable segments, low point count)
- OPD regulatory flags (UN R13-H, brake blending)
- Stale results after settings changes

Click **Scan session** on the AI Advisor page. Enable **Enhance with OpenAI** and provide an API key (or set `OPENAI_API_KEY` in `.streamlit/secrets.toml`) for a natural-language action plan.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```
