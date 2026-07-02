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

## AI Advisor (Microsoft Copilot)

Rule-based diagnostics run automatically. For natural-language action plans, enable **Enhance with Microsoft Copilot** on the AI Advisor page.

Configure your company's Azure OpenAI deployment in `.streamlit/secrets.toml`:

```toml
[copilot]
azure_endpoint = "https://YOUR-RESOURCE.openai.azure.com/"
api_key = "YOUR-KEY"
deployment = "YOUR-DEPLOYMENT"
api_version = "2024-08-01-preview"
```

See `.streamlit/secrets.toml.example` for a template.

## Shared signal mapping

Speed, acceleration, brake, and pedal channels selected on **Data Source** or **OPD Analysis** are **automatically synced** to the other workspace — you only map them once per session.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```
