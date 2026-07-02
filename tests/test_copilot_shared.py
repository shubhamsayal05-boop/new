"""Tests for shared signal sync and Copilot config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_plotter.copilot_client import CopilotConfig, load_copilot_config


def test_copilot_config_not_configured_by_default():
    cfg = CopilotConfig(azure_endpoint="", api_key="", deployment="")
    assert not cfg.is_configured


def test_copilot_config_complete():
    cfg = CopilotConfig(
        azure_endpoint="https://test.openai.azure.com/",
        api_key="key",
        deployment="gpt-4o",
    )
    assert cfg.is_configured


def test_load_copilot_config_returns_dataclass():
    cfg = load_copilot_config()
    assert isinstance(cfg, CopilotConfig)
