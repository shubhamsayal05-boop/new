"""Microsoft Copilot (Azure OpenAI) client for AI Advisor enhancements."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class CopilotConfig:
    """Azure OpenAI settings used by enterprise Microsoft Copilot deployments."""

    azure_endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-08-01-preview"

    @property
    def is_configured(self) -> bool:
        return bool(self.azure_endpoint.strip() and self.api_key.strip() and self.deployment.strip())


def load_copilot_config() -> CopilotConfig:
    """Load Copilot/Azure OpenAI settings from Streamlit secrets and environment."""

    endpoint = ""
    api_key = ""
    deployment = ""
    api_version = "2024-08-01-preview"

    try:
        import streamlit as st

        copilot = st.secrets.get("copilot", {})
        if isinstance(copilot, dict):
            endpoint = str(copilot.get("azure_endpoint") or copilot.get("endpoint") or "")
            api_key = str(copilot.get("api_key") or "")
            deployment = str(copilot.get("deployment") or copilot.get("deployment_name") or "")
            api_version = str(copilot.get("api_version") or api_version)
        endpoint = endpoint or str(st.secrets.get("AZURE_OPENAI_ENDPOINT", ""))
        api_key = api_key or str(st.secrets.get("AZURE_OPENAI_API_KEY", ""))
        deployment = deployment or str(st.secrets.get("AZURE_OPENAI_DEPLOYMENT", ""))
        api_version = str(st.secrets.get("AZURE_OPENAI_API_VERSION", api_version))
    except Exception:
        pass

    endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", api_version)

    return CopilotConfig(
        azure_endpoint=endpoint.strip(),
        api_key=api_key.strip(),
        deployment=deployment.strip(),
        api_version=api_version.strip(),
    )


def complete_with_copilot(system_prompt: str, user_prompt: str, config: CopilotConfig) -> str:
    """Send a chat completion request to Azure OpenAI (Microsoft Copilot backend)."""

    if not config.is_configured:
        raise RuntimeError(
            "Microsoft Copilot is not configured. Set copilot.azure_endpoint, copilot.api_key, "
            "and copilot.deployment in .streamlit/secrets.toml (or AZURE_OPENAI_* environment variables)."
        )

    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for Microsoft Copilot integration. Run: pip install openai"
        ) from exc

    client = AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.azure_endpoint,
    )
    response = client.chat.completions.create(
        model=config.deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return (response.choices[0].message.content or "").strip()
