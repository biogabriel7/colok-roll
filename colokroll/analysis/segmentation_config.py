"""
Utilities for configuring Cellpose segmentation, including retrieving the
Hugging Face token from environment variables or a local YAML config.
"""

import os
from pathlib import Path
from typing import Optional


def _read_token_from_yaml(config_path: Path) -> Optional[str]:
    """Try to read Hugging Face token from analysis/config.yaml.

    Expected structure:
    huggingface:
      token: "..."
    """
    try:
        import yaml  # Lazy import to avoid hard dependency elsewhere
    except Exception:
        return None

    if not config_path.exists():
        return None

    try:
        with config_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return (
            data.get("huggingface", {}).get("token")
            if isinstance(data, dict)
            else None
        )
    except Exception:
        return None


def get_hf_token(env_var: str = "HUGGINGFACE_TOKEN") -> str:
    """Return a Hugging Face token for authenticating to Spaces.

    Order of precedence:
    1) Environment variable `env_var` (default: HUGGINGFACE_TOKEN)
    2) Common alternates: HF_TOKEN, HUGGINGFACEHUB_API_TOKEN
    3) YAML file at analysis/config.yaml

    Raises:
        RuntimeError: if no token can be found.
    """
    # 1) Primary env var
    token = os.getenv(env_var)
    if token:
        return token

    # 2) Alternate env vars
    for alt in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        token = os.getenv(alt)
        if token:
            return token

    # 3) YAML fallback
    config_path = Path(__file__).with_name("config.yaml")
    token = _read_token_from_yaml(config_path)
    if token:
        return token

    raise RuntimeError(
        "Hugging Face token not found. Set HUGGINGFACE_TOKEN or provide analysis/config.yaml."
    )


__all__ = ["get_hf_token"]


