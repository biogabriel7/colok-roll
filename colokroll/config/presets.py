"""Factory and persistence helpers for configuration objects."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from .base import RuntimeConfig


def create_runtime_config(sections: Optional[Iterable[str]] = None) -> RuntimeConfig:
    """Create a new runtime configuration.

    Args:
        sections: Optional iterable of section names to initialize explicitly. This
            argument is accepted for compatibility but currently does not change the
            behaviour because all sections are always available.
    """

    if sections is not None:
        # We accept the parameter to stay API compatible with the old phase-based
        # helper but it no longer affects the result.
        sections = list(sections)  # consumption keeps behaviour defined
    return RuntimeConfig()


def save_config(config: RuntimeConfig, filepath: Union[str, Path], format: str = "auto") -> None:
    """Persist a configuration object to disk."""

    config.save(filepath, format=format)


def load_config(filepath: Union[str, Path]) -> RuntimeConfig:
    """Load a configuration object from a JSON or YAML file."""

    return RuntimeConfig.load(filepath)


