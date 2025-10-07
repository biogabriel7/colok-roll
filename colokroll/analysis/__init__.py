"""Analysis API for colokroll."""

from __future__ import annotations

import importlib
from typing import Any, Optional

__all__ = [
    "get_cell_segmenter",
    "get_colocalization_module",
    "NucleiDetector",
]


def _optional_import(module: str, attr: Optional[str] = None) -> Any:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr) if attr else mod
    except ImportError as exc:  # pragma: no cover - dependency missing path
        raise ImportError(
            f"Optional dependency for '{module}' is not installed. "
            "Install extras with `pip install colokroll[segmentation]`."
        ) from exc


def get_cell_segmenter() -> Any:
    """Return the CellSegmenter class, loading optional deps on demand."""
    return _optional_import("colokroll.analysis.segmentation", "CellSegmenter")


def get_colocalization_module() -> Any:
    """Return the colocalization interface module."""
    return _optional_import("colokroll.analysis.colocalization")


from .nuclei_detection import NucleiDetector  # noqa: E402