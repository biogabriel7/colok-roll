"""Z-slice selection package for microscopy stacks.

This package provides comprehensive tools for identifying and selecting
in-focus Z-slices from microscopy image stacks.
"""

from .core import (
    ZSliceSelectionResult,
    select_z_slices,
    FocusMethod,
    AggregationMethod,
    DetectionStrategy,
    NormalizationMode,
)
from .comparison import (
    StrategyComparisonResult,
    auto_select_best_method,
    benchmark_focus_methods,
    compare_strategies,
)
from .focus_metrics import compute_focus_scores
from .slice_selection import aggregate_focus_scores, detect_slices_to_keep

__all__ = [
    # Main API
    "select_z_slices",
    "ZSliceSelectionResult",
    # Comparison/benchmarking
    "auto_select_best_method",
    "benchmark_focus_methods",
    "compare_strategies",
    "StrategyComparisonResult",
    # Component functions (advanced users)
    "compute_focus_scores",
    "aggregate_focus_scores",
    "detect_slices_to_keep",
    # Type aliases
    "FocusMethod",
    "AggregationMethod",
    "DetectionStrategy",
    "NormalizationMode",
]

