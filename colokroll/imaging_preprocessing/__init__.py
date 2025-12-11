"""Imaging preprocessing utilities."""

from .background_subtraction import BackgroundSubtractor
from .z_slice_detection import (
    ZSliceSelectionResult,
    StrategyComparisonResult,
    aggregate_focus_scores,
    compute_focus_scores,
    detect_slices_to_keep,
    select_z_slices,
    compare_strategies,
    benchmark_focus_methods,
    auto_select_best_method,
)
from .focus_measure_quality import (
    FocusMeasureQuality,
    CurveSegmentation,
    compute_focus_measure_quality,
    extend_z_slice_result_with_quality,
    plot_focus_curve_analysis,
)

__all__ = [
    "BackgroundSubtractor",
    "ZSliceSelectionResult",
    "StrategyComparisonResult",
    "aggregate_focus_scores",
    "compute_focus_scores",
    "detect_slices_to_keep",
    "select_z_slices",
    "compare_strategies",
    "benchmark_focus_methods",
    "auto_select_best_method",
    # Quality metrics
    "FocusMeasureQuality",
    "CurveSegmentation",
    "compute_focus_measure_quality",
    "extend_z_slice_result_with_quality",
    "plot_focus_curve_analysis",
]


