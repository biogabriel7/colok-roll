"""Z-slice selection for microscopy stacks.

This module provides lightweight, per-slice focus metrics to identify which
Z-slices to retain for downstream analysis. It computes focus scores per slice
and applies configurable threshold-based selection strategies. Slices with
scores below the threshold are typically retained, while those above are excluded.

Designed to work with `ImageLoader` outputs, which are shaped `(Z, Y, X, C)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

# Import submodules
from .focus_metrics import compute_focus_scores
from .slice_selection import aggregate_focus_scores, detect_slices_to_keep
from .visualization import save_z_slice_plots

# Re-export for backward compatibility
from .comparison import (
    auto_select_best_method,
    benchmark_focus_methods,
    compare_strategies,
    StrategyComparisonResult,
)

if TYPE_CHECKING:
    from ..focus_measure_quality import FocusMeasureQuality


# Type aliases
FocusMethod = Literal["laplacian", "tenengrad", "fft", "combined"]
AggregationMethod = Literal["median", "mean", "max", "min", "weighted"]
DetectionStrategy = Literal["relative", "percentile", "topk", "closest_to_peak"]
NormalizationMode = Literal["per_slice", "per_stack", "percentile"]


@dataclass(frozen=True)
class ZSliceSelectionResult:
    """Container for Z-slice selection outputs.
    
    Attributes
    ----------
    scores_zc : np.ndarray
        Per-channel focus scores, shape (Z, C).
    scores_agg : np.ndarray
        Aggregated focus scores, shape (Z,).
    mask_keep : np.ndarray
        Boolean mask indicating which slices to keep, shape (Z,).
    indices_keep : np.ndarray
        Indices of slices to keep.
    indices_remove : np.ndarray
        Indices of slices to remove.
    mask_keep_zc : np.ndarray
        Per-channel keep masks, shape (Z, C).
    threshold_used : float
        Threshold value used for selection.
    smoothed_scores : np.ndarray
        Smoothed aggregated scores, shape (Z,).
    strategy : DetectionStrategy
        Detection strategy used.
    method : FocusMethod
        Focus method used.
    aggregation : AggregationMethod
        Aggregation method used.
    quality_metrics : Optional[FocusMeasureQuality]
        Piao et al. quality metrics for the focus curve. Only present if
        `compute_quality=True` was passed to `select_z_slices()`.
    """

    scores_zc: np.ndarray
    scores_agg: np.ndarray
    mask_keep: np.ndarray
    indices_keep: np.ndarray
    indices_remove: np.ndarray
    mask_keep_zc: np.ndarray
    threshold_used: float
    smoothed_scores: np.ndarray
    strategy: DetectionStrategy
    method: FocusMethod
    aggregation: AggregationMethod
    quality_metrics: Optional["FocusMeasureQuality"] = None


def select_z_slices(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    method: FocusMethod = "combined",
    aggregation: AggregationMethod = "median",
    strategy: DetectionStrategy = "relative",
    threshold: float = 0.6,
    smooth: int = 3,
    keep_top: Optional[int] = None,
    auto_keep_fraction: float = 0.8,
    normalize: bool = True,
    normalization_mode: NormalizationMode = "per_slice",
    clip_percent: float = 0.0,
    border_width: int = 0,
    gaussian_sigma: float = 0.0,
    min_variance_threshold: float = 0.0,
    fft_cutoff: float = 0.15,
    weights: Optional[Iterable[float]] = None,
    save_plot: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    compute_quality: bool = False,
    step_distance: float = 1.0,
    n_fitting_points: int = 5,
) -> ZSliceSelectionResult:
    """End-to-end helper: compute focus scores, aggregate, and select slices.
    
    This function computes focus scores for each Z-slice, aggregates them across
    channels, and applies a threshold-based selection strategy to determine which
    slices to keep for downstream analysis.

    Parameters
    ----------
    img:
        Input stack. Expected shape `(Z, Y, X, C)`.
    axes:
        Optional axis order string (e.g. ``"CZYX"``).
    method:
        Focus metric to use. Options: ``"laplacian"``, ``"tenengrad"``, ``"fft"``,
        or ``"combined"`` (default).
    aggregation:
        How to aggregate per-channel scores. Options: ``"median"`` (default),
        ``"mean"``, ``"max"``, ``"min"``, or ``"weighted"``.
    strategy:
        Detection strategy. Options: ``"relative"`` (default), ``"percentile"``,
        ``"topk"``, or ``"closest_to_peak"``.
    threshold:
        Threshold value. Interpretation depends on ``strategy``.
    smooth:
        Window size for smoothing scores before detection.
    keep_top:
        Number of slices to keep. Only used when ``strategy="topk"`` or
        ``strategy="closest_to_peak"``. If ``None`` with ``"closest_to_peak"``,
        automatically determines the number.
    auto_keep_fraction:
        For ``strategy="closest_to_peak"`` with ``keep_top=None``, keeps slices
        with scores >= this fraction of the peak score. Default 0.8 (80% of peak).
    normalize:
        If ``True``, each slice is normalized before scoring.
    normalization_mode:
        Normalization strategy. ``"per_slice"`` (default), ``"per_stack"``, or
        ``"percentile"``.
    clip_percent:
        Percentile clipping applied before normalization.
    border_width:
        Width of border (in pixels) to mask out. Default 0 (no masking).
    gaussian_sigma:
        Gaussian blur sigma applied before scoring. Default 0 (no blur).
    min_variance_threshold:
        Minimum variance for valid slices. Default 0 (no filtering).
    fft_cutoff:
        FFT high-frequency cutoff (0-1). Default 0.15.
    weights:
        Channel weights for ``aggregation="weighted"``.
    save_plot:
        If ``True``, automatically generates and saves visualization plots showing
        focus scores and a gallery of all Z-slices. Default ``False``.
    output_path:
        Directory path where plots will be saved. If ``None`` and ``save_plot=True``,
        saves to current working directory. Ignored if ``save_plot=False``.
    compute_quality:
        If ``True``, compute Piao et al. (2025) quality metrics (Ws, Rsg, Cp, etc.)
        for the focus curve. Useful for comparing/selecting focus methods.
        Default ``False``.
    step_distance:
        Physical distance between z-slices in micrometers. Used for computing
        quality metrics. Default 1.0 µm.
    n_fitting_points:
        Number of points to use for multi-point linear fitting in quality metrics.
        Typical values: 3 for small stacks (<20 slices), 5-7 for larger stacks.
        Default 5.

    Returns
    -------
    ZSliceSelectionResult:
        Result object containing scores, masks, indices, and metadata. If
        ``compute_quality=True``, the result will include quality metrics in
        the ``quality_metrics`` field.
    """

    # Step 1: Compute focus scores for each slice and channel
    scores_zc, scores_median = compute_focus_scores(
        img,
        axes=axes,
        method=method,
        normalize=normalize,
        normalization_mode=normalization_mode,
        clip_percent=clip_percent,
        border_width=border_width,
        gaussian_sigma=gaussian_sigma,
        min_variance_threshold=min_variance_threshold,
        fft_cutoff=fft_cutoff,
    )

    # Step 2: Aggregate scores across channels
    scores_agg = (
        scores_median
        if aggregation == "median" and weights is None
        else aggregate_focus_scores(scores_zc, aggregation=aggregation, weights=weights)
    )

    # Step 3: Detect which slices to keep
    detection = detect_slices_to_keep(
        scores_agg,
        strategy=strategy,
        threshold=threshold,
        smooth=smooth,
        keep_top=keep_top,
        auto_keep_fraction=auto_keep_fraction,
    )

    # Step 4: Per-channel masks (for advanced use cases)
    mask_keep_zc = np.empty_like(scores_zc, dtype=bool)
    for channel in range(scores_zc.shape[1]):
        channel_detection = detect_slices_to_keep(
            scores_zc[:, channel],
            strategy=strategy,
            threshold=threshold,
            smooth=smooth,
            keep_top=keep_top,
            auto_keep_fraction=auto_keep_fraction,
        )
        mask_keep_zc[:, channel] = channel_detection["mask_keep"]

    # Step 5: Compute quality metrics if requested
    quality_metrics = None
    if compute_quality:
        from ..focus_measure_quality import compute_focus_measure_quality
        quality_metrics = compute_focus_measure_quality(
            scores_agg,
            method=str(method),
            step_distance=step_distance,
            n_fitting_points=n_fitting_points,
        )
    
    # Step 6: Build result object
    result = ZSliceSelectionResult(
        scores_zc=scores_zc,
        scores_agg=scores_agg,
        mask_keep=detection["mask_keep"],
        indices_keep=detection["indices_keep"],
        indices_remove=detection["indices_remove"],
        mask_keep_zc=mask_keep_zc,
        threshold_used=float(detection["threshold_used"]),
        smoothed_scores=detection["smoothed_scores"],
        strategy=strategy,
        method=method,
        aggregation=aggregation,
        quality_metrics=quality_metrics,
    )
    
    # Step 7: Generate visualization if requested
    if save_plot:
        save_z_slice_plots(
            img=img,
            result=result,
            output_path=output_path,
        )
    
    return result


# Re-export these for backward compatibility
__all__ = [
    # Main API
    "select_z_slices",
    "ZSliceSelectionResult",
    # Comparison/benchmarking
    "auto_select_best_method",
    "benchmark_focus_methods",
    "compare_strategies",
    "StrategyComparisonResult",
    # Utility functions (advanced users)
    "compute_focus_scores",
    "aggregate_focus_scores",
    "detect_slices_to_keep",
    # Type aliases
    "FocusMethod",
    "AggregationMethod",
    "DetectionStrategy",
    "NormalizationMode",
]
