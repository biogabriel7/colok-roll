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
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


FocusMethod = Literal["laplacian", "tenengrad", "fft", "combined"]
AggregationMethod = Literal["median", "mean", "max", "min", "weighted"]
DetectionStrategy = Literal["relative", "percentile", "topk", "closest_to_peak"]
NormalizationMode = Literal["per_slice", "per_stack", "percentile"]


@dataclass(frozen=True)
class ZSliceSelectionResult:
    """Container for Z-slice selection outputs."""

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


@dataclass(frozen=True)
class StrategyComparisonResult:
    """Container for multi-strategy comparison outputs."""
    
    strategy_names: Tuple[str, ...]
    results: Dict[str, ZSliceSelectionResult]
    decision_matrix: np.ndarray
    score_matrix: np.ndarray
    n_slices: int
    n_strategies: int


def compute_focus_scores(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    method: FocusMethod = "combined",
    normalize: bool = True,
    normalization_mode: NormalizationMode = "per_slice",
    clip_percent: float = 0.0,
    border_width: int = 0,
    gaussian_sigma: float = 0.0,
    min_variance_threshold: float = 0.0,
    fft_cutoff: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-slice focus scores for each channel.

    Parameters
    ----------
    img:
        Input stack. Expected shape `(Z, Y, X, C)` when sourced from
        :class:`~colokroll.data_processing.image_loader.ImageLoader`. If a
        different axis order is supplied, specify it with ``axes``.
    axes:
        Optional axis order string (e.g. ``"CZYX"``). Axes beyond ``Z``, ``Y``,
        ``X``, ``C`` must be singleton; they are squeezed automatically.
    method:
        Focus metric to use. ``"combined"`` applies Laplacian variance and
        Tenengrad, z-normalises them per channel, and averages the results.
    normalize:
        If ``True`` (default), each slice is min-max normalised prior to focus
        scoring to reduce intensity bias across the stack.
    normalization_mode:
        Normalization strategy. ``"per_slice"`` (default) normalizes each slice
        independently; ``"per_stack"`` uses global min/max across Z;
        ``"percentile"`` uses 1st-99th percentile range per slice.
    clip_percent:
        Symmetric percentile clipping applied before normalisation. For example,
        ``clip_percent=1`` clips to the 1st–99th percentile range.
    border_width:
        Width of border (in pixels) to mask out before scoring. Helps avoid
        coverslip and edge artifacts. Default 0 (no masking).
    gaussian_sigma:
        Standard deviation for Gaussian blur applied before scoring. Helps
        suppress sensor noise. Default 0 (no blur).
    min_variance_threshold:
        Minimum variance threshold. Slices with variance below this value are
        assigned a score of 0 (treated as flat/empty). Default 0 (no filtering).
    fft_cutoff:
        Radial frequency cutoff for FFT method, as fraction of maximum frequency.
        Default 0.15. Only used when ``method="fft"``.

    Returns
    -------
    scores_zc, scores_agg:
        ``scores_zc`` has shape ``(Z, C)`` (per-channel scores). ``scores_agg``
        collapses the channel axis via the median and has shape ``(Z,)``.
    """

    stack = _ensure_zyxc(np.asarray(img), axes)
    stack = stack.astype(np.float32, copy=False)

    if clip_percent < 0 or clip_percent >= 50:
        raise ValueError("clip_percent must be within [0, 50).")
    
    if border_width < 0:
        raise ValueError("border_width must be non-negative.")
    
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be non-negative.")
    
    if min_variance_threshold < 0:
        raise ValueError("min_variance_threshold must be non-negative.")
    
    if not 0 < fft_cutoff < 1:
        raise ValueError("fft_cutoff must be in (0, 1).")

    # Compute global statistics if needed for per-stack normalization
    global_stats = None
    if normalize and normalization_mode == "per_stack":
        global_stats = _compute_global_stats(stack, clip_percent)

    if method == "combined":
        laplacian = _score_volume(
            stack,
            _laplacian_variance,
            normalize=normalize,
            normalization_mode=normalization_mode,
            clip_percent=clip_percent,
            border_width=border_width,
            gaussian_sigma=gaussian_sigma,
            min_variance_threshold=min_variance_threshold,
            global_stats=global_stats,
            fft_cutoff=fft_cutoff,
        )
        tenengrad = _score_volume(
            stack,
            _tenengrad_score,
            normalize=normalize,
            normalization_mode=normalization_mode,
            clip_percent=clip_percent,
            border_width=border_width,
            gaussian_sigma=gaussian_sigma,
            min_variance_threshold=min_variance_threshold,
            global_stats=global_stats,
            fft_cutoff=fft_cutoff,
        )
        scores_zc = _combine_scores(laplacian, tenengrad)
    else:
        scorer = _resolve_scorer(method, fft_cutoff=fft_cutoff)
        scores_zc = _score_volume(
            stack,
            scorer,
            normalize=normalize,
            normalization_mode=normalization_mode,
            clip_percent=clip_percent,
            border_width=border_width,
            gaussian_sigma=gaussian_sigma,
            min_variance_threshold=min_variance_threshold,
            global_stats=global_stats,
            fft_cutoff=fft_cutoff,
        )

    scores_agg = np.median(scores_zc, axis=1).astype(np.float32, copy=False)

    return scores_zc, scores_agg


def aggregate_focus_scores(
    scores_zc: np.ndarray,
    *,
    aggregation: AggregationMethod = "median",
    weights: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Aggregate per-channel focus scores into a single per-slice series."""

    scores_zc = np.asarray(scores_zc, dtype=np.float32)
    if scores_zc.ndim != 2:
        raise ValueError(f"scores_zc must be 2-D (Z, C); got shape {scores_zc.shape}")

    if aggregation == "median":
        return np.median(scores_zc, axis=1).astype(np.float32, copy=False)
    if aggregation == "mean":
        return np.mean(scores_zc, axis=1).astype(np.float32, copy=False)
    if aggregation == "max":
        return np.max(scores_zc, axis=1).astype(np.float32, copy=False)
    if aggregation == "min":
        return np.min(scores_zc, axis=1).astype(np.float32, copy=False)
    if aggregation == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for weighted aggregation")
        weights_arr = np.asarray(list(weights), dtype=np.float32)
        if weights_arr.ndim != 1:
            raise ValueError("weights must be a 1-D iterable")
        if weights_arr.shape[0] != scores_zc.shape[1]:
            raise ValueError(
                "weights length must match number of channels; "
                f"expected {scores_zc.shape[1]}, got {weights_arr.shape[0]}"
            )
        weight_sum = weights_arr.sum()
        if not np.isfinite(weight_sum) or weight_sum == 0:
            raise ValueError("weights must sum to a finite, non-zero value")
        normalized = weights_arr / weight_sum
        return np.matmul(scores_zc, normalized).astype(np.float32, copy=False)

    raise ValueError(f"Unknown aggregation method: {aggregation}")


def detect_slices_to_keep(
    scores: np.ndarray,
    *,
    strategy: DetectionStrategy = "relative",
    threshold: float = 0.6,
    smooth: int = 3,
    keep_top: Optional[int] = None,
    auto_keep_fraction: float = 0.8,
) -> Dict[str, Any]:
    """Determine which slices to keep for analysis based on their scores.
    
    Slices with scores below the threshold are typically kept for downstream
    analysis, while those above are excluded.

    Parameters
    ----------
    scores:
        1-D array of per-slice focus scores.
    strategy:
        Detection strategy. ``"relative"`` computes threshold as a fraction of
        the median score; ``"percentile"`` uses a percentile cutoff; ``"topk"``
        keeps the bottom-k slices; ``"closest_to_peak"`` keeps the k slices
        with scores closest to the peak score.
    threshold:
        Threshold value. For ``"relative"``, this is a multiplier of the median
        score. For ``"percentile"``, this is a percentile value in [0, 100].
        Ignored for ``"topk"`` and ``"closest_to_peak"``.
    smooth:
        Window size for smoothing scores before detection. Must be odd.
    keep_top:
        Number of slices to keep. Only used when ``strategy="topk"`` or
        ``strategy="closest_to_peak"``. If ``None`` and using ``"closest_to_peak"``,
        automatically determines the number based on ``auto_keep_fraction``.
    auto_keep_fraction:
        For ``strategy="closest_to_peak"`` with ``keep_top=None``, keeps all slices
        with scores >= ``auto_keep_fraction * peak_score``. Default 0.8 (80% of peak).
        Only used for automatic determination.

    Returns
    -------
    dict:
        Dictionary with keys: ``"scores"``, ``"smoothed_scores"``, ``"mask_keep"``,
        ``"indices_keep"``, ``"indices_remove"``, ``"threshold_used"``, ``"strategy"``.
    """

    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D; got shape {scores.shape}")

    smoothed = _smooth_scores(scores, window=smooth)

    if strategy == "relative":
        if threshold <= 0:
            raise ValueError("threshold must be > 0 for relative strategy")
        baseline = float(np.median(smoothed))
        threshold_value = baseline * float(threshold)
    elif strategy == "percentile":
        if not 0 <= threshold <= 100:
            raise ValueError("threshold must be in [0,100] for percentile strategy")
        threshold_value = float(np.percentile(smoothed, threshold))
    elif strategy == "topk":
        if keep_top is None or keep_top <= 0:
            raise ValueError("keep_top must be a positive integer for topk strategy")
        keep_top = min(keep_top, smoothed.shape[0])
        # Find bottom-k slices (lowest scores)
        sort_idx = np.argsort(smoothed)
        indices_keep = np.sort(sort_idx[:keep_top])
        indices_remove = np.sort(sort_idx[keep_top:])
        mask_keep = np.zeros_like(smoothed, dtype=bool)
        mask_keep[indices_keep] = True
        return {
            "scores": scores,
            "smoothed_scores": smoothed,
            "mask_keep": mask_keep,
            "indices_keep": indices_keep,
            "indices_remove": indices_remove,
            "threshold_used": float(np.max(smoothed[indices_keep])),
            "strategy": strategy,
        }
    elif strategy == "closest_to_peak":
        # Find peak (highest score)
        peak_idx = int(np.argmax(smoothed))
        peak_score = float(smoothed[peak_idx])
        
        # Automatic determination if keep_top not specified
        if keep_top is None:
            # Keep all slices with scores >= auto_keep_fraction * peak_score
            score_threshold = peak_score * auto_keep_fraction
            indices_keep = np.nonzero(smoothed >= score_threshold)[0]
            
            # Ensure at least 1 slice is kept
            if len(indices_keep) == 0:
                indices_keep = np.array([peak_idx], dtype=np.intp)
            
            # Cap at 80% of total slices to avoid keeping too many
            max_keep = max(1, int(0.8 * smoothed.shape[0]))
            if len(indices_keep) > max_keep:
                # Sort by score and keep top max_keep
                score_sorted_idx = np.argsort(smoothed)[::-1]
                indices_keep = np.sort(score_sorted_idx[:max_keep])
            
            indices_remove = np.setdiff1d(np.arange(smoothed.shape[0]), indices_keep)
            mask_keep = np.zeros_like(smoothed, dtype=bool)
            mask_keep[indices_keep] = True
            
            return {
                "scores": scores,
                "smoothed_scores": smoothed,
                "mask_keep": mask_keep,
                "indices_keep": indices_keep,
                "indices_remove": indices_remove,
                "threshold_used": score_threshold,
                "strategy": strategy,
            }
        
        # Manual specification of keep_top
        if keep_top <= 0:
            raise ValueError("keep_top must be a positive integer for closest_to_peak strategy")
        keep_top = min(keep_top, smoothed.shape[0])
        
        # Calculate distance from peak score
        dist = np.abs(smoothed - smoothed[peak_idx])
        # Keep k slices with scores closest to peak
        keep_order = np.argsort(dist)
        indices_keep = np.sort(keep_order[:keep_top])
        indices_remove = np.sort(keep_order[keep_top:])
        mask_keep = np.zeros_like(smoothed, dtype=bool)
        mask_keep[indices_keep] = True
        return {
            "scores": scores,
            "smoothed_scores": smoothed,
            "mask_keep": mask_keep,
            "indices_keep": indices_keep,
            "indices_remove": indices_remove,
            "threshold_used": float(np.max(smoothed[indices_keep])),
            "strategy": strategy,
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Slices below threshold are kept
    mask_keep = smoothed < threshold_value
    indices_keep = np.nonzero(mask_keep)[0]
    indices_remove = np.nonzero(~mask_keep)[0]

    return {
        "scores": scores,
        "smoothed_scores": smoothed,
        "mask_keep": mask_keep,
        "indices_keep": indices_keep,
        "indices_remove": indices_remove,
        "threshold_used": threshold_value,
        "strategy": strategy,
    }


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

    Returns
    -------
    ZSliceSelectionResult:
        Result object containing scores, masks, indices, and metadata.
    """

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

    scores_agg = (
        scores_median
        if aggregation == "median" and weights is None
        else aggregate_focus_scores(scores_zc, aggregation=aggregation, weights=weights)
    )

    detection = detect_slices_to_keep(
        scores_agg,
        strategy=strategy,
        threshold=threshold,
        smooth=smooth,
        keep_top=keep_top,
        auto_keep_fraction=auto_keep_fraction,
    )

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
    )
    
    if save_plot:
        _save_z_slice_plots(
            img=img,
            result=result,
            output_path=output_path,
        )
    
    return result


def compare_strategies(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    strategies: Optional[list] = None,
    output_path: Optional[Union[str, Path]] = None,
    save_plots: bool = True,
) -> StrategyComparisonResult:
    """Compare multiple detection strategies side-by-side.
    
    Runs multiple strategy configurations and generates comparison visualizations
    including a decision matrix heatmap and slice gallery for visual inspection.
    
    Parameters
    ----------
    img:
        Input stack. Expected shape `(Z, Y, X, C)`.
    axes:
        Optional axis order string (e.g. ``"CZYX"``).
    strategies:
        List of strategy configurations to compare. Each item should be a dict
        with parameters for `select_z_slices()`. If None, uses a default set
        of common strategies.
    output_path:
        Directory path where comparison plots will be saved. If None, uses
        current working directory.
    save_plots:
        If True (default), generates and saves comparison visualizations.
    
    Returns
    -------
    StrategyComparisonResult:
        Object containing results for all strategies plus comparison matrices.
    
    Examples
    --------
    >>> # Compare with default strategies
    >>> comparison = cr.compare_strategies(stack, output_path="comparison_results")
    >>> 
    >>> # Compare custom strategies
    >>> strategies = [
    ...     {"name": "FFT Auto", "method": "fft", "strategy": "closest_to_peak"},
    ...     {"name": "Combined k=15", "method": "combined", "strategy": "closest_to_peak", "keep_top": 15},
    ...     {"name": "Laplacian", "method": "laplacian", "strategy": "relative", "threshold": 0.6},
    ... ]
    >>> comparison = cr.compare_strategies(stack, strategies=strategies)
    """
    
    # Default strategies if none provided
    if strategies is None:
        strategies = [
            {
                "name": "FFT + Closest (Auto 0.8)",
                "method": "fft",
                "strategy": "closest_to_peak",
                "auto_keep_fraction": 0.8,
            },
            {
                "name": "FFT + Closest (Auto 0.7)",
                "method": "fft",
                "strategy": "closest_to_peak",
                "auto_keep_fraction": 0.7,
            },
            {
                "name": "FFT + Closest (k=14)",
                "method": "fft",
                "strategy": "closest_to_peak",
                "keep_top": 14,
            },
            {
                "name": "Combined + Closest (Auto)",
                "method": "combined",
                "strategy": "closest_to_peak",
                "auto_keep_fraction": 0.8,
            },
            {
                "name": "FFT + TopK (k=20)",
                "method": "fft",
                "strategy": "topk",
                "keep_top": 20,
            },
            {
                "name": "FFT + Relative (0.6)",
                "method": "fft",
                "strategy": "relative",
                "threshold": 0.6,
            },
        ]
    
    # Run all strategies
    print(f"Comparing {len(strategies)} strategies...")
    results = {}
    strategy_names = []
    
    for i, config in enumerate(strategies, 1):
        name = config.pop("name", f"Strategy {i}")
        strategy_names.append(name)
        print(f"  [{i}/{len(strategies)}] Running: {name}")
        
        result = select_z_slices(img, axes=axes, **config)
        results[name] = result
        
        print(f"    → Kept {len(result.indices_keep)} / {len(result.scores_agg)} slices")
    
    # Build comparison matrices
    n_slices = len(next(iter(results.values())).scores_agg)
    n_strategies = len(strategy_names)
    
    decision_matrix = np.zeros((n_slices, n_strategies), dtype=bool)
    score_matrix = np.zeros((n_slices, n_strategies), dtype=np.float32)
    
    for j, name in enumerate(strategy_names):
        decision_matrix[:, j] = results[name].mask_keep
        score_matrix[:, j] = results[name].smoothed_scores
    
    comparison_result = StrategyComparisonResult(
        strategy_names=tuple(strategy_names),
        results=results,
        decision_matrix=decision_matrix,
        score_matrix=score_matrix,
        n_slices=n_slices,
        n_strategies=n_strategies,
    )
    
    # Generate comparison plots if requested
    if save_plots:
        if output_path is None:
            output_dir = Path.cwd() / "strategy_comparison"
        else:
            output_dir = Path(output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating comparison visualizations in {output_dir}...")
        _plot_strategy_comparison(img, comparison_result, output_dir, axes)
        print("✓ Comparison complete!")
    
    return comparison_result


def _save_z_slice_plots(
    img: np.ndarray,
    result: ZSliceSelectionResult,
    output_path: Optional[Union[str, Path]],
) -> None:
    """Generate and save visualization plots for Z-slice selection results.
    
    Creates two plots:
    1. Score plot: Shows raw and smoothed focus scores with threshold
    2. Gallery plot: Grid of all Z-slices with their scores
    
    Parameters
    ----------
    img:
        Original input stack (Z, Y, X, C).
    result:
        Z-slice selection result object.
    output_path:
        Directory to save plots. If None, uses current working directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn(
            "matplotlib not available; skipping plot generation. "
            "Install matplotlib to enable visualization.",
            UserWarning,
        )
        return
    
    # Determine output directory
    if output_path is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure image is in ZYXC format
    stack = _ensure_zyxc(img, axes=None)
    
    # 1. Generate score plot
    _plot_z_slice_scores(result, output_dir)
    
    # 2. Generate gallery plot
    _plot_z_slice_gallery(stack, result, output_dir)


def _plot_z_slice_scores(
    result: ZSliceSelectionResult,
    output_dir: Path,
) -> None:
    """Plot focus scores over Z with threshold and selection.
    
    Parameters
    ----------
    result:
        Z-slice selection result object.
    output_dir:
        Directory to save the plot.
    """
    import matplotlib.pyplot as plt
    
    scores = result.scores_agg
    smoothed = result.smoothed_scores
    mask_keep = result.mask_keep
    z = np.arange(scores.shape[0])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z, scores, label="scores", alpha=0.4, color="#1f77b4")
    ax.plot(z, smoothed, label="smoothed", color="#d62728")
    ax.hlines(
        result.threshold_used,
        z.min(),
        z.max(),
        colors="#7f7f7f",
        linestyles="--",
        label="threshold",
    )
    ax.scatter(z[mask_keep], scores[mask_keep], color="#2ca02c", s=20, label="keep")
    ax.scatter(z[~mask_keep], scores[~mask_keep], color="#ff7f0e", s=20, label="remove")
    ax.set_xlabel("Z index")
    ax.set_ylabel("Focus score (aggregated)")
    ax.legend()
    ax.set_title("Z-slice focus scores and selection")
    fig.tight_layout()
    
    out_path = output_dir / "z_slice_scores.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved score plot to {out_path}")


def _plot_z_slice_gallery(
    stack: np.ndarray,
    result: ZSliceSelectionResult,
    output_dir: Path,
) -> None:
    """Plot gallery of all Z-slices with their scores.
    
    Parameters
    ----------
    stack:
        Image stack in ZYXC format.
    result:
        Z-slice selection result object.
    output_dir:
        Directory to save the plot.
    """
    import matplotlib.pyplot as plt
    
    z_total = stack.shape[0]
    n_channels = stack.shape[-1]
    cols = 6
    rows = int(np.ceil(z_total / cols))
    
    # Create RGB composite for each slice
    composite_slices = _create_channel_composites(stack)
    
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    
    for ax, idx in zip(axes, range(z_total)):
        ax.imshow(composite_slices[idx])
        ax.set_title(f"z={idx} score={result.scores_agg[idx]:.3f}", fontsize=8)
        ax.axis("off")
    
    # Hide unused subplots
    for ax in axes[z_total:]:
        ax.axis("off")
    
    fig2.suptitle(f"Per-slice composite ({n_channels} channels)", y=0.99)
    fig2.tight_layout()
    
    out_path2 = output_dir / "z_slice_gallery.png"
    fig2.savefig(out_path2, dpi=200)
    plt.close(fig2)
    print(f"Saved gallery to {out_path2}")


def _ensure_zyxc(img: np.ndarray, axes: Optional[str]) -> np.ndarray:
    arr = np.asarray(img)

    if axes is None:
        if arr.ndim == 4:
            return arr
        if arr.ndim == 3:
            return arr[..., np.newaxis]
        raise ValueError(f"Image must be 3-D or 4-D when axes=None; got shape {arr.shape}")

    axes = axes.upper()
    axes = axes.replace(" ", "")

    if arr.ndim != len(axes):
        raise ValueError(
            f"axes string '{axes}' has length {len(axes)}, but array has ndim={arr.ndim}"
        )

    # Remove singleton ancillary axes (e.g., T=1)
    removable = [i for i, ax in enumerate(axes) if ax not in {"Z", "Y", "X", "C"}]
    for offset, idx in enumerate(removable):
        actual_idx = idx - offset
        if arr.shape[actual_idx] != 1:
            raise ValueError(
                f"Cannot squeeze axis '{axes[actual_idx]}' of length {arr.shape[actual_idx]}"
            )
        arr = np.take(arr, 0, axis=actual_idx)
        axes = axes[:actual_idx] + axes[actual_idx + 1 :]

    for required in ("Z", "Y", "X"):
        if required not in axes:
            raise ValueError(f"Axis '{required}' missing from axes='{axes}'")

    order = [axes.index("Z"), axes.index("Y"), axes.index("X")]
    if "C" in axes:
        order.append(axes.index("C"))

    arr = np.transpose(arr, order)

    if "C" not in axes:
        arr = arr[..., np.newaxis]

    return arr


def _score_volume(
    stack: np.ndarray,
    scorer: Callable[[np.ndarray], float],
    *,
    normalize: bool,
    normalization_mode: NormalizationMode,
    clip_percent: float,
    border_width: int,
    gaussian_sigma: float,
    min_variance_threshold: float,
    global_stats: Optional[Dict[str, Tuple[float, float]]],
    fft_cutoff: float,
) -> np.ndarray:
    z, _, _, c = stack.shape
    scores = np.empty((z, c), dtype=np.float32)

    for channel in range(c):
        for plane in range(z):
            slice_2d = stack[plane, :, :, channel]
            
            # Check minimum variance threshold (sanity filter)
            if min_variance_threshold > 0:
                slice_var = float(np.var(slice_2d, dtype=np.float32))
                if slice_var < min_variance_threshold:
                    scores[plane, channel] = 0.0
                    continue
            
            # Get global stats for this channel if needed
            channel_stats = global_stats.get(channel) if global_stats else None
            
            prepared = _prepare_slice(
                slice_2d,
                normalize=normalize,
                normalization_mode=normalization_mode,
                clip_percent=clip_percent,
                border_width=border_width,
                gaussian_sigma=gaussian_sigma,
                global_stats=channel_stats,
            )
            scores[plane, channel] = float(scorer(prepared))

    return scores


def _prepare_slice(
    slice_2d: np.ndarray,
    *,
    normalize: bool,
    normalization_mode: NormalizationMode,
    clip_percent: float,
    border_width: int,
    gaussian_sigma: float,
    global_stats: Optional[Tuple[float, float]],
) -> np.ndarray:
    arr = np.asarray(slice_2d, dtype=np.float32)

    # Apply border mask if requested
    if border_width > 0:
        mask = np.ones_like(arr, dtype=bool)
        mask[:border_width, :] = False
        mask[-border_width:, :] = False
        mask[:, :border_width] = False
        mask[:, -border_width:] = False
        arr = arr * mask

    # Apply Gaussian blur if requested
    if gaussian_sigma > 0:
        arr = ndimage.gaussian_filter(arr, sigma=gaussian_sigma, mode="reflect")

    # Apply clipping
    if clip_percent > 0:
        if normalization_mode == "percentile":
            # For percentile mode, we'll use 1st-99th percentile in normalization
            pass
        else:
            lower = np.percentile(arr, clip_percent)
            upper = np.percentile(arr, 100 - clip_percent)
            if upper > lower:
                arr = np.clip(arr, lower, upper)

    # Apply normalization
    if normalize:
        if normalization_mode == "per_slice":
            # Original per-slice min-max normalization
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
        
        elif normalization_mode == "per_stack":
            # Use global min/max across the entire stack
            if global_stats is not None:
                mn, mx = global_stats
                if mx > mn:
                    arr = np.clip(arr, mn, mx)
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
            else:
                # Fallback to per-slice if global stats not provided
                mn = float(np.min(arr))
                mx = float(np.max(arr))
                if mx > mn:
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
        
        elif normalization_mode == "percentile":
            # Use 1st-99th percentile for robust normalization
            p1 = float(np.percentile(arr, 1))
            p99 = float(np.percentile(arr, 99))
            if p99 > p1:
                arr = np.clip(arr, p1, p99)
                arr = (arr - p1) / (p99 - p1)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)

    return arr


def _combine_scores(laplacian: np.ndarray, tenengrad: np.ndarray) -> np.ndarray:
    if laplacian.shape != tenengrad.shape:
        raise ValueError(
            "laplacian and tenengrad scores must share the same shape; "
            f"got {laplacian.shape} vs {tenengrad.shape}"
        )
    laplacian_z = _zscore(laplacian)
    tenengrad_z = _zscore(tenengrad)
    combined = 0.5 * (laplacian_z + tenengrad_z)
    return combined.astype(np.float32, copy=False)


def _compute_global_stats(
    stack: np.ndarray,
    clip_percent: float,
) -> Dict[int, Tuple[float, float]]:
    """Compute global min/max statistics per channel across the entire stack.
    
    Parameters
    ----------
    stack:
        Input stack with shape (Z, Y, X, C).
    clip_percent:
        Percentile clipping to apply before computing min/max.
    
    Returns
    -------
    dict:
        Dictionary mapping channel index to (min, max) tuple.
    """
    z, y, x, c = stack.shape
    stats = {}
    
    for channel in range(c):
        channel_data = stack[:, :, :, channel].ravel()
        
        if clip_percent > 0:
            lower = np.percentile(channel_data, clip_percent)
            upper = np.percentile(channel_data, 100 - clip_percent)
            channel_data = np.clip(channel_data, lower, upper)
        
        mn = float(np.min(channel_data))
        mx = float(np.max(channel_data))
        stats[channel] = (mn, mx)
    
    return stats


def _resolve_scorer(method: FocusMethod, fft_cutoff: float = 0.15):
    if method == "laplacian":
        return _laplacian_variance
    if method == "tenengrad":
        return _tenengrad_score
    if method == "fft":
        # Return a partial function with the cutoff parameter
        return lambda slice_2d: _fft_high_frequency_energy(slice_2d, cutoff=fft_cutoff)

    raise ValueError(f"Unknown focus method: {method}")


def _laplacian_variance(slice_2d: np.ndarray) -> float:
    lap = ndimage.laplace(slice_2d, mode="reflect")
    return float(np.var(lap, dtype=np.float32))


def _tenengrad_score(slice_2d: np.ndarray) -> float:
    gx = ndimage.sobel(slice_2d, axis=0, mode="reflect")
    gy = ndimage.sobel(slice_2d, axis=1, mode="reflect")
    return float(np.mean(gx * gx + gy * gy, dtype=np.float32))


def _fft_high_frequency_energy(slice_2d: np.ndarray, cutoff: float = 0.15) -> float:
    freq = np.fft.rfftn(slice_2d)
    power = np.abs(freq) ** 2
    fy = np.fft.fftfreq(slice_2d.shape[0])[:, None]
    fx = np.fft.rfftfreq(slice_2d.shape[1])[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    mask = radius >= cutoff * np.max(radius)
    high_freq = float(np.sum(power[mask]))
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    return high_freq / total


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return scores
    window = int(window)
    window = window + 1 if window % 2 == 0 else window
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(scores, pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std_safe = np.where(std < 1e-6, 1.0, std)
    return ((arr - mean) / std_safe).astype(np.float32)


def _create_channel_composites(stack: np.ndarray) -> np.ndarray:
    """Create RGB composite images from multi-channel stack.
    
    Parameters
    ----------
    stack:
        Image stack in ZYXC format.
    
    Returns
    -------
    composites:
        RGB composite images, shape (Z, Y, X, 3).
    """
    z, y, x, c = stack.shape
    composites = np.zeros((z, y, x, 3), dtype=np.float32)
    
    # Default color mapping for up to 6 channels
    # Common microscopy colors: DAPI (blue), GFP (green), RFP (red), Cy5 (magenta), etc.
    default_colors = [
        (0, 0, 1),      # Channel 0: Blue (DAPI)
        (0, 1, 0),      # Channel 1: Green (GFP/FITC)
        (1, 0, 0),      # Channel 2: Red (RFP/mCherry)
        (1, 0, 1),      # Channel 3: Magenta (Cy5)
        (1, 1, 0),      # Channel 4: Yellow
        (0, 1, 1),      # Channel 5: Cyan
    ]
    
    # Normalize stack per channel
    for ch in range(c):
        channel_data = stack[:, :, :, ch].astype(np.float32)
        
        # Per-channel normalization to 0-1
        ch_min = channel_data.min()
        ch_max = channel_data.max()
        if ch_max > ch_min:
            channel_normalized = (channel_data - ch_min) / (ch_max - ch_min)
        else:
            channel_normalized = np.zeros_like(channel_data)
        
        # Get color for this channel
        color = default_colors[ch % len(default_colors)]
        
        # Add to composite (additive blending)
        for rgb_idx in range(3):
            composites[:, :, :, rgb_idx] += channel_normalized * color[rgb_idx]
    
    # Normalize composite to 0-1 range (clip to handle overflow from additive blending)
    composites = np.clip(composites, 0, 1)
    
    return composites


def _plot_strategy_comparison(
    img: np.ndarray,
    comparison: StrategyComparisonResult,
    output_dir: Path,
    axes: Optional[str],
) -> None:
    """Generate comparison visualizations for multiple strategies.
    
    Creates:
    1. Decision matrix heatmap (which slices each strategy keeps/removes)
    2. Slice gallery (visual inspection of all slices)
    
    Parameters
    ----------
    img:
        Original input stack.
    comparison:
        Strategy comparison result object.
    output_dir:
        Directory to save plots.
    axes:
        Axis order string.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        import warnings
        warnings.warn(
            "matplotlib not available; skipping comparison plots. "
            "Install matplotlib to enable visualization.",
            UserWarning,
        )
        return
    
    # Ensure image is in ZYXC format
    stack = _ensure_zyxc(img, axes)
    
    # 1. Generate decision matrix heatmap
    print("  - Creating decision matrix heatmap...")
    fig1, ax1 = plt.subplots(figsize=(max(12, comparison.n_strategies * 1.2), 
                                      max(8, comparison.n_slices * 0.25)))
    
    # Custom colormap: remove=red, KEEP=green
    colors = ['#ff6b6b', '#51cf66']
    cmap = LinearSegmentedColormap.from_list('decision', colors, N=2)
    
    im1 = ax1.imshow(comparison.decision_matrix.astype(int), 
                     aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Set ticks
    ax1.set_xticks(np.arange(comparison.n_strategies))
    ax1.set_yticks(np.arange(comparison.n_slices))
    ax1.set_xticklabels(comparison.strategy_names, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(range(comparison.n_slices), fontsize=8)
    
    # Labels
    ax1.set_xlabel('Detection Strategy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Z-Slice Index', fontsize=11, fontweight='bold')
    ax1.set_title('Z-Slice Selection Decisions Across Strategies\n(Green = KEEP, Red = remove)',
                  fontsize=13, fontweight='bold', pad=15)
    
    # Add grid
    ax1.set_xticks(np.arange(comparison.n_strategies + 1) - 0.5, minor=True)
    ax1.set_yticks(np.arange(comparison.n_slices + 1) - 0.5, minor=True)
    ax1.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Add text annotations
    for i in range(comparison.n_slices):
        for j in range(comparison.n_strategies):
            text = '✓' if comparison.decision_matrix[i, j] else '✗'
            ax1.text(j, i, text, ha='center', va='center', color='white',
                    fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    out_path1 = output_dir / "decision_matrix_heatmap.png"
    fig1.savefig(out_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"    ✓ Saved: {out_path1}")
    
    # 2. Generate slice gallery
    print("  - Creating slice gallery...")
    
    z_total = stack.shape[0]
    n_channels = stack.shape[-1]
    cols = 6
    rows = int(np.ceil(z_total / cols))
    
    # Create RGB composites for each slice
    composite_slices = _create_channel_composites(stack)
    
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    
    for ax, idx in zip(axes, range(z_total)):
        ax.imshow(composite_slices[idx])
        ax.set_title(f"z={idx}", fontsize=9, fontweight='bold')
        ax.axis("off")
    
    for ax in axes[z_total:]:
        ax.axis("off")
    
    fig2.suptitle(f"Per-slice composite ({n_channels} channels) - For Visual Inspection", y=0.99)
    fig2.tight_layout()
    
    out_path2 = output_dir / "z_slice_gallery.png"
    fig2.savefig(out_path2, dpi=200)
    plt.close(fig2)
    print(f"    ✓ Saved: {out_path2}")
    
    # 3. Generate summary table
    print("  - Creating strategy summary...")
    
    summary_lines = []
    summary_lines.append("Strategy Comparison Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Total Z-slices: {comparison.n_slices}")
    summary_lines.append(f"Strategies compared: {comparison.n_strategies}")
    summary_lines.append("")
    summary_lines.append(f"{'Strategy':<35} {'Kept':<10} {'Removed':<10} {'% Kept':<10}")
    summary_lines.append("-" * 80)
    
    for name in comparison.strategy_names:
        result = comparison.results[name]
        n_kept = len(result.indices_keep)
        n_removed = len(result.indices_remove)
        pct_kept = (n_kept / comparison.n_slices) * 100
        summary_lines.append(f"{name:<35} {n_kept:<10} {n_removed:<10} {pct_kept:<10.1f}%")
    
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    out_path3 = output_dir / "comparison_summary.txt"
    with open(out_path3, 'w') as f:
        f.write(summary_text)
    
    print(f"    ✓ Saved: {out_path3}")
    print("\n" + summary_text)

