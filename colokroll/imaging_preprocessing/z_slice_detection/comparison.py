"""Method comparison and automatic selection for Z-slice detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..focus_measure_quality import FocusMeasureQuality


@dataclass(frozen=True)
class StrategyComparisonResult:
    """Container for multi-strategy comparison outputs.
    
    Attributes
    ----------
    strategy_names : Tuple[str, ...]
        Names of the strategies compared.
    results : Dict[str, ZSliceSelectionResult]
        Mapping from strategy name to its selection result.
    decision_matrix : np.ndarray
        Boolean matrix of shape (n_slices, n_strategies) indicating keep decisions.
    score_matrix : np.ndarray
        Matrix of shape (n_slices, n_strategies) with smoothed scores.
    n_slices : int
        Total number of Z-slices.
    n_strategies : int
        Number of strategies compared.
    quality_comparison : Optional[Dict[str, FocusMeasureQuality]]
        Quality metrics for each strategy. Only present if quality metrics
        were computed for the results.
    """
    
    strategy_names: Tuple[str, ...]
    results: Dict[str, "ZSliceSelectionResult"]
    decision_matrix: np.ndarray
    score_matrix: np.ndarray
    n_slices: int
    n_strategies: int
    quality_comparison: Optional[Dict[str, "FocusMeasureQuality"]] = None


def auto_select_best_method(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    methods: Optional[list] = None,
    step_distance: float = 1.0,
    n_fitting_points: int = 5,
    ranking_metric: str = "Rsg",
    require_unimodal: bool = False,
    verbose: bool = True,
    **kwargs
) -> "ZSliceSelectionResult":
    """Automatically select and apply the best focus method for the data.
    
    This function benchmarks multiple focus methods, evaluates them using Piao et al.
    quality metrics, and automatically selects the best one. No manual method selection
    required!
    
    Parameters
    ----------
    img:
        Input stack. Expected shape `(Z, Y, X, C)`.
    axes:
        Optional axis order string (e.g. ``"CZYX"``).
    methods:
        List of focus methods to evaluate. If ``None``, evaluates all methods:
        ``["laplacian", "tenengrad", "fft", "combined"]``.
    step_distance:
        Physical z-step distance in micrometers. Default 1.0 µm.
    n_fitting_points:
        Number of points for multi-point linear fitting. Default 5.
    ranking_metric:
        Quality metric to use for ranking methods. Options:
        - ``"Rsg"`` (default): Steep-to-gradual ratio (higher = better discrimination)
        - ``"Ws"`` : Steep width (lower = higher sensitivity)
        - ``"Cp"`` : Peak curvature (higher = more sensitive near peak)
        - ``"composite"``: Weighted combination of metrics
    require_unimodal:
        If ``True``, only consider methods with unimodal focus curves.
        Default ``False``.
    verbose:
        If ``True``, print evaluation results and selection rationale.
        Default ``True``.
    **kwargs:
        Additional arguments for ``select_z_slices()`` (e.g., ``strategy``,
        ``threshold``, etc.).
    
    Returns
    -------
    ZSliceSelectionResult:
        Result from the automatically selected best method, including quality metrics.
    
    Examples
    --------
    >>> from colokroll.imaging_preprocessing import auto_select_best_method
    >>> 
    >>> # Fully automatic - just load and go!
    >>> result = auto_select_best_method(z_stack, step_distance=0.5)
    >>> print(f"Auto-selected method: {result.method}")
    >>> print(f"Kept {len(result.indices_keep)} slices: {list(result.indices_keep)}")
    >>> 
    >>> # Require unimodal curves for reliability
    >>> result = auto_select_best_method(
    ...     z_stack,
    ...     step_distance=0.5,
    ...     require_unimodal=True,
    ...     ranking_metric="composite",
    ... )
    
    Notes
    -----
    The automatic selection is based on objective quality metrics (Piao et al. 2025):
    
    - **Rsg (default)**: Higher values indicate better ability to distinguish
      in-focus from out-of-focus. Recommended for most applications.
    - **Ws**: Lower values indicate higher sensitivity to focus changes.
      Good for precise focal plane detection.
    - **Cp**: Higher values indicate sensitivity near the focal position.
      Useful for applications requiring precise peak localization.
    - **composite**: Balanced ranking using normalized metrics.
    
    The function evaluates all methods and automatically selects the one with
    the best quality metrics, eliminating the need for manual trial-and-error.
    """
    # Import here to avoid circular dependency
    from .core import select_z_slices
    
    if methods is None:
        methods = ["laplacian", "tenengrad", "fft", "combined"]
    
    if verbose:
        print(f"🔍 Auto-selecting best focus method from {methods}...")
        print(f"   Ranking by: {ranking_metric}")
        if require_unimodal:
            print(f"   Constraint: Unimodal curves only")
    
    # Benchmark all methods
    results = benchmark_focus_methods(
        img,
        axes=axes,
        methods=methods,
        step_distance=step_distance,
        n_fitting_points=n_fitting_points,
        **kwargs
    )
    
    # Filter by unimodality if required
    candidates = {}
    for method, result in results.items():
        if require_unimodal and not result.quality_metrics.is_unimodal:
            if verbose:
                print(f"   ✗ {method}: Excluded (non-unimodal)")
            continue
        candidates[method] = result
    
    if not candidates:
        if verbose:
            print(f"   ⚠ No unimodal methods found, using all methods")
        candidates = results
    
    # Rank methods by quality metric
    if ranking_metric == "Rsg":
        # Higher is better
        ranked = sorted(
            candidates.items(),
            key=lambda x: x[1].quality_metrics.Rsg,
            reverse=True
        )
    elif ranking_metric == "Ws":
        # Lower is better
        ranked = sorted(
            candidates.items(),
            key=lambda x: x[1].quality_metrics.Ws,
            reverse=False
        )
    elif ranking_metric == "Cp":
        # Higher is better
        ranked = sorted(
            candidates.items(),
            key=lambda x: x[1].quality_metrics.Cp,
            reverse=True
        )
    elif ranking_metric == "composite":
        # Composite score: normalize and combine metrics
        # Score = (Rsg_norm + Cp_norm) - Ws_norm
        metrics_data = {
            method: result.quality_metrics
            for method, result in candidates.items()
        }
        
        # Normalize each metric to [0, 1]
        rsg_values = [q.Rsg for q in metrics_data.values()]
        ws_values = [q.Ws for q in metrics_data.values()]
        cp_values = [q.Cp for q in metrics_data.values()]
        
        rsg_min, rsg_max = min(rsg_values), max(rsg_values)
        ws_min, ws_max = min(ws_values), max(ws_values)
        cp_min, cp_max = min(cp_values), max(cp_values)
        
        composite_scores = {}
        for method, q in metrics_data.items():
            # Normalize (handle edge case where all values are the same)
            rsg_norm = (q.Rsg - rsg_min) / (rsg_max - rsg_min) if rsg_max > rsg_min else 1.0
            ws_norm = (q.Ws - ws_min) / (ws_max - ws_min) if ws_max > ws_min else 0.0
            cp_norm = (q.Cp - cp_min) / (cp_max - cp_min) if cp_max > cp_min else 1.0
            
            # Composite: maximize Rsg and Cp, minimize Ws
            # Weight Rsg higher (2x) as it's the most important metric
            composite_scores[method] = (2.0 * rsg_norm + cp_norm) - ws_norm
        
        ranked = sorted(
            candidates.items(),
            key=lambda x: composite_scores[x[0]],
            reverse=True
        )
    else:
        raise ValueError(
            f"Unknown ranking_metric: {ranking_metric}. "
            f"Choose from: 'Rsg', 'Ws', 'Cp', 'composite'"
        )
    
    # Print ranking if verbose
    if verbose:
        print(f"\n   Method Ranking:")
        print(f"   {'Rank':<6} {'Method':<12} {'Rsg':<8} {'Ws (µm)':<9} {'Cp':<9} {'Unimodal'}")
        print(f"   {'-'*60}")
        for i, (method, result) in enumerate(ranked, 1):
            q = result.quality_metrics
            uni = "Yes" if q.is_unimodal else "No"
            marker = "→" if i == 1 else " "
            print(f"   {marker} {i:<4} {method:<12} {q.Rsg:<8.3f} {q.Ws:<9.2f} {q.Cp:<9.4f} {uni}")
    
    # Select best method
    best_method_name, best_result = ranked[0]
    
    if verbose:
        print(f"\n✓ Selected method: {best_method_name}")
        q = best_result.quality_metrics
        print(f"  Rsg={q.Rsg:.3f}, Ws={q.Ws:.2f} µm, Cp={q.Cp:.4f}")
        print(f"  Kept {len(best_result.indices_keep)}/{len(best_result.scores_agg)} slices: {list(best_result.indices_keep)}")
    
    return best_result


def benchmark_focus_methods(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    methods: Optional[list] = None,
    step_distance: float = 1.0,
    n_fitting_points: int = 5,
    **kwargs
) -> Dict[str, "ZSliceSelectionResult"]:
    """Benchmark multiple focus methods on the same z-stack.
    
    Returns results with quality metrics for each method, allowing objective
    comparison based on Ws, Rsg, Cp, and other metrics from Piao et al. (2025).
    
    Parameters
    ----------
    img:
        Input stack. Expected shape `(Z, Y, X, C)`.
    axes:
        Optional axis order string (e.g. ``"CZYX"``).
    methods:
        List of focus methods to compare. Options: ``"laplacian"``, ``"tenengrad"``,
        ``"fft"``, ``"combined"``. If ``None``, compares all methods.
    step_distance:
        Physical z-step distance in micrometers. Used for quality metrics.
        Default 1.0 µm.
    n_fitting_points:
        Number of points for multi-point linear fitting in quality metrics.
        Default 5.
    **kwargs:
        Additional arguments passed to ``select_z_slices()``, such as ``strategy``,
        ``threshold``, ``smooth``, etc.
    
    Returns
    -------
    dict:
        Mapping from method name to ``ZSliceSelectionResult`` with quality metrics.
    
    Examples
    --------
    >>> from colokroll.imaging_preprocessing import benchmark_focus_methods
    >>> 
    >>> # Compare all methods
    >>> results = benchmark_focus_methods(z_stack, step_distance=0.5)
    >>> 
    >>> # Print comparison table
    >>> for method, result in results.items():
    ...     q = result.quality_metrics
    ...     print(f"{method}: Ws={q.Ws:.2f}, Rsg={q.Rsg:.3f}, Cp={q.Cp:.4f}")
    >>> 
    >>> # Select best method by Rsg (steep-to-gradual ratio)
    >>> best = max(results.items(), key=lambda x: x[1].quality_metrics.Rsg)
    >>> print(f"Best method: {best[0]}")
    """
    # Import here to avoid circular dependency
    from .core import select_z_slices
    
    if methods is None:
        methods = ["laplacian", "tenengrad", "fft", "combined"]
    
    results = {}
    for method in methods:
        results[method] = select_z_slices(
            img,
            axes=axes,
            method=method,
            compute_quality=True,
            step_distance=step_distance,
            n_fitting_points=n_fitting_points,
            **kwargs
        )
    
    return results


def compare_strategies(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    strategies: Optional[list] = None,
    output_path: Optional[Union[str, Path]] = None,
    save_plots: bool = True,
    compute_quality: bool = False,
    step_distance: float = 1.0,
    n_fitting_points: int = 5,
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
    compute_quality:
        If True, compute Piao et al. quality metrics for each strategy.
        Default False.
    step_distance:
        Physical z-step distance in micrometers. Used for quality metrics.
        Default 1.0 µm.
    n_fitting_points:
        Number of points for multi-point linear fitting in quality metrics.
        Default 5.
    
    Returns
    -------
    StrategyComparisonResult:
        Object containing results for all strategies plus comparison matrices.
        If ``compute_quality=True``, includes quality metrics comparison.
    
    Examples
    --------
    >>> # Compare with default strategies
    >>> comparison = compare_strategies(stack, output_path="comparison_results")
    >>> 
    >>> # Compare custom strategies with quality metrics
    >>> strategies = [
    ...     {"name": "FFT Auto", "method": "fft", "strategy": "closest_to_peak"},
    ...     {"name": "Combined k=15", "method": "combined", "strategy": "closest_to_peak", "keep_top": 15},
    ...     {"name": "Laplacian", "method": "laplacian", "strategy": "relative", "threshold": 0.6},
    ... ]
    >>> comparison = compare_strategies(stack, strategies=strategies, compute_quality=True)
    >>> 
    >>> # Access quality metrics
    >>> for name, quality in comparison.quality_comparison.items():
    ...     print(f"{name}: Rsg={quality.Rsg:.3f}")
    """
    # Import here to avoid circular dependency
    from .core import select_z_slices
    from .visualization import plot_strategy_comparison
    
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
        
        # Add quality computation parameters if requested
        if compute_quality:
            config["compute_quality"] = True
            config["step_distance"] = step_distance
            config["n_fitting_points"] = n_fitting_points
        
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
    
    # Collect quality metrics if computed
    quality_comparison = None
    if compute_quality:
        quality_comparison = {}
        for name in strategy_names:
            if results[name].quality_metrics is not None:
                quality_comparison[name] = results[name].quality_metrics
    
    comparison_result = StrategyComparisonResult(
        strategy_names=tuple(strategy_names),
        results=results,
        decision_matrix=decision_matrix,
        score_matrix=score_matrix,
        n_slices=n_slices,
        n_strategies=n_strategies,
        quality_comparison=quality_comparison,
    )
    
    # Generate comparison plots if requested
    if save_plots:
        if output_path is None:
            output_dir = Path.cwd() / "strategy_comparison"
        else:
            output_dir = Path(output_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating comparison visualizations in {output_dir}...")
        plot_strategy_comparison(img, comparison_result, output_dir, axes)
        print("✓ Comparison complete!")
    
    return comparison_result

