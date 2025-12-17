"""Z-slice selection strategies and score aggregation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from .utils import _smooth_scores


AggregationMethod = str  # "median", "mean", "max", "min", "weighted"
DetectionStrategy = str  # "relative", "percentile", "topk", "closest_to_peak"


def aggregate_focus_scores(
    scores_zc: np.ndarray,
    *,
    aggregation: AggregationMethod = "median",
    weights: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Aggregate per-channel focus scores into a single per-slice series.
    
    Parameters
    ----------
    scores_zc:
        Per-channel focus scores, shape (Z, C).
    aggregation:
        Aggregation method. Options: "median", "mean", "max", "min", "weighted".
    weights:
        Channel weights for weighted aggregation. Required if aggregation="weighted".
    
    Returns
    -------
    scores:
        Aggregated scores, shape (Z,).
    """

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
        
        For ``"relative"`` and ``"percentile"`` strategies, ``"threshold_used"``
        is the computed threshold value (slices with scores below this are kept).
        For ``"topk"``, it is the maximum score among kept slices.
        For ``"closest_to_peak"``, it is the minimum score among kept slices
        (the effective lower boundary for the kept slices).
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
        
        # We track the *decision* threshold in score space so `threshold_used`
        # has consistent semantics across strategies: the minimum score among
        # kept slices (i.e., the effective lower boundary).
        
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
            
            # Effective decision boundary: lowest score among kept slices
            decision_threshold = float(np.min(smoothed[indices_keep]))
            
            return {
                "scores": scores,
                "smoothed_scores": smoothed,
                "mask_keep": mask_keep,
                "indices_keep": indices_keep,
                "indices_remove": indices_remove,
                "threshold_used": decision_threshold,
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
        
        # Effective decision boundary: lowest score among kept slices
        decision_threshold = float(np.min(smoothed[indices_keep]))
        
        return {
            "scores": scores,
            "smoothed_scores": smoothed,
            "mask_keep": mask_keep,
            "indices_keep": indices_keep,
            "indices_remove": indices_remove,
            "threshold_used": decision_threshold,
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

