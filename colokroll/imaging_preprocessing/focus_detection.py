"""Focus assessment for microscopy Z-stacks.

This module provides lightweight, per-slice focus metrics to help exclude
out-of-focus planes prior to downstream analysis. It is designed to work with
`ImageLoader` outputs, which are shaped `(Z, Y, X, C)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
from scipy import ndimage


FocusMethod = Literal["laplacian", "tenengrad", "fft", "combined"]
AggregationMethod = Literal["median", "mean", "max", "min", "weighted"]
DetectionStrategy = Literal["relative", "percentile", "topk"]


@dataclass(frozen=True)
class FocusDetectionResult:
    """Container for focus detection outputs."""

    scores_zc: np.ndarray
    scores_agg: np.ndarray
    mask_oof: np.ndarray
    indices_in_focus: np.ndarray
    mask_oof_zc: np.ndarray
    threshold_used: float
    smoothed_scores: np.ndarray
    strategy: DetectionStrategy
    method: FocusMethod
    aggregation: AggregationMethod


def compute_focus_scores(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    method: FocusMethod = "combined",
    normalize: bool = True,
    clip_percent: float = 0.0,
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
    clip_percent:
        Symmetric percentile clipping applied before normalisation. For example,
        ``clip_percent=1`` clips to the 1st–99th percentile range.

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

    if method == "combined":
        laplacian = _score_volume(
            stack,
            _laplacian_variance,
            normalize=normalize,
            clip_percent=clip_percent,
        )
        tenengrad = _score_volume(
            stack,
            _tenengrad_score,
            normalize=normalize,
            clip_percent=clip_percent,
        )
        scores_zc = _combine_scores(laplacian, tenengrad)
    else:
        scorer = _resolve_scorer(method)
        scores_zc = _score_volume(
            stack,
            scorer,
            normalize=normalize,
            clip_percent=clip_percent,
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


def detect_oof_slices(
    scores: np.ndarray,
    *,
    strategy: DetectionStrategy = "relative",
    threshold: float = 0.6,
    smooth: int = 3,
    keep_top: Optional[int] = None,
) -> Dict[str, Any]:
    """Determine which slices are out-of-focus based on their scores."""

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
        sort_idx = np.argsort(smoothed)[::-1]
        in_focus_idx = np.sort(sort_idx[:keep_top])
        mask_oof = np.ones_like(smoothed, dtype=bool)
        mask_oof[in_focus_idx] = False
        return {
            "scores": scores,
            "smoothed_scores": smoothed,
            "mask_oof": mask_oof,
            "indices_in_focus": in_focus_idx,
            "threshold_used": float(np.min(smoothed[in_focus_idx])),
            "strategy": strategy,
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    mask_oof = smoothed < threshold_value
    indices_in_focus = np.nonzero(~mask_oof)[0]

    return {
        "scores": scores,
        "smoothed_scores": smoothed,
        "mask_oof": mask_oof,
        "indices_in_focus": indices_in_focus,
        "threshold_used": threshold_value,
        "strategy": strategy,
    }


def find_oof_slices(
    img: np.ndarray,
    *,
    axes: Optional[str] = None,
    method: FocusMethod = "combined",
    aggregation: AggregationMethod = "median",
    strategy: DetectionStrategy = "relative",
    threshold: float = 0.6,
    smooth: int = 3,
    keep_top: Optional[int] = None,
    normalize: bool = True,
    clip_percent: float = 0.0,
    weights: Optional[Iterable[float]] = None,
) -> FocusDetectionResult:
    """End-to-end helper: compute focus, aggregate, and classify slices."""

    scores_zc, scores_median = compute_focus_scores(
        img,
        axes=axes,
        method=method,
        normalize=normalize,
        clip_percent=clip_percent,
    )

    scores_agg = (
        scores_median
        if aggregation == "median" and weights is None
        else aggregate_focus_scores(scores_zc, aggregation=aggregation, weights=weights)
    )

    detection = detect_oof_slices(
        scores_agg,
        strategy=strategy,
        threshold=threshold,
        smooth=smooth,
        keep_top=keep_top,
    )

    mask_oof_zc = np.empty_like(scores_zc, dtype=bool)
    for channel in range(scores_zc.shape[1]):
        channel_detection = detect_oof_slices(
            scores_zc[:, channel],
            strategy=strategy,
            threshold=threshold,
            smooth=smooth,
            keep_top=keep_top,
        )
        mask_oof_zc[:, channel] = channel_detection["mask_oof"]

    return FocusDetectionResult(
        scores_zc=scores_zc,
        scores_agg=scores_agg,
        mask_oof=detection["mask_oof"],
        indices_in_focus=detection["indices_in_focus"],
        mask_oof_zc=mask_oof_zc,
        threshold_used=float(detection["threshold_used"]),
        smoothed_scores=detection["smoothed_scores"],
        strategy=strategy,
        method=method,
        aggregation=aggregation,
    )


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
    clip_percent: float,
) -> np.ndarray:
    z, _, _, c = stack.shape
    scores = np.empty((z, c), dtype=np.float32)

    for channel in range(c):
        for plane in range(z):
            slice_2d = stack[plane, :, :, channel]
            prepared = _prepare_slice(slice_2d, normalize=normalize, clip_percent=clip_percent)
            scores[plane, channel] = float(scorer(prepared))

    return scores


def _prepare_slice(
    slice_2d: np.ndarray,
    *,
    normalize: bool,
    clip_percent: float,
) -> np.ndarray:
    arr = np.asarray(slice_2d, dtype=np.float32)

    if clip_percent > 0:
        lower = np.percentile(arr, clip_percent)
        upper = np.percentile(arr, 100 - clip_percent)
        if upper > lower:
            arr = np.clip(arr, lower, upper)

    if normalize:
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
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


def _resolve_scorer(method: FocusMethod):
    if method == "laplacian":
        return _laplacian_variance
    if method == "tenengrad":
        return _tenengrad_score
    if method == "fft":
        return _fft_high_frequency_energy

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


