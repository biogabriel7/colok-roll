"""Focus scoring metrics for Z-slice selection.

This module provides various focus measurement methods to quantify sharpness
of individual Z-slices in a microscopy stack.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from .utils import (
    _compute_global_stats,
    _ensure_zyxc,
    _prepare_slice,
    _zscore,
)

FocusMethod = str  # "laplacian", "tenengrad", "fft", "combined"
NormalizationMode = str  # "per_slice", "per_stack", "percentile"


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
    global_stats: Optional[Dict[int, Tuple[float, float]]],
    fft_cutoff: float,
) -> np.ndarray:
    """Score all slices in a volume using the given scorer function."""
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


def _resolve_scorer(method: FocusMethod, fft_cutoff: float = 0.15) -> Callable:
    """Get the scorer function for the given method."""
    if method == "laplacian":
        return _laplacian_variance
    if method == "tenengrad":
        return _tenengrad_score
    if method == "fft":
        # Return a partial function with the cutoff parameter
        return lambda slice_2d: _fft_high_frequency_energy(slice_2d, cutoff=fft_cutoff)

    raise ValueError(f"Unknown focus method: {method}")


def _laplacian_variance(slice_2d: np.ndarray) -> float:
    """Laplacian variance focus metric."""
    lap = ndimage.laplace(slice_2d, mode="reflect")
    return float(np.var(lap, dtype=np.float32))


def _tenengrad_score(slice_2d: np.ndarray) -> float:
    """Tenengrad (gradient energy) focus metric."""
    gx = ndimage.sobel(slice_2d, axis=0, mode="reflect")
    gy = ndimage.sobel(slice_2d, axis=1, mode="reflect")
    return float(np.mean(gx * gx + gy * gy, dtype=np.float32))


def _fft_high_frequency_energy(slice_2d: np.ndarray, cutoff: float = 0.15) -> float:
    """FFT high-frequency energy focus metric."""
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


def _combine_scores(laplacian: np.ndarray, tenengrad: np.ndarray) -> np.ndarray:
    """Combine laplacian and tenengrad scores using z-score normalization."""
    if laplacian.shape != tenengrad.shape:
        raise ValueError(
            "laplacian and tenengrad scores must share the same shape; "
            f"got {laplacian.shape} vs {tenengrad.shape}"
        )
    laplacian_z = _zscore(laplacian)
    tenengrad_z = _zscore(tenengrad)
    combined = 0.5 * (laplacian_z + tenengrad_z)
    return combined.astype(np.float32, copy=False)

