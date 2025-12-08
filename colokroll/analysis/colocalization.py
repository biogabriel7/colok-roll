from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Prefer GPU (CuPy) when available for heavy numeric work
try:
    import cupy as cp  # type: ignore
    xp = cp  # device-agnostic array module alias
    HAS_CUDA = True
except ImportError:  # pragma: no cover - CuPy not installed
    cp = None  # type: ignore
    xp = np
    HAS_CUDA = False

try:
    # Prefer package import when used inside perinuclear_analysis
    from ..data_processing.image_loader import ImageLoader
except ImportError:
    # Allow running standalone in a notebook if needed
    ImageLoader = None  # type: ignore

# Optional preprocessing utilities (bleed-through unmixing and ROI percentile background)
try:
    from ..imaging_preprocessing.background_subtraction import (
        apply_bleedthrough_unmix,
        subtract_background_percentile_roi,
    )
except ImportError:
    apply_bleedthrough_unmix = None  # type: ignore
    subtract_background_percentile_roi = None  # type: ignore


# =============================================================================
# Module-level constants
# =============================================================================
_EPSILON = 1e-12  # Small value to prevent division by zero
_DEFAULT_HISTOGRAM_BINS = 256  # Default bins for Otsu thresholding
_COSTES_MIN_POINTS = 100  # Minimum voxels for Costes threshold estimation
_COSTES_STEPS = 256  # Number of steps for Costes threshold search


def _to_python_float(val) -> float:
    """Safely convert CuPy/NumPy scalar or array element to Python float."""
    if HAS_CUDA:
        if hasattr(val, 'get'):
            # CuPy array - transfer to CPU
            return float(val.get())
        if hasattr(val, 'item'):
            # NumPy scalar or 0-d array
            return float(val.item())
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)

logger = logging.getLogger(__name__)


def _as_zyxc(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        # Assume (Y, X, C) -> promote to (1, Y, X, C)
        return image[None, ...]
    if image.ndim != 4:
        raise ValueError(f"Expected image shaped (Z,Y,X,C); got {image.shape}")
    return image


def _load_image_and_channels(
    image: Union[str, Path, np.ndarray, Dict[str, Any]],
    channel_a: Union[int, str],
    channel_b: Union[int, str],
    channel_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, int, int, List[str]]:
    # Load image
    if isinstance(image, dict):
        # Expect mapping: channel_name -> (array, meta) or -> array
        names = list(image.keys())
        arrays: List[np.ndarray] = []

        def _to_backend_array(a: Any) -> np.ndarray:
            # Convert inputs to preferred backend
            if HAS_CUDA:
                return cp.asarray(a)  # type: ignore[return-value]
            return np.asarray(a)

        for nm in names:
            val = image[nm]
            arr = val[0] if (isinstance(val, (tuple, list)) and len(val) >= 1) else val
            arr_np = _to_backend_array(arr)
            
            # Promote 2D (Y,X) to 3D (1,Y,X) - check ndim using int() for compatibility
            ndim = int(arr_np.ndim)
            if ndim == 2:
                arr_np = arr_np[None, ...]
            elif ndim == 4 and arr_np.shape[-1] == 1:
                # Squeeze trailing singleton channel dimension
                arr_np = arr_np[..., 0]
            
            # Validate final shape
            if int(arr_np.ndim) != 3:
                raise ValueError(f"Per-channel array must be 3D (Z,Y,X); got {arr_np.shape} for channel '{nm}'")
            arrays.append(arr_np)

        # Validate consistent shapes and stack into ZYXC
        zyx_shapes = {a.shape for a in arrays}
        if len(zyx_shapes) != 1:
            raise ValueError(f"All channels must share the same Z,Y,X shape; got {zyx_shapes}")
        img = (xp.stack if HAS_CUDA else np.stack)(arrays, axis=-1)  # type: ignore[arg-type]

    elif isinstance(image, (str, Path)):
        if ImageLoader is None:
            raise RuntimeError("ImageLoader not available; pass a numpy array instead.")
        loader = ImageLoader()
        img = loader.load_image(str(image))
        names = loader.get_channel_names()
        logger.info(f"Loaded image {str(image)} with channels: {names}")
    else:
        img = image
        names = channel_names or []
        if not names and isinstance(channel_a, str):
            raise ValueError("When passing a numpy image, provide channel_names if selecting by name.")

    img = _as_zyxc(img)
    # Convert to backend array (GPU if available)
    if HAS_CUDA:
        img = cp.asarray(img)  # type: ignore[assignment]

    # Resolve channel indices
    if isinstance(channel_a, str):
        if channel_a not in names:
            raise ValueError(f"Channel '{channel_a}' not in metadata: {names}")
        ch_a = names.index(channel_a)
    else:
        ch_a = int(channel_a)

    if isinstance(channel_b, str):
        if channel_b not in names:
            raise ValueError(f"Channel '{channel_b}' not in metadata: {names}")
        ch_b = names.index(channel_b)
    else:
        ch_b = int(channel_b)

    if ch_a == ch_b:
        raise ValueError("channel_a and channel_b must be different.")

    if ch_a < 0 or ch_a >= img.shape[-1] or ch_b < 0 or ch_b >= img.shape[-1]:
        raise ValueError(f"Channel indices out of range for image with C={img.shape[-1]}.")

    return img, ch_a, ch_b, names


def _reduce_3d_mask_to_2d(m: np.ndarray) -> np.ndarray:
    """Select Z-slice with largest labeled area from a 3D mask."""
    labeled_counts = [(z_idx, int((m[z_idx] > 0).sum())) for z_idx in range(m.shape[0])]
    z_best = max(labeled_counts, key=lambda t: t[1])[0]
    logger.info(
        f"3D mask detected; reducing to 2D by selecting z={z_best} "
        f"(largest labeled area) and broadcasting across Z."
    )
    return m[z_best]


def _coerce_mask_dtype(m: np.ndarray) -> np.ndarray:
    """Convert mask to int32, handling various input dtypes."""
    # Integer dtype: direct conversion
    if np.issubdtype(m.dtype, np.integer):
        logger.info("Loaded labeled mask with integer dtype: %s", str(m.dtype))
        return m.astype(np.int32)

    # Float or other non-integer dtype: decide between binary and labeled
    m_min = float(np.nanmin(m)) if m.size > 0 else 0.0
    m_max = float(np.nanmax(m)) if m.size > 0 else 0.0

    if 0.0 <= m_min <= m_max <= 1.0:
        # Likely binary/probability mask; threshold at 0.5
        logger.info("Loaded non-integer mask in [0,1]; converting to binary with threshold 0.5")
        return (m > 0.5).astype(np.int32)

    # Otherwise, assume labeled mask stored as float; round to nearest int
    logger.info("Loaded non-integer mask with range [%s, %s]; rounding to int labels", m_min, m_max)
    return np.rint(m).astype(np.int32)


def _load_mask(mask: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load and validate a 2D labeled mask."""
    # Load from file if needed
    if isinstance(mask, (str, Path)):
        if ImageLoader is None:
            raise RuntimeError("ImageLoader not available; pass a numpy array instead.")
        m = ImageLoader().load_tif_mask(str(mask))
    else:
        m = np.asarray(mask)

    # Reduce 3D to best 2D slice
    if m.ndim == 3:
        m = _reduce_3d_mask_to_2d(m)

    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after reduction; got {m.shape}")

    return _coerce_mask_dtype(m)


def estimate_min_area_threshold(
    mask: Union[str, Path, np.ndarray],
    *,
    fraction_of_median: float = 0.4,
) -> int:
    """
    Estimate a label area threshold as a fraction of the median labeled area.

    - Loads a 2D labeled mask (reduces 3D to best z-slice via _load_mask).
    - Excludes background label 0 when computing areas.
    - Returns int(fraction_of_median * median_area). Returns 0 if no labels.
    """
    m2d = _load_mask(mask)
    labels, counts = np.unique(m2d, return_counts=True)
    if labels.size == 0:
        logger.info("No labels found in mask; min_area threshold set to 0")
        return 0
    areas = counts[labels != 0]
    if areas.size == 0:
        logger.info("No non-background labels found; min_area threshold set to 0")
        return 0
    median_area = int(np.median(areas))
    thr = int(float(fraction_of_median) * float(median_area))
    logger.info(
        "Estimated min_area threshold: fraction=%s, median_area=%d -> thr=%d",
        str(fraction_of_median),
        median_area,
        thr,
    )
    return max(0, thr)


def _broadcast_mask_to_z(mask_2d: np.ndarray, z: int) -> np.ndarray:
    # Return backend-compatible broadcast mask for indexing
    m = mask_2d
    if HAS_CUDA:
        m = cp.asarray(m)
        return cp.broadcast_to(m[cp.newaxis, ...], (z, m.shape[0], m.shape[1]))  # type: ignore[return-value]
    return np.broadcast_to(m[np.newaxis, ...], (z, m.shape[0], m.shape[1]))


def _safe_corrcoef(
    a: np.ndarray,
    b: np.ndarray,
    *,
    winsor_clip: Optional[float] = None,
    min_count: int = 2,
    min_range: float = 1e-12,
) -> float:
    """Compute Pearson correlation coefficient with safety checks.

    Args:
        a, b: input vectors (any shape, flattened internally)
        winsor_clip: optional symmetric percentile clip (e.g., 0.1 -> clip to 0.1–99.9 percentiles)
        min_count: minimum number of finite samples required
        min_range: treat vectors with range < min_range as constant
    """
    mod = xp
    if getattr(a, "size", 0) < min_count or getattr(b, "size", 0) < min_count:
        return float("nan")

    a = mod.asarray(a, dtype=mod.float64).reshape(-1)
    b = mod.asarray(b, dtype=mod.float64).reshape(-1)

    # Drop non-finite
    finite_mask = mod.isfinite(a) & mod.isfinite(b)
    if not bool(mod.any(finite_mask)):
        return float("nan")
    a = a[finite_mask]
    b = b[finite_mask]
    if a.size < min_count or b.size < min_count:
        return float("nan")

    # Optional winsorization
    if winsor_clip is not None and winsor_clip > 0:
        lower = winsor_clip
        upper = 100.0 - winsor_clip
        a = mod.clip(a, mod.percentile(a, lower), mod.percentile(a, upper))
        b = mod.clip(b, mod.percentile(b, lower), mod.percentile(b, upper))

    # Guard near-constant vectors
    a_range = _to_python_float(mod.max(a) - mod.min(a))
    b_range = _to_python_float(mod.max(b) - mod.min(b))
    if a_range < min_range or b_range < min_range:
        return float("nan")

    r = mod.corrcoef(a, b)[0, 1]
    return _to_python_float(r)


def _overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Compute overlap coefficient between two vectors."""
    mod = xp
    num = _to_python_float(mod.sum(a * b))
    den = _to_python_float(mod.sqrt(mod.sum(a * a) * mod.sum(b * b)))
    return num / den if den > 0 else float("nan")


def _manders_m1_m2(
    a: np.ndarray,
    b: np.ndarray,
    *,
    t_a: Optional[float] = None,
    t_b: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute Manders' M1 and M2.

    - If t_a/t_b are None, use the historical no-threshold behavior (>0).
    - If thresholds are provided, apply them asymmetrically:
      M1 uses threshold on B (b > t_b), M2 uses threshold on A (a > t_a).
    """
    if t_a is None and t_b is None:
        a_pos = a > 0
        b_pos = b > 0
    else:
        a_pos = a > (0.0 if t_a is None else t_a)
        b_pos = b > (0.0 if t_b is None else t_b)

    sum_a = float(np.sum(a))
    sum_b = float(np.sum(b))
    m1 = float(np.sum(a[b_pos])) / sum_a if sum_a > 0 else float("nan")
    m2 = float(np.sum(b[a_pos])) / sum_b if sum_b > 0 else float("nan")
    return m1, m2


def _jaccard_on_positive(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Jaccard index on positive (>0) regions."""
    mod = xp
    a_bin = a > 0
    b_bin = b > 0
    inter = _to_python_float(mod.sum(a_bin & b_bin))
    union = _to_python_float(mod.sum(a_bin | b_bin))
    return inter / union if union > 0 else float("nan")


def _fit_minmax(a_vals: np.ndarray, b_vals: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Fit min-max scalers for two vectors."""
    mod = xp
    a_vals = mod.asarray(a_vals).reshape(-1)
    b_vals = mod.asarray(b_vals).reshape(-1)
    a_min = _to_python_float(mod.nanmin(a_vals))
    a_max = _to_python_float(mod.nanmax(a_vals))
    b_min = _to_python_float(mod.nanmin(b_vals))
    b_max = _to_python_float(mod.nanmax(b_vals))
    return (a_min, a_max), (b_min, b_max)


def _normalize(
    a_mm: Tuple[float, float],
    b_mm: Tuple[float, float],
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize vectors using pre-fitted min-max ranges."""
    mod = xp
    a_min, a_max = a_mm
    b_min, b_max = b_mm
    a = mod.asarray(a).reshape(-1)
    b = mod.asarray(b).reshape(-1)
    a_n = (a - a_min) / (a_max - a_min + _EPSILON)
    b_n = (b - b_min) / (b_max - b_min + _EPSILON)
    return a_n.astype(mod.float32, copy=False), b_n.astype(mod.float32, copy=False)


def _otsu_threshold_1d(values: np.ndarray, *, bins: int = _DEFAULT_HISTOGRAM_BINS) -> float:
    """
    Compute Otsu threshold on a 1D array.
    
    Works for arbitrary value ranges. Returns a threshold in the same scale
    as the input values.
    
    Args:
        values: 1D array of intensity values.
        bins: Number of histogram bins for threshold computation.
    
    Returns:
        Optimal threshold value, or nan if computation fails.
    """
    if values.size == 0:
        return float("nan")
    
    # Prefer GPU histogram for large arrays; CPU for very small to avoid overhead
    if HAS_CUDA and values.size >= 4096:
        vals = cp.asarray(values)
        vals = vals[xp.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        vmin = _to_python_float(xp.min(vals))
        vmax = _to_python_float(xp.max(vals))
        if vmax <= vmin:
            return vmin
        hist, bin_edges = xp.histogram(vals, bins=bins, range=(vmin, vmax))
        hist = hist.astype(xp.float64)
        prob = hist / xp.sum(hist)
        omega = xp.cumsum(prob)
        mu = xp.cumsum(prob * (xp.arange(bins)))
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + _EPSILON)
        idx = int(xp.nanargmax(sigma_b2))
        t = _to_python_float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)
        return t

    # CPU path: explicitly convert CuPy arrays to NumPy if needed
    if HAS_CUDA and isinstance(values, cp.ndarray):
        vals = np.asarray(values.get())  # Explicit GPU->CPU transfer
    else:
        vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        return vmin
    hist, bin_edges = np.histogram(vals, bins=bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / np.sum(hist)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (np.arange(bins)))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + _EPSILON)
    idx = int(np.nanargmax(sigma_b2))
    # Place threshold between idx and idx+1
    t = float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)
    return t


def _linear_regression_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute linear regression slope and intercept."""
    mod = xp
    if x.size == 0 or y.size == 0:
        # Safe conversion: handle CuPy arrays explicitly
        y_cpu = y.get() if (HAS_CUDA and isinstance(y, cp.ndarray)) else y
        return 0.0, float(np.nanmean(np.asarray(y_cpu)) if y.size else 0.0)
    x = mod.asarray(x)
    y = mod.asarray(y)
    x_mean = _to_python_float(mod.mean(x))
    y_mean = _to_python_float(mod.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    denom = _to_python_float(mod.sum(dx * dx))
    if denom == 0.0:
        return 0.0, y_mean
    slope = _to_python_float(mod.sum(dx * dy) / denom)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _costes_thresholds(
    a: np.ndarray,
    b: np.ndarray,
    *,
    steps: int = _COSTES_STEPS,
    min_points: int = _COSTES_MIN_POINTS,
) -> Tuple[float, float]:
    """
    Compute Costes automatic thresholds for two vectors a, b.

    Strategy (pragmatic implementation):
    - Fit linear regression b ~= s * a + c over all voxels in ROI.
    - Descend a-threshold from max(a) to min(a) in 'steps'.
    - For each t_a, set t_b = s * t_a + c (clipped to b-range).
    - Select the largest thresholds where Pearson(a<=t_a, b<=t_b) <= 0
      with at least 'min_points' voxels; otherwise choose the pair with
      Pearson closest to 0 from the positive side.
    
    Args:
        a: First channel intensity vector.
        b: Second channel intensity vector.
        steps: Number of threshold steps to search.
        min_points: Minimum number of points required for correlation.
    
    Returns:
        Tuple of (threshold_a, threshold_b).
    """
    mod = xp
    a_f = a[mod.isfinite(a)]
    b_f = b[mod.isfinite(b)]
    if a_f.size == 0 or b_f.size == 0:
        return float("nan"), float("nan")
    a_min = _to_python_float(mod.min(a_f))
    a_max = _to_python_float(mod.max(a_f))
    b_min = _to_python_float(mod.min(b_f))
    b_max = _to_python_float(mod.max(b_f))
    if a_max <= a_min or b_max <= b_min:
        return a_min, b_min

    slope, intercept = _linear_regression_slope_intercept(a_f, b_f)

    best_ta = a_min
    best_tb = b_min
    best_r = float("inf")

    for k in range(steps, 0, -1):
        t_a = a_min + (a_max - a_min) * (k / steps)
        t_b = slope * t_a + intercept
        # clip into observed range
        t_b = max(b_min, min(b_max, t_b))
        mask = (a <= t_a) & (b <= t_b)
        if int(xp.count_nonzero(mask)) < min_points:
            continue
        r_bg = _safe_corrcoef(a[mask], b[mask])
        if np.isnan(r_bg):
            continue
        if r_bg <= 0.0:
            best_ta = t_a
            best_tb = t_b
            best_r = r_bg
            break
        # Track smallest positive r to fall back to
        if r_bg < best_r:
            best_r = r_bg
            best_ta = t_a
            best_tb = t_b

    return float(best_ta), float(best_tb)


def _extract_channel_vectors(
    img: np.ndarray, ch_a: int, ch_b: int, roi_2d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # img: (Z,Y,X,C), roi_2d: (Y,X) bool or label==k selection
    z = img.shape[0]
    roi_zyx = _broadcast_mask_to_z(roi_2d, z)  # (Z,Y,X)
    a = img[..., ch_a][roi_zyx]
    b = img[..., ch_b][roi_zyx]
    if HAS_CUDA:
        return a.astype(cp.float32), b.astype(cp.float32)  # type: ignore[return-value]
    return a.astype(np.float32), b.astype(np.float32)


def _extract_channel_vectors_single_z(
    img: np.ndarray, ch_a: int, ch_b: int, roi_2d: np.ndarray, z_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract channel vectors from a single Z-slice."""
    # img: (Z,Y,X,C), roi_2d: (Y,X) bool or label==k selection, z_idx: specific Z-slice
    a = img[z_idx, ..., ch_a][roi_2d]
    b = img[z_idx, ..., ch_b][roi_2d]
    if HAS_CUDA:
        return a.astype(cp.float32), b.astype(cp.float32)  # type: ignore[return-value]
    return a.astype(np.float32), b.astype(np.float32)


def _filter_labels(
    mask_2d: np.ndarray,
    *,
    min_area: int = 0,
    max_border_fraction: Optional[float] = None,
    border_margin_px: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Filter labels by minimum area and by fraction overlapping an edge band.

    - min_area: keep labels with area >= min_area (pixels)
    - max_border_fraction: if set, remove labels whose fraction of pixels within a
      border band (thickness=border_margin_px) exceeds this threshold.
    - border_margin_px: thickness of the border band in pixels.

    Returns (filtered_mask, info_dict).
    """
    if mask_2d.ndim != 2:
        raise ValueError("Expected a 2D labeled mask.")

    labels, counts = np.unique(mask_2d, return_counts=True)
    area: Dict[int, int] = {int(l): int(c) for l, c in zip(labels.tolist(), counts.tolist())}

    h, w = mask_2d.shape
    m = max(1, int(border_margin_px))
    edge = np.zeros((h, w), dtype=bool)
    edge[:m, :] = True
    edge[-m:, :] = True
    edge[:, :m] = True
    edge[:, -m:] = True

    keep: List[int] = []
    border_frac: Dict[int, float] = {}
    for l in labels.tolist():
        if l == 0:
            continue
        a = area[int(l)]
        if a < int(min_area):
            continue
        if max_border_fraction is not None:
            on_edge = int(np.count_nonzero(edge & (mask_2d == l)))
            frac = on_edge / a if a > 0 else 1.0
            border_frac[int(l)] = frac
            if frac > float(max_border_fraction):
                continue
        keep.append(int(l))

    out = mask_2d.copy()
    if keep:
        out[~np.isin(out, keep)] = 0
    else:
        out[:] = 0

    info: Dict[str, Any] = {
        "min_area": int(min_area),
        "max_border_fraction": None if max_border_fraction is None else float(max_border_fraction),
        "border_margin_px": int(border_margin_px),
        "kept_labels": keep,
        "removed_labels": [int(l) for l in labels.tolist() if l != 0 and int(l) not in keep],
        "border_fraction_by_label": border_frac,
    }
    logger.info(
        "Filtering labels: min_area=%s, max_border_fraction=%s, border_margin_px=%s -> kept=%d, removed=%d",
        info["min_area"],
        info["max_border_fraction"],
        info["border_margin_px"],
        len(info["kept_labels"]),
        len(info["removed_labels"]),
    )
    return out.astype(np.int32), info


def _compute_thresholds(
    a: np.ndarray,
    b: np.ndarray,
    thresholding: str,
    fixed_thresholds: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute thresholds for Manders coefficients based on selected strategy.
    
    Args:
        a: First channel intensity vector.
        b: Second channel intensity vector.
        thresholding: Strategy - 'none', 'otsu', 'costes', or 'fixed'.
        fixed_thresholds: Required if thresholding='fixed'.
    
    Returns:
        Tuple of (threshold_a, threshold_b), or (None, None) for 'none'.
    
    Raises:
        ValueError: If thresholding method is unknown or fixed_thresholds missing.
    """
    if thresholding == "none":
        return None, None
    elif thresholding == "otsu":
        return _otsu_threshold_1d(a), _otsu_threshold_1d(b)
    elif thresholding == "costes":
        return _costes_thresholds(a, b)
    elif thresholding == "fixed":
        if not fixed_thresholds:
            raise ValueError("fixed_thresholds must be provided when thresholding='fixed'")
        return float(fixed_thresholds[0]), float(fixed_thresholds[1])
    else:
        raise ValueError(f"thresholding must be one of {{'none','otsu','costes','fixed'}}, got '{thresholding}'")


def _plot_mask_with_indices(
    mask_original: np.ndarray,
    kept: List[int],
    removed: List[int],
    title: str = "Mask (labels)",
    show: bool = False,
):
    """Plot mask with label indices annotated (kept=black, removed=red).

    Returns a Matplotlib Figure; does not call plt.show() unless explicitly requested.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        logger.warning(f"matplotlib not available; skipping plot: {e}")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask_original, cmap="tab20")
    ax.axis("off")
    ax.set_title(title)

    labels = [int(l) for l in np.unique(mask_original) if l != 0]
    removed_set = set(removed)
    for l in labels:
        ys, xs = np.where(mask_original == l)
        if xs.size == 0:
            continue
        y = float(ys.mean())
        x = float(xs.mean())
        color = "red" if l in removed_set else "black"
        ax.text(x, y, str(l), color=color, fontsize=10, ha="center", va="center")
    if show:
        plt.show()
    logger.info(
        "Prepared mask plot (kept=%d, removed=%d, show=%s)",
        len(kept),
        len(removed),
        show,
    )
    return fig


def compute_colocalization(
    image: Union[str, Path, np.ndarray],
    mask: Union[str, Path, np.ndarray],
    channel_a: Union[int, str],
    channel_b: Union[int, str],
    *,
    normalization_scope: str = "none",  # deprecated: raw domain is enforced
    thresholding: str = "none",  # 'none'|'otsu'|'costes'|'fixed'
    fixed_thresholds: Optional[Tuple[float, float]] = None,  # used if thresholding='fixed' -> (t_a, t_b)
    channel_names: Optional[List[str]] = None,
    # Filtering and plotting controls
    min_area: Union[int, str] = "auto",  # int or "auto" to estimate from mask
    min_area_fraction: float = 0.7,  # fraction of median area when min_area="auto"
    max_border_fraction: Optional[float] = None,
    border_margin_px: int = 1,
    drop_label_1: bool = True,
    plot_mask: bool = False,
    plot_mask_save: Optional[Union[str, Path]] = None,
    # Preprocessing controls
    bleedthrough_matrix: Optional[List[List[float]]] = None,
    background_subtract: Optional[Dict[str, Any]] = None,  # {"percentile": 1.0}
    # Threshold domain selection
    threshold_domain: str = "raw",  # enforced to raw for consistency
    # Aggregation controls
    manders_weighting: str = "voxel",  # 'voxel' (global) or 'slice' (legacy mean of per-z)
    pearson_winsor_clip: Optional[float] = None,  # symmetric percentile clip for Pearson (e.g., 0.1)
    pearson_min_count: int = 2,
    pearson_min_range: float = 1e-12,
    # Output options
    output_json: Optional[Union[str, Path]] = None,  # save results to JSON if provided
) -> Dict[str, Any]:
    """
    Compute colocalization metrics between two channels on a ZYXC post-processed image.

    - image: path to OME-TIFF (preferred) or np.ndarray (ZYXC). If array and selecting by name, pass channel_names.
    - mask: path to labeled TIF (2D or 3D) or np.ndarray. 3D will be reduced to best z-slice and broadcast across Z.
    - channel_a, channel_b: channel indices or names (names require metadata/channel_names).
    - normalization_scope:
        'mask'   -> MinMaxScaler fit on all voxels inside union of labels (recommended).
        'global' -> MinMaxScaler fit on all voxels (entire ZYX).
        'none'   -> no normalization (computations on raw intensities).
    - min_area: int for fixed threshold, or "auto" to estimate from median cell area.
    - min_area_fraction: when min_area="auto", use this fraction of median area (default 0.7).
    - output_json: optional path to save results as JSON.
    Returns a JSON-serializable dict with per-label and total_image metrics.
    """
    img, ch_a, ch_b, names = _load_image_and_channels(image, channel_a, channel_b, channel_names)
    mask_2d = _load_mask(mask)
    
    # Auto-estimate min_area from mask if requested
    if normalization_scope.lower() != "none":
        logger.warning(
            "normalization_scope is deprecated; raw intensities are always used. "
            "Provided value '%s' will be ignored.",
            normalization_scope,
        )

    if threshold_domain.lower() != "raw":
        logger.warning(
            "threshold_domain is forced to 'raw' for all metrics. Provided value '%s' will be ignored.",
            threshold_domain,
        )
        threshold_domain = "raw"

    if min_area == "auto":
        min_area = estimate_min_area_threshold(mask_2d, fraction_of_median=min_area_fraction)
        logger.info(f"Auto-estimated min_area threshold: {min_area} (fraction={min_area_fraction})")
    else:
        min_area = int(min_area)
    
    logger.info(
        "Starting compute_colocalization(ch_a=%s, ch_b=%s, normalization_scope=%s, min_area=%d, max_border_fraction=%s, border_margin_px=%d, plot_mask=%s)",
        str(channel_a),
        str(channel_b),
        normalization_scope,
        int(min_area),
        str(max_border_fraction),
        int(border_margin_px),
        str(plot_mask),
    )
    logger.info(
        "Image loaded: shape=%s, channels=%s | Mask loaded: shape=%s, unique_labels=%d",
        tuple(int(x) for x in img.shape),
        names,
        tuple(int(x) for x in mask_2d.shape),
        int(len(np.unique(mask_2d)) - (1 if np.any(mask_2d == 0) else 0)),
    )
    logger.info(
        "Manders coefficients (M1/M2) weighting: %s (thresholding=%s, z-slices=%d)",
        manders_weighting,
        thresholding,
        img.shape[0],
    )

    if mask_2d.shape != img.shape[1:3]:
        raise ValueError(f"Mask (H,W) {mask_2d.shape} must match image spatial size {img.shape[1:3]}")

    # Keep a copy of original mask for visualization
    mask_original = mask_2d.copy()
    
    # Automatically remove label 1 (background in Cellpose)
    has_label_1 = np.any(mask_2d == 1)
    if has_label_1 and drop_label_1:
        mask_2d[mask_2d == 1] = 0
        logger.warning(
            "Removed label 1 from mask (assumed background, e.g., Cellpose). "
            "Set drop_label_1=False to keep it."
        )

    # Optional filtering BEFORE normalization/metrics
    filter_info: Dict[str, Any] = {
        "min_area": int(min_area),
        "max_border_fraction": None if max_border_fraction is None else float(max_border_fraction),
        "border_margin_px": int(border_margin_px),
        "kept_labels": [int(l) for l in np.unique(mask_2d) if l > 1],  # Exclude 0 (background) and 1 (already removed)
        "removed_labels": [1] if (has_label_1 and drop_label_1) else [],
    }
    if min_area > 0 or max_border_fraction is not None:
        mask_2d, filter_info = _filter_labels(
            mask_2d,
            min_area=min_area,
            max_border_fraction=max_border_fraction,
            border_margin_px=border_margin_px,
        )
        # Preserve label 1 in removed list (it was already removed before filtering)
        if has_label_1 and 1 not in filter_info["removed_labels"]:
            filter_info["removed_labels"].append(1)
    else:
        logger.info("No label filtering applied.")

    # Optional visualization of labels (kept=black, removed=red)
    if plot_mask:
        fig = _plot_mask_with_indices(
            mask_original,
            kept=filter_info.get("kept_labels", []),
            removed=filter_info.get("removed_labels", []),
            title="Mask (labels)",
            show=False,
        )
        if fig is not None and plot_mask_save is not None:
            plot_path = Path(plot_mask_save)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, bbox_inches="tight")
            filter_info["mask_plot_path"] = str(plot_path)
        elif fig is not None:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.close(fig)
            except Exception:
                pass

    # Build union mask (labels>0) AFTER filtering
    union_mask = (mask_2d > 0)

    # Optional preprocessing before metrics
    if bleedthrough_matrix is not None:
        if apply_bleedthrough_unmix is None:
            raise RuntimeError("Bleed-through unmix utility unavailable")
        mat = np.asarray(bleedthrough_matrix, dtype=np.float32)
        img = apply_bleedthrough_unmix(img, mat)  # type: ignore[arg-type]

    if background_subtract is not None:
        if subtract_background_percentile_roi is None:
            raise RuntimeError("Background subtraction utility unavailable")
        perc = float(background_subtract.get("percentile", 1.0))
        img = subtract_background_percentile_roi(img, union_mask, percentile=perc)  # type: ignore[arg-type]

    # Normalization is deprecated; metrics use raw values
    a_mm = b_mm = None  # type: ignore

    # Helper to compute metrics for a boolean ROI (2D)
    def metrics_for_roi(roi_2d: np.ndarray) -> Dict[str, float]:
        a_raw, b_raw = _extract_channel_vectors(img, ch_a, ch_b, roi_2d)
        if a_raw.size == 0:
            return {
                "pearson_r": float("nan"),
                "manders_m1": float("nan"),
                "manders_m2": float("nan"),
                "manders_m1_per_z": [],
                "manders_m2_per_z": [],
                "overlap_r": float("nan"),
                "jaccard": float("nan"),
                "n_voxels": 0.0,
                "thresholds_per_z": [],
            }

        if a_mm is not None:
            a_norm, b_norm = _normalize(a_mm, b_mm, a_raw, b_raw)  # type: ignore[arg-type]
        else:
            a_norm, b_norm = a_raw, b_raw

        # Pearson, overlap, and Jaccard computed on all Z-slices together
        r = _safe_corrcoef(
            a_norm,
            b_norm,
            winsor_clip=pearson_winsor_clip,
            min_count=pearson_min_count,
            min_range=pearson_min_range,
        )
        ov = _overlap_coefficient(a_norm, b_norm)
        jac = _jaccard_on_positive(a_norm, b_norm)

        # Manders computed per Z-slice with independent thresholds
        num_z = img.shape[0]
        m1_per_z = []
        m2_per_z = []
        thresholds_per_z = []
        voxel_weights = []
        sum_a_over_b = 0.0
        sum_b_over_a = 0.0
        total_a = float(np.sum(a_tsrc := a_raw))  # type: ignore[assignment]
        total_b = float(np.sum(b_tsrc := b_raw))  # type: ignore[assignment]
        
        for z_idx in range(num_z):
            a_raw_z, b_raw_z = _extract_channel_vectors_single_z(img, ch_a, ch_b, roi_2d, z_idx)
            
            if a_raw_z.size == 0:
                continue
            
            # Normalize if needed
            if a_mm is not None:
                a_norm_z, b_norm_z = _normalize(a_mm, b_mm, a_raw_z, b_raw_z)  # type: ignore[arg-type]
            else:
                a_norm_z, b_norm_z = a_raw_z, b_raw_z
            
            # Select domain for threshold discovery
            if threshold_domain == "raw":
                a_tsrc_z, b_tsrc_z = a_raw_z, b_raw_z
            elif threshold_domain == "normalized":
                a_tsrc_z, b_tsrc_z = a_norm_z, b_norm_z
            else:
                raise ValueError("threshold_domain must be one of {'raw','normalized'}")
            
            # Determine thresholds for this Z-slice using extracted helper
            t_a_z, t_b_z = _compute_thresholds(a_tsrc_z, b_tsrc_z, thresholding, fixed_thresholds)
            
            # Compute Manders for this Z-slice
            m1_z, m2_z = _manders_m1_m2(a_tsrc_z, b_tsrc_z, t_a=t_a_z, t_b=t_b_z)
            m1_per_z.append(m1_z)
            m2_per_z.append(m2_z)
            voxel_weights.append(int(a_tsrc_z.size))

            # Accumulate voxel-weighted numerators for global M1/M2
            sum_a_over_b += float(np.sum(a_tsrc_z[b_tsrc_z > (t_b_z if t_b_z is not None else 0)]))
            sum_b_over_a += float(np.sum(b_tsrc_z[a_tsrc_z > (t_a_z if t_a_z is not None else 0)]))
            thresholds_per_z.append({
                "z": z_idx,
                "t_a": None if t_a_z is None else float(t_a_z),
                "t_b": None if t_b_z is None else float(t_b_z),
            })
        
        # Aggregate Manders
        if manders_weighting == "slice":
            m1 = float(np.nanmean(m1_per_z)) if m1_per_z else float("nan")
            m2 = float(np.nanmean(m2_per_z)) if m2_per_z else float("nan")
        elif manders_weighting == "voxel":
            m1 = sum_a_over_b / total_a if total_a > 0 else float("nan")
            m2 = sum_b_over_a / total_b if total_b > 0 else float("nan")
        else:
            raise ValueError("manders_weighting must be one of {'voxel','slice'}")

        return {
            "pearson_r": float(r),
            "manders_m1": float(m1),
            "manders_m2": float(m2),
            "manders_m1_per_z": [float(x) for x in m1_per_z],
            "manders_m2_per_z": [float(x) for x in m2_per_z],
            "manders_weighting": manders_weighting,
            "overlap_r": float(ov),
            "jaccard": float(jac),
            # number of voxels used in this ROI (use channel vector length)
            "n_voxels": float(a_raw.size),
            "thresholds_per_z": thresholds_per_z,
        }

    # Per-label metrics (after filtering)
    labels = np.unique(mask_2d)
    labels = labels[labels != 0]
    per_label: List[Dict[str, Any]] = []
    for lab in labels.tolist():
        roi = (mask_2d == lab)
        m = metrics_for_roi(roi)
        m["type"] = f"cell{int(lab)}"
        m["label"] = int(lab)
        per_label.append(m)

    # Total image metrics (union of labels)
    total_metrics = metrics_for_roi(union_mask)
    total_metrics["type"] = "total_image"
    logger.info(
        "Computed metrics for %d labels; total_image n_voxels=%s",
        int(len(labels)),
        str(total_metrics.get("n_voxels")),
    )

    # Aggregate (mean over labels) for convenience
    def _mean_over_labels(key: str) -> float:
        vals = [x[key] for x in per_label if np.isfinite(x[key])]
        return float(np.mean(vals)) if len(vals) else float("nan")

    summary = {
        "labels_count": int(len(labels)),
        "mean_over_labels": {
            "pearson_r": _mean_over_labels("pearson_r"),
            "manders_m1": _mean_over_labels("manders_m1"),
            "manders_m2": _mean_over_labels("manders_m2"),
            "overlap_r": _mean_over_labels("overlap_r"),
            "jaccard": _mean_over_labels("jaccard"),
        },
    }

    out: Dict[str, Any] = {
        "image_shape": tuple(int(x) for x in img.shape),
        "channels": names,
        "channel_a": names[ch_a] if names and ch_a < len(names) else ch_a,
        "channel_b": names[ch_b] if names and ch_b < len(names) else ch_b,
        "normalization_scope": normalization_scope,
        "thresholding": thresholding,
        "threshold_domain": threshold_domain,
        "filtering": filter_info,
        "results": {
            "per_label": per_label,
            "total_image": total_metrics,
            "summary": summary,
        },
    }
    
    # Save to JSON if output path provided
    if output_json is not None:
        export_colocalization_json(out, output_json)
    
    logger.info("compute_colocalization finished.")
    return out


def export_colocalization_json(result: Dict[str, Any], out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote colocalization JSON to {str(out_path)}")