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
except Exception:  # pragma: no cover - fallback path
    cp = None  # type: ignore
    xp = np
    HAS_CUDA = False

try:
    # Prefer package import when used inside perinuclear_analysis
    from ..data_processing.image_loader import ImageLoader
except Exception:
    # Allow running standalone in a notebook if needed
    ImageLoader = None  # type: ignore

# Optional preprocessing utilities (bleed-through unmixing and ROI percentile background)
try:
    from ..imaging_preprocessing.background_subtraction import (
        apply_bleedthrough_unmix,
        subtract_background_percentile_roi,
    )
except Exception:
    apply_bleedthrough_unmix = None  # type: ignore
    subtract_background_percentile_roi = None  # type: ignore

logger = logging.getLogger(__name__)


def _as_zyxc(image: np.ndarray) -> np.ndarray:
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
            if arr_np.ndim == 4 and arr_np.shape[-1] == 1:
                arr_np = arr_np[..., 0]
            if arr_np.ndim != 3:
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


def _load_mask(mask: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        if ImageLoader is None:
            raise RuntimeError("ImageLoader not available; pass a numpy array instead.")
        loader = ImageLoader()
        m = loader.load_tif_mask(str(mask))
    else:
        m = mask

    if m.ndim == 3:
        # Reduce labeled 3D mask to a representative 2D slice (max labeled area)
        labeled_counts = [(z_idx, int((m[z_idx] > 0).sum())) for z_idx in range(m.shape[0])]
        z_best = max(labeled_counts, key=lambda t: t[1])[0]
        logger.info(
            f"3D mask detected; reducing to 2D by selecting z={z_best} "
            f"(largest labeled area) and broadcasting across Z."
        )
        m = m[z_best]

    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after reduction; got {m.shape}")

    # Coerce dtype robustly while preserving label integers when possible
    if np.issubdtype(m.dtype, np.integer):
        m_out = m.astype(np.int32)
        logger.info("Loaded labeled mask with integer dtype: %s", str(m.dtype))
        return m_out

    # Float or other non-integer dtype: decide between binary and labeled
    try:
        m_min = float(np.nanmin(m))
        m_max = float(np.nanmax(m))
    except Exception:
        m_min, m_max = 0.0, 0.0

    if m_min >= 0.0 and m_max <= 1.0:
        # Likely binary/probability mask; threshold at 0.5
        logger.info("Loaded non-integer mask in [0,1]; converting to binary with threshold 0.5")
        return (m > 0.5).astype(np.int32)

    # Otherwise, assume labeled mask stored as float; round to nearest int
    logger.info("Loaded non-integer mask with range [%s, %s]; rounding to int labels", m_min, m_max)
    return np.rint(m).astype(np.int32)


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


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    mod = xp
    if getattr(a, "size", 0) < 2 or getattr(b, "size", 0) < 2:
        return float("nan")
    a = mod.asarray(a).reshape(-1)
    b = mod.asarray(b).reshape(-1)
    # constant-vector guard using range to avoid indexing/reshape pitfalls
    if float(mod.max(a) - mod.min(a)) == 0.0 or float(mod.max(b) - mod.min(b)) == 0.0:
        return float("nan")
    r = mod.corrcoef(a, b)[0, 1]
    return float(cp.asnumpy(r)) if HAS_CUDA else float(r)


def _overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    mod = xp
    num = float(mod.sum(a * b))
    den = float(mod.sqrt(mod.sum(a * a) * mod.sum(b * b)))
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
    mod = xp
    a_bin = a > 0
    b_bin = b > 0
    inter = float(mod.sum(a_bin & b_bin))
    union = float(mod.sum(a_bin | b_bin))
    return inter / union if union > 0 else float("nan")


def _fit_minmax(a_vals: np.ndarray, b_vals: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    mod = xp
    a_vals = mod.asarray(a_vals).reshape(-1)
    b_vals = mod.asarray(b_vals).reshape(-1)
    a_min = float(mod.nanmin(a_vals))
    a_max = float(mod.nanmax(a_vals))
    b_min = float(mod.nanmin(b_vals))
    b_max = float(mod.nanmax(b_vals))
    return (a_min, a_max), (b_min, b_max)


def _normalize(a_mm: Tuple[float, float], b_mm: Tuple[float, float], a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mod = xp
    a_min, a_max = a_mm
    b_min, b_max = b_mm
    a = mod.asarray(a).reshape(-1)
    b = mod.asarray(b).reshape(-1)
    a_n = (a - a_min) / (a_max - a_min + 1e-12)
    b_n = (b - b_min) / (b_max - b_min + 1e-12)
    return a_n.astype(mod.float32, copy=False), b_n.astype(mod.float32, copy=False)


def _otsu_threshold_1d(values: np.ndarray, *, bins: int = 256) -> float:
    """
    Compute Otsu threshold on a 1D array. Works for arbitrary value ranges.
    Returns a threshold in the same scale as the input values.
    """
    if values.size == 0:
        return float("nan")
    # Prefer GPU histogram for large arrays; CPU for very small to avoid overhead
    if HAS_CUDA and values.size >= 4096:
        vals = cp.asarray(values)
        vals = vals[xp.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        vmin = float(xp.min(vals))
        vmax = float(xp.max(vals))
        if vmax <= vmin:
            return vmin
        hist, bin_edges = xp.histogram(vals, bins=bins, range=(vmin, vmax))
        hist = hist.astype(xp.float64)
        prob = hist / xp.sum(hist)
        omega = xp.cumsum(prob)
        mu = xp.cumsum(prob * (xp.arange(bins)))
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
        idx = int(xp.nanargmax(sigma_b2))
        t = float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)
        return t

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
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    # Place threshold between idx and idx+1
    t = float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)
    return t


def _linear_regression_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mod = xp
    if x.size == 0 or y.size == 0:
        return 0.0, float(np.nanmean(np.asarray(y)) if y.size else 0.0)
    x = mod.asarray(x)
    y = mod.asarray(y)
    x_mean = float(mod.mean(x))
    y_mean = float(mod.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    denom = float(mod.sum(dx * dx))
    if denom == 0.0:
        return 0.0, y_mean
    slope = float(mod.sum(dx * dy) / denom)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _costes_thresholds(
    a: np.ndarray,
    b: np.ndarray,
    *,
    steps: int = 256,
    min_points: int = 100,
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
    """
    mod = xp
    a_f = a[mod.isfinite(a)]
    b_f = b[mod.isfinite(b)]
    if a_f.size == 0 or b_f.size == 0:
        return float("nan"), float("nan")
    a_min = float(mod.min(a_f))
    a_max = float(mod.max(a_f))
    b_min = float(mod.min(b_f))
    b_max = float(mod.max(b_f))
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
        if t_b < b_min:
            t_b = b_min
        if t_b > b_max:
            t_b = b_max
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


def _plot_mask_with_indices(
    mask_original: np.ndarray,
    kept: List[int],
    removed: List[int],
    title: str = "Mask (labels)",
    show: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.warning(f"matplotlib not available; skipping plot: {e}")
        return

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_original, cmap="tab20")
    plt.axis("off")
    plt.title(title)

    labels = [int(l) for l in np.unique(mask_original) if l != 0]
    removed_set = set(removed)
    for l in labels:
        ys, xs = np.where(mask_original == l)
        if xs.size == 0:
            continue
        y = float(ys.mean())
        x = float(xs.mean())
        color = "red" if l in removed_set else "black"
        plt.text(x, y, str(l), color=color, fontsize=10, ha="center", va="center")
    if show:
        plt.show()
        logger.info(
            "Displayed mask with label indices (kept=%d, removed=%d)",
            len(kept),
            len(removed),
        )


def compute_colocalization(
    image: Union[str, Path, np.ndarray],
    mask: Union[str, Path, np.ndarray],
    channel_a: Union[int, str],
    channel_b: Union[int, str],
    *,
    normalization_scope: str = "none",  # default to none to avoid affecting thresholds
    thresholding: str = "none",  # 'none'|'otsu'|'costes'|'fixed'
    fixed_thresholds: Optional[Tuple[float, float]] = None,  # used if thresholding='fixed' -> (t_a, t_b)
    channel_names: Optional[List[str]] = None,
    # Filtering and plotting controls
    min_area: int = 0,
    max_border_fraction: Optional[float] = None,
    border_margin_px: int = 1,
    plot_mask: bool = True,
    # Preprocessing controls
    bleedthrough_matrix: Optional[List[List[float]]] = None,
    background_subtract: Optional[Dict[str, Any]] = None,  # {"percentile": 1.0}
    # Threshold domain selection
    threshold_domain: str = "raw",  # 'raw'|'normalized'
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
    Returns a JSON-serializable dict with per-label and total_image metrics.
    """
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
    img, ch_a, ch_b, names = _load_image_and_channels(image, channel_a, channel_b, channel_names)
    mask_2d = _load_mask(mask)
    logger.info(
        "Image loaded: shape=%s, channels=%s | Mask loaded: shape=%s, unique_labels=%d",
        tuple(int(x) for x in img.shape),
        names,
        tuple(int(x) for x in mask_2d.shape),
        int(len(np.unique(mask_2d)) - (1 if np.any(mask_2d == 0) else 0)),
    )

    if mask_2d.shape != img.shape[1:3]:
        raise ValueError(f"Mask (H,W) {mask_2d.shape} must match image spatial size {img.shape[1:3]}")

    # Keep a copy of original mask for visualization
    mask_original = mask_2d.copy()

    # Optional filtering BEFORE normalization/metrics
    filter_info: Dict[str, Any] = {
        "min_area": int(min_area),
        "max_border_fraction": None if max_border_fraction is None else float(max_border_fraction),
        "border_margin_px": int(border_margin_px),
        "kept_labels": [int(l) for l in np.unique(mask_2d) if l != 0],
        "removed_labels": [],
    }
    if min_area > 0 or max_border_fraction is not None:
        mask_2d, filter_info = _filter_labels(
            mask_2d,
            min_area=min_area,
            max_border_fraction=max_border_fraction,
            border_margin_px=border_margin_px,
        )
    else:
        logger.info("No label filtering applied.")

    # Optional visualization of labels (kept=black, removed=red)
    if plot_mask:
        _plot_mask_with_indices(
            mask_original,
            kept=filter_info.get("kept_labels", []),
            removed=filter_info.get("removed_labels", []),
            title="Mask (labels)",
            show=True,
        )

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

    # Collect vectors for normalization
    if normalization_scope == "mask":
        a_all, b_all = _extract_channel_vectors(img, ch_a, ch_b, union_mask)
    elif normalization_scope == "global":
        # all ZYX
        z, h, w, _ = img.shape
        all_true = np.ones((h, w), dtype=bool)
        a_all, b_all = _extract_channel_vectors(img, ch_a, ch_b, all_true)
    elif normalization_scope == "none":
        a_all = b_all = None  # type: ignore
    else:
        raise ValueError("normalization_scope must be one of {'mask','global','none'}")

    # Fit scalers (if enabled)
    if normalization_scope != "none":
        a_mm, b_mm = _fit_minmax(a_all, b_all)  # type: ignore[arg-type]
        logger.info(
            "Normalization fitted (scope=%s) on %d voxels per channel",
            normalization_scope,
            int(a_all.size) if hasattr(a_all, "size") else 0,
        )
    else:
        a_mm = b_mm = None  # type: ignore
        logger.info("Normalization disabled (scope='none')")

    # Helper to compute metrics for a boolean ROI (2D)
    def metrics_for_roi(roi_2d: np.ndarray) -> Dict[str, float]:
        a_raw, b_raw = _extract_channel_vectors(img, ch_a, ch_b, roi_2d)
        if a_raw.size == 0:
            return {
                "pearson_r": float("nan"),
                "manders_m1": float("nan"),
                "manders_m2": float("nan"),
                "overlap_r": float("nan"),
                "jaccard": float("nan"),
                "n_voxels": 0.0,
            }

        if a_mm is not None:
            a_norm, b_norm = _normalize(a_mm, b_mm, a_raw, b_raw)  # type: ignore[arg-type]
        else:
            a_norm, b_norm = a_raw, b_raw

        r = _safe_corrcoef(a_norm, b_norm)
        # Select domain for threshold discovery and Manders
        if threshold_domain == "raw":
            a_tsrc, b_tsrc = a_raw, b_raw
        elif threshold_domain == "normalized":
            a_tsrc, b_tsrc = a_norm, b_norm
        else:
            raise ValueError("threshold_domain must be one of {'raw','normalized'}")

        # Determine thresholds for Manders
        t_a: Optional[float]
        t_b: Optional[float]
        if thresholding == "none":
            t_a = None
            t_b = None
        elif thresholding == "otsu":
            t_a = _otsu_threshold_1d(a_tsrc)
            t_b = _otsu_threshold_1d(b_tsrc)
        elif thresholding == "costes":
            ta_c, tb_c = _costes_thresholds(a_tsrc, b_tsrc)
            t_a = ta_c
            t_b = tb_c
        elif thresholding == "fixed":
            if not fixed_thresholds:
                raise ValueError("fixed_thresholds must be provided when thresholding='fixed'")
            t_a, t_b = float(fixed_thresholds[0]), float(fixed_thresholds[1])
        else:
            raise ValueError("thresholding must be one of {'none','otsu','costes','fixed'}")

        m1, m2 = _manders_m1_m2(a_tsrc, b_tsrc, t_a=t_a, t_b=t_b)
        ov = _overlap_coefficient(a_norm, b_norm)
        jac = _jaccard_on_positive(a_norm, b_norm)

        return {
            "pearson_r": float(r),
            "manders_m1": float(m1),
            "manders_m2": float(m2),
            "overlap_r": float(ov),
            "jaccard": float(jac),
            # number of voxels used in this ROI (use channel vector length)
            "n_voxels": float(a_raw.size),
            "thresholds": {"t_a": None if t_a is None else float(t_a), "t_b": None if t_b is None else float(t_b)},
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
    logger.info("compute_colocalization finished.")
    return out


def export_colocalization_json(result: Dict[str, Any], out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote colocalization JSON to {str(out_path)}")