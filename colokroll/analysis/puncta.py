"""
Puncta detection and analysis module.

Provides automated puncta (spot) detection and segmentation with per-punctum and per-cell metrics.
Uses LoG/DoG bandpass filtering, MAD-based SNR thresholding, and seeded watershed segmentation.

Workflow:
- Accept background-corrected image (path, array, or results dict)
- Project to 2D (MIP/SME) or use provided 2D image
- Detect puncta seeds via LoG filtering + local maxima
- Threshold foreground via robust background + MAD-based SNR
- Segment puncta via seeded watershed
- Measure per-punctum and per-cell metrics
- Return JSON-serializable dict compatible with existing pipeline
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import distance as sp_distance

try:
    from skimage import measure, morphology
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from skimage.segmentation import watershed
    HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    HAS_SKIMAGE = False

try:
    from ..data_processing.image_loader import ImageLoader
except ImportError:
    ImageLoader = None  # type: ignore

try:
    from ..data_processing.projection import MIPCreator
except ImportError:
    MIPCreator = None  # type: ignore

try:
    from bigfish import detection as bigfish_detection
    HAS_BIGFISH = True
except ImportError:  # pragma: no cover
    bigfish_detection = None  # type: ignore
    HAS_BIGFISH = False


# =============================================================================
# Module-level constants (imported from centralized constants module)
# =============================================================================
from .constants import (
    EPSILON as _EPSILON,
    DEFAULT_MIN_DISTANCE_PX as _DEFAULT_MIN_DISTANCE_PX,
    DEFAULT_SNR_THRESHOLD as _DEFAULT_SNR_THRESHOLD,
    DEFAULT_EXPECTED_DIAMETER_UM as _DEFAULT_EXPECTED_DIAMETER_UM,
    DEFAULT_MIN_DIAMETER_UM as _DEFAULT_MIN_DIAMETER_UM,
    DEFAULT_MAX_DIAMETER_UM as _DEFAULT_MAX_DIAMETER_UM,
    MAD_TO_STD,
    LOG_SIGMA_FACTOR,
    MIN_BACKGROUND_SAMPLES,
    FALLBACK_EXPECTED_DIAMETER_PX,
    FALLBACK_MIN_DIAMETER_PX,
    FALLBACK_MAX_DIAMETER_PX,
    CHANNEL_DIM_HEURISTIC,
    BIGFISH_THRESHOLD_SAMPLES,
    MAX_AREA_SAFETY_MARGIN,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions
# =============================================================================

def _to_numpy(arr: Any) -> np.ndarray:
    """Convert CuPy or array-like to NumPy ndarray."""
    try:
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)


def _um_to_px(um: float, pixel_size_um: float) -> float:
    """Convert micrometers to pixels."""
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    return um / pixel_size_um


def _px_to_um(px: float, pixel_size_um: float) -> float:
    """Convert pixels to micrometers."""
    if pixel_size_um is None or pixel_size_um <= 0:
        return float("nan")
    return px * pixel_size_um


def _area_px_to_um2(area_px: float, pixel_size_um: float) -> float:
    """Convert area from pixels to µm²."""
    if pixel_size_um is None or pixel_size_um <= 0:
        return float("nan")
    return area_px * (pixel_size_um ** 2)


def _load_single_channel(
    image: Union[str, Path, np.ndarray, Dict[str, Any]],
    channel: Union[int, str],
    channel_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, str, List[str]]:
    """
    Load a single channel from various input formats.

    Args:
        image: Path, array (Z,Y,X,C) or (Y,X,C), or dict[channel_name -> array].
        channel: Channel index or name.
        channel_names: Optional list of channel names (required if array + string channel).

    Returns:
        Tuple of (channel_data as (Z,Y,X) or (Y,X), channel_name, all_names).
    """
    if isinstance(image, dict):
        # Dict mapping channel_name -> (array, meta) or -> array
        names = list(image.keys())
        if isinstance(channel, str):
            if channel not in names:
                raise ValueError(f"Channel '{channel}' not in dict keys: {names}")
            ch_name = channel
        else:
            if channel < 0 or channel >= len(names):
                raise ValueError(f"Channel index {channel} out of range for {len(names)} channels")
            ch_name = names[channel]

        val = image[ch_name]
        arr = val[0] if (isinstance(val, (tuple, list)) and len(val) >= 1) else val
        arr = _to_numpy(arr)
        return arr, ch_name, names

    elif isinstance(image, (str, Path)):
        if ImageLoader is None:
            raise RuntimeError("ImageLoader not available; pass a numpy array instead.")
        loader = ImageLoader()
        img = loader.load_image(str(image))
        names = loader.get_channel_names()
        logger.info(f"Loaded image {str(image)} with channels: {names}")

        if isinstance(channel, str):
            if channel not in names:
                raise ValueError(f"Channel '{channel}' not in metadata: {names}")
            ch_idx = names.index(channel)
            ch_name = channel
        else:
            ch_idx = int(channel)
            ch_name = names[ch_idx] if ch_idx < len(names) else f"channel_{ch_idx}"

        return img[..., ch_idx], ch_name, names

    else:
        # NumPy array
        arr = _to_numpy(image)
        names = channel_names or []

        if isinstance(channel, str):
            if not names:
                raise ValueError("channel_names required when passing array and selecting by name")
            if channel not in names:
                raise ValueError(f"Channel '{channel}' not in channel_names: {names}")
            ch_idx = names.index(channel)
            ch_name = channel
        else:
            ch_idx = int(channel)
            ch_name = names[ch_idx] if ch_idx < len(names) else f"channel_{ch_idx}"

        # Handle different array shapes
        if arr.ndim == 2:
            # Already 2D, assume single channel
            if ch_idx != 0:
                raise ValueError("2D image has only 1 channel (index 0)")
            return arr, ch_name, names
        elif arr.ndim == 3:
            # Could be (Z,Y,X) single channel or (Y,X,C) multichannel
            # Heuristic: if last dim is small (<= CHANNEL_DIM_HEURISTIC), treat as channels
            if arr.shape[-1] <= CHANNEL_DIM_HEURISTIC:
                # (Y, X, C)
                return arr[..., ch_idx], ch_name, names
            else:
                # (Z, Y, X) single channel
                if ch_idx != 0:
                    raise ValueError("3D single-channel image has only 1 channel (index 0)")
                return arr, ch_name, names
        elif arr.ndim == 4:
            # (Z, Y, X, C)
            return arr[..., ch_idx], ch_name, names
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")


def _project_to_2d(
    channel_data: np.ndarray,
    projection: str = "mip",
    sme_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Project a 3D channel to 2D.

    Args:
        channel_data: (Z, Y, X) array.
        projection: "mip" for max projection, "sme" for surface manifold, "none" if already 2D.
        sme_reference: Optional full image for SME manifold computation.

    Returns:
        2D array (Y, X).
    """
    if channel_data.ndim == 2:
        return channel_data

    if channel_data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {channel_data.ndim}D")

    if projection == "none":
        # Take middle slice
        z_mid = channel_data.shape[0] // 2
        logger.info(f"projection='none' but 3D data; using middle slice z={z_mid}")
        return channel_data[z_mid]

    elif projection == "mip":
        return np.max(channel_data, axis=0)

    elif projection == "sme":
        if MIPCreator is None:
            raise RuntimeError("MIPCreator not available for SME projection")
        mip_creator = MIPCreator()
        # SME expects (Z, Y, X, C) or (Z, Y, X)
        if channel_data.ndim == 3:
            result = mip_creator.create_sme(channel_data)
            return result.projection
        else:
            raise ValueError("SME projection requires 3D data")

    else:
        raise ValueError(f"projection must be 'mip', 'sme', or 'none', got '{projection}'")


def _load_mask(mask: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load and validate a 2D labeled mask using shared utilities."""
    from ..core.mask_utils import load_and_validate_mask
    
    # Create loader function if ImageLoader is available
    loader_func = None
    if ImageLoader is not None:
        loader_func = lambda path: ImageLoader().load_tif_mask(path)
    
    return load_and_validate_mask(mask, image_loader_func=loader_func)


def _laplacian_of_gaussian(
    image: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Apply Laplacian of Gaussian (LoG) filter for blob enhancement.

    Args:
        image: 2D input image.
        sigma: Gaussian sigma (related to expected blob radius).

    Returns:
        LoG-filtered image (negative at blob centers).
    """
    # LoG = Laplacian(Gaussian(image))
    # For blob detection, we want the negative of LoG (positive at blob centers)
    smoothed = gaussian(image.astype(np.float64), sigma=sigma, preserve_range=True)
    log_result = ndi.laplace(smoothed)
    return -log_result  # Invert so blobs are positive


def _estimate_background_mad(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    percentile_low: float = 5.0,
    percentile_high: float = 50.0,
) -> Tuple[float, float]:
    """
    Estimate background mean and MAD from lower intensity pixels.

    Args:
        image: 2D input image.
        mask: Optional binary mask (True = include).
        percentile_low: Lower percentile for background region.
        percentile_high: Upper percentile for background region.

    Returns:
        Tuple of (background_mean, background_mad).
    """
    if mask is not None:
        vals = image[mask]
    else:
        vals = image.ravel()

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    # Use lower percentile range as background
    p_lo = np.percentile(vals, percentile_low)
    p_hi = np.percentile(vals, percentile_high)
    bg_vals = vals[(vals >= p_lo) & (vals <= p_hi)]

    if bg_vals.size < MIN_BACKGROUND_SAMPLES:
        bg_vals = vals

    bg_mean = float(np.median(bg_vals))
    # MAD = median absolute deviation
    mad = float(np.median(np.abs(bg_vals - bg_mean)))
    # Convert MAD to approximate std (for Gaussian: std ≈ 1.4826 * MAD)
    mad_std = max(mad * MAD_TO_STD, _EPSILON)

    return bg_mean, mad_std


def _detect_puncta_seeds(
    log_image: np.ndarray,
    min_distance: int,
    threshold_abs: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect puncta seed points using local maxima on LoG-filtered image.

    Args:
        log_image: LoG-filtered image (blobs are positive).
        min_distance: Minimum distance between peaks (pixels).
        threshold_abs: Absolute threshold for peak detection.
        mask: Optional binary mask restricting detection region.

    Returns:
        Array of seed coordinates (N, 2) as (row, col).
    """
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image required for puncta detection")

    # Apply mask if provided
    if mask is not None:
        log_masked = log_image.copy()
        log_masked[~mask] = log_image.min()
    else:
        log_masked = log_image

    # Find local maxima
    coordinates = peak_local_max(
        log_masked,
        min_distance=max(1, int(min_distance)),
        threshold_abs=threshold_abs,
        exclude_border=True,
    )

    return coordinates


def _segment_puncta_watershed(
    image: np.ndarray,
    seeds: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    """
    Segment puncta using seeded watershed.

    Args:
        image: Original or filtered 2D image (for gradient).
        seeds: (N, 2) array of seed coordinates.
        foreground_mask: Binary mask of foreground (puncta regions).

    Returns:
        Labeled mask where each punctum has a unique integer label.
    """
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image required for watershed segmentation")

    if seeds.size == 0:
        return np.zeros(image.shape, dtype=np.int32)

    # Create seed markers
    markers = np.zeros(image.shape, dtype=np.int32)
    for i, (r, c) in enumerate(seeds, start=1):
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
            markers[r, c] = i

    # Watershed on inverted image (so puncta are basins)
    # Use negative of image if we want bright regions to be segmented
    watershed_input = -image.astype(np.float64)

    # Run watershed
    labels = watershed(
        watershed_input,
        markers=markers,
        mask=foreground_mask,
        connectivity=2,
    )

    return labels.astype(np.int32)


def _detect_spots_bigfish(
    image: np.ndarray,
    spot_radius_px: float,
    cell_mask: Optional[np.ndarray] = None,
    return_threshold_data: bool = False,
    min_snr_threshold: float = 5.0,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Detect spots using BigFISH with automatic thresholding.

    BigFISH uses LoG filtering with automatic threshold detection based on
    spot statistics, which is more robust than manual threshold selection.

    Args:
        image: 2D input image.
        spot_radius_px: Expected spot radius in pixels (used for LoG sigma).
        cell_mask: Optional labeled cell mask to restrict detection.
        return_threshold_data: If True, return elbow curve data for plotting.
        min_snr_threshold: Minimum SNR threshold to apply when BigFISH auto-threshold
            fails (returns 0 or None). This prevents detecting noise as spots.

    Returns:
        If return_threshold_data is False:
            Tuple of (spots array (N, 2) as (y, x), puncta_labels 2D array).
        If return_threshold_data is True:
            Tuple of (spots, puncta_labels, threshold_data dict).
            threshold_data contains: threshold, thresholds, spot_counts for elbow plot.
    """
    if not HAS_BIGFISH:
        raise RuntimeError(
            "BigFISH is required for this detection method. "
            "Install with: pip install big-fish"
        )

    # Import additional BigFISH modules for threshold data
    from bigfish import stack as bigfish_stack

    # Compute LoG kernel size and minimum distance from spot radius
    # LoG kernel size should be ~2x the spot radius (diameter)
    log_kernel_size = max(3, int(round(spot_radius_px * 2)))
    # Ensure odd kernel size
    if log_kernel_size % 2 == 0:
        log_kernel_size += 1
    min_distance = max(1, int(round(spot_radius_px)))
    
    # For 2D images, use (y, x) tuple format
    log_kernel_2d = (log_kernel_size, log_kernel_size)
    min_distance_2d = (min_distance, min_distance)

    # Use the high-level detect_spots API which handles thresholding correctly
    spots, threshold = bigfish_detection.detect_spots(
        image.astype(np.float64),
        threshold=None,  # Auto-threshold using elbow method
        return_threshold=True,
        log_kernel_size=log_kernel_2d,
        minimum_distance=min_distance_2d,
    )

    # Handle edge cases: threshold can be None (no spots)
    if threshold is None:
        threshold_scalar = 0.0
        logger.warning("BigFISH detected 0 spots (no valid threshold found)")
    elif isinstance(threshold, np.ndarray):
        # Handle numpy arrays (shouldn't happen with detect_spots, but be safe)
        if threshold.size == 0:
            threshold_scalar = 0.0
            logger.warning("BigFISH returned empty threshold array")
        elif threshold.size == 1:
            threshold_scalar = float(threshold.flat[0])
        else:
            threshold_scalar = float(threshold.flat[0])
            logger.warning(f"BigFISH returned multi-element threshold (shape={threshold.shape})")
    else:
        threshold_scalar = float(threshold)

    # Compute LoG-filtered image for fallback validation
    sigma = (spot_radius_px, spot_radius_px)
    image_filtered = bigfish_stack.log_filter(image.astype(np.float64), sigma=sigma)
    
    # Compute percentiles of LoG-filtered intensities (only positive values)
    positive_intensities = image_filtered[image_filtered > 0]
    if len(positive_intensities) > 0:
        p95 = float(np.percentile(positive_intensities, 95))
        p99 = float(np.percentile(positive_intensities, 99))
    else:
        p95 = p99 = 0.0

    # FALLBACK: If BigFISH couldn't find a valid threshold (returns 0 or very low),
    # apply an SNR-based minimum threshold to prevent detecting noise as spots.
    # This is critical for negative control images where signal is minimal.
    used_fallback = False
    fallback_reason = None
    original_spot_count = len(spots)
    
    if threshold_scalar <= 0.01 and len(spots) > 0:
        # FALLBACK 1: Threshold too LOW
        # Estimate background from LoG-filtered image
        # Use median + MAD for robust estimation
        median_val = float(np.median(image_filtered))
        mad = float(np.median(np.abs(image_filtered - median_val)))
        robust_std = 1.4826 * mad  # MAD to std conversion
        
        # Compute SNR-based threshold
        fallback_threshold = median_val + min_snr_threshold * robust_std
        
        logger.warning(
            f"BigFISH auto-threshold TOO LOW (threshold={threshold_scalar:.4f}, "
            f"detected {original_spot_count} spots). "
            f"Applying fallback SNR-based threshold: {fallback_threshold:.2f} "
            f"(median={median_val:.2f}, std={robust_std:.2f}, SNR={min_snr_threshold})"
        )
        
        # Re-detect spots with the fallback threshold
        spots, _ = bigfish_detection.detect_spots(
            image.astype(np.float64),
            threshold=fallback_threshold,
            return_threshold=True,
            log_kernel_size=log_kernel_2d,
            minimum_distance=min_distance_2d,
        )
        threshold_scalar = fallback_threshold
        used_fallback = True
        fallback_reason = "threshold_too_low"
        
        # If fallback still detects many spots (e.g., >50% of original), 
        # it's likely noise - return 0 spots
        if len(spots) > 0 and original_spot_count > 0:
            ratio = len(spots) / original_spot_count
            if ratio > 0.3:  # More than 30% of "noise spots" still detected
                logger.warning(
                    f"Fallback still detected {len(spots)} spots ({ratio:.0%} of original). "
                    f"Image likely contains only noise - returning 0 spots."
                )
                spots = np.empty((0, 2), dtype=np.int64)
    
    elif threshold_scalar > p99 and p99 > 0:
        # FALLBACK 2: Threshold too HIGH (NEW)
        # This catches cases where extreme outliers skew the LoG-filtered distribution
        fallback_threshold = p95
        
        logger.warning(
            f"BigFISH auto-threshold TOO HIGH (threshold={threshold_scalar:.4f} > "
            f"p99={p99:.4f}, detected {original_spot_count} spots). "
            f"Elbow detection likely failed due to extreme outliers. "
            f"Applying percentile-based fallback: {fallback_threshold:.4f} (95th percentile)"
        )
        
        # Re-detect spots with the fallback threshold
        spots_fallback, _ = bigfish_detection.detect_spots(
            image.astype(np.float64),
            threshold=fallback_threshold,
            return_threshold=True,
            log_kernel_size=log_kernel_2d,
            minimum_distance=min_distance_2d,
        )
        
        # Validate: fallback should detect significantly more spots
        # At least 10x more, or at least 50 spots if original was very low
        min_expected_increase = max(10, original_spot_count * 10)
        
        if len(spots_fallback) >= min_expected_increase or (len(spots_fallback) >= 50 and original_spot_count < 10):
            logger.info(
                f"Fallback successful: detected {len(spots_fallback)} spots "
                f"(vs {original_spot_count} with auto-threshold)"
            )
            spots = spots_fallback
            threshold_scalar = fallback_threshold
            used_fallback = True
            fallback_reason = "threshold_too_high"
        else:
            logger.warning(
                f"Fallback validation failed: only detected {len(spots_fallback)} spots "
                f"(expected >={min_expected_increase}). "
                f"Keeping original threshold={threshold_scalar:.4f} with {original_spot_count} spots."
            )
    
    logger.info(
        f"BigFISH detected {len(spots)} spots with "
        f"{'fallback' if used_fallback else 'auto'}-threshold={threshold_scalar:.2f}"
        + (f" (reason: {fallback_reason})" if fallback_reason else "")
    )

    # Compute elbow curve data if requested
    threshold_data = None
    if return_threshold_data:
        # Apply LoG filter to get the filtered image for threshold analysis
        sigma = (spot_radius_px, spot_radius_px)
        image_filtered = bigfish_stack.log_filter(image.astype(np.float64), sigma=sigma)
        
        # Generate threshold range for elbow curve
        max_val = float(image_filtered.max())
        min_val = float(image_filtered[image_filtered > 0].min()) if np.any(image_filtered > 0) else 0
        thresholds = np.linspace(min_val, max_val, num=BIGFISH_THRESHOLD_SAMPLES)

        # Get local maxima mask and convert to coordinates
        local_maxima_mask = bigfish_detection.local_maximum_detection(
            image_filtered,
            min_distance=min_distance_2d,
        )
        # Convert boolean mask to coordinates
        local_maxima_coords = np.argwhere(local_maxima_mask)
        
        # Count spots at each threshold
        spot_counts = []
        if len(local_maxima_coords) > 0:
            for t in thresholds:
                count = int(np.sum(image_filtered[local_maxima_coords[:, 0], local_maxima_coords[:, 1]] >= t))
                spot_counts.append(count)
            local_maxima_count = len(local_maxima_coords)
        else:
            spot_counts = [0] * len(thresholds)
            local_maxima_count = 0

        threshold_data = {
            "threshold": threshold_scalar,
            "thresholds": thresholds.tolist(),
            "spot_counts": spot_counts,
            "local_maxima_count": local_maxima_count,
            "filtered_spots_count": len(spots),
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "log_intensity_percentiles": {
                "p95": p95,
                "p99": p99,
            },
        }

    # Filter spots to only those within cell mask if provided
    if cell_mask is not None:
        mask_filter = np.array([
            cell_mask[int(y), int(x)] > 0
            for y, x in spots
            if 0 <= int(y) < cell_mask.shape[0] and 0 <= int(x) < cell_mask.shape[1]
        ])
        spots = spots[mask_filter] if len(mask_filter) > 0 else spots
        logger.info(f"After cell mask filtering: {len(spots)} spots")

    # Create labeled mask from spots (simple dilation for now)
    puncta_labels = np.zeros(image.shape, dtype=np.int32)
    radius_int = max(1, int(round(spot_radius_px)))

    for i, (y, x) in enumerate(spots, start=1):
        y, x = int(y), int(x)
        # Create small circular region around each spot
        ymin = max(0, y - radius_int)
        ymax = min(image.shape[0], y + radius_int + 1)
        xmin = max(0, x - radius_int)
        xmax = min(image.shape[1], x + radius_int + 1)

        for dy in range(ymin, ymax):
            for dx in range(xmin, xmax):
                if (dy - y) ** 2 + (dx - x) ** 2 <= radius_int ** 2:
                    if puncta_labels[dy, dx] == 0:  # Don't overwrite
                        puncta_labels[dy, dx] = i

    if return_threshold_data:
        return spots, puncta_labels, threshold_data
    return spots, puncta_labels


def _measure_puncta(
    image_raw: np.ndarray,
    puncta_labels: np.ndarray,
    cell_mask: np.ndarray,
    pixel_size_um: Optional[float] = None,
    bg_mean: float = 0.0,
    bg_std: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Measure properties of each punctum.

    Args:
        image_raw: Raw/background-corrected 2D image.
        puncta_labels: Labeled puncta mask.
        cell_mask: Labeled cell mask (for cell assignment).
        pixel_size_um: Pixel size in micrometers (optional).
        bg_mean: Background mean for SNR calculation.
        bg_std: Background std for SNR calculation.

    Returns:
        List of dicts, one per punctum.
    """
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image required for puncta measurement")

    props = measure.regionprops(puncta_labels, intensity_image=image_raw)
    results: List[Dict[str, Any]] = []

    for prop in props:
        # Centroid
        cy, cx = prop.centroid
        
        # Assign to cell
        cell_label = int(cell_mask[int(cy), int(cx)]) if cell_mask is not None else 0

        # Geometry
        area_px = int(prop.area)
        area_um2 = _area_px_to_um2(area_px, pixel_size_um) if pixel_size_um else float("nan")
        equiv_diameter_px = prop.equivalent_diameter
        equiv_diameter_um = _px_to_um(equiv_diameter_px, pixel_size_um) if pixel_size_um else float("nan")

        # Shape metrics (guard against degenerate cases like single-pixel regions)
        try:
            eccentricity = float(prop.eccentricity)
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            eccentricity = float("nan")

        try:
            solidity = float(prop.solidity)
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            solidity = float("nan")

        # Intensity
        mean_intensity = float(prop.mean_intensity)
        max_intensity = float(image_raw[puncta_labels == prop.label].max())
        integrated_intensity = float(prop.mean_intensity * prop.area)

        # Background-subtracted intensity
        mean_intensity_bgsub = mean_intensity - bg_mean
        integrated_intensity_bgsub = mean_intensity_bgsub * area_px

        # SNR
        snr = mean_intensity_bgsub / bg_std if bg_std > _EPSILON else float("nan")

        # Border touch (check if punctum touches image edge)
        bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
        touches_border = (
            bbox[0] == 0 or
            bbox[1] == 0 or
            bbox[2] >= image_raw.shape[0] or
            bbox[3] >= image_raw.shape[1]
        )

        results.append({
            "punctum_id": int(prop.label),
            "cell_label": cell_label,
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "area_px": area_px,
            "area_um2": area_um2,
            "equivalent_diameter_px": float(equiv_diameter_px),
            "equivalent_diameter_um": equiv_diameter_um,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "mean_intensity": mean_intensity,
            "max_intensity": max_intensity,
            "integrated_intensity": integrated_intensity,
            "mean_intensity_bgsub": mean_intensity_bgsub,
            "integrated_intensity_bgsub": integrated_intensity_bgsub,
            "snr": snr,
            "touches_border": touches_border,
        })

    return results


def _aggregate_per_cell(
    puncta_list: List[Dict[str, Any]],
    cell_mask: np.ndarray,
    pixel_size_um: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregate puncta metrics per cell.

    Args:
        puncta_list: List of per-punctum dicts.
        cell_mask: Labeled cell mask.
        pixel_size_um: Pixel size in micrometers.

    Returns:
        List of dicts, one per cell.
    """
    # Get all cell labels (excluding background)
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]

    results: List[Dict[str, Any]] = []

    for cell_label in cell_labels:
        cell_label = int(cell_label)

        # Get puncta in this cell
        cell_puncta = [p for p in puncta_list if p["cell_label"] == cell_label]

        # Cell area
        cell_area_px = int(np.sum(cell_mask == cell_label))
        cell_area_um2 = _area_px_to_um2(cell_area_px, pixel_size_um) if pixel_size_um else float("nan")

        # Puncta count
        puncta_count = len(cell_puncta)

        # Puncta density
        puncta_density_per_px = puncta_count / cell_area_px if cell_area_px > 0 else 0.0
        puncta_density_per_um2 = puncta_count / cell_area_um2 if cell_area_um2 > 0 and np.isfinite(cell_area_um2) else float("nan")

        if puncta_count > 0:
            areas = [p["area_px"] for p in cell_puncta]
            intensities = [p["mean_intensity"] for p in cell_puncta]
            integrated = [p["integrated_intensity"] for p in cell_puncta]
            snrs = [p["snr"] for p in cell_puncta if np.isfinite(p["snr"])]

            # Statistics
            total_integrated_intensity = float(sum(integrated))
            mean_area_px = float(np.mean(areas))
            median_area_px = float(np.median(areas))
            mean_intensity = float(np.mean(intensities))
            median_intensity = float(np.median(intensities))
            mean_snr = float(np.mean(snrs)) if snrs else float("nan")

            # Nearest-neighbor distances
            if puncta_count >= 2:
                coords = np.array([[p["centroid_y"], p["centroid_x"]] for p in cell_puncta])
                dists = sp_distance.cdist(coords, coords)
                np.fill_diagonal(dists, np.inf)
                nn_dists = dists.min(axis=1)
                mean_nn_distance_px = float(np.mean(nn_dists))
                median_nn_distance_px = float(np.median(nn_dists))
                mean_nn_distance_um = _px_to_um(mean_nn_distance_px, pixel_size_um) if pixel_size_um else float("nan")
                median_nn_distance_um = _px_to_um(median_nn_distance_px, pixel_size_um) if pixel_size_um else float("nan")
            else:
                mean_nn_distance_px = float("nan")
                median_nn_distance_px = float("nan")
                mean_nn_distance_um = float("nan")
                median_nn_distance_um = float("nan")
        else:
            total_integrated_intensity = 0.0
            mean_area_px = float("nan")
            median_area_px = float("nan")
            mean_intensity = float("nan")
            median_intensity = float("nan")
            mean_snr = float("nan")
            mean_nn_distance_px = float("nan")
            median_nn_distance_px = float("nan")
            mean_nn_distance_um = float("nan")
            median_nn_distance_um = float("nan")

        results.append({
            "cell_label": cell_label,
            "cell_area_px": cell_area_px,
            "cell_area_um2": cell_area_um2,
            "puncta_count": puncta_count,
            "puncta_density_per_px": puncta_density_per_px,
            "puncta_density_per_um2": puncta_density_per_um2,
            "total_integrated_intensity": total_integrated_intensity,
            "mean_puncta_area_px": mean_area_px,
            "median_puncta_area_px": median_area_px,
            "mean_puncta_intensity": mean_intensity,
            "median_puncta_intensity": median_intensity,
            "mean_puncta_snr": mean_snr,
            "mean_nn_distance_px": mean_nn_distance_px,
            "median_nn_distance_px": median_nn_distance_px,
            "mean_nn_distance_um": mean_nn_distance_um,
            "median_nn_distance_um": median_nn_distance_um,
        })

    return results


def _compute_total_image_metrics(
    puncta_list: List[Dict[str, Any]],
    per_cell_list: List[Dict[str, Any]],
    cell_mask: np.ndarray,
    pixel_size_um: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute total-image aggregate metrics."""
    # Total area of all cells
    total_cell_area_px = int(np.sum(cell_mask > 0))
    total_cell_area_um2 = _area_px_to_um2(total_cell_area_px, pixel_size_um) if pixel_size_um else float("nan")

    # Total puncta count (only counting those inside cells)
    puncta_in_cells = [p for p in puncta_list if p["cell_label"] > 0]
    total_puncta_count = len(puncta_in_cells)

    # Total density
    total_density_per_px = total_puncta_count / total_cell_area_px if total_cell_area_px > 0 else 0.0
    total_density_per_um2 = total_puncta_count / total_cell_area_um2 if total_cell_area_um2 > 0 and np.isfinite(total_cell_area_um2) else float("nan")

    # Total integrated intensity
    total_integrated = sum(p["integrated_intensity"] for p in puncta_in_cells)

    return {
        "total_cell_area_px": total_cell_area_px,
        "total_cell_area_um2": total_cell_area_um2,
        "total_puncta_count": total_puncta_count,
        "total_puncta_density_per_px": total_density_per_px,
        "total_puncta_density_per_um2": total_density_per_um2,
        "total_integrated_intensity": float(total_integrated),
        "n_cells": len(per_cell_list),
    }


def _compute_summary(per_cell_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary (mean over cells) like colocalization."""
    if not per_cell_list:
        return {
            "cells_count": 0,
            "mean_over_cells": {},
        }

    def _mean_finite(key: str) -> float:
        vals = [c[key] for c in per_cell_list if np.isfinite(c.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "cells_count": len(per_cell_list),
        "mean_over_cells": {
            "puncta_count": _mean_finite("puncta_count"),
            "puncta_density_per_um2": _mean_finite("puncta_density_per_um2"),
            "total_integrated_intensity": _mean_finite("total_integrated_intensity"),
            "mean_puncta_area_px": _mean_finite("mean_puncta_area_px"),
            "mean_puncta_intensity": _mean_finite("mean_puncta_intensity"),
            "mean_puncta_snr": _mean_finite("mean_puncta_snr"),
            "mean_nn_distance_um": _mean_finite("mean_nn_distance_um"),
        },
    }


# =============================================================================
# Detection Parameter Helpers
# =============================================================================

def _convert_detection_params_to_pixels(
    pixel_size_um: Optional[float],
    expected_diameter_um: float,
    min_diameter_um: float,
    max_diameter_um: float,
    min_distance_um: Optional[float],
) -> Tuple[float, float, float, float]:
    """Convert detection parameters from µm to pixels."""
    if pixel_size_um is not None and pixel_size_um > 0:
        expected_diameter_px = _um_to_px(expected_diameter_um, pixel_size_um)
        min_diameter_px = _um_to_px(min_diameter_um, pixel_size_um)
        max_diameter_px = _um_to_px(max_diameter_um, pixel_size_um)
        if min_distance_um is not None:
            min_distance_px = _um_to_px(min_distance_um, pixel_size_um)
        else:
            min_distance_px = expected_diameter_px
    else:
        expected_diameter_px = FALLBACK_EXPECTED_DIAMETER_PX
        min_diameter_px = FALLBACK_MIN_DIAMETER_PX
        max_diameter_px = FALLBACK_MAX_DIAMETER_PX
        min_distance_px = expected_diameter_px if min_distance_um is None else min_distance_um
        logger.warning("pixel_size_um not provided; using pixel-based defaults for detection")
    
    return expected_diameter_px, min_diameter_px, max_diameter_px, min_distance_px


def _detect_puncta_log_method(
    image_2d: np.ndarray,
    log_sigma: float,
    min_distance_px: float,
    snr_threshold: float,
    bg_mean: float,
    bg_std: float,
    cell_region_mask: np.ndarray,
) -> np.ndarray:
    """Detect puncta using LoG + watershed method."""
    # LoG filter for blob enhancement
    log_image = _laplacian_of_gaussian(image_2d, sigma=log_sigma)
    
    # Create foreground mask using SNR threshold
    snr_image = (image_2d - bg_mean) / bg_std
    foreground_mask = (snr_image >= snr_threshold) & cell_region_mask
    
    # Detect seeds (local maxima on LoG)
    log_threshold = 0.0
    seeds = _detect_puncta_seeds(
        log_image,
        min_distance=int(max(1, min_distance_px)),
        threshold_abs=log_threshold,
        mask=foreground_mask,
    )
    logger.info(f"Detected {len(seeds)} puncta seeds")
    
    # Segment puncta via watershed
    return _segment_puncta_watershed(image_2d, seeds, foreground_mask)


def _detect_puncta_bigfish_method(
    image_2d: np.ndarray,
    expected_diameter_px: float,
    cell_mask: np.ndarray,
    return_threshold_data: bool,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Detect puncta using BigFISH method."""
    if not HAS_BIGFISH:
        raise RuntimeError(
            "BigFISH is required for detection_method='bigfish'. "
            "Install with: pip install big-fish"
        )
    
    spot_radius_px = expected_diameter_px / 2.0
    threshold_data = None
    
    if return_threshold_data:
        _spots, puncta_labels, threshold_data = _detect_spots_bigfish(
            image_2d,
            spot_radius_px=spot_radius_px,
            cell_mask=cell_mask,
            return_threshold_data=True,
        )
    else:
        _spots, puncta_labels = _detect_spots_bigfish(
            image_2d,
            spot_radius_px=spot_radius_px,
            cell_mask=cell_mask,
        )
    
    return puncta_labels, threshold_data


def _filter_puncta_by_size(
    puncta_labels: np.ndarray,
    min_area_px: int,
    max_area_px: Optional[int],
) -> np.ndarray:
    """Filter puncta by size constraints."""
    if min_area_px <= 0 and max_area_px is None:
        return puncta_labels
    
    props = measure.regionprops(puncta_labels)
    keep_labels = []
    
    for prop in props:
        if prop.area < min_area_px:
            continue
        if max_area_px is not None and prop.area > max_area_px:
            continue
        keep_labels.append(prop.label)
    
    # Relabel to keep only valid puncta
    filtered_labels = np.zeros_like(puncta_labels)
    for new_id, old_id in enumerate(keep_labels, start=1):
        filtered_labels[puncta_labels == old_id] = new_id
    
    logger.info(f"After size filtering: {len(keep_labels)} puncta remain")
    return filtered_labels


# =============================================================================
# Main API
# =============================================================================

def compute_puncta(
    image: Union[str, Path, np.ndarray, Dict[str, Any]],
    mask: Union[str, Path, np.ndarray],
    channel: Union[int, str],
    *,
    channel_names: Optional[List[str]] = None,
    projection: str = "mip",  # "mip" | "sme" | "none"
    detection_method: str = "log",  # "log" | "bigfish"
    pixel_size_um: Optional[float] = None,
    expected_diameter_um: float = _DEFAULT_EXPECTED_DIAMETER_UM,
    min_diameter_um: float = _DEFAULT_MIN_DIAMETER_UM,
    max_diameter_um: float = _DEFAULT_MAX_DIAMETER_UM,
    snr_threshold: float = _DEFAULT_SNR_THRESHOLD,
    min_distance_um: Optional[float] = None,
    min_area_px: int = 4,
    max_area_px: Optional[int] = None,
    drop_label_1: bool = True,
    return_threshold_data: bool = False,
    output_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Detect and analyze puncta in a single channel.

    This function mirrors the API pattern of compute_colocalization: it accepts
    background-corrected image data (path, array, or results dict), a labeled
    cell mask, and returns a JSON-serializable dict with per-punctum, per-cell,
    and total-image metrics.

    Args:
        image: Input image. Can be:
            - Path to OME-TIFF or other supported format
            - NumPy array (Z,Y,X,C), (Y,X,C), (Z,Y,X), or (Y,X)
            - Dict mapping channel_name -> (array, meta) or -> array
              (same format as background subtraction results)
        mask: 2D labeled cell mask (path or array). Labels > 0 are cells.
        channel: Channel index or name to analyze.
        channel_names: Optional list of channel names (required if array + name).
        projection: How to project 3D to 2D:
            - "mip": Maximum intensity projection (default)
            - "sme": Surface manifold extraction
            - "none": Expect already 2D or use middle slice
        detection_method: Spot detection algorithm to use:
            - "log": Laplacian of Gaussian + watershed (default, requires scikit-image)
            - "bigfish": BigFISH automatic thresholding (requires big-fish package)
        pixel_size_um: Pixel size in micrometers. Required for µm-based
            filtering and metrics.
        expected_diameter_um: Expected punctum diameter in µm (for LoG sigma).
        min_diameter_um: Minimum punctum diameter in µm (for filtering).
        max_diameter_um: Maximum punctum diameter in µm (for filtering).
        snr_threshold: SNR threshold for foreground detection (default 3.0).
        min_distance_um: Minimum distance between puncta in µm. If None,
            defaults to expected_diameter_um.
        min_area_px: Minimum punctum area in pixels (default 4).
        max_area_px: Maximum punctum area in pixels. If None, computed from
            max_diameter_um.
        drop_label_1: If True, remove cell label 1 from mask (Cellpose background).
        return_threshold_data: If True and detection_method="bigfish", include
            elbow curve data in the result for visualization. Ignored for "log".
        output_json: Optional path to save results as JSON.

    Returns:
        Dict with structure:
        {
            "image_shape": tuple,
            "channel": str,
            "projection": str,
            "pixel_size_um": float or None,
            "detection_params": {...},
            "threshold_data": {...},  # Only if return_threshold_data=True and bigfish
            "results": {
                "puncta": [...],  # per-punctum metrics
                "per_label": [...],  # per-cell aggregates
                "total_image": {...},
                "summary": {...},
            }
        }

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If required dependencies are missing.
    """
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required for puncta detection. Install with: pip install scikit-image")

    # Phase 1: Load and prepare data
    channel_data, ch_name, all_names = _load_single_channel(image, channel, channel_names)
    original_shape = channel_data.shape
    logger.info(f"Loaded channel '{ch_name}' with shape {original_shape}")

    image_2d = _project_to_2d(channel_data, projection=projection)
    image_2d = image_2d.astype(np.float64)
    logger.info(f"Projected to 2D: {image_2d.shape} (projection='{projection}')")

    cell_mask = _load_mask(mask)
    if cell_mask.shape != image_2d.shape:
        raise ValueError(f"Mask shape {cell_mask.shape} must match image shape {image_2d.shape}")

    if drop_label_1 and np.any(cell_mask == 1):
        cell_mask[cell_mask == 1] = 0
        logger.info("Removed cell label 1 from mask (assumed Cellpose background)")

    # Phase 2: Convert parameters to pixels
    expected_diameter_px, min_diameter_px, max_diameter_px, min_distance_px = _convert_detection_params_to_pixels(
        pixel_size_um, expected_diameter_um, min_diameter_um, max_diameter_um, min_distance_um
    )

    log_sigma = max(0.5, expected_diameter_px / LOG_SIGMA_FACTOR)
    
    if max_area_px is None:
        max_area_px = int(math.pi * (max_diameter_px / 2) ** 2 * MAX_AREA_SAFETY_MARGIN)

    logger.info(
        f"Detection params: method={detection_method}, log_sigma={log_sigma:.2f}px, "
        f"min_distance={min_distance_px:.2f}px, snr_threshold={snr_threshold}, "
        f"min_area={min_area_px}px, max_area={max_area_px}px"
    )

    # Phase 3: Estimate background
    cell_region_mask = cell_mask > 0
    bg_mean, bg_std = _estimate_background_mad(image_2d, mask=cell_region_mask)
    logger.info(f"Background estimate: mean={bg_mean:.2f}, std={bg_std:.2f}")

    # Phase 4: Detect puncta
    threshold_data = None
    if detection_method.lower() == "bigfish":
        puncta_labels, threshold_data = _detect_puncta_bigfish_method(
            image_2d, expected_diameter_px, cell_mask, return_threshold_data
        )
    elif detection_method.lower() == "log":
        puncta_labels = _detect_puncta_log_method(
            image_2d, log_sigma, min_distance_px, snr_threshold, 
            bg_mean, bg_std, cell_region_mask
        )
    else:
        raise ValueError(
            f"Unknown detection_method '{detection_method}'. "
            "Supported: 'log', 'bigfish'"
        )

    # Phase 5: Filter by size
    puncta_labels = _filter_puncta_by_size(puncta_labels, min_area_px, max_area_px)

    # Step 7: Measure puncta
    puncta_list = _measure_puncta(
        image_2d, puncta_labels, cell_mask,
        pixel_size_um=pixel_size_um,
        bg_mean=bg_mean,
        bg_std=bg_std,
    )
    logger.info(f"Measured {len(puncta_list)} puncta")

    # Step 8: Aggregate per cell
    per_cell_list = _aggregate_per_cell(puncta_list, cell_mask, pixel_size_um=pixel_size_um)

    # Step 9: Compute total image metrics
    total_image = _compute_total_image_metrics(puncta_list, per_cell_list, cell_mask, pixel_size_um)

    # Step 10: Compute summary
    summary = _compute_summary(per_cell_list)

    # Build output
    result: Dict[str, Any] = {
        "image_shape": original_shape,
        "channel": ch_name,
        "projection": projection,
        "pixel_size_um": pixel_size_um,
        "detection_params": {
            "detection_method": detection_method,
            "expected_diameter_um": expected_diameter_um,
            "min_diameter_um": min_diameter_um,
            "max_diameter_um": max_diameter_um,
            "log_sigma_px": float(log_sigma),
            "min_distance_px": float(min_distance_px),
            "snr_threshold": snr_threshold,
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "background_mean": float(bg_mean),
            "background_std": float(bg_std),
        },
        "results": {
            "puncta": puncta_list,
            "per_label": per_cell_list,
            "total_image": total_image,
            "summary": summary,
        },
    }

    # Add threshold data if available (BigFISH with return_threshold_data=True)
    if threshold_data is not None:
        result["threshold_data"] = threshold_data

    # Optionally save to JSON
    if output_json is not None:
        export_puncta_json(result, output_json)

    logger.info(
        f"compute_puncta finished: {total_image['total_puncta_count']} puncta in "
        f"{total_image['n_cells']} cells"
    )

    return result


def export_puncta_json(result: Dict[str, Any], out_path: Union[str, Path]) -> None:
    """
    Export puncta analysis results to JSON file.

    Args:
        result: Output from compute_puncta().
        out_path: Path to save JSON file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all values are JSON-serializable
    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        elif isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    serializable = _make_serializable(result)

    with out_path.open("w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Wrote puncta JSON to {out_path}")


def plot_puncta_elbow(
    result: Dict[str, Any],
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
) -> Any:
    """
    Plot the elbow curve from BigFISH threshold detection.

    This shows the relationship between threshold value and number of detected
    spots, with the automatically selected threshold marked.

    Args:
        result: Output from compute_puncta() with return_threshold_data=True.
        ax: Optional matplotlib axes. If None, creates a new figure.
        figsize: Figure size if creating new figure.
        title: Optional title for the plot.

    Returns:
        matplotlib axes object.

    Raises:
        ValueError: If result doesn't contain threshold_data.

    Example:
        >>> result = compute_puncta(
        ...     image, mask, "ALIX",
        ...     detection_method="bigfish",
        ...     return_threshold_data=True,
        ... )
        >>> plot_puncta_elbow(result)
        >>> plt.show()
    """
    if "threshold_data" not in result:
        raise ValueError(
            "Result does not contain threshold_data. "
            "Use compute_puncta(..., detection_method='bigfish', return_threshold_data=True)"
        )

    threshold_data = result["threshold_data"]
    thresholds = threshold_data["thresholds"]
    spot_counts = threshold_data["spot_counts"]
    selected_threshold = threshold_data["threshold"]

    # Import matplotlib lazily
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the elbow curve
    ax.plot(thresholds, spot_counts, "b-", linewidth=2, label="Spot count vs threshold")

    # Mark the selected threshold
    # Find the spot count at the selected threshold
    selected_idx = np.argmin(np.abs(np.array(thresholds) - selected_threshold))
    selected_count = spot_counts[selected_idx]

    ax.axvline(
        x=selected_threshold,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"Auto threshold: {selected_threshold:.1f}",
    )
    ax.scatter(
        [selected_threshold],
        [selected_count],
        color="r",
        s=100,
        zorder=5,
        marker="o",
    )

    # Labels and formatting
    ax.set_xlabel("Threshold (LoG-filtered intensity)", fontsize=11)
    ax.set_ylabel("Number of spots detected", fontsize=11)

    if title is None:
        channel = result.get("channel", "Unknown")
        title = f"BigFISH Elbow Curve - {channel}"
    ax.set_title(title, fontsize=12)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add annotation for final count
    final_count = threshold_data.get("filtered_spots_count", selected_count)
    ax.annotate(
        f"Detected: {final_count} spots",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return ax


def plot_puncta_detection(
    result: Dict[str, Any],
    image_2d: np.ndarray,
    cell_mask: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (15, 5),
    radius_px: int = 5,
    title: Optional[str] = None,
) -> Any:
    """
    Plot detected puncta overlaid on image using BigFISH visualization.
    
    Creates a multi-panel figure showing:
    - Panel A: MIP image with detected spots (circles) using BigFISH plot_detection
    - Panel B: Cell mask with spots colored by cell assignment
    - Panel C: Elbow curve (if threshold_data available)
    
    Args:
        result: Output from compute_puncta() with puncta coordinates.
        image_2d: 2D image used for detection (typically MIP).
        cell_mask: Optional 2D labeled cell mask.
        figsize: Figure size (width, height) in inches.
        radius_px: Radius for spot markers in pixels.
        title: Optional title for the figure.
        
    Returns:
        matplotlib figure object.
        
    Raises:
        ValueError: If result doesn't contain puncta data.
        RuntimeError: If required dependencies are missing.
        
    Example:
        >>> # After puncta detection
        >>> alix_mip = np.max(bg_results["ALIX"][0], axis=0)
        >>> fig = plot_puncta_detection(puncta_result, alix_mip, cell_mask=mask)
        >>> fig.savefig("detection_overlay.png", dpi=150)
        >>> plt.close(fig)
    """
    if "results" not in result or "puncta" not in result["results"]:
        raise ValueError(
            "Result does not contain puncta data. "
            "Ensure compute_puncta() completed successfully."
        )
    
    # Import dependencies
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    try:
        from bigfish import plot as bigfish_plot
    except ImportError:
        raise RuntimeError("bigfish is required for plotting. Install with: pip install big-fish")
    
    # Extract spot coordinates from puncta results
    puncta_list = result["results"]["puncta"]
    spots = np.array([
        [p["centroid_y"], p["centroid_x"]]
        for p in puncta_list
    ])
    
    # Determine number of panels based on available data
    has_threshold_data = "threshold_data" in result
    n_panels = 3 if has_threshold_data else 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    
    # Panel A: Detection overlay (manual plotting for better control)
    ax_detection = axes[0]
    # Display image with contrast adjustment (similar to BigFISH)
    vmin, vmax = np.percentile(image_2d, [1, 99])
    ax_detection.imshow(image_2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    
    # Overlay detected spots as circles
    if len(spots) > 0:
        for spot in spots:
            y, x = spot
            circle = plt.Circle((x, y), radius_px, color="red", fill=False, 
                              linewidth=1.5, alpha=0.8)
            ax_detection.add_patch(circle)
    ax_detection.set_title(f"Detected Spots (n={len(spots)})", fontsize=12)
    ax_detection.axis("off")
    
    # Panel B: Cell mask with spots
    ax_mask = axes[1]
    if cell_mask is not None:
        # Create a colorful display of cells
        ax_mask.imshow(cell_mask, cmap="nipy_spectral", interpolation="nearest")
        
        # Overlay spots, colored by cell assignment
        if len(spots) > 0:
            spot_cell_labels = []
            for p in puncta_list:
                spot_cell_labels.append(p.get("cell_label", 0))
            
            # Plot spots with marker
            for i, (spot, cell_label) in enumerate(zip(spots, spot_cell_labels)):
                y, x = spot
                color = "white" if cell_label == 0 else "red"
                marker_size = 40 if cell_label > 0 else 20
                ax_mask.scatter(x, y, s=marker_size, c=color, marker="o", 
                              edgecolors="black", linewidths=0.5, alpha=0.7)
        
        ax_mask.set_title(f"Cell Segmentation with Spots", fontsize=12)
    else:
        # No mask, just show the image with spots as scatter
        ax_mask.imshow(image_2d, cmap="gray")
        if len(spots) > 0:
            ax_mask.scatter(spots[:, 1], spots[:, 0], s=40, c="red", 
                          marker="o", edgecolors="white", linewidths=0.5, alpha=0.7)
        ax_mask.set_title(f"Spots Overlay", fontsize=12)
    ax_mask.axis("off")
    
    # Panel C: Elbow curve (if available)
    if has_threshold_data:
        ax_elbow = axes[2]
        try:
            plot_puncta_elbow(result, ax=ax_elbow, title=None)
        except Exception as e:
            logger.warning(f"Could not plot elbow curve: {e}")
            ax_elbow.text(0.5, 0.5, "Elbow curve\nnot available", 
                         ha="center", va="center", transform=ax_elbow.transAxes)
            ax_elbow.axis("off")
    
    # Set main title
    if title is None:
        channel = result.get("channel", "Unknown")
        method = result.get("detection_params", {}).get("detection_method", "unknown")
        title = f"Puncta Detection - {channel} ({method})"
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


__all__ = [
    "compute_puncta",
    "export_puncta_json",
    "plot_puncta_elbow",
    "plot_puncta_detection",
]


