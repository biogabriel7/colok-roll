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


# =============================================================================
# Module-level constants
# =============================================================================
_EPSILON = 1e-12
_DEFAULT_MIN_DISTANCE_PX = 3  # Minimum distance between puncta seeds (pixels)
_DEFAULT_SNR_THRESHOLD = 3.0  # Default SNR threshold for foreground
_DEFAULT_EXPECTED_DIAMETER_UM = 0.5  # Expected puncta diameter in µm
_DEFAULT_MIN_DIAMETER_UM = 0.2  # Minimum puncta diameter in µm
_DEFAULT_MAX_DIAMETER_UM = 2.0  # Maximum puncta diameter in µm

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
            # Heuristic: if last dim is small (<= 6), treat as channels
            if arr.shape[-1] <= 6:
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
    """Load and validate a 2D labeled mask (replicates colocalization helper)."""
    if isinstance(mask, (str, Path)):
        if ImageLoader is None:
            raise RuntimeError("ImageLoader not available; pass a numpy array instead.")
        m = ImageLoader().load_tif_mask(str(mask))
    else:
        m = np.asarray(mask)

    # Reduce 3D to best 2D slice
    if m.ndim == 3:
        labeled_counts = [(z_idx, int((m[z_idx] > 0).sum())) for z_idx in range(m.shape[0])]
        z_best = max(labeled_counts, key=lambda t: t[1])[0]
        logger.info(f"3D mask detected; reducing to 2D by selecting z={z_best}")
        m = m[z_best]

    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after reduction; got {m.shape}")

    # Coerce dtype
    if np.issubdtype(m.dtype, np.integer):
        return m.astype(np.int32)
    elif 0.0 <= float(np.nanmin(m)) <= float(np.nanmax(m)) <= 1.0:
        return (m > 0.5).astype(np.int32)
    else:
        return np.rint(m).astype(np.int32)


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

    if bg_vals.size < 10:
        bg_vals = vals

    bg_mean = float(np.median(bg_vals))
    # MAD = median absolute deviation
    mad = float(np.median(np.abs(bg_vals - bg_mean)))
    # Convert MAD to approximate std (for Gaussian: std ≈ 1.4826 * MAD)
    mad_std = max(mad * 1.4826, _EPSILON)

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

        # Shape metrics (guard against degenerate cases)
        try:
            eccentricity = float(prop.eccentricity)
        except Exception:
            eccentricity = float("nan")

        try:
            solidity = float(prop.solidity)
        except Exception:
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
# Main API
# =============================================================================

def compute_puncta(
    image: Union[str, Path, np.ndarray, Dict[str, Any]],
    mask: Union[str, Path, np.ndarray],
    channel: Union[int, str],
    *,
    channel_names: Optional[List[str]] = None,
    projection: str = "mip",  # "mip" | "sme" | "none"
    pixel_size_um: Optional[float] = None,
    expected_diameter_um: float = _DEFAULT_EXPECTED_DIAMETER_UM,
    min_diameter_um: float = _DEFAULT_MIN_DIAMETER_UM,
    max_diameter_um: float = _DEFAULT_MAX_DIAMETER_UM,
    snr_threshold: float = _DEFAULT_SNR_THRESHOLD,
    min_distance_um: Optional[float] = None,
    min_area_px: int = 4,
    max_area_px: Optional[int] = None,
    drop_label_1: bool = True,
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
        output_json: Optional path to save results as JSON.

    Returns:
        Dict with structure:
        {
            "image_shape": tuple,
            "channel": str,
            "projection": str,
            "pixel_size_um": float or None,
            "detection_params": {...},
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

    # Load channel data
    channel_data, ch_name, all_names = _load_single_channel(image, channel, channel_names)
    original_shape = channel_data.shape
    logger.info(f"Loaded channel '{ch_name}' with shape {original_shape}")

    # Project to 2D if needed
    image_2d = _project_to_2d(channel_data, projection=projection)
    image_2d = image_2d.astype(np.float64)
    logger.info(f"Projected to 2D: {image_2d.shape} (projection='{projection}')")

    # Load cell mask
    cell_mask = _load_mask(mask)
    if cell_mask.shape != image_2d.shape:
        raise ValueError(f"Mask shape {cell_mask.shape} must match image shape {image_2d.shape}")

    # Optionally remove label 1 (Cellpose background)
    if drop_label_1 and np.any(cell_mask == 1):
        cell_mask[cell_mask == 1] = 0
        logger.info("Removed cell label 1 from mask (assumed Cellpose background)")

    # Convert µm parameters to pixels
    if pixel_size_um is not None and pixel_size_um > 0:
        expected_diameter_px = _um_to_px(expected_diameter_um, pixel_size_um)
        min_diameter_px = _um_to_px(min_diameter_um, pixel_size_um)
        max_diameter_px = _um_to_px(max_diameter_um, pixel_size_um)
        if min_distance_um is not None:
            min_distance_px = _um_to_px(min_distance_um, pixel_size_um)
        else:
            min_distance_px = expected_diameter_px
    else:
        # Use pixel-based defaults
        expected_diameter_px = 5.0
        min_diameter_px = 2.0
        max_diameter_px = 20.0
        min_distance_px = expected_diameter_px if min_distance_um is None else min_distance_um
        logger.warning("pixel_size_um not provided; using pixel-based defaults for detection")

    # Compute sigma for LoG (sigma ≈ diameter / (2 * sqrt(2)))
    log_sigma = expected_diameter_px / (2.0 * math.sqrt(2))
    log_sigma = max(0.5, log_sigma)

    # Compute max_area_px if not provided
    if max_area_px is None:
        max_area_px = int(math.pi * (max_diameter_px / 2) ** 2 * 2)  # 2x for safety margin

    logger.info(
        f"Detection params: log_sigma={log_sigma:.2f}px, min_distance={min_distance_px:.2f}px, "
        f"snr_threshold={snr_threshold}, min_area={min_area_px}px, max_area={max_area_px}px"
    )

    # Step 1: Estimate background (within cell regions)
    cell_region_mask = cell_mask > 0
    bg_mean, bg_std = _estimate_background_mad(image_2d, mask=cell_region_mask)
    logger.info(f"Background estimate: mean={bg_mean:.2f}, std={bg_std:.2f}")

    # Step 2: LoG filter for blob enhancement
    log_image = _laplacian_of_gaussian(image_2d, sigma=log_sigma)

    # Step 3: Create foreground mask using SNR threshold
    snr_image = (image_2d - bg_mean) / bg_std
    foreground_mask = (snr_image >= snr_threshold) & cell_region_mask

    # Step 4: Detect seeds (local maxima on LoG)
    # Threshold LoG at 0 (enhanced blobs should be positive)
    log_threshold = 0.0
    seeds = _detect_puncta_seeds(
        log_image,
        min_distance=int(max(1, min_distance_px)),
        threshold_abs=log_threshold,
        mask=foreground_mask,
    )
    logger.info(f"Detected {len(seeds)} puncta seeds")

    # Step 5: Segment puncta via watershed
    puncta_labels = _segment_puncta_watershed(image_2d, seeds, foreground_mask)

    # Step 6: Filter by size
    if min_area_px > 0 or max_area_px is not None:
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
        puncta_labels = filtered_labels
        logger.info(f"After size filtering: {len(keep_labels)} puncta remain")

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


__all__ = [
    "compute_puncta",
    "export_puncta_json",
]

