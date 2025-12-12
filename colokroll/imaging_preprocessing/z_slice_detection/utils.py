"""Utility functions for Z-slice detection and processing."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


NormalizationMode = str  # "per_slice", "per_stack", "percentile"


def _ensure_zyxc(img: np.ndarray, axes: Optional[str]) -> np.ndarray:
    """Ensure image array is in ZYXC format."""
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
    """Prepare a single slice for focus scoring.
    
    Applies border masking, Gaussian blur, clipping, and normalization.
    """
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


def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing to scores."""
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
    """Compute z-scores along the first axis."""
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

