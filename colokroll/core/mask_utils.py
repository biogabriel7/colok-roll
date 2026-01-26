"""
Shared mask loading and processing utilities.

This module provides common functions for loading and processing labeled masks
used across different analysis modules (colocalization, puncta, etc.).
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def reduce_3d_mask_to_2d(mask: np.ndarray) -> np.ndarray:
    """Select Z-slice with largest labeled area from a 3D mask.
    
    Args:
        mask: 3D labeled mask array (Z, Y, X).
        
    Returns:
        2D mask from the Z-slice with the most labeled pixels.
    """
    labeled_counts = [(z_idx, int((mask[z_idx] > 0).sum())) for z_idx in range(mask.shape[0])]
    z_best = max(labeled_counts, key=lambda t: t[1])[0]
    logger.info(
        f"3D mask detected; reducing to 2D by selecting z={z_best} "
        f"(largest labeled area) and broadcasting across Z."
    )
    return mask[z_best]


def coerce_mask_dtype(mask: np.ndarray) -> np.ndarray:
    """Convert mask to int32, handling various input dtypes.
    
    Args:
        mask: Input mask array.
        
    Returns:
        Mask as int32 array with proper label values.
    """
    # Integer dtype: direct conversion
    if np.issubdtype(mask.dtype, np.integer):
        logger.info("Loaded labeled mask with integer dtype: %s", str(mask.dtype))
        return mask.astype(np.int32)

    # Float or other non-integer dtype: decide between binary and labeled
    m_min = float(np.nanmin(mask)) if mask.size > 0 else 0.0
    m_max = float(np.nanmax(mask)) if mask.size > 0 else 0.0

    if 0.0 <= m_min <= m_max <= 1.0:
        # Likely binary/probability mask; threshold at 0.5
        logger.info("Loaded non-integer mask in [0,1]; converting to binary with threshold 0.5")
        return (mask > 0.5).astype(np.int32)

    # Otherwise, assume labeled mask stored as float; round to nearest int
    logger.info("Loaded non-integer mask with range [%s, %s]; rounding to int labels", m_min, m_max)
    return np.rint(mask).astype(np.int32)


def load_and_validate_mask(
    mask: Union[str, Path, np.ndarray],
    image_loader_func=None,
) -> np.ndarray:
    """Load and validate a 2D labeled mask from various input formats.
    
    Args:
        mask: Path to mask file or numpy array. 3D masks are reduced to 2D.
        image_loader_func: Optional function to load mask from file path.
            Should take a string path and return a numpy array.
            
    Returns:
        2D int32 labeled mask array.
        
    Raises:
        ValueError: If mask cannot be reduced to 2D.
        RuntimeError: If image loader is required but not provided.
    """
    # Load from file if needed
    if isinstance(mask, (str, Path)):
        if image_loader_func is None:
            raise RuntimeError(
                "Image loader function required for file paths. "
                "Pass a numpy array instead or provide image_loader_func."
            )
        m = image_loader_func(str(mask))
    else:
        m = np.asarray(mask)

    # Reduce 3D to best 2D slice
    if m.ndim == 3:
        m = reduce_3d_mask_to_2d(m)

    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after reduction; got {m.shape}")

    return coerce_mask_dtype(m)


def broadcast_mask_to_z(mask_2d: np.ndarray, z_depth: int) -> np.ndarray:
    """Broadcast a 2D mask to 3D for Z-stack operations.
    
    Args:
        mask_2d: 2D mask array (Y, X).
        z_depth: Number of Z-slices.
        
    Returns:
        3D mask array (Z, Y, X) with mask_2d repeated along Z.
    """
    return np.broadcast_to(mask_2d[np.newaxis, ...], (z_depth, mask_2d.shape[0], mask_2d.shape[1]))
