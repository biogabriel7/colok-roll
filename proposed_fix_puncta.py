"""
PROPOSED FIX: Improved threshold fallback for BigFISH puncta detection

Problem:
--------
The current fallback mechanism in _detect_spots_bigfish() only triggers when
threshold <= 0.01. However, the auto-threshold can also FAIL by selecting an
unreasonably HIGH threshold (e.g., 5.9 instead of 0.2), causing it to miss
most puncta.

Example: anti_ALIX_15_min_3.ome
- Auto-threshold: 5.88 → detected 2 puncta
- Expected threshold: ~0.2-0.5 → should detect ~700 puncta
- Background stats are normal (bg_mean=8.3, bg_std=2.9)

Root Cause:
-----------
The BigFISH elbow detection algorithm can fail when the LoG-filtered intensity
distribution has an unusual shape. The elbow curve becomes nearly vertical,
and the algorithm picks a threshold that's too high.

Solution:
---------
Add a SECOND fallback that detects when auto-threshold is unreasonably high:

1. Check if threshold > 99th percentile of LoG-filtered intensities
2. If so, use a robust percentile-based fallback (e.g., 95th percentile)
3. Log a warning about the fallback
4. Validate the new threshold produces reasonable spot counts

This catches BOTH failure modes:
- threshold too low (existing fallback)
- threshold too high (NEW fallback)
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


def _detect_spots_bigfish_improved(
    image: np.ndarray,
    spot_radius_px: float,
    cell_mask: Optional[np.ndarray] = None,
    return_threshold_data: bool = False,
    min_snr_threshold: float = 5.0,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Detect spots using BigFISH with IMPROVED automatic thresholding.
    
    Improvements:
    1. Detects when auto-threshold is unreasonably HIGH (new)
    2. Detects when auto-threshold is unreasonably LOW (existing)
    3. Uses percentile-based fallback for both failure modes
    4. Validates fallback produces reasonable results
    
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
    # NOTE: This requires bigfish to be installed
    try:
        from bigfish import detection as bigfish_detection
        from bigfish import stack as bigfish_stack
    except ImportError:
        raise RuntimeError(
            "BigFISH is required for this detection method. "
            "Install with: pip install big-fish"
        )
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Compute LoG kernel size and minimum distance from spot radius
    log_kernel_size = max(3, int(round(spot_radius_px * 2)))
    if log_kernel_size % 2 == 0:
        log_kernel_size += 1
    min_distance = max(1, int(round(spot_radius_px)))
    
    log_kernel_2d = (log_kernel_size, log_kernel_size)
    min_distance_2d = (min_distance, min_distance)
    
    # Use the high-level detect_spots API which handles thresholding
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
    
    # NEW: Compute LoG-filtered image statistics for validation
    sigma = (spot_radius_px, spot_radius_px)
    image_filtered = bigfish_stack.log_filter(image.astype(np.float64), sigma=sigma)
    
    # Compute percentiles of LoG-filtered intensities (only positive values)
    positive_intensities = image_filtered[image_filtered > 0]
    if len(positive_intensities) > 0:
        p50 = float(np.percentile(positive_intensities, 50))  # median
        p90 = float(np.percentile(positive_intensities, 90))
        p95 = float(np.percentile(positive_intensities, 95))
        p99 = float(np.percentile(positive_intensities, 99))
        p999 = float(np.percentile(positive_intensities, 99.9))
    else:
        p50 = p90 = p95 = p99 = p999 = 0.0
    
    # Track if we used fallback
    used_fallback = False
    fallback_reason = None
    original_spot_count = len(spots)
    
    # =================================================================
    # FALLBACK 1: Threshold too LOW (existing logic)
    # =================================================================
    if threshold_scalar <= 0.01 and len(spots) > 0:
        # Compute SNR-based fallback threshold
        median_val = float(np.median(image_filtered))
        mad = float(np.median(np.abs(image_filtered - median_val)))
        robust_std = 1.4826 * mad
        fallback_threshold = median_val + min_snr_threshold * robust_std
        
        logger.warning(
            f"BigFISH auto-threshold TOO LOW (threshold={threshold_scalar:.4f}, "
            f"detected {original_spot_count} spots). "
            f"Applying fallback SNR-based threshold: {fallback_threshold:.2f}"
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
        
        # If fallback still detects many spots, likely noise
        if len(spots) > 0 and original_spot_count > 0:
            ratio = len(spots) / original_spot_count
            if ratio > 0.3:
                logger.warning(
                    f"Fallback still detected {len(spots)} spots ({ratio:.0%} of original). "
                    f"Image likely contains only noise - returning 0 spots."
                )
                spots = np.empty((0, 2), dtype=np.int64)
    
    # =================================================================
    # FALLBACK 2: Threshold too HIGH (NEW logic)
    # =================================================================
    # Check if threshold is unreasonably high (above 99th percentile)
    # This indicates the elbow detection failed
    elif threshold_scalar > p99 and p99 > 0:
        logger.warning(
            f"BigFISH auto-threshold TOO HIGH (threshold={threshold_scalar:.4f} > "
            f"p99={p99:.4f}, detected {original_spot_count} spots). "
            f"Elbow detection likely failed."
        )
        
        # Use 95th percentile as fallback (more conservative than 90th)
        fallback_threshold = p95
        
        logger.warning(
            f"Applying percentile-based fallback threshold: {fallback_threshold:.4f} "
            f"(95th percentile of LoG-filtered intensities)"
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
        # (at least 10x more, or at least 50 spots if original was very low)
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
                f"Fallback validation failed: only detected {len(spots_fallback)} spots. "
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
        # Generate threshold range for elbow curve
        max_val = float(image_filtered.max())
        min_val = float(image_filtered[image_filtered > 0].min()) if np.any(image_filtered > 0) else 0
        thresholds = np.linspace(min_val, max_val, num=100)
        
        # Get local maxima mask and convert to coordinates
        local_maxima_mask = bigfish_detection.local_maximum_detection(
            image_filtered,
            min_distance=min_distance_2d,
        )
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
            # Add diagnostic info
            "log_intensity_percentiles": {
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "p99": p99,
                "p99.9": p999,
            },
        }
    
    # Filter spots to only those within cell mask if provided
    # ... (rest of the function remains the same as original)
    
    # For now, create dummy puncta labels
    puncta_labels = np.zeros_like(image, dtype=np.int32)
    
    if return_threshold_data:
        return spots, puncta_labels, threshold_data
    else:
        return spots, puncta_labels


# =============================================================================
# Summary of changes to apply to colokroll/analysis/puncta.py
# =============================================================================
"""
CHANGES TO APPLY:

1. In _detect_spots_bigfish() function (starting around line 418):
   
   a) After computing threshold_scalar, add LoG-filtered percentile computation:
      
      ```python
      # NEW: Compute LoG-filtered image statistics for validation
      sigma = (spot_radius_px, spot_radius_px)
      image_filtered = bigfish_stack.log_filter(image.astype(np.float64), sigma=sigma)
      
      positive_intensities = image_filtered[image_filtered > 0]
      if len(positive_intensities) > 0:
          p99 = float(np.percentile(positive_intensities, 99))
          p95 = float(np.percentile(positive_intensities, 95))
      else:
          p99 = p95 = 0.0
      ```
   
   b) After the existing "FALLBACK 1" (threshold too low) block, add:
      
      ```python
      # FALLBACK 2: Threshold too HIGH (NEW)
      elif threshold_scalar > p99 and p99 > 0:
          fallback_threshold = p95
          
          logger.warning(
              f"BigFISH auto-threshold TOO HIGH (threshold={threshold_scalar:.4f} > "
              f"p99={p99:.4f}). Applying percentile-based fallback: {fallback_threshold:.4f}"
          )
          
          spots_fallback, _ = bigfish_detection.detect_spots(
              image.astype(np.float64),
              threshold=fallback_threshold,
              return_threshold=True,
              log_kernel_size=log_kernel_2d,
              minimum_distance=min_distance_2d,
          )
          
          # Validate: fallback should detect significantly more spots
          if len(spots_fallback) >= max(10, original_spot_count * 10) or \
             (len(spots_fallback) >= 50 and original_spot_count < 10):
              spots = spots_fallback
              threshold_scalar = fallback_threshold
              used_fallback = True
              logger.info(f"Fallback successful: detected {len(spots_fallback)} spots")
      ```

2. Update threshold_data dict to include diagnostic info:
   
   ```python
   threshold_data = {
       ...existing fields...
       "used_fallback": used_fallback,
       "log_intensity_percentiles": {"p95": p95, "p99": p99},
   }
   ```

This fix will catch the failure mode seen in anti_ALIX_15_min_3.ome where the
auto-threshold was 5.88 instead of the expected ~0.2-0.5.
"""
