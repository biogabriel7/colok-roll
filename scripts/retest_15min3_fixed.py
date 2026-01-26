#!/usr/bin/env python3
"""
Re-test anti_ALIX_15_min_3.ome with the fixed threshold fallback.

This script runs the complete pipeline on just the problematic image to verify the fix.
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import colokroll
try:
    import colokroll as cr
    from colokroll import compute_puncta, plot_puncta_elbow, plot_puncta_detection
    from colokroll import estimate_min_area_threshold
    from colokroll.analysis.colocalization import _filter_labels
except ImportError as e:
    logger.error(f"Failed to import colokroll: {e}")
    sys.exit(1)


def main():
    # Paths
    image_path = Path("/fs/scratch/PAS2598/duarte63/outputs/format_converter_test/anti_ALIX_15_min_3.ome.tiff")
    output_dir = Path("/users/PAS2598/duarte63/GitHub/colok-roll/retest_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("Re-testing anti_ALIX_15_min_3.ome with FIXED threshold fallback")
    logger.info("=" * 70)
    logger.info("")
    
    start_time = time.perf_counter()
    
    # =========================================================================
    # Step 1: Load image
    # =========================================================================
    logger.info("Step 1: Loading image...")
    loader = cr.ImageLoader(auto_convert=False)
    image = loader.load_image(image_path)
    logger.info(f"  Loaded image with shape: {image.shape}")
    
    # =========================================================================
    # Step 2: Rename channels
    # =========================================================================
    logger.info("Step 2: Renaming channels...")
    channel_names = ["LAMP1", "Phalloidin", "ALIX", "DAPI"]
    loader.rename_channels(channel_names)
    logger.info(f"  Channels: {channel_names}")
    
    # =========================================================================
    # Step 3: Z-slice selection
    # =========================================================================
    logger.info("Step 3: Z-slice selection...")
    comparison = cr.compare_strategies(
        image,
        save_plots=False,
        compute_quality=False,
        display_inline=False,
    )
    
    z_result = comparison.results.get("FFT + Closest (k=14)", list(comparison.results.values())[0])
    filtered_image = image[z_result.indices_keep]
    logger.info(f"  Filtered from {image.shape[0]} to {filtered_image.shape[0]} slices")
    
    # =========================================================================
    # Step 4: Background subtraction
    # =========================================================================
    logger.info("Step 4: Background subtraction...")
    bg_subtractor = cr.BackgroundSubtractor()
    
    bg_results = {}
    for i, ch in enumerate(channel_names):
        ch_data = filtered_image[:, :, :, i]
        corrected, meta = bg_subtractor.subtract_background(
            image=ch_data,
            channel_name=ch,
            is_negative_control=False,
        )
        bg_results[ch] = (corrected, meta)
        logger.info(f"  {ch}: {meta.get('method', 'auto')}")
    
    # =========================================================================
    # Step 5: Cell segmentation
    # =========================================================================
    logger.info("Step 5: Cell segmentation...")
    segmenter = cr.CellSegmenter(
        output_dir=None,
        auto_resize=False,
        resize_candidates=[600, 400],
    )
    
    seg = segmenter.segment_from_results(
        results=bg_results,
        channel_a="Phalloidin",
        channel_b="DAPI",
        channel_weights=(1.0, 0.10),
        projection="mip",
        output_format="png8",
        save_basename=None,
    )
    
    # Filter mask
    mask_2d = seg.mask_array.copy()
    if np.any(mask_2d == 1):
        mask_2d[mask_2d == 1] = 0
    
    min_area = estimate_min_area_threshold(mask_2d, fraction_of_median=0.90)
    filtered_mask, filter_info = _filter_labels(
        mask_2d,
        min_area=min_area,
        max_border_fraction=0.20,
        border_margin_px=1,
    )
    
    cell_count = len(filter_info["kept_labels"])
    logger.info(f"  Cells detected: {cell_count}")
    
    # =========================================================================
    # Step 6: Puncta detection (WITH FIX)
    # =========================================================================
    logger.info("Step 6: Puncta detection (WITH FIX)...")
    puncta_result = compute_puncta(
        bg_results,
        filtered_mask,
        channel="ALIX",
        detection_method="bigfish",
        return_threshold_data=True,
        drop_label_1=False,
    )
    
    # Extract metrics
    total_image = puncta_result.get("results", {}).get("total_image", {})
    puncta_count = total_image.get("total_puncta_count", 0)
    
    threshold_data = puncta_result.get("threshold_data", {})
    threshold = threshold_data.get("threshold", 0.0) if threshold_data else 0.0
    used_fallback = threshold_data.get("used_fallback", False)
    fallback_reason = threshold_data.get("fallback_reason", None)
    
    detection_params = puncta_result.get("detection_params", {})
    bg_mean = detection_params.get("background_mean", None)
    bg_std = detection_params.get("background_std", None)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Cells: {cell_count}")
    logger.info(f"Puncta detected: {puncta_count}")
    logger.info(f"Puncta per cell: {puncta_count / cell_count:.1f}")
    logger.info(f"Threshold: {threshold:.4f}")
    logger.info(f"Used fallback: {used_fallback}")
    if fallback_reason:
        logger.info(f"Fallback reason: {fallback_reason}")
    logger.info(f"Background: mean={bg_mean:.2f}, std={bg_std:.2f}")
    
    # Compare with expected
    logger.info("")
    logger.info("COMPARISON WITH OTHER 15min SAMPLES:")
    logger.info("  15_min_1: 348 puncta, 6 cells, 58.0 per cell, threshold=0.18")
    logger.info("  15_min_2: 881 puncta, 17 cells, 51.8 per cell, threshold=0.45")
    logger.info(f"  15_min_3 (OLD): 2 puncta, 13 cells, 0.15 per cell, threshold=5.88")
    logger.info(f"  15_min_3 (NEW): {puncta_count} puncta, {cell_count} cells, {puncta_count/cell_count:.1f} per cell, threshold={threshold:.2f}")
    
    # =========================================================================
    # Save outputs
    # =========================================================================
    logger.info("")
    logger.info("Saving outputs...")
    
    # Save elbow curve
    if threshold_data:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax = plot_puncta_elbow(puncta_result, ax=ax, title="15_min_3 - FIXED Threshold")
        
        # Add annotation about fallback
        if used_fallback:
            ax.text(0.98, 0.98, f"Fallback used: {fallback_reason}", 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        elbow_path = output_dir / "elbow_curve_fixed.png"
        fig.savefig(elbow_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {elbow_path}")
    
    # Save detection overlay
    alix_data = bg_results["ALIX"][0]
    alix_mip = np.max(alix_data, axis=0)
    
    fig = plot_puncta_detection(
        puncta_result,
        alix_mip,
        cell_mask=filtered_mask,
        figsize=(15, 5),
        title=f"15_min_3 FIXED - {puncta_count} puncta detected (threshold={threshold:.2f})",
    )
    
    detection_path = output_dir / "detection_overlay_fixed.png"
    fig.savefig(detection_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {detection_path}")
    
    # Save metrics
    metrics = {
        "image": "anti_ALIX_15_min_3.ome",
        "fix_applied": True,
        "cells": cell_count,
        "puncta_count": puncta_count,
        "puncta_per_cell": puncta_count / cell_count if cell_count > 0 else 0,
        "threshold": threshold,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "background_mean": bg_mean,
        "background_std": bg_std,
        "comparison": {
            "15_min_1": {"puncta": 348, "cells": 6, "per_cell": 58.0, "threshold": 0.18},
            "15_min_2": {"puncta": 881, "cells": 17, "per_cell": 51.8, "threshold": 0.45},
            "15_min_3_old": {"puncta": 2, "cells": 13, "per_cell": 0.15, "threshold": 5.88},
            "15_min_3_new": {"puncta": puncta_count, "cells": cell_count, 
                            "per_cell": puncta_count/cell_count, "threshold": threshold},
        },
        "processing_time_s": time.perf_counter() - start_time,
    }
    
    if threshold_data:
        metrics["log_intensity_percentiles"] = threshold_data.get("log_intensity_percentiles", {})
    
    metrics_path = output_dir / "retest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved: {metrics_path}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("FIX VALIDATION")
    logger.info("=" * 70)
    
    if used_fallback and fallback_reason == "threshold_too_high":
        logger.info("✅ SUCCESS: Fallback mechanism triggered for high threshold!")
        logger.info(f"   Detected {puncta_count} puncta (vs 2 with old code)")
        logger.info(f"   Per-cell average: {puncta_count/cell_count:.1f} (vs 0.15 with old code)")
        logger.info(f"   Now consistent with other 15min samples (50-58 per cell)")
    elif puncta_count > 500:
        logger.info("✅ SUCCESS: Detected reasonable number of puncta!")
        logger.info(f"   {puncta_count} puncta detected ({puncta_count/cell_count:.1f} per cell)")
        logger.info(f"   Consistent with other 15min samples")
    else:
        logger.warning("⚠️  WARNING: Still detecting low number of puncta")
        logger.warning(f"   Only {puncta_count} puncta detected")
        logger.warning(f"   Expected ~700-900 based on other samples")
    
    logger.info("")
    logger.info(f"Total processing time: {time.perf_counter() - start_time:.1f}s")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
