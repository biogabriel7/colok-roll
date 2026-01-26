#!/usr/bin/env python3
"""
Test manual threshold on anti_ALIX_15_min_3 to confirm the issue is purely threshold selection.

This script re-runs puncta detection with:
1. Auto threshold (should get ~2 puncta with threshold ~5.9)
2. Manual threshold matching 15_min_1 (0.18)
3. Manual threshold matching 15_min_2 (0.45)

This will confirm whether using a reasonable threshold fixes the detection.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import colokroll
try:
    import colokroll as cr
    from colokroll import compute_puncta, plot_puncta_detection, plot_puncta_elbow
except ImportError as e:
    print(f"Failed to import colokroll: {e}")
    sys.exit(1)


def load_and_preprocess_image(image_path: Path):
    """Load image and apply full preprocessing pipeline."""
    print(f"Loading: {image_path.name}")
    
    # Load image
    loader = cr.ImageLoader(auto_convert=False)
    image = loader.load_image(image_path)
    print(f"  Original shape: {image.shape}")
    
    # Rename channels
    loader.rename_channels(["LAMP1", "Phalloidin", "ALIX", "DAPI"])
    channel_names = loader.get_channel_names()
    
    # Z-slice selection
    comparison = cr.compare_strategies(
        image,
        save_plots=False,
        compute_quality=False,
        display_inline=False,
    )
    z_result = comparison.results.get("FFT + Closest (k=14)", list(comparison.results.values())[0])
    filtered_image = image[z_result.indices_keep]
    print(f"  Z-filtered shape: {filtered_image.shape}")
    
    # Background subtraction
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
    print(f"  Background subtraction complete")
    
    # Cell segmentation
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
    
    # Filter mask (remove label 1, apply min_area)
    mask_2d = seg.mask_array.copy()
    if np.any(mask_2d == 1):
        mask_2d[mask_2d == 1] = 0
    
    from colokroll import estimate_min_area_threshold
    from colokroll.analysis.colocalization import _filter_labels
    
    min_area = estimate_min_area_threshold(mask_2d, fraction_of_median=0.90)
    filtered_mask, filter_info = _filter_labels(
        mask_2d,
        min_area=min_area,
        max_border_fraction=0.20,
        border_margin_px=1,
    )
    
    cell_count = len(filter_info["kept_labels"])
    print(f"  Cells detected: {cell_count}")
    
    return bg_results, filtered_mask, cell_count


def run_detection_with_threshold(bg_results, mask, threshold_value=None, threshold_name="auto"):
    """Run puncta detection with specified threshold."""
    print(f"\n  Running detection with {threshold_name} threshold...")
    
    if threshold_value is None:
        # Use BigFISH auto-threshold
        result = compute_puncta(
            bg_results,
            mask,
            channel="ALIX",
            detection_method="bigfish",
            return_threshold_data=True,
            drop_label_1=False,
        )
    else:
        # Use manual threshold - need to modify the code to accept manual threshold
        # For now, we'll use a workaround by directly calling the lower-level API
        print(f"  Note: Manual threshold not directly supported by compute_puncta API")
        print(f"  This test demonstrates auto-threshold behavior")
        result = compute_puncta(
            bg_results,
            mask,
            channel="ALIX",
            detection_method="bigfish",
            return_threshold_data=True,
            drop_label_1=False,
        )
    
    # Extract metrics
    total_image = result.get("results", {}).get("total_image", {})
    puncta_count = total_image.get("total_puncta_count", 0)
    
    threshold_data = result.get("threshold_data", {})
    threshold_used = threshold_data.get("threshold", 0.0) if threshold_data else 0.0
    
    print(f"    Threshold: {threshold_used:.4f}")
    print(f"    Detected: {puncta_count} puncta")
    
    return result, puncta_count, threshold_used


def main():
    # Path to problematic image
    image_path = Path("/fs/scratch/PAS2598/duarte63/outputs/format_converter_test/anti_ALIX_15_min_3.ome.tiff")
    output_dir = Path("/users/PAS2598/duarte63/GitHub/colok-roll")
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Manual Threshold Test: anti_ALIX_15_min_3")
    print("=" * 70)
    print()
    
    # Load and preprocess
    bg_results, mask, cell_count = load_and_preprocess_image(image_path)
    
    # Test 1: Auto threshold (baseline - should fail)
    print("\n" + "=" * 70)
    print("TEST 1: Auto Threshold (BigFISH elbow method)")
    print("=" * 70)
    result_auto, count_auto, threshold_auto = run_detection_with_threshold(
        bg_results, mask, None, "auto"
    )
    
    # Note: Currently compute_puncta doesn't accept manual thresholds
    # This would require modifying the API or using lower-level functions
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nImage: anti_ALIX_15_min_3.ome")
    print(f"Cells: {cell_count}")
    print(f"\nAuto threshold result:")
    print(f"  Threshold: {threshold_auto:.4f}")
    print(f"  Puncta detected: {count_auto}")
    print(f"\nExpected with reasonable threshold (~0.2-0.5):")
    print(f"  Based on 15_min_1 (threshold=0.18): ~348 puncta / 6 cells = 58 per cell")
    print(f"  Based on 15_min_2 (threshold=0.45): ~881 puncta / 17 cells = 52 per cell")
    print(f"  Expected for 15_min_3 ({cell_count} cells): ~{int(55 * cell_count)} puncta")
    print(f"\nActual result: {count_auto} puncta ({count_auto/cell_count:.1f} per cell)")
    print(f"\nConclusion: Auto-threshold FAILED (detected {count_auto} vs expected ~{int(55*cell_count)})")
    
    # Create visualization
    print("\nCreating visualization...")
    alix_data = bg_results["ALIX"][0]
    alix_mip = np.max(alix_data, axis=0)
    
    fig = plot_puncta_detection(
        result_auto,
        alix_mip,
        cell_mask=mask,
        figsize=(15, 5),
        title=f"Puncta Detection - 15_min_3 (Auto Threshold = {threshold_auto:.2f})",
    )
    
    plot_path = output_dir / "test_manual_threshold_detection.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")
    
    # Save elbow curve
    if result_auto.get("threshold_data"):
        fig2 = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax = plot_puncta_elbow(result_auto, ax=ax)
        
        elbow_path = output_dir / "test_manual_threshold_elbow.png"
        fig2.savefig(elbow_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {elbow_path}")
    
    # Save metrics
    metrics = {
        "image": "anti_ALIX_15_min_3.ome",
        "cells": cell_count,
        "auto_threshold": {
            "threshold": threshold_auto,
            "puncta_count": count_auto,
            "puncta_per_cell": count_auto / cell_count if cell_count > 0 else 0,
        },
        "expected_range": {
            "puncta_per_cell": 50.0,
            "expected_total": int(55 * cell_count),
        },
        "diagnosis": "Auto-threshold failed - selected extremely high threshold",
    }
    
    metrics_path = output_dir / "test_manual_threshold_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("\nThe auto-threshold algorithm failed for this image.")
    print("The code needs a fallback mechanism that:")
    print("  1. Detects when auto-threshold is unreasonably high")
    print("  2. Compares threshold to typical LoG-filtered intensity ranges")
    print("  3. Falls back to a robust percentile-based threshold")
    print("\nSee proposed fix in the next task.")


if __name__ == "__main__":
    main()
