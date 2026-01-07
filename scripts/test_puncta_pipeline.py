#!/usr/bin/env python3
"""
Test puncta detection pipeline with visualization and colocalization.

Runs the complete colokroll workflow on all images in a directory:
1. Load image
2. Rename channels (LAMP1, Phalloidin, ALIX, DAPI)
3. Z-slice selection (FFT + Closest k=14)
4. Background subtraction (automatic per channel) + visualization
5. Cell segmentation (Cellpose) + visualization
6. Puncta detection (BigFISH on ALIX)
7. Colocalization metrics (ALIX vs LAMP1)
8. Save results

Usage:
    python scripts/test_puncta_pipeline.py \
        --input-dir /path/to/images \
        --output-dir /path/to/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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
    from colokroll import compute_puncta, plot_puncta_elbow, compute_colocalization, estimate_min_area_threshold
    from colokroll.analysis.colocalization import _filter_labels
except ImportError as e:
    logger.error(f"Failed to import colokroll: {e}")
    sys.exit(1)

# Try to import cupy for GPU sync
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# Try to import matplotlib for saving plots
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

# Try to import pandas for per-cell CSV
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


# Constants
CHANNEL_NAMES = ["LAMP1", "Phalloidin", "ALIX", "DAPI"]
Z_SLICE_STRATEGY = "FFT + Closest (k=14)"


def is_control_image(image_path: Path) -> bool:
    """Check if image is a control (contains 'ctrl' in filename)."""
    return "ctrl" in image_path.stem.lower()


def discover_images(input_dir: Path) -> List[Path]:
    """Discover all ome.tiff images in the input directory."""
    patterns = ["*.ome.tiff", "*.ome.tif"]
    images = []
    for pattern in patterns:
        images.extend(sorted(input_dir.glob(pattern)))
    return images


def run_pipeline(
    image_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run the complete puncta detection pipeline on a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for outputs
        
    Returns:
        Dictionary with pipeline results and metrics
    """
    image_name = image_path.stem
    is_ctrl = is_control_image(image_path)
    
    logger.info("=" * 60)
    logger.info(f"Processing: {image_name}")
    logger.info(f"Control image: {is_ctrl}")
    logger.info("=" * 60)
    
    # Create output directories
    image_output_dir = output_dir / image_name
    bg_output_dir = image_output_dir / "background"
    seg_output_dir = image_output_dir / "segmentation"
    puncta_output_dir = image_output_dir / "puncta"
    coloc_output_dir = image_output_dir / "colocalization"
    
    for d in [image_output_dir, bg_output_dir, seg_output_dir, puncta_output_dir, coloc_output_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize result dict
    result = {
        "image_name": image_name,
        "image_path": str(image_path),
        "is_control": is_ctrl,
        "success": False,
        "error": None,
        "z_slices_original": None,
        "z_slices_kept": None,
        "cell_count": None,
        "puncta_count": None,
        "threshold": None,
        "bg_mean": None,
        "bg_std": None,
        # Colocalization metrics (both Otsu and Costes thresholding)
        "pearson_r": None,
        "manders_m1_otsu": None,
        "manders_m2_otsu": None,
        "manders_m1_costes": None,
        "manders_m2_costes": None,
        "jaccard": None,
        "processing_time_s": None,
    }
    
    start_time = time.perf_counter()
    
    try:
        # =====================================================================
        # Step 1: Load image
        # =====================================================================
        logger.info("Step 1: Loading image...")
        loader = cr.ImageLoader(auto_convert=False)
        image = loader.load_image(image_path)
        result["z_slices_original"] = image.shape[0]
        logger.info(f"  Loaded image with shape: {image.shape}")
        
        # =====================================================================
        # Step 2: Rename channels
        # =====================================================================
        logger.info("Step 2: Renaming channels...")
        loader.rename_channels(CHANNEL_NAMES)
        channel_names = loader.get_channel_names()
        logger.info(f"  Channels: {channel_names}")
        
        # =====================================================================
        # Step 3: Z-slice selection
        # =====================================================================
        logger.info("Step 3: Z-slice selection...")
        comparison = cr.compare_strategies(
            image,
            save_plots=False,
            compute_quality=False,
            display_inline=False,
        )
        
        # Pick the strategy
        if Z_SLICE_STRATEGY in comparison.results:
            z_result = comparison.results[Z_SLICE_STRATEGY]
        else:
            # Fallback to first available strategy
            strategy_name = comparison.strategy_names[0]
            z_result = comparison.results[strategy_name]
            logger.warning(f"  Strategy '{Z_SLICE_STRATEGY}' not found, using '{strategy_name}'")
        
        filtered_image = image[z_result.indices_keep]
        result["z_slices_kept"] = filtered_image.shape[0]
        logger.info(f"  Filtered from {image.shape[0]} to {filtered_image.shape[0]} slices")
        
        # =====================================================================
        # Step 4: Background subtraction (automatic)
        # =====================================================================
        logger.info("Step 4: Background subtraction...")
        bg_subtractor = cr.BackgroundSubtractor()
        
        bg_results = {}
        for i, ch in enumerate(channel_names):
            ch_data = filtered_image[:, :, :, i]
            t0 = time.perf_counter()
            
            # Use is_negative_control for ALIX on control images
            use_negative_control = (ch == "ALIX" and is_ctrl)
            
            corrected, meta = bg_subtractor.subtract_background(
                image=ch_data,
                channel_name=ch,
                is_negative_control=use_negative_control,
            )
            
            # Sync GPU if available
            if HAS_CUPY:
                cp.cuda.Stream.null.synchronize()
            
            dt = time.perf_counter() - t0
            bg_results[ch] = (corrected, meta)
            logger.info(f"  {ch}: {meta.get('method', 'auto')} ({dt:.2f}s)")
        
        # Save background subtraction comparison plot
        if HAS_MATPLOTLIB:
            try:
                middle_slice_idx = filtered_image.shape[0] // 2
                fig = bg_subtractor.plot_background_subtraction_comparison(
                    original_data=filtered_image,
                    corrected_results=bg_results,
                    channel_names=channel_names,
                    z_slice=middle_slice_idx,
                    figsize=(5 * len(channel_names), 12),
                )
                bg_plot_path = bg_output_dir / "background_comparison.png"
                fig.savefig(bg_plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"  Saved background comparison: {bg_plot_path}")
            except Exception as e:
                logger.warning(f"  Could not save background plot: {e}")
        
        # =====================================================================
        # Step 5: Cell segmentation
        # =====================================================================
        logger.info("Step 5: Cell segmentation...")
        segmenter = cr.CellSegmenter(
            output_dir=seg_output_dir,
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
            save_basename=image_name,
        )
        
        # Count cells before filtering
        unique_labels = np.unique(seg.mask_array)
        cell_count_raw = len(unique_labels) - 1  # Exclude background (0)
        logger.info(f"  Segmented {cell_count_raw} cells (before filtering)")
        logger.info(f"  Mask saved: {seg.mask_path}")
        
        # Save segmentation overlay if outlines available
        if seg.outlines_path and Path(seg.outlines_path).exists():
            logger.info(f"  Outlines saved: {seg.outlines_path}")
        
        # =====================================================================
        # Step 5b: Filter cells (same as colocalization)
        # =====================================================================
        logger.info("Step 5b: Filtering cells (min_area=auto, max_border_fraction=0.20)...")
        
        # Load mask and apply same filtering as colocalization
        mask_2d = seg.mask_array.copy()
        
        # Remove label 1 (Cellpose background) if present
        if np.any(mask_2d == 1):
            mask_2d[mask_2d == 1] = 0
            logger.info("  Removed label 1 (Cellpose background)")
        
        # Estimate min_area threshold (90% of median cell area)
        min_area = estimate_min_area_threshold(mask_2d, fraction_of_median=0.90)
        logger.info(f"  Auto min_area threshold: {min_area} (90% of median)")
        
        # Apply filtering
        filtered_mask, filter_info = _filter_labels(
            mask_2d,
            min_area=min_area,
            max_border_fraction=0.20,
            border_margin_px=1,
        )
        
        # Save filtered mask
        filtered_mask_path = seg_output_dir / f"{image_name}_masks_filtered.tif"
        from tifffile import imwrite
        imwrite(str(filtered_mask_path), filtered_mask.astype(np.uint16))
        logger.info(f"  Saved filtered mask: {filtered_mask_path}")
        
        # Update cell count after filtering
        cell_count = len(filter_info["kept_labels"])
        result["cell_count"] = cell_count
        logger.info(f"  Cells after filtering: {cell_count} (removed {len(filter_info['removed_labels'])})")
        
        # =====================================================================
        # Step 6: Puncta detection (using filtered mask)
        # =====================================================================
        logger.info("Step 6: Puncta detection...")
        puncta_result = compute_puncta(
            bg_results,
            filtered_mask,  # Use filtered mask instead of original
            channel="ALIX",
            detection_method="bigfish",
            return_threshold_data=True,
            drop_label_1=False,  # Already removed during filtering
        )
        
        # Extract metrics from puncta_result structure
        total_image = puncta_result.get("results", {}).get("total_image", {})
        puncta_count = total_image.get("total_puncta_count", 0)
        
        threshold_data = puncta_result.get("threshold_data", {})
        threshold = threshold_data.get("threshold", 0.0) if threshold_data else 0.0
        
        # Get background stats from detection_params
        detection_params = puncta_result.get("detection_params", {})
        bg_mean = detection_params.get("background_mean", None)
        bg_std = detection_params.get("background_std", None)
        
        result["puncta_count"] = puncta_count
        result["threshold"] = threshold
        result["bg_mean"] = bg_mean
        result["bg_std"] = bg_std
        
        logger.info(f"  Detected {puncta_count} puncta")
        
        # Save elbow curve plot
        if HAS_MATPLOTLIB and threshold_data:
            try:
                ax = plot_puncta_elbow(puncta_result)
                if ax is not None:
                    fig = ax.get_figure()
                    elbow_path = puncta_output_dir / "elbow_curve.png"
                    fig.savefig(elbow_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"  Saved elbow curve: {elbow_path}")
            except Exception as e:
                logger.warning(f"  Could not save elbow curve: {e}")
        
        # Save puncta metrics JSON
        puncta_metrics_path = puncta_output_dir / "metrics.json"
        summary = puncta_result.get("results", {}).get("summary", {})
        puncta_metrics = {
            "image_name": image_name,
            "is_control": is_ctrl,
            "puncta_count": puncta_count,
            "threshold": threshold,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "puncta_per_cell": puncta_count / cell_count if cell_count > 0 else 0,
            "summary": summary,  # mean_over_cells stats
        }
        with open(puncta_metrics_path, "w") as f:
            json.dump(puncta_metrics, f, indent=2)
        logger.info(f"  Saved puncta metrics: {puncta_metrics_path}")
        
        # Save per-cell puncta data
        puncta_per_label = puncta_result.get("results", {}).get("per_label", [])
        if puncta_per_label and HAS_PANDAS:
            df_puncta_cells = pd.DataFrame(puncta_per_label)
            puncta_per_cell_path = puncta_output_dir / "per_cell.csv"
            df_puncta_cells.to_csv(puncta_per_cell_path, index=False)
            logger.info(f"  Saved per-cell puncta data: {puncta_per_cell_path}")
        
        # =====================================================================
        # Step 7: Colocalization (ALIX vs LAMP1) - Both Otsu and Costes
        # =====================================================================
        logger.info("Step 7: Colocalization analysis (ALIX vs LAMP1)...")
        
        # Reconstruct 4D array from results
        corrected_stack = np.stack([
            bg_results[ch][0]
            for ch in channel_names
        ], axis=-1)
        
        # Use different min_threshold_sigma: 3.0 for control (strict), 2.0 for treatment (more inclusive)
        min_sigma = 2.0 if is_ctrl else 3.0
        logger.info(f"  Using min_threshold_sigma={min_sigma} ({'control - strict' if is_ctrl else 'treatment - inclusive'})")
        
        # Path for mask visualization
        coloc_mask_plot_path = coloc_output_dir / "mask_filtered.png"
        
        # Common parameters for both thresholding methods
        # Use the same filtered mask as puncta detection
        common_params = dict(
            image=corrected_stack,
            mask=filtered_mask,  # Use pre-filtered mask (same as puncta)
            channel_a="ALIX",
            channel_b="LAMP1",
            channel_names=channel_names,
            pearson_winsor_clip=0.1,
            min_threshold_sigma=min_sigma,
            min_area=0,  # Already filtered
            max_border_fraction=None,  # Already filtered
            drop_label_1=False,  # Already removed during filtering
        )
        
        try:
            # --- Run with OTSU thresholding ---
            logger.info("  Running colocalization with Otsu thresholding...")
            coloc_otsu = compute_colocalization(
                **common_params,
                thresholding="otsu",
                plot_mask=True,
                plot_mask_save=str(coloc_mask_plot_path),
            )
            
            otsu_total = coloc_otsu.get("results", {}).get("total_image", {})
            result["pearson_r"] = otsu_total.get("pearson_r", None)
            result["manders_m1_otsu"] = otsu_total.get("manders_m1", None)
            result["manders_m2_otsu"] = otsu_total.get("manders_m2", None)
            result["jaccard"] = otsu_total.get("jaccard", None)
            
            logger.info(f"    Pearson r: {result['pearson_r']:.4f}" if result['pearson_r'] else "    Pearson r: N/A")
            logger.info(f"    Manders M1 (Otsu): {result['manders_m1_otsu']:.4f}" if result['manders_m1_otsu'] else "    Manders M1: N/A")
            logger.info(f"    Manders M2 (Otsu): {result['manders_m2_otsu']:.4f}" if result['manders_m2_otsu'] else "    Manders M2: N/A")
            
            # --- Run with COSTES thresholding ---
            logger.info("  Running colocalization with Costes thresholding...")
            coloc_costes = compute_colocalization(
                **common_params,
                thresholding="costes",
                plot_mask=False,  # Already saved mask from Otsu run
            )
            
            costes_total = coloc_costes.get("results", {}).get("total_image", {})
            result["manders_m1_costes"] = costes_total.get("manders_m1", None)
            result["manders_m2_costes"] = costes_total.get("manders_m2", None)
            
            logger.info(f"    Manders M1 (Costes): {result['manders_m1_costes']:.4f}" if result['manders_m1_costes'] else "    Manders M1: N/A")
            logger.info(f"    Manders M2 (Costes): {result['manders_m2_costes']:.4f}" if result['manders_m2_costes'] else "    Manders M2: N/A")
            logger.info(f"    Jaccard: {result['jaccard']:.4f}" if result['jaccard'] else "    Jaccard: N/A")
            
            if coloc_mask_plot_path.exists():
                logger.info(f"  Saved mask plot: {coloc_mask_plot_path}")
            
            # Save colocalization metrics JSON (includes both methods)
            coloc_metrics_path = coloc_output_dir / "metrics.json"
            coloc_metrics = {
                "image_name": image_name,
                "is_control": is_ctrl,
                "channel_a": "ALIX",
                "channel_b": "LAMP1",
                "min_threshold_sigma": min_sigma,
                "pearson_r": result["pearson_r"],
                "manders_m1_otsu": result["manders_m1_otsu"],
                "manders_m2_otsu": result["manders_m2_otsu"],
                "manders_m1_costes": result["manders_m1_costes"],
                "manders_m2_costes": result["manders_m2_costes"],
                "jaccard": result["jaccard"],
                "thresholds_per_z_otsu": otsu_total.get("thresholds_per_z", []),
                "thresholds_per_z_costes": costes_total.get("thresholds_per_z", []),
            }
            with open(coloc_metrics_path, "w") as f:
                json.dump(coloc_metrics, f, indent=2)
            logger.info(f"  Saved colocalization metrics: {coloc_metrics_path}")
            
            # Save per-cell colocalization data (from Otsu - includes all metrics)
            per_label = coloc_otsu.get("results", {}).get("per_label", [])
            if per_label and HAS_PANDAS:
                df_cells = pd.DataFrame(per_label)
                per_cell_path = coloc_output_dir / "per_cell.csv"
                df_cells.to_csv(per_cell_path, index=False)
                logger.info(f"  Saved per-cell data: {per_cell_path}")
            
        except Exception as e:
            logger.warning(f"  Colocalization analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        result["success"] = True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        result["error"] = str(e)
        import traceback
        traceback.print_exc()
    
    # Record processing time
    result["processing_time_s"] = time.perf_counter() - start_time
    logger.info(f"Processing time: {result['processing_time_s']:.1f}s")
    
    return result


def save_summary_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary CSV with results from all images."""
    if not results:
        return
    
    columns = [
        "image_name",
        "is_control",
        "success",
        "z_slices_original",
        "z_slices_kept",
        "cell_count",
        "puncta_count",
        "threshold",
        "bg_mean",
        "bg_std",
        # Colocalization columns (both Otsu and Costes)
        "pearson_r",
        "manders_m1_otsu",
        "manders_m2_otsu",
        "manders_m1_costes",
        "manders_m2_costes",
        "jaccard",
        "processing_time_s",
        "error",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    logger.info(f"Saved summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test puncta detection pipeline on all images in a directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input ome.tiff images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Auto-discover all ome.tiff images
    images = discover_images(args.input_dir)
    
    if not images:
        logger.error(f"No ome.tiff images found in {args.input_dir}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Puncta Detection Pipeline - Full Batch Test")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Discovered {len(images)} images:")
    for img in images:
        ctrl_tag = " (control)" if is_control_image(img) else ""
        logger.info(f"  - {img.name}{ctrl_tag}")
    logger.info(f"Channel names: {CHANNEL_NAMES}")
    logger.info(f"Z-slice strategy: {Z_SLICE_STRATEGY}")
    logger.info("=" * 60)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_results = []
    
    for image_path in images:
        result = run_pipeline(image_path, args.output_dir)
        all_results.append(result)
    
    # Save summary CSV
    summary_path = args.output_dir / "summary.csv"
    save_summary_csv(all_results, summary_path)
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for r in all_results:
        status = "✓" if r.get("success") else "✗"
        puncta = r.get("puncta_count", "N/A")
        cells = r.get("cell_count", "N/A")
        pearson = r.get("pearson_r")
        m1_otsu = r.get("manders_m1_otsu")
        m1_costes = r.get("manders_m1_costes")
        pearson_str = f"r={pearson:.3f}" if pearson is not None else "r=N/A"
        m1_str = f"M1: otsu={m1_otsu:.3f}/costes={m1_costes:.3f}" if m1_otsu is not None else "M1=N/A"
        ctrl = "(ctrl)" if r.get("is_control") else ""
        logger.info(f"  {status} {r['image_name']} {ctrl}: {puncta} puncta, {cells} cells, {pearson_str}, {m1_str}")
    
    # Count successes/failures
    n_success = sum(1 for r in all_results if r.get("success"))
    n_total = len(all_results)
    logger.info("")
    logger.info(f"Completed: {n_success}/{n_total} images processed successfully")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
