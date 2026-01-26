#!/usr/bin/env python3
"""
Diagnostic script to investigate why BigFISH auto-threshold failed for anti_ALIX_15_min_3.

This script:
1. Loads the three 15min images
2. Compares raw and LoG-filtered intensity distributions
3. Visualizes the elbow curves side-by-side
4. Identifies differences that caused threshold failure
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import colokroll
try:
    import colokroll as cr
    from colokroll.analysis.puncta import _detect_spots_bigfish
except ImportError as e:
    print(f"Failed to import colokroll: {e}")
    sys.exit(1)

# Import bigfish
try:
    from bigfish import detection as bigfish_detection
    from bigfish import stack as bigfish_stack
except ImportError:
    print("BigFISH is required. Install with: pip install big-fish")
    sys.exit(1)


def load_and_preprocess_image(image_path: Path, channel_idx: int = 2):
    """Load image and extract ALIX channel (index 2) with Z-slice selection and bg subtraction."""
    loader = cr.ImageLoader(auto_convert=False)
    image = loader.load_image(image_path)
    
    # Rename channels
    loader.rename_channels(["LAMP1", "Phalloidin", "ALIX", "DAPI"])
    
    # Z-slice selection (FFT + Closest, k=14)
    comparison = cr.compare_strategies(
        image,
        save_plots=False,
        compute_quality=False,
        display_inline=False,
    )
    z_result = comparison.results.get("FFT + Closest (k=14)", list(comparison.results.values())[0])
    filtered_image = image[z_result.indices_keep]
    
    # Background subtraction
    bg_subtractor = cr.BackgroundSubtractor()
    alix_data = filtered_image[:, :, :, channel_idx]
    corrected, meta = bg_subtractor.subtract_background(
        image=alix_data,
        channel_name="ALIX",
        is_negative_control=False,
    )
    
    # Get MIP
    mip = np.max(corrected, axis=0)
    
    return mip, meta


def analyze_log_filtering(image_2d: np.ndarray, spot_radius_px: float = 1.5):
    """Apply LoG filtering and analyze the intensity distribution."""
    sigma = (spot_radius_px, spot_radius_px)
    
    # Apply LoG filter (same as BigFISH)
    image_filtered = bigfish_stack.log_filter(image_2d.astype(np.float64), sigma=sigma)
    
    # Compute statistics
    stats = {
        "raw_min": float(image_2d.min()),
        "raw_max": float(image_2d.max()),
        "raw_mean": float(image_2d.mean()),
        "raw_std": float(image_2d.std()),
        "log_min": float(image_filtered.min()),
        "log_max": float(image_filtered.max()),
        "log_mean": float(image_filtered.mean()),
        "log_std": float(image_filtered.std()),
        "log_positive_count": int(np.sum(image_filtered > 0)),
    }
    
    return image_filtered, stats


def compute_elbow_data(image_2d: np.ndarray, spot_radius_px: float = 1.5):
    """Compute elbow curve data for threshold analysis."""
    sigma = (spot_radius_px, spot_radius_px)
    log_kernel_size = max(3, int(round(spot_radius_px * 2)))
    if log_kernel_size % 2 == 0:
        log_kernel_size += 1
    min_distance = max(1, int(round(spot_radius_px)))
    
    log_kernel_2d = (log_kernel_size, log_kernel_size)
    min_distance_2d = (min_distance, min_distance)
    
    # Apply LoG filter
    image_filtered = bigfish_stack.log_filter(image_2d.astype(np.float64), sigma=sigma)
    
    # Get local maxima
    local_maxima_mask = bigfish_detection.local_maximum_detection(
        image_filtered,
        min_distance=min_distance_2d,
    )
    local_maxima_coords = np.argwhere(local_maxima_mask)
    
    # Generate threshold range
    max_val = float(image_filtered.max())
    min_val = float(image_filtered[image_filtered > 0].min()) if np.any(image_filtered > 0) else 0
    thresholds = np.linspace(min_val, max_val, num=200)
    
    # Count spots at each threshold
    spot_counts = []
    if len(local_maxima_coords) > 0:
        for t in thresholds:
            count = int(np.sum(image_filtered[local_maxima_coords[:, 0], local_maxima_coords[:, 1]] >= t))
            spot_counts.append(count)
    else:
        spot_counts = [0] * len(thresholds)
    
    # Get BigFISH auto threshold
    spots, threshold = bigfish_detection.detect_spots(
        image_2d.astype(np.float64),
        threshold=None,
        return_threshold=True,
        log_kernel_size=log_kernel_2d,
        minimum_distance=min_distance_2d,
    )
    
    threshold_scalar = float(threshold) if threshold is not None else 0.0
    
    return {
        "thresholds": thresholds,
        "spot_counts": spot_counts,
        "auto_threshold": threshold_scalar,
        "local_maxima_count": len(local_maxima_coords),
        "detected_spots": len(spots),
        "image_filtered": image_filtered,
    }


def main():
    # Paths
    converted_dir = Path("/fs/scratch/PAS2598/duarte63/outputs/format_converter_test")
    output_dir = Path("/fs/scratch/PAS2598/duarte63/outputs/puncta_pipeline_converted")
    
    images = {
        "15_min_1": converted_dir / "anti_ALIX_15_min_1.ome.tiff",
        "15_min_2": converted_dir / "anti_ALIX_15_min_2.ome.tiff",
        "15_min_3": converted_dir / "anti_ALIX_15_min_3.ome.tiff",  # Problematic
    }
    
    print("=" * 70)
    print("Diagnostic Analysis: BigFISH Threshold Failure")
    print("=" * 70)
    print()
    
    # Load and analyze each image
    results = {}
    for name, path in images.items():
        print(f"Processing {name}...")
        
        # Load image
        mip, bg_meta = load_and_preprocess_image(path)
        print(f"  Loaded: {mip.shape}, dtype={mip.dtype}")
        
        # Analyze LoG filtering
        image_filtered, stats = analyze_log_filtering(mip)
        print(f"  Raw stats: min={stats['raw_min']:.2f}, max={stats['raw_max']:.2f}, "
              f"mean={stats['raw_mean']:.2f}, std={stats['raw_std']:.2f}")
        print(f"  LoG stats: min={stats['log_min']:.4f}, max={stats['log_max']:.4f}, "
              f"mean={stats['log_mean']:.4f}, std={stats['log_std']:.4f}")
        
        # Compute elbow data
        elbow_data = compute_elbow_data(mip)
        print(f"  Local maxima: {elbow_data['local_maxima_count']}")
        print(f"  Auto threshold: {elbow_data['auto_threshold']:.4f}")
        print(f"  Detected spots: {elbow_data['detected_spots']}")
        print()
        
        results[name] = {
            "mip": mip,
            "filtered": image_filtered,
            "stats": stats,
            "elbow": elbow_data,
            "bg_meta": bg_meta,
        }
    
    # Print comparison table
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'15_min_1':>12} {'15_min_2':>12} {'15_min_3':>12}")
    print("-" * 70)
    
    metrics = [
        ("Raw mean", "stats", "raw_mean", "{:.2f}"),
        ("Raw std", "stats", "raw_std", "{:.2f}"),
        ("LoG max", "stats", "log_max", "{:.4f}"),
        ("LoG mean", "stats", "log_mean", "{:.4f}"),
        ("LoG std", "stats", "log_std", "{:.4f}"),
        ("Local maxima", "elbow", "local_maxima_count", "{:d}"),
        ("Auto threshold", "elbow", "auto_threshold", "{:.4f}"),
        ("Detected spots", "elbow", "detected_spots", "{:d}"),
    ]
    
    for label, category, key, fmt in metrics:
        values = [results[name][category][key] for name in ["15_min_1", "15_min_2", "15_min_3"]]
        print(f"{label:<30} {fmt.format(values[0]):>12} {fmt.format(values[1]):>12} {fmt.format(values[2]):>12}")
    
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()
    
    # Analyze the problematic image
    ratio_1_to_3 = results["15_min_1"]["elbow"]["auto_threshold"] / results["15_min_3"]["elbow"]["auto_threshold"]
    ratio_2_to_3 = results["15_min_2"]["elbow"]["auto_threshold"] / results["15_min_3"]["elbow"]["auto_threshold"]
    
    print(f"15_min_3 threshold is {1/ratio_1_to_3:.1f}× higher than 15_min_1")
    print(f"15_min_3 threshold is {1/ratio_2_to_3:.1f}× higher than 15_min_2")
    print()
    
    # Check if LoG distributions are different
    log_max_ratio = results["15_min_3"]["stats"]["log_max"] / results["15_min_1"]["stats"]["log_max"]
    print(f"LoG max ratio (15_min_3 / 15_min_1): {log_max_ratio:.2f}")
    print()
    
    # Create visualization
    print("Creating diagnostic plots...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for idx, name in enumerate(["15_min_1", "15_min_2", "15_min_3"]):
        data = results[name]
        
        # Row 1: Raw MIP
        ax = axes[0, idx]
        im = ax.imshow(data["mip"], cmap="gray", vmin=0, vmax=np.percentile(data["mip"], 99.5))
        ax.set_title(f"{name}\nRaw MIP (99.5% clip)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 2: LoG-filtered
        ax = axes[1, idx]
        im = ax.imshow(data["filtered"], cmap="viridis")
        ax.set_title(f"LoG-filtered\nmax={data['stats']['log_max']:.4f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 3: Elbow curve
        ax = axes[2, idx]
        elbow = data["elbow"]
        ax.plot(elbow["thresholds"], elbow["spot_counts"], "b-", linewidth=2)
        ax.axvline(elbow["auto_threshold"], color="r", linestyle="--", linewidth=2,
                   label=f"Auto: {elbow['auto_threshold']:.2f}")
        ax.set_xlabel("Threshold (LoG-filtered intensity)")
        ax.set_ylabel("Number of spots")
        ax.set_title(f"Elbow Curve\n{elbow['detected_spots']} spots detected")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight the problematic case
        if name == "15_min_3":
            for row in range(3):
                axes[row, idx].patch.set_edgecolor('red')
                axes[row, idx].patch.set_linewidth(3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path("/users/PAS2598/duarte63/GitHub/colok-roll/diagnostic_threshold_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {output_path}")
    
    # Save detailed statistics as JSON
    stats_output = Path("/users/PAS2598/duarte63/GitHub/colok-roll/diagnostic_threshold_stats.json")
    stats_export = {}
    for name, data in results.items():
        stats_export[name] = {
            "stats": data["stats"],
            "auto_threshold": data["elbow"]["auto_threshold"],
            "local_maxima_count": data["elbow"]["local_maxima_count"],
            "detected_spots": data["elbow"]["detected_spots"],
        }
    
    with open(stats_output, "w") as f:
        json.dump(stats_export, f, indent=2)
    print(f"Saved statistics: {stats_output}")
    
    print()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
