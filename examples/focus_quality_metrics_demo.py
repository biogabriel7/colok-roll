"""Demo: Focus Quality Metrics Integration with Colokroll

This example demonstrates how to use the Piao et al. (2025) focus quality metrics
to objectively evaluate and compare different focus measure operators.

Usage:
    python examples/focus_quality_metrics_demo.py
"""

import numpy as np
from pathlib import Path

# Import colokroll imaging preprocessing functions
from colokroll.imaging_preprocessing import (
    select_z_slices,
    benchmark_focus_methods,
    compare_strategies,
    plot_focus_curve_analysis,
)


def load_example_data():
    """Load or simulate example z-stack data."""
    # For demo purposes, create a synthetic z-stack
    # In real use, load your microscopy data with ImageLoader
    
    print("Generating synthetic z-stack for demo...")
    np.random.seed(42)
    
    # Create a synthetic z-stack (40 slices, 256x256, 3 channels)
    n_z = 40
    height, width = 256, 256
    n_channels = 3
    
    # Simulate focus curve - peaked around slice 20
    z_indices = np.arange(n_z)
    focus_center = 20
    focus_width = 8
    focus_profile = np.exp(-0.5 * ((z_indices - focus_center) / focus_width) ** 2)
    
    stack = np.zeros((n_z, height, width, n_channels), dtype=np.float32)
    
    for z in range(n_z):
        # Base signal intensity varies with focus
        base_intensity = focus_profile[z] * 200 + 50
        
        # Add some structure (simulated cells/features)
        y, x = np.ogrid[:height, :width]
        for _ in range(10):  # 10 random "cells"
            cy, cx = np.random.randint(30, height-30, 2)
            radius = np.random.randint(10, 30)
            mask = ((y - cy)**2 + (x - cx)**2) < radius**2
            
            for ch in range(n_channels):
                stack[z, mask, ch] += base_intensity * (0.5 + 0.5 * np.random.rand())
        
        # Add noise
        stack[z] += np.random.randn(height, width, n_channels) * 10
    
    # Clip to valid range
    stack = np.clip(stack, 0, 255)
    
    print(f"Generated stack shape: {stack.shape}")
    return stack


def example_1_basic_usage_with_quality():
    """Example 1: Basic z-slice selection with quality metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage with Quality Metrics")
    print("="*80)
    
    stack = load_example_data()
    
    # Select z-slices with quality metrics enabled
    result = select_z_slices(
        stack,
        method="combined",
        strategy="closest_to_peak",
        compute_quality=True,      # Enable quality metrics
        step_distance=0.5,          # 0.5 µm between slices
    )
    
    print(f"\nSelection Results:")
    print(f"  - Kept {len(result.indices_keep)} / {len(result.scores_agg)} slices")
    print(f"  - Removed slices: {list(result.indices_remove)}")
    
    # Access quality metrics
    q = result.quality_metrics
    print(f"\nFocus Curve Quality Metrics (Piao et al. 2025):")
    print(f"  - Ws (steep width):        {q.Ws:.2f} µm")
    print(f"  - Rsg (steep/gradual):     {q.Rsg:.3f}")
    print(f"  - Cp (peak curvature):     {q.Cp:.4f}")
    print(f"  - FWHM:                    {q.FWHM:.2f} µm")
    print(f"  - Sp (steep slope):        {q.Sp:.3f}")
    print(f"  - Unimodal curve:          {q.is_unimodal}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if q.Ws < 5.0:
        print(f"  ✓ Narrow steep region (Ws={q.Ws:.2f} µm) → High sensitivity to focus changes")
    else:
        print(f"  ⚠ Wide steep region (Ws={q.Ws:.2f} µm) → Lower sensitivity")
    
    if q.Rsg > 2.0:
        print(f"  ✓ High Rsg ({q.Rsg:.3f}) → Good discrimination of focus vs. defocus")
    else:
        print(f"  ⚠ Low Rsg ({q.Rsg:.3f}) → Poor discrimination")
    
    if q.Cp > 0.01:
        print(f"  ✓ High curvature (Cp={q.Cp:.4f}) → Sensitive to small focal deviations")
    else:
        print(f"  ⚠ Low curvature (Cp={q.Cp:.4f}) → Less sensitive near peak")


def example_2_benchmark_focus_methods():
    """Example 2: Compare different focus methods objectively."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Benchmark Focus Methods")
    print("="*80)
    
    stack = load_example_data()
    
    # Compare all available focus methods
    results = benchmark_focus_methods(
        stack,
        methods=["laplacian", "tenengrad", "fft", "combined"],
        step_distance=0.5,
        strategy="closest_to_peak",
    )
    
    # Print comparison table
    print("\nFocus Method Comparison:")
    print("-" * 90)
    print(f"{'Method':<12} | {'Ws (µm)':<8} | {'Rsg':<7} | {'Cp':<8} | {'FWHM (µm)':<10} | {'Unimodal'}")
    print("-" * 90)
    
    for method, result in results.items():
        q = result.quality_metrics
        unimodal_str = "Yes" if q.is_unimodal else "No"
        print(f"{method:<12} | {q.Ws:>8.2f} | {q.Rsg:>7.3f} | {q.Cp:>8.4f} | {q.FWHM:>10.2f} | {unimodal_str}")
    
    # Select best method by Rsg (steep-to-gradual ratio)
    best_method = max(results.items(), key=lambda x: x[1].quality_metrics.Rsg)
    print(f"\n✓ Best method (by Rsg): {best_method[0]} (Rsg={best_method[1].quality_metrics.Rsg:.3f})")
    
    # Select best method by Ws (narrower is better)
    best_by_ws = min(results.items(), key=lambda x: x[1].quality_metrics.Ws)
    print(f"✓ Best method (by Ws):  {best_by_ws[0]} (Ws={best_by_ws[1].quality_metrics.Ws:.2f} µm)")
    
    # Select best method by Cp (higher is better)
    best_by_cp = max(results.items(), key=lambda x: x[1].quality_metrics.Cp)
    print(f"✓ Best method (by Cp):  {best_by_cp[0]} (Cp={best_by_cp[1].quality_metrics.Cp:.4f})")


def example_3_strategy_comparison_with_quality():
    """Example 3: Compare strategies with quality metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Strategy Comparison with Quality Metrics")
    print("="*80)
    
    stack = load_example_data()
    
    # Define strategies to compare
    strategies = [
        {
            "name": "FFT Auto 80%",
            "method": "fft",
            "strategy": "closest_to_peak",
            "auto_keep_fraction": 0.8,
        },
        {
            "name": "Combined Auto 70%",
            "method": "combined",
            "strategy": "closest_to_peak",
            "auto_keep_fraction": 0.7,
        },
        {
            "name": "Tenengrad Top-15",
            "method": "tenengrad",
            "strategy": "topk",
            "keep_top": 15,
        },
    ]
    
    # Compare strategies with quality metrics
    comparison = compare_strategies(
        stack,
        strategies=strategies,
        save_plots=False,  # Set to True to save visualizations
        compute_quality=True,
        step_distance=0.5,
    )
    
    print(f"\nStrategy Comparison Results:")
    print(f"Total slices: {comparison.n_slices}")
    print(f"Strategies compared: {comparison.n_strategies}")
    print()
    
    # Print results for each strategy
    for name in comparison.strategy_names:
        result = comparison.results[name]
        n_kept = len(result.indices_keep)
        pct_kept = (n_kept / comparison.n_slices) * 100
        
        print(f"{name}:")
        print(f"  - Kept: {n_kept}/{comparison.n_slices} slices ({pct_kept:.1f}%)")
        
        if comparison.quality_comparison:
            q = comparison.quality_comparison[name]
            print(f"  - Quality: Ws={q.Ws:.2f} µm, Rsg={q.Rsg:.3f}, Cp={q.Cp:.4f}")


def example_4_visualize_focus_curve():
    """Example 4: Visualize focus curve analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Visualize Focus Curve Analysis")
    print("="*80)
    
    stack = load_example_data()
    
    # Get result with quality metrics
    result = select_z_slices(
        stack,
        method="combined",
        compute_quality=True,
        step_distance=0.5,
    )
    
    # Create output directory
    output_dir = Path("focus_quality_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize the focus curve with quality metrics annotations
    output_path = output_dir / "focus_curve_analysis.png"
    plot_focus_curve_analysis(
        result.scores_agg,
        result.quality_metrics,
        output_path=str(output_path),
    )
    
    print(f"\n✓ Saved focus curve analysis to: {output_path}")
    print(f"  The plot shows:")
    print(f"  - Focus measure curve")
    print(f"  - Steep vs. gradual region segmentation")
    print(f"  - Quality metric annotations (Ws, Rsg, Cp, FWHM)")


def example_5_tissue_specific_method_selection():
    """Example 5: Tissue-specific method selection workflow."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Tissue-Specific Method Selection Workflow")
    print("="*80)
    
    print("\nWorkflow for selecting optimal focus method for your tissue type:")
    print("1. Collect representative samples (3-5 z-stacks)")
    print("2. Benchmark all focus methods on each sample")
    print("3. Average quality metrics across samples")
    print("4. Select method with highest average Rsg or lowest Ws")
    print("5. Use selected method for all subsequent analysis")
    
    # Simulate with one sample (in practice, use multiple)
    stack = load_example_data()
    
    print("\nBenchmarking on Sample 1...")
    results = benchmark_focus_methods(
        stack,
        step_distance=0.5,
    )
    
    # Collect metrics
    print("\nMetrics Summary:")
    metrics_table = []
    for method, result in results.items():
        q = result.quality_metrics
        metrics_table.append({
            'method': method,
            'Ws': q.Ws,
            'Rsg': q.Rsg,
            'Cp': q.Cp,
            'FWHM': q.FWHM,
        })
    
    # In practice, repeat for multiple samples and average
    print("\nRecommendation Criteria:")
    print("  - High Rsg (>2.0): Better focus/defocus discrimination")
    print("  - Low Ws (<5.0 µm): Higher sensitivity to focus changes")
    print("  - High Cp (>0.01): More sensitive near focal position")
    print("  - Unimodal curve: Reliable peak detection")
    
    # Simple ranking by Rsg
    best_method = max(metrics_table, key=lambda x: x['Rsg'])
    print(f"\n✓ Recommended method: {best_method['method']}")
    print(f"  Rsg={best_method['Rsg']:.3f}, Ws={best_method['Ws']:.2f} µm")


def main():
    """Run all examples."""
    print("="*80)
    print("Focus Quality Metrics Integration Demo")
    print("Piao et al. (2025) Metrics for Colokroll")
    print("="*80)
    
    # Run examples
    example_1_basic_usage_with_quality()
    example_2_benchmark_focus_methods()
    example_3_strategy_comparison_with_quality()
    example_4_visualize_focus_curve()
    example_5_tissue_specific_method_selection()
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Use compute_quality=True to get objective focus method evaluation")
    print("  2. Use benchmark_focus_methods() to compare methods on your data")
    print("  3. Select methods based on Ws, Rsg, and Cp metrics")
    print("  4. Different tissues may favor different focus methods")
    print("  5. Quality metrics help make reproducible, data-driven decisions")


if __name__ == "__main__":
    main()

