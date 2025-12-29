"""Visualization functions for Z-slice selection results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .utils import _create_channel_composites, _ensure_zyxc

if TYPE_CHECKING:
    from .core import ZSliceSelectionResult, StrategyComparisonResult


def save_z_slice_plots(
    img: np.ndarray,
    result: "ZSliceSelectionResult",
    output_path: Optional[Union[str, Path]],
) -> None:
    """Generate and save visualization plots for Z-slice selection results.
    
    Creates two plots:
    1. Score plot: Shows raw and smoothed focus scores with threshold
    2. Gallery plot: Grid of all Z-slices with their scores
    
    Parameters
    ----------
    img:
        Original input stack (Z, Y, X, C).
    result:
        Z-slice selection result object.
    output_path:
        Directory to save plots. If None, uses current working directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import warnings
        warnings.warn(
            "matplotlib not available; skipping plot generation. "
            "Install matplotlib to enable visualization.",
            UserWarning,
        )
        return
    
    # Determine output directory
    if output_path is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure image is in ZYXC format
    stack = _ensure_zyxc(img, axes=None)
    
    # 1. Generate score plot
    plot_z_slice_scores(result, output_dir)
    
    # 2. Generate gallery plot
    plot_z_slice_gallery(stack, result, output_dir)


def plot_z_slice_scores(
    result: "ZSliceSelectionResult",
    output_dir: Path,
) -> None:
    """Plot focus scores over Z with threshold and selection.
    
    Parameters
    ----------
    result:
        Z-slice selection result object.
    output_dir:
        Directory to save the plot.
    """
    import matplotlib.pyplot as plt
    
    scores = result.scores_agg
    smoothed = result.smoothed_scores
    mask_keep = result.mask_keep
    z = np.arange(scores.shape[0])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z, scores, label="scores", alpha=0.4, color="#1f77b4")
    ax.plot(z, smoothed, label="smoothed", color="#d62728")
    ax.hlines(
        result.threshold_used,
        z.min(),
        z.max(),
        colors="#7f7f7f",
        linestyles="--",
        label="threshold",
    )
    ax.scatter(z[mask_keep], scores[mask_keep], color="#2ca02c", s=20, label="keep")
    ax.scatter(z[~mask_keep], scores[~mask_keep], color="#ff7f0e", s=20, label="remove")
    ax.set_xlabel("Z index")
    ax.set_ylabel("Focus score (aggregated)")
    ax.legend()
    ax.set_title("Z-slice focus scores and selection")
    fig.tight_layout()
    
    out_path = output_dir / "z_slice_scores.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved score plot to {out_path}")


def plot_z_slice_gallery(
    stack: np.ndarray,
    result: "ZSliceSelectionResult",
    output_dir: Path,
) -> None:
    """Plot gallery of all Z-slices with their scores.
    
    Parameters
    ----------
    stack:
        Image stack in ZYXC format.
    result:
        Z-slice selection result object.
    output_dir:
        Directory to save the plot.
    """
    import matplotlib.pyplot as plt
    
    z_total = stack.shape[0]
    n_channels = stack.shape[-1]
    cols = 6
    rows = int(np.ceil(z_total / cols))
    
    # Create RGB composite for each slice
    composite_slices = _create_channel_composites(stack)
    
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    
    for ax, idx in zip(axes, range(z_total)):
        ax.imshow(composite_slices[idx])
        ax.set_title(f"z={idx} score={result.scores_agg[idx]:.3f}", fontsize=8)
        ax.axis("off")
    
    # Hide unused subplots
    for ax in axes[z_total:]:
        ax.axis("off")
    
    fig2.suptitle(f"Per-slice composite ({n_channels} channels)", y=0.99)
    fig2.tight_layout()
    
    out_path2 = output_dir / "z_slice_gallery.png"
    fig2.savefig(out_path2, dpi=200)
    plt.close(fig2)
    print(f"Saved gallery to {out_path2}")


def plot_strategy_comparison(
    img: np.ndarray,
    comparison: "StrategyComparisonResult",
    output_dir: Path,
    axes: Optional[str],
    *,
    display_inline: bool = False,
) -> None:
    """Generate comparison visualizations for multiple strategies.
    
    Creates:
    1. Decision matrix heatmap (which slices each strategy keeps/removes)
    2. Slice gallery (visual inspection of all slices)
    3. Summary text file
    
    Parameters
    ----------
    img:
        Original input stack.
    comparison:
        Strategy comparison result object.
    output_dir:
        Directory to save plots.
    axes:
        Axis order string.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        import warnings
        warnings.warn(
            "matplotlib not available; skipping comparison plots. "
            "Install matplotlib to enable visualization.",
            UserWarning,
        )
        return
    
    # Ensure image is in ZYXC format
    stack = _ensure_zyxc(img, axes)
    
    # 1. Generate decision matrix heatmap
    print("  - Creating decision matrix heatmap...")
    fig1, ax1 = plt.subplots(figsize=(max(12, comparison.n_strategies * 1.2), 
                                      max(8, comparison.n_slices * 0.25)))
    
    # Custom colormap: remove=red, KEEP=green
    colors = ['#ff6b6b', '#51cf66']
    cmap = LinearSegmentedColormap.from_list('decision', colors, N=2)
    
    im1 = ax1.imshow(comparison.decision_matrix.astype(int), 
                     aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Set ticks
    ax1.set_xticks(np.arange(comparison.n_strategies))
    ax1.set_yticks(np.arange(comparison.n_slices))
    ax1.set_xticklabels(comparison.strategy_names, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(range(comparison.n_slices), fontsize=8)
    
    # Labels
    ax1.set_xlabel('Detection Strategy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Z-Slice Index', fontsize=11, fontweight='bold')
    ax1.set_title('Z-Slice Selection Decisions Across Strategies\n(Green = KEEP, Red = remove)',
                  fontsize=13, fontweight='bold', pad=15)
    
    # Add grid
    ax1.set_xticks(np.arange(comparison.n_strategies + 1) - 0.5, minor=True)
    ax1.set_yticks(np.arange(comparison.n_slices + 1) - 0.5, minor=True)
    ax1.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Add text annotations.
    # By default we show just ✓/✗. For small-ish comparisons we also include the score
    # to make it easy to visually validate thresholds/strategies (not just decisions).
    annotate_scores = (comparison.n_slices <= 60 and comparison.n_strategies <= 14)
    for i in range(comparison.n_slices):
        for j in range(comparison.n_strategies):
            keep = bool(comparison.decision_matrix[i, j])
            mark = "✓" if keep else "✗"
            if annotate_scores:
                score = float(comparison.score_matrix[i, j])
                text = f"{mark}\n{score:.4f}"
                fontsize = 6
            else:
                text = mark
                fontsize = 7
            ax1.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
                fontweight="bold",
                linespacing=0.9,
            )
    
    plt.tight_layout()
    out_path1 = output_dir / "decision_matrix_heatmap.png"
    fig1.savefig(out_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"    ✓ Saved: {out_path1}")

    if display_inline:
        try:
            from IPython.display import Image as _IPyImage, display as _display  # type: ignore

            _display(_IPyImage(filename=str(out_path1)))
        except Exception:
            # If IPython isn't available (or we're not in a notebook), just skip inline display.
            pass
    
    # 2. Generate slice gallery
    print("  - Creating slice gallery...")
    
    z_total = stack.shape[0]
    n_channels = stack.shape[-1]
    cols = 6
    rows = int(np.ceil(z_total / cols))
    
    # Create RGB composites for each slice
    composite_slices = _create_channel_composites(stack)
    
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    
    for ax, idx in zip(axes, range(z_total)):
        ax.imshow(composite_slices[idx])
        ax.set_title(f"z={idx}", fontsize=9, fontweight='bold')
        ax.axis("off")
    
    for ax in axes[z_total:]:
        ax.axis("off")
    
    fig2.suptitle(f"Per-slice composite ({n_channels} channels) - For Visual Inspection", y=0.99)
    fig2.tight_layout()
    
    out_path2 = output_dir / "z_slice_gallery.png"
    fig2.savefig(out_path2, dpi=200)
    plt.close(fig2)
    print(f"    ✓ Saved: {out_path2}")

    if display_inline:
        try:
            from IPython.display import Image as _IPyImage, display as _display  # type: ignore

            _display(_IPyImage(filename=str(out_path2)))
        except Exception:
            pass
    
    # 3. Generate summary table
    print("  - Creating strategy summary...")
    
    summary_lines = []
    summary_lines.append("Strategy Comparison Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Total Z-slices: {comparison.n_slices}")
    summary_lines.append(f"Strategies compared: {comparison.n_strategies}")
    summary_lines.append("")
    summary_lines.append(f"{'Strategy':<35} {'Kept':<10} {'Removed':<10} {'% Kept':<10}")
    summary_lines.append("-" * 80)
    
    for name in comparison.strategy_names:
        result = comparison.results[name]
        n_kept = len(result.indices_keep)
        n_removed = len(result.indices_remove)
        pct_kept = (n_kept / comparison.n_slices) * 100
        summary_lines.append(f"{name:<35} {n_kept:<10} {n_removed:<10} {pct_kept:<10.1f}%")
    
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    out_path3 = output_dir / "comparison_summary.txt"
    with open(out_path3, 'w') as f:
        f.write(summary_text)
    
    print(f"    ✓ Saved: {out_path3}")
    print("\n" + summary_text)

