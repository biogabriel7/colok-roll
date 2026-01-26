"""
Visualization utilities for background subtraction.

This module provides plotting and comparison functions for visualizing
background subtraction results and comparing different parameter settings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


class VisualizationHelper:
    """
    Provides visualization utilities for background subtraction results.
    
    Supports:
    - Before/after comparison plots
    - Histogram comparisons
    - Multi-slice visualization
    - Parameter comparison grids
    """
    
    def __init__(self):
        """Initialize the visualization helper."""
        self.logger = logging.getLogger(__name__)
    
    def plot_comparison(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        z_slice: Optional[int] = None,
        channel_name: str = "Unknown",
        method: str = "Unknown",
        figsize: Tuple[int, int] = (15, 10),
        show_histogram: bool = True,
        vmax_percentile: float = 99.5,
    ) -> "Figure":
        """
        Create a comparison plot of original vs corrected images.
        
        Args:
            original: Original 3D image
            corrected: Background-corrected 3D image
            z_slice: Z-slice to display (defaults to middle)
            channel_name: Channel name for title
            method: Method name for title
            figsize: Figure size
            show_histogram: Whether to include histogram comparison
            vmax_percentile: Percentile for intensity scaling
            
        Returns:
            Matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        if z_slice is None:
            z_slice = original.shape[0] // 2
        
        orig_slice = original[z_slice]
        corr_slice = corrected[z_slice]
        
        # Compute display range
        vmax_orig = np.percentile(orig_slice, vmax_percentile)
        vmax_corr = np.percentile(corr_slice[corr_slice > 0], vmax_percentile) if (corr_slice > 0).any() else 1
        
        if show_histogram:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(figsize[0], figsize[1] // 2))
            axes = axes.reshape(1, -1)
        
        # Original image
        im0 = axes[0, 0].imshow(orig_slice, cmap='gray', vmin=0, vmax=vmax_orig)
        axes[0, 0].set_title(f'Original (z={z_slice})')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
        
        # Corrected image
        im1 = axes[0, 1].imshow(corr_slice, cmap='gray', vmin=0, vmax=vmax_corr)
        axes[0, 1].set_title(f'Corrected ({method})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Difference (background removed)
        diff = orig_slice.astype(np.float32) - corr_slice.astype(np.float32)
        im2 = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-vmax_orig/4, vmax=vmax_orig/4)
        axes[0, 2].set_title('Background Removed')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        if show_histogram:
            # Histogram of original
            axes[1, 0].hist(orig_slice.ravel(), bins=256, color='blue', alpha=0.7, log=True)
            axes[1, 0].set_title('Original Histogram')
            axes[1, 0].set_xlabel('Intensity')
            axes[1, 0].set_ylabel('Count (log)')
            
            # Histogram of corrected
            corr_valid = corr_slice[corr_slice > 0]
            if corr_valid.size > 0:
                axes[1, 1].hist(corr_valid.ravel(), bins=256, color='green', alpha=0.7, log=True)
            axes[1, 1].set_title('Corrected Histogram')
            axes[1, 1].set_xlabel('Intensity')
            axes[1, 1].set_ylabel('Count (log)')
            
            # Overlay histogram
            axes[1, 2].hist(orig_slice.ravel(), bins=256, color='blue', alpha=0.5, 
                          label='Original', log=True)
            if corr_valid.size > 0:
                axes[1, 2].hist(corr_valid.ravel(), bins=256, color='green', alpha=0.5,
                              label='Corrected', log=True)
            axes[1, 2].set_title('Overlay Comparison')
            axes[1, 2].set_xlabel('Intensity')
            axes[1, 2].legend()
        
        fig.suptitle(f'Background Subtraction: {channel_name}', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_multi_slice(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        num_slices: int = 5,
        channel_name: str = "Unknown",
        method: str = "Unknown",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> "Figure":
        """
        Create a multi-slice comparison showing several Z-slices.
        
        Args:
            original: Original 3D image
            corrected: Background-corrected 3D image
            num_slices: Number of Z-slices to display
            channel_name: Channel name for title
            method: Method name for title
            figsize: Figure size (computed automatically if None)
            
        Returns:
            Matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        z_total = original.shape[0]
        z_indices = np.linspace(0, z_total - 1, num_slices, dtype=int)
        
        if figsize is None:
            figsize = (4 * num_slices, 8)
        
        fig, axes = plt.subplots(2, num_slices, figsize=figsize)
        
        # Compute global display ranges
        vmax_orig = np.percentile(original, 99.5)
        corr_positive = corrected[corrected > 0]
        vmax_corr = np.percentile(corr_positive, 99.5) if corr_positive.size > 0 else 1
        
        for i, z_idx in enumerate(z_indices):
            # Original
            axes[0, i].imshow(original[z_idx], cmap='gray', vmin=0, vmax=vmax_orig)
            axes[0, i].set_title(f'z={z_idx}')
            axes[0, i].axis('off')
            
            # Corrected
            axes[1, i].imshow(corrected[z_idx], cmap='gray', vmin=0, vmax=vmax_corr)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('Original', fontsize=12)
        axes[1, 0].set_ylabel('Corrected', fontsize=12)
        
        fig.suptitle(f'{channel_name} - {method}', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def plot_parameter_comparison(
        self,
        original: np.ndarray,
        results: List[Dict[str, Any]],
        z_slice: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> "Figure":
        """
        Create a comparison grid of different parameter settings.
        
        Args:
            original: Original 3D image
            results: List of dicts with 'params', 'corrected', 'score' keys
            z_slice: Z-slice to display
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        if z_slice is None:
            z_slice = original.shape[0] // 2
        
        n_results = len(results)
        n_cols = min(4, n_results + 1)  # +1 for original
        n_rows = (n_results + n_cols) // n_cols
        
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Original
        orig_slice = original[z_slice]
        vmax = np.percentile(orig_slice, 99.5)
        
        axes[0].imshow(orig_slice, cmap='gray', vmin=0, vmax=vmax)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Results
        for i, result in enumerate(results, start=1):
            if i >= len(axes):
                break
            
            corrected = result.get('corrected')
            params = result.get('params', {})
            score = result.get('score', 0)
            
            if corrected is not None:
                corr_slice = corrected[z_slice]
                corr_vmax = np.percentile(corr_slice[corr_slice > 0], 99.5) if (corr_slice > 0).any() else 1
                axes[i].imshow(corr_slice, cmap='gray', vmin=0, vmax=corr_vmax)
            
            # Create parameter label
            param_str = ', '.join(f'{k}={v}' for k, v in list(params.items())[:3])
            axes[i].set_title(f'Score: {score:.3f}\n{param_str}', fontsize=9)
            axes[i].axis('off')
        
        # Hide unused axes
        for i in range(len(results) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_intensity_profile(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        axis: int = 0,
        position: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 4),
    ) -> "Figure":
        """
        Plot intensity profiles along an axis for original vs corrected.
        
        Args:
            original: Original 3D image
            corrected: Corrected 3D image
            axis: Axis along which to take profile (0=Z, 1=Y, 2=X)
            position: Position along other axes (defaults to center)
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        # Get center position if not specified
        if position is None:
            other_axes = [i for i in range(3) if i != axis]
            position = tuple(original.shape[i] // 2 for i in other_axes)
        
        # Extract profiles
        if axis == 0:  # Z profile
            orig_profile = original[:, position[0], position[1]]
            corr_profile = corrected[:, position[0], position[1]]
            xlabel = 'Z slice'
        elif axis == 1:  # Y profile
            orig_profile = original[position[0], :, position[1]]
            corr_profile = corrected[position[0], :, position[1]]
            xlabel = 'Y position'
        else:  # X profile
            orig_profile = original[position[0], position[1], :]
            corr_profile = corrected[position[0], position[1], :]
            xlabel = 'X position'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(orig_profile, label='Original', alpha=0.7)
        ax.plot(corr_profile, label='Corrected', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Intensity')
        ax.set_title(f'Intensity Profile along axis {axis}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_summary_report(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        metadata: Dict[str, Any],
        channel_name: str = "Unknown",
        figsize: Tuple[int, int] = (16, 12),
    ) -> "Figure":
        """
        Create a comprehensive summary report figure.
        
        Args:
            original: Original 3D image
            corrected: Corrected 3D image
            metadata: Background subtraction metadata
            channel_name: Channel name
            figsize: Figure size
            
        Returns:
            Matplotlib Figure with comprehensive summary
        """
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=figsize)
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        z_mid = original.shape[0] // 2
        
        # Top row: images
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        vmax = np.percentile(original, 99.5)
        corr_pos = corrected[corrected > 0]
        corr_vmax = np.percentile(corr_pos, 99.5) if corr_pos.size > 0 else 1
        
        ax1.imshow(original[z_mid], cmap='gray', vmin=0, vmax=vmax)
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(corrected[z_mid], cmap='gray', vmin=0, vmax=corr_vmax)
        ax2.set_title('Corrected')
        ax2.axis('off')
        
        diff = original[z_mid].astype(np.float32) - corrected[z_mid].astype(np.float32)
        ax3.imshow(diff, cmap='RdBu_r')
        ax3.set_title('Background')
        ax3.axis('off')
        
        # Z projection
        ax4.imshow(np.max(corrected, axis=0), cmap='gray')
        ax4.set_title('MIP (Corrected)')
        ax4.axis('off')
        
        # Middle row: histograms and profiles
        ax5 = fig.add_subplot(gs[1, 0:2])
        ax5.hist(original.ravel(), bins=256, alpha=0.5, label='Original', log=True)
        if corr_pos.size > 0:
            ax5.hist(corr_pos.ravel(), bins=256, alpha=0.5, label='Corrected', log=True)
        ax5.set_xlabel('Intensity')
        ax5.set_ylabel('Count (log)')
        ax5.set_title('Intensity Distribution')
        ax5.legend()
        
        ax6 = fig.add_subplot(gs[1, 2:4])
        z_profile_orig = np.mean(original, axis=(1, 2))
        z_profile_corr = np.mean(corrected, axis=(1, 2))
        ax6.plot(z_profile_orig, label='Original')
        ax6.plot(z_profile_corr, label='Corrected')
        ax6.set_xlabel('Z slice')
        ax6.set_ylabel('Mean Intensity')
        ax6.set_title('Z Profile')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Bottom row: metadata text
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Format metadata as text
        method = metadata.get('background_method', 'Unknown')
        stats = metadata.get('background_stats', {})
        
        text_lines = [
            f"Channel: {channel_name}",
            f"Method: {method}",
            f"Image Shape: {original.shape}",
        ]
        
        if 'mean_background' in stats:
            text_lines.append(f"Mean Background: {stats['mean_background']:.2f}")
        if 'std_background' in stats:
            text_lines.append(f"Std Background: {stats['std_background']:.2f}")
        
        # Add method-specific parameters
        for key in ['radius_pixels', 'sigma', 'size', 'sigma_stage1', 'radius_stage2']:
            if key in metadata:
                text_lines.append(f"{key}: {metadata[key]}")
        
        text = '\n'.join(text_lines)
        ax7.text(0.1, 0.9, text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(f'Background Subtraction Report: {channel_name}', fontsize=14, y=0.98)
        
        return fig
