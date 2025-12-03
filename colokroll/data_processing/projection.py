"""
Maximum Intensity Projection (MIP) and Surface Manifold Extraction (SME) 
creation for z-stack images. Supports different projection methods and 
multi-channel data.

SME implementation based on:
Shihavuddin, ASM and Basu, Sreetama - "Makes a 2D reconstruction which is 
spatially continuous out of a 3D image volume"
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any, List
from pathlib import Path
import logging
import warnings
import json

import numpy as np
import tifffile
from sklearn.cluster import KMeans

from ..core.config import Phase2Config
from .image_loader import ImageLoader
from ..core.utils import ensure_directory


logger = logging.getLogger(__name__)


@dataclass
class SMEResult:
    """Container for Surface Manifold Extraction results.
    
    Attributes:
        projection: The SME composite image (Y, X) or (Y, X, C).
        manifold: Z-map showing optimal z-position for each pixel (Y, X).
        classmap: K-means classification map (Y, X) with 3 classes:
                  1=background, 2=uncertain, 3=foreground.
        initial_zmap: Initial MIP-based z-positions before optimization (Y, X).
        cost_history: List of cost values during optimization iterations.
        parameters: Dictionary containing algorithm parameters (C1, C2, C3, lambda1).
    """
    projection: np.ndarray
    manifold: np.ndarray
    classmap: np.ndarray
    initial_zmap: np.ndarray
    cost_history: List[float] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)


class MIPCreator:
    """Create Maximum Intensity Projections from z-stack images."""
    
    def __init__(self, config: Optional[Phase2Config] = None):
        """Initialize the MIP creator.
        
        Args:
            config: Phase 2 configuration. If None, uses defaults.
        """
        self.config = config or Phase2Config()
        self.quality_metrics: Dict[str, Any] = {}
    
    def create_mip(self, 
                   image: np.ndarray,
                   method: Optional[str] = None,
                   z_range: Optional[Tuple[int, int]] = None,
                   channel: Optional[Union[int, str]] = None) -> np.ndarray:
        """Create a Maximum Intensity Projection from a z-stack.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            method: Projection method ('max', 'mean', 'sum', 'median', 'std').
                   If None, uses config default.
            z_range: Optional range of z-slices to use (start, end).
                    If None, uses all slices or config default.
            channel: Optional specific channel to project. If None, projects all.
            
        Returns:
            np.ndarray: Projected image (Y, X, C) or (Y, X).
            
        Raises:
            ValueError: If image dimensions are invalid or method unknown.
        """
        # Validate input
        if image.ndim not in [3, 4]:
            raise ValueError(f"Image must be 3D or 4D z-stack, got {image.ndim}D")
        
        if image.ndim == 3:
            # Add channel dimension if missing
            image = image[..., np.newaxis]
            single_channel = True
        else:
            single_channel = False
        
        # Use config defaults if not specified
        method = method or self.config.projection_method
        z_range = z_range or self.config.z_range
        
        # Apply z-range if specified
        if z_range is not None:
            start, end = z_range
            if start < 0 or end > image.shape[0]:
                raise ValueError(
                    f"Invalid z_range {z_range}. Valid range: (0, {image.shape[0]})"
                )
            image = image[start:end]
            logger.info(f"Using z-slices {start} to {end}")
        
        # Extract specific channel if requested
        if channel is not None:
            if isinstance(channel, str):
                raise ValueError("Channel names require ImageLoader context")
            if channel < 0 or channel >= image.shape[-1]:
                raise ValueError(f"Invalid channel {channel}. Valid range: 0-{image.shape[-1]-1}")
            image = image[..., channel:channel+1]
        
        # Perform projection
        logger.info(f"Creating {method} projection from shape {image.shape}")
        
        if method == 'max':
            projection = np.max(image, axis=0)
        elif method == 'mean':
            projection = np.mean(image, axis=0)
        elif method == 'sum':
            projection = np.sum(image, axis=0)
        elif method == 'median':
            projection = np.median(image, axis=0)
        elif method == 'std':
            projection = np.std(image, axis=0)
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
        # Remove channel dimension if single channel
        if single_channel and projection.shape[-1] == 1:
            projection = projection[..., 0]
        
        # Calculate quality metrics if requested
        if self.config.calculate_quality_metrics:
            self._calculate_quality_metrics(image, projection, method)
        
        logger.info(f"Created {method} projection with shape {projection.shape}")
        
        return projection
    
    def create_color_mip(self,
                         image: np.ndarray,
                         channel_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                         method: str = 'max') -> np.ndarray:
        """Create a color MIP with specified colors for each channel.
        
        Args:
            image: Input z-stack image (Z, Y, X, C).
            channel_colors: Dict mapping channel index to RGB color (0-1).
                          If None, uses default colors.
            method: Projection method.
            
        Returns:
            np.ndarray: Color MIP image (Y, X, 3) in RGB format.
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        # Create MIP for each channel
        mip = self.create_mip(image, method=method)
        
        if mip.ndim == 2:
            mip = mip[..., np.newaxis]
        
        # Setup colors
        if channel_colors is None:
            # Default colors: blue, green, red, cyan, magenta, yellow
            default_colors = [
                (0, 0, 1),  # Blue
                (0, 1, 0),  # Green
                (1, 0, 0),  # Red
                (0, 1, 1),  # Cyan
                (1, 0, 1),  # Magenta
                (1, 1, 0),  # Yellow
            ]
            channel_colors = {i: default_colors[i % len(default_colors)] 
                            for i in range(mip.shape[-1])}
        
        # Create color image
        height, width, n_channels = mip.shape
        color_image = np.zeros((height, width, 3), dtype=np.float32)
        
        for c in range(n_channels):
            if c in channel_colors:
                color = channel_colors[c]
            else:
                # Default to grayscale for unmapped channels
                color = (1, 1, 1)
            
            # Normalize channel
            channel_data = mip[..., c].astype(np.float32)
            if channel_data.max() > 0:
                channel_data = channel_data / channel_data.max()
            
            # Apply color
            for i, color_val in enumerate(color):
                color_image[..., i] += channel_data * color_val
        
        # Clip to valid range
        color_image = np.clip(color_image, 0, 1)
        
        return color_image
    
    def create_multi_method_mip(self,
                                image: np.ndarray,
                                methods: List[str] = ['max', 'mean', 'std']) -> Dict[str, np.ndarray]:
        """Create multiple projections using different methods.
        
        Args:
            image: Input z-stack image.
            methods: List of projection methods to apply.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping method names to projections.
        """
        projections = {}
        
        for method in methods:
            try:
                projections[method] = self.create_mip(image, method=method)
            except Exception as e:
                logger.warning(f"Failed to create {method} projection: {e}")
                continue
        
        return projections
    
    def _calculate_quality_metrics(self,
                                   z_stack: np.ndarray,
                                   projection: np.ndarray,
                                   method: str) -> None:
        """Calculate quality metrics for the projection.
        
        Args:
            z_stack: Original z-stack.
            projection: Created projection.
            method: Projection method used.
        """
        metrics = {}
        
        # Basic statistics
        metrics['method'] = method
        metrics['z_slices'] = z_stack.shape[0]
        metrics['shape'] = projection.shape
        
        # Intensity statistics
        metrics['min_intensity'] = float(projection.min())
        metrics['max_intensity'] = float(projection.max())
        metrics['mean_intensity'] = float(projection.mean())
        metrics['std_intensity'] = float(projection.std())
        
        # Dynamic range
        if projection.max() > projection.min():
            metrics['dynamic_range'] = float(
                (projection.max() - projection.min()) / projection.max()
            )
        else:
            metrics['dynamic_range'] = 0.0
        
        # Signal-to-noise ratio estimate
        if projection.std() > 0:
            metrics['snr_estimate'] = float(projection.mean() / projection.std())
        else:
            metrics['snr_estimate'] = 0.0
        
        # Focus quality score (higher std along z indicates better focus variation)
        z_std = np.std(z_stack, axis=0)
        metrics['focus_variation'] = float(z_std.mean())
        
        # Coverage (fraction of non-zero pixels)
        metrics['coverage'] = float(np.sum(projection > 0) / projection.size)
        
        # Information content (entropy-based)
        if projection.max() > 0:
            normalized = projection / projection.max()
            hist, _ = np.histogram(normalized.flatten(), bins=256)
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist = hist / hist.sum()
                metrics['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            else:
                metrics['entropy'] = 0.0
        else:
            metrics['entropy'] = 0.0
        
        # Overall quality score (0-1)
        quality_score = 0.0
        if metrics['dynamic_range'] > 0.3:
            quality_score += 0.25
        if metrics['snr_estimate'] > 5:
            quality_score += 0.25
        if metrics['coverage'] > 0.1:
            quality_score += 0.25
        if metrics['entropy'] > 2:
            quality_score += 0.25
        
        metrics['quality_score'] = quality_score
        
        # Store metrics
        self.quality_metrics = metrics
        
        if quality_score < self.config.quality_threshold:
            warnings.warn(
                f"Low quality projection (score: {quality_score:.2f}). "
                f"Consider adjusting parameters or checking input data.",
                UserWarning,
                stacklevel=2
            )
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get the quality metrics from the last projection.
        
        Returns:
            Dict[str, Any]: Quality metrics dictionary.
        """
        return self.quality_metrics.copy()
    
    def optimize_z_range(self,
                         image: np.ndarray,
                         metric: str = 'focus',
                         percentile: float = 90) -> Tuple[int, int]:
        """Automatically determine optimal z-range for projection.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            metric: Metric to use ('focus', 'intensity', 'variance').
            percentile: Percentile threshold for selecting slices.
            
        Returns:
            Tuple[int, int]: Optimal z-range (start, end).
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        n_slices = image.shape[0]
        scores = np.zeros(n_slices)
        
        for z in range(n_slices):
            slice_img = image[z]
            
            if metric == 'focus':
                # Use variance of Laplacian as focus measure
                from scipy.ndimage import laplace
                for c in range(slice_img.shape[-1]):
                    lap = laplace(slice_img[..., c])
                    scores[z] += lap.var()
            
            elif metric == 'intensity':
                scores[z] = slice_img.mean()
            
            elif metric == 'variance':
                scores[z] = slice_img.var()
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        # Find range containing high-scoring slices
        threshold = np.percentile(scores, 100 - percentile)
        good_slices = np.where(scores >= threshold)[0]
        
        if len(good_slices) > 0:
            z_start = good_slices.min()
            z_end = good_slices.max() + 1
        else:
            z_start = 0
            z_end = n_slices
        
        logger.info(f"Optimized z-range: {z_start}-{z_end} (using {metric} metric)")
        
        return (z_start, z_end)
    
    def create_depth_coded_mip(self,
                               image: np.ndarray,
                               colormap: str = 'viridis') -> Tuple[np.ndarray, np.ndarray]:
        """Create a depth-coded MIP where color represents z-position.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            colormap: Matplotlib colormap name.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Depth-coded color image (Y, X, 3)
                - Depth map (Y, X) with z-indices
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        # Find maximum intensity position for each pixel
        if image.shape[-1] == 1:
            # Single channel
            max_z_indices = np.argmax(image[..., 0], axis=0)
            max_intensities = np.max(image[..., 0], axis=0)
        else:
            # Multi-channel: use mean across channels
            mean_image = np.mean(image, axis=-1)
            max_z_indices = np.argmax(mean_image, axis=0)
            max_intensities = np.max(mean_image, axis=0)
        
        # Normalize depth to 0-1
        n_slices = image.shape[0]
        depth_normalized = max_z_indices.astype(np.float32) / (n_slices - 1)
        
        # Apply colormap
        import matplotlib.pyplot as plt
        cmap = plt.colormaps.get_cmap(colormap)
        depth_colored = cmap(depth_normalized)[..., :3]  # RGB only
        
        # Weight by intensity
        max_intensities_norm = max_intensities / (max_intensities.max() + 1e-10)
        depth_colored = depth_colored * max_intensities_norm[..., np.newaxis]
        
        return depth_colored.astype(np.float32), max_z_indices
    
    def create_extended_depth_of_field(self,
                                       image: np.ndarray,
                                       window_size: int = 5) -> np.ndarray:
        """Create an extended depth of field projection using local focus measures.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            window_size: Window size for local focus calculation.
            
        Returns:
            np.ndarray: Extended depth of field image.
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        
        from scipy.ndimage import generic_filter
        
        n_slices, height, width, n_channels = image.shape
        result = np.zeros((height, width, n_channels))
        
        for c in range(n_channels):
            # Calculate focus measure for each slice
            focus_stack = np.zeros((n_slices, height, width))
            
            for z in range(n_slices):
                # Use variance as focus measure
                focus_stack[z] = generic_filter(
                    image[z, ..., c],
                    np.var,
                    size=window_size
                )
            
            # Select pixel from slice with best focus
            best_focus_indices = np.argmax(focus_stack, axis=0)
            
            for y in range(height):
                for x in range(width):
                    best_z = best_focus_indices[y, x]
                    result[y, x, c] = image[best_z, y, x, c]
        
        if result.shape[-1] == 1:
            result = result[..., 0]
        
        return result

    def save_mip_tiff(self,
                      mip: np.ndarray,
                      filepath: Union[str, Path],
                      dtype: str = 'auto',
                      normalize: bool = True,
                      compress: Union[int, None] = 6,
                      metadata: Optional[Dict[str, Any]] = None,
                      overwrite: bool = False) -> Path:
        """Save a MIP image to a TIFF file.
        
        Args:
            mip: MIP image array (Y, X) or (Y, X, C).
            filepath: Output TIFF path.
            dtype: Output dtype: 'auto', 'uint8', 'uint16', or 'float32'.
            normalize: If True, scale intensities to full range for integer dtypes.
            compress: Deflate compression level (0-9) or None for no compression.
            metadata: Optional metadata dict stored in TIFF tags where supported.
            overwrite: If False, raise if file exists.
        
        Returns:
            Path: The written file path.
        
        Raises:
            ValueError: If image shape is invalid or dtype unsupported.
            FileExistsError: If file exists and overwrite is False.
        """
        # Validate input dimensions
        if mip.ndim not in (2, 3):
            raise ValueError(f"MIP must be 2D or 3D (Y, X[, C]), got shape {mip.shape}")

        out_path = Path(filepath)
        ensure_directory(out_path.parent)

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {out_path}")

        # Determine target dtype
        target_dtype: str
        if dtype == 'auto':
            if np.issubdtype(mip.dtype, np.floating):
                target_dtype = 'uint16'
            elif np.issubdtype(mip.dtype, np.integer):
                target_dtype = 'uint8' if np.max(mip) <= 255 else 'uint16'
            else:
                target_dtype = 'uint16'
        else:
            target_dtype = dtype

        # Prepare array for saving
        array_to_save: np.ndarray
        if target_dtype in ('uint8', 'uint16'):
            # Convert to float for scaling if needed
            arr = mip.astype(np.float32)
            if normalize:
                min_val = float(arr.min())
                max_val = float(arr.max())
                if max_val > min_val:
                    arr = (arr - min_val) / (max_val - min_val)
                else:
                    arr = np.zeros_like(arr)
            else:
                # If values are outside [0,1] and we're writing integer, clip sensibly
                arr = np.clip(arr, 0, 1) if np.issubdtype(mip.dtype, np.floating) else arr

            if target_dtype == 'uint8':
                array_to_save = (arr * 255.0 + 0.5).astype(np.uint8)
            else:
                array_to_save = (arr * 65535.0 + 0.5).astype(np.uint16)
        elif target_dtype == 'float32':
            arr = mip.astype(np.float32)
            if normalize:
                min_val = float(arr.min())
                max_val = float(arr.max())
                if max_val > min_val:
                    arr = (arr - min_val) / (max_val - min_val)
            array_to_save = arr
        else:
            raise ValueError(f"Unsupported dtype: {target_dtype}")

        # Save using tifffile - keep original (Y,X,C) format for multichannel
        if array_to_save.ndim == 3:
            # Save as OME-TIFF for proper multichannel handling
            # Keep original (Y, X, C) format - don't transpose
            tifffile.imwrite(
                str(out_path),
                array_to_save,
                ome=True,
                metadata={
                    'axes': 'YXC',
                    'Channel': {'Name': [f'Channel_{i}' for i in range(array_to_save.shape[-1])]},
                    **({k: v for k, v in (metadata or {}).items() if k != 'axes'})
                },
                compression=('deflate' if compress is not None else None),
                compressionargs=({'level': int(compress)} if compress is not None else None)
            )
        else:
            # Single channel image
            tifffile.imwrite(
                str(out_path),
                array_to_save,
                photometric='minisblack',
                compression=('deflate' if compress is not None else None),
                compressionargs=({'level': int(compress)} if compress is not None else None),
                description=json.dumps(metadata or {})
            )

        logger.info(f"Saved MIP to {out_path} ({array_to_save.dtype}, shape={array_to_save.shape})")
        return out_path

    # =========================================================================
    # Surface Manifold Extraction (SME) Methods
    # =========================================================================
    
    def create_sme(self,
                   image: np.ndarray,
                   reference_channel: Optional[int] = None,
                   layers_above: int = 0,
                   layers_below: int = 0,
                   convergence_threshold: float = 1e-6,
                   max_iterations: int = 1000,
                   n_classes: int = 3) -> SMEResult:
        """Create a Surface Manifold Extraction projection from a z-stack.
        
        SME creates a spatially continuous 2D reconstruction by computing
        an optimal z-manifold that balances signal fidelity with spatial
        smoothness, rather than pixel-wise max projection.
        
        Args:
            image: Input z-stack image (Z, Y, X, C) or (Z, Y, X).
            reference_channel: Channel to use for manifold computation.
                             If None and multi-channel, uses channel 0.
            layers_above: Number of z-layers above manifold to include in projection.
            layers_below: Number of z-layers below manifold to include in projection.
            convergence_threshold: Cost change threshold for stopping optimization.
            max_iterations: Maximum number of optimization iterations.
            n_classes: Number of k-means classes (default 3: background, uncertain, foreground).
            
        Returns:
            SMEResult: Container with projection, manifold, classmap, and metrics.
            
        Raises:
            ValueError: If image dimensions are invalid.
        """
        # Validate input
        if image.ndim not in [3, 4]:
            raise ValueError(f"Image must be 3D or 4D z-stack, got {image.ndim}D")
        
        if image.ndim == 3:
            image = image[..., np.newaxis]
            single_channel = True
        else:
            single_channel = False
        
        n_slices, height, width, n_channels = image.shape
        
        # Select reference channel for manifold computation
        if reference_channel is None:
            reference_channel = 0
        
        if reference_channel < 0 or reference_channel >= n_channels:
            raise ValueError(
                f"Invalid reference_channel {reference_channel}. "
                f"Valid range: 0-{n_channels-1}"
            )
        
        logger.info(
            f"Creating SME projection from shape {image.shape}, "
            f"reference channel: {reference_channel}"
        )
        
        # Extract reference channel for manifold computation
        ref_image = image[..., reference_channel].astype(np.float64)
        
        # Step 1: Classify pixels using FFT + K-means
        classmap, edgeflag2, edgeflag3k = self._classify_pixels_fft_kmeans(
            ref_image, n_classes=n_classes
        )
        
        # Step 2: Get initial z-map from max intensity positions
        max_values, initial_zmap = np.max(ref_image, axis=0), np.argmax(ref_image, axis=0)
        initial_zmap = initial_zmap.astype(np.float64)
        
        # Step 3: Compute lambda parameters
        C1, C2, C3, lambda1 = self._compute_sme_lambda_parameters(
            ref_image, max_values, initial_zmap, edgeflag2
        )
        
        # Update weight map with computed parameters
        edgeflag3k[edgeflag2 == 1.0] = C1
        edgeflag3k[edgeflag2 == 0.5] = C2
        edgeflag3k[edgeflag2 == 0.0] = C3
        
        # Step 4: Optimize manifold
        manifold, cost_history = self._optimize_sme_manifold(
            ref_image, initial_zmap.copy(), edgeflag2, edgeflag3k,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations
        )
        
        # Step 5: Create projection from manifold
        if single_channel:
            projection = self._project_from_manifold(
                ref_image, manifold, layers_above, layers_below
            )
        else:
            # Apply manifold to all channels
            projection = np.zeros((height, width, n_channels), dtype=image.dtype)
            for c in range(n_channels):
                projection[..., c] = self._project_from_manifold(
                    image[..., c], manifold, layers_above, layers_below
                )
        
        # Remove channel dimension if single channel
        if single_channel:
            projection = projection
        elif projection.shape[-1] == 1:
            projection = projection[..., 0]
        
        final_cost_str = f"{cost_history[-1]:.6f}" if cost_history else "N/A"
        logger.info(
            f"SME completed: {len(cost_history)} iterations, "
            f"final cost: {final_cost_str}"
        )
        
        return SMEResult(
            projection=projection,
            manifold=manifold,
            classmap=classmap,
            initial_zmap=initial_zmap.astype(np.int32),
            cost_history=cost_history,
            parameters={'C1': C1, 'C2': C2, 'C3': C3, 'lambda1': lambda1}
        )
    
    def create_multichannel_sme(self,
                                image: np.ndarray,
                                reference_channel: int = 0,
                                layers_above: int = 0,
                                layers_below: int = 0,
                                convergence_threshold: float = 1e-6,
                                max_iterations: int = 1000) -> Tuple[np.ndarray, SMEResult]:
        """Create SME projection for multi-channel image.
        
        Computes the manifold using the reference channel, then applies
        it to all channels to preserve spatial coherence.
        
        Args:
            image: Input z-stack image (Z, Y, X, C).
            reference_channel: Channel index used for manifold computation.
            layers_above: Number of z-layers above manifold to include.
            layers_below: Number of z-layers below manifold to include.
            convergence_threshold: Cost change threshold for stopping.
            max_iterations: Maximum optimization iterations.
            
        Returns:
            Tuple containing:
                - Multi-channel projection array (Y, X, C)
                - SMEResult with manifold computed from reference channel
        """
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image (Z, Y, X, C), got {image.ndim}D")
        
        n_slices, height, width, n_channels = image.shape
        
        # Compute SME using reference channel
        result = self.create_sme(
            image,
            reference_channel=reference_channel,
            layers_above=layers_above,
            layers_below=layers_below,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations
        )
        
        # Apply manifold to all channels
        multichannel_projection = np.zeros((height, width, n_channels), dtype=image.dtype)
        
        for c in range(n_channels):
            multichannel_projection[..., c] = self._project_from_manifold(
                image[..., c], result.manifold, layers_above, layers_below
            )
        
        # Update the result with full multichannel projection
        result.projection = multichannel_projection
        
        return multichannel_projection, result
    
    def _classify_pixels_fft_kmeans(self,
                                    image: np.ndarray,
                                    n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify pixels using FFT of z-profiles and K-means clustering.
        
        Analyzes the frequency content of each pixel's z-profile to classify
        pixels into background, uncertain, and foreground classes.
        
        Args:
            image: Single-channel z-stack (Z, Y, X).
            n_classes: Number of k-means clusters (default 3).
            
        Returns:
            Tuple containing:
                - classmap: Classification map (Y, X) with values 1, 2, 3
                - edgeflag2: Normalized class weights (Y, X) with values 0, 0.5, 1
                - edgeflag3k: Weight map for cost function (Y, X)
        """
        n_slices, height, width = image.shape
        n_pixels = height * width
        
        # Reshape to (n_pixels, n_slices) for z-profile analysis
        z_profiles = image.reshape(n_slices, n_pixels).T.astype(np.float64)
        
        # Compute FFT of z-profiles
        fft_profiles = np.abs(np.fft.fft(z_profiles, n=n_slices, axis=1))
        
        # Keep only first half of FFT (excluding DC and mirror)
        half_point = n_slices // 2
        fft_profiles = fft_profiles[:, 1:half_point + 1]
        
        # Normalize FFT features
        fft_range = fft_profiles.max(axis=0) - fft_profiles.min(axis=0)
        fft_range[fft_range == 0] = 1  # Avoid division by zero
        fft_normalized = fft_profiles / fft_range
        
        # Handle NaN values
        fft_normalized = np.nan_to_num(fft_normalized, nan=0.0)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(fft_normalized)
        
        # Sort clusters by sum of centroids (background has lowest, foreground highest)
        centroid_sums = kmeans.cluster_centers_.sum(axis=1)
        sorted_indices = np.argsort(centroid_sums)
        
        # Remap labels so that 0=background, 1=uncertain, 2=foreground
        label_mapping = {old: new for new, old in enumerate(sorted_indices)}
        remapped_labels = np.array([label_mapping[l] for l in labels])
        
        # Reshape to image dimensions (labels are 0, 1, 2)
        classmap = remapped_labels.reshape(height, width) + 1  # 1, 2, 3
        
        # Create normalized edge flag (0=background, 0.5=uncertain, 1=foreground)
        edgeflag2 = (classmap - 1) / 2.0  # 0, 0.5, 1
        
        # Initialize weight map (will be updated with C1, C2, C3 later)
        edgeflag3k = edgeflag2.copy()
        
        logger.debug(
            f"K-means classification: "
            f"bg={np.sum(classmap == 1)}, "
            f"uncertain={np.sum(classmap == 2)}, "
            f"fg={np.sum(classmap == 3)}"
        )
        
        return classmap, edgeflag2, edgeflag3k
    
    def _compute_sme_lambda_parameters(self,
                                       image: np.ndarray,
                                       max_values: np.ndarray,
                                       initial_zmap: np.ndarray,
                                       edgeflag2: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute adaptive lambda parameters for SME cost function.
        
        Analyzes the overlap between intensity distributions of different
        classes to determine appropriate weighting parameters.
        
        Args:
            image: Single-channel z-stack (Z, Y, X).
            max_values: Maximum intensity values at each pixel (Y, X).
            initial_zmap: Initial z-map from argmax (Y, X).
            edgeflag2: Normalized class weights (Y, X).
            
        Returns:
            Tuple of (C1, C2, C3, lambda1) parameters.
        """
        n_slices = image.shape[0]
        
        # Get intensity distributions for foreground and uncertain classes
        fg_mask = edgeflag2 == 1.0
        uncertain_mask = edgeflag2 == 0.5
        bg_mask = edgeflag2 == 0.0
        
        fg_values = max_values[fg_mask]
        uncertain_values = max_values[uncertain_mask]
        bg_values = max_values[bg_mask]
        
        # Compute histograms to find overlap
        hist_range = (max_values.min(), max_values.max())
        n_bins = 100
        
        if len(fg_values) > 0:
            hist_fg, bin_edges = np.histogram(fg_values, bins=n_bins, range=hist_range)
            hist_fg = hist_fg / (hist_fg.sum() + 1e-10)
        else:
            hist_fg = np.zeros(n_bins)
            bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
        
        if len(uncertain_values) > 0:
            hist_uncertain, _ = np.histogram(uncertain_values, bins=n_bins, range=hist_range)
            hist_uncertain = hist_uncertain / (hist_uncertain.sum() + 1e-10)
        else:
            hist_uncertain = np.zeros(n_bins)
        
        # Find threshold where uncertain exceeds foreground
        overlap_indices = np.where(hist_uncertain > hist_fg)[0]
        if len(overlap_indices) > 0:
            threshold_idx = overlap_indices[-1]
            threshold = bin_edges[threshold_idx]
        else:
            threshold = hist_range[1]
        
        # Compute overlap ratio
        if len(fg_values) > 0:
            fg_below_threshold = np.sum(fg_values <= threshold)
            fg_above_threshold = np.sum(fg_values > threshold)
            if fg_above_threshold > 0:
                overlap2 = fg_below_threshold / fg_above_threshold
            else:
                overlap2 = 1.0
        else:
            overlap2 = 0.5
        
        overlap2 = np.clip(overlap2, 0.0, 1.0)
        
        # Compute neighbor-based statistics for lambda estimation
        padded_edgeflag = np.pad(edgeflag2, 1, mode='symmetric')
        padded_zmap = np.pad(initial_zmap, 1, mode='symmetric')
        
        neighbors = self._get_8_neighbors(padded_zmap)
        neighbor_mean = np.mean(neighbors, axis=2)
        neighbor_var = np.sum((neighbors - neighbor_mean[..., np.newaxis]) ** 2, axis=2)
        
        # Find fully surrounded foreground pixels
        neighbor_flags = self._get_8_neighbors(padded_edgeflag)
        fg_neighbor_count = np.sum(neighbor_flags == 1.0, axis=2)
        
        # Compute statistics for lambda
        valid_mask = (fg_neighbor_count[1:-1, 1:-1] > 7) & (edgeflag2 == 1.0)
        
        if np.sum(valid_mask) > 0:
            z_deviation = np.abs(initial_zmap - neighbor_mean[1:-1, 1:-1])
            M10 = initial_zmap - neighbor_mean[1:-1, 1:-1]
            s01 = np.sqrt(
                (neighbor_var[1:-1, 1:-1] + M10 * (initial_zmap - (neighbor_mean[1:-1, 1:-1] + M10 / 9))) / 8
            )
            
            sgain = s01[valid_mask]
            dg = z_deviation[valid_mask]
            
            # Remove zero gains
            valid_gain = sgain > 0
            if np.sum(valid_gain) > 0:
                WA = dg[valid_gain] / sgain[valid_gain]
                lambda1 = np.abs(np.quantile(WA, overlap2))
            else:
                lambda1 = 1.0
        else:
            lambda1 = 1.0
        
        # Ensure lambda1 is reasonable
        lambda1 = max(lambda1, 0.01)
        
        # Compute mean intensities per class
        mean_fg = np.mean(fg_values) if len(fg_values) > 0 else 1.0
        mean_uncertain = np.mean(uncertain_values) if len(uncertain_values) > 0 else 0.5
        mean_bg = np.mean(bg_values) if len(bg_values) > 0 else 0.0
        
        # Compute relative intensity ratio
        if (mean_fg - mean_bg) > 0:
            RT = (mean_uncertain - mean_bg) / (mean_fg - mean_bg)
        else:
            RT = 0.5
        
        # Compute C1, C2, C3 weights
        CD = 1.0
        C1 = CD / lambda1
        C2 = CD * RT / lambda1
        C3 = 0.0
        
        logger.debug(
            f"SME parameters: lambda1={lambda1:.4f}, "
            f"C1={C1:.4f}, C2={C2:.4f}, C3={C3:.4f}, overlap={overlap2:.4f}"
        )
        
        return C1, C2, C3, lambda1
    
    def _get_8_neighbors(self, image: np.ndarray) -> np.ndarray:
        """Extract 8-connected neighborhood values for each pixel.
        
        Args:
            image: 2D array (Y, X).
            
        Returns:
            3D array (Y, X, 8) containing neighbor values for each pixel.
        """
        # Use np.roll for efficient neighbor extraction
        neighbors = np.zeros((*image.shape, 8), dtype=image.dtype)
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        shifts = [
            (-1, 0),   # N
            (-1, 1),   # NE
            (0, 1),    # E
            (1, 1),    # SE
            (1, 0),    # S
            (1, -1),   # SW
            (0, -1),   # W
            (-1, -1),  # NW
        ]
        
        for i, (dy, dx) in enumerate(shifts):
            neighbors[..., i] = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)
        
        return neighbors
    
    def _optimize_sme_manifold(self,
                               image: np.ndarray,
                               z_map: np.ndarray,
                               edgeflag2: np.ndarray,
                               edgeflag3k: np.ndarray,
                               convergence_threshold: float = 1e-6,
                               max_iterations: int = 1000) -> Tuple[np.ndarray, List[float]]:
        """Optimize the z-manifold using iterative cost minimization.
        
        Minimizes a cost function that balances data fidelity (staying close
        to initial max positions) with spatial smoothness (local neighborhood
        consistency).
        
        Args:
            image: Single-channel z-stack (Z, Y, X).
            z_map: Initial z-map to optimize (Y, X).
            edgeflag2: Normalized class weights (Y, X).
            edgeflag3k: Cost weighting map (Y, X).
            convergence_threshold: Stop when cost change is below this.
            max_iterations: Maximum number of iterations.
            
        Returns:
            Tuple containing:
                - Optimized z-map (Y, X)
                - List of cost values per iteration
        """
        n_slices, height, width = image.shape
        n_pixels = height * width
        
        # Get initial z-map (for data term)
        initial_zmap = np.argmax(image, axis=0).astype(np.float64)
        
        # Compute initial step size based on z-range in foreground
        fg_mask = edgeflag2 > 0
        if np.sum(fg_mask) > 0:
            KE = np.max(initial_zmap[fg_mask]) - np.min(initial_zmap[fg_mask]) + 1
        else:
            KE = n_slices
        
        step = KE / 100.0
        WW = 1.0  # Smoothness weight
        
        # Initialize optimization
        cost_history = []
        z_current = z_map.astype(np.float64)
        
        # Initial costs to guarantee entering loop at least once
        prev_cost = float('inf')
        current_cost = 0.0  # Finite value so abs(inf - 0) = inf > threshold
        
        iteration = 0
        
        while abs(prev_cost - current_cost) > (convergence_threshold * KE) and iteration < max_iterations:
            iteration += 1
            prev_cost = current_cost
            
            # Compute neighbor statistics
            padded_z = np.pad(z_current, 1, mode='symmetric')
            neighbors = self._get_8_neighbors(padded_z)
            neighbor_mean = np.mean(neighbors, axis=2)[1:-1, 1:-1]
            neighbor_var = np.sum(
                (neighbors - np.mean(neighbors, axis=2, keepdims=True)) ** 2, 
                axis=2
            )[1:-1, 1:-1]
            
            # Candidate z-positions
            z_plus = z_current + step
            z_minus = z_current - step
            
            # Data term: distance from initial z-map, weighted by class
            d0 = np.abs(initial_zmap - z_current) * edgeflag3k
            d1 = np.abs(initial_zmap - z_plus) * edgeflag3k
            d2 = np.abs(initial_zmap - z_minus) * edgeflag3k
            
            # Smoothness term: increase in local variance
            M10 = z_current - neighbor_mean
            M11 = z_plus - neighbor_mean
            M12 = z_minus - neighbor_mean
            
            s0 = WW * np.sqrt(np.maximum(0, (neighbor_var + M10 * (z_current - (neighbor_mean + M10 / 9))) / 8))
            s1 = WW * np.sqrt(np.maximum(0, (neighbor_var + M11 * (z_plus - (neighbor_mean + M11 / 9))) / 8))
            s2 = WW * np.sqrt(np.maximum(0, (neighbor_var + M12 * (z_minus - (neighbor_mean + M12 / 9))) / 8))
            
            # Total cost for each option
            c0 = d0 + s0
            c1 = d1 + s1
            c2 = d2 + s2
            
            # Find minimum cost option
            costs_stack = np.stack([c0, c1, c2], axis=-1)
            min_cost = np.min(costs_stack, axis=-1)
            best_option = np.argmin(costs_stack, axis=-1)
            
            # Apply shifts based on best option
            shift = np.zeros_like(z_current)
            shift[best_option == 1] = step
            shift[best_option == 2] = -step
            
            z_current = z_current + shift
            
            # Clamp to valid range
            z_current = np.clip(z_current, 0, n_slices - 1)
            
            # Compute total cost
            current_cost = np.sum(np.abs(min_cost)) / n_pixels
            cost_history.append(current_cost)
            
            # Decay step size
            step = step * 0.99
            
            if iteration % 100 == 0:
                logger.debug(f"SME iteration {iteration}: cost={current_cost:.6f}")
        
        logger.info(f"SME optimization converged after {iteration} iterations")
        
        return np.round(z_current).astype(np.float64), cost_history
    
    def _project_from_manifold(self,
                               image: np.ndarray,
                               z_map: np.ndarray,
                               layers_above: int = 0,
                               layers_below: int = 0) -> np.ndarray:
        """Create projection by sampling from z-manifold with optional layers.
        
        Args:
            image: Single-channel z-stack (Z, Y, X).
            z_map: Z-map indicating which slice to sample from (Y, X).
            layers_above: Number of additional layers above manifold.
            layers_below: Number of additional layers below manifold.
            
        Returns:
            2D projection image (Y, X).
        """
        n_slices, height, width = image.shape
        z_map_int = np.round(z_map).astype(np.int32)
        
        # Clamp z_map to valid range
        z_map_int = np.clip(z_map_int, 0, n_slices - 1)
        
        if layers_above == 0 and layers_below == 0:
            # Simple case: sample directly from manifold
            projection = np.zeros((height, width), dtype=image.dtype)
            
            for z in range(n_slices):
                mask = z_map_int == z
                if np.any(mask):
                    projection[mask] = image[z][mask]
            
            return projection
        
        else:
            # Multi-layer case: pad stack and take max over layer range
            total_layers = layers_above + layers_below + 1
            
            # Pad the image stack
            padded_image = np.concatenate([
                np.zeros((layers_above, height, width), dtype=image.dtype),
                image,
                np.zeros((layers_below, height, width), dtype=image.dtype)
            ], axis=0)
            
            # Adjust z_map for padding
            z_map_padded = z_map_int + layers_above
            
            # Collect values from each layer
            layer_values = np.zeros((height, width, total_layers), dtype=image.dtype)
            
            for layer_idx in range(total_layers):
                offset = layer_idx - layers_above
                z_indices = np.clip(z_map_padded + offset, 0, padded_image.shape[0] - 1)
                
                for z in range(padded_image.shape[0]):
                    mask = z_indices == z
                    if np.any(mask):
                        layer_values[mask, layer_idx] = padded_image[z][mask]
            
            # Take maximum across layers
            projection = np.max(layer_values, axis=-1)
            
            return projection