"""
Configuration dataclasses for analysis functions.

This module provides typed configuration objects to group related parameters
and reduce the number of function arguments, improving readability and maintainability.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ColocalizationConfig:
    """Configuration for colocalization analysis.
    
    Groups the many parameters of compute_colocalization() into logical sections.
    
    Example:
        >>> config = ColocalizationConfig(
        ...     channel_a=0,
        ...     channel_b=1,
        ...     thresholding="otsu",
        ...     min_area="auto",
        ... )
        >>> result = compute_colocalization(image, mask, config=config)
    """
    
    # Channel selection
    channel_a: Union[int, str] = 0
    channel_b: Union[int, str] = 1
    channel_names: Optional[List[str]] = None
    
    # Thresholding options
    thresholding: str = "none"  # 'none', 'otsu', 'costes', 'fixed'
    fixed_thresholds: Optional[Tuple[float, float]] = None
    threshold_domain: str = "raw"  # enforced to 'raw'
    
    # Filtering options
    min_area: Union[int, str] = "auto"
    min_area_fraction: float = 0.7
    max_border_fraction: Optional[float] = None
    border_margin_px: int = 1
    drop_label_1: bool = True
    
    # Preprocessing options
    bleedthrough_matrix: Optional[List[List[float]]] = None
    background_subtract: Optional[Dict[str, Any]] = None
    
    # Aggregation options
    manders_weighting: str = "voxel"  # 'voxel' or 'slice'
    pearson_winsor_clip: Optional[float] = None
    pearson_min_count: int = 2
    pearson_min_range: float = 1e-12
    
    # Visualization options
    plot_mask: bool = False
    plot_mask_save: Optional[Union[str, Path]] = None
    
    # Output options
    output_json: Optional[Union[str, Path]] = None
    
    # Deprecated (kept for backward compatibility)
    normalization_scope: str = "none"


@dataclass
class PunctaConfig:
    """Configuration for puncta detection and analysis.
    
    Groups the many parameters of compute_puncta() into logical sections.
    
    Example:
        >>> config = PunctaConfig(
        ...     channel=2,
        ...     detection_method="bigfish",
        ...     pixel_size_um=0.1,
        ... )
        >>> result = compute_puncta(image, mask, config=config)
    """
    
    # Channel selection
    channel: Union[int, str] = 0
    channel_names: Optional[List[str]] = None
    
    # Projection options
    projection: str = "mip"  # 'mip', 'sme', 'none'
    
    # Detection options
    detection_method: str = "log"  # 'log', 'bigfish'
    pixel_size_um: Optional[float] = None
    
    # Size parameters (in µm)
    expected_diameter_um: float = 0.5
    min_diameter_um: float = 0.2
    max_diameter_um: float = 2.0
    
    # Detection thresholds
    snr_threshold: float = 3.0
    min_distance_um: Optional[float] = None
    
    # Size filtering (in pixels)
    min_area_px: int = 4
    max_area_px: Optional[int] = None
    
    # Mask options
    drop_label_1: bool = True
    
    # Output options
    return_threshold_data: bool = False
    output_json: Optional[Union[str, Path]] = None


@dataclass
class ZSliceSelectionConfig:
    """Configuration for Z-slice selection.
    
    Groups the many parameters of select_z_slices() into logical sections.
    
    Example:
        >>> config = ZSliceSelectionConfig(
        ...     method="combined",
        ...     strategy="relative",
        ...     threshold=0.6,
        ... )
        >>> result = select_z_slices(image, config=config)
    """
    
    # Method selection
    method: str = "combined"  # 'laplacian', 'tenengrad', 'fft', 'combined'
    aggregation: str = "median"  # 'median', 'mean', 'max', 'min', 'weighted'
    
    # Detection strategy
    strategy: str = "relative"  # 'relative', 'percentile', 'topk', 'closest_to_peak'
    threshold: float = 0.6
    keep_top: Optional[int] = None
    auto_keep_fraction: float = 0.8
    
    # Smoothing
    smooth: int = 3
    
    # Normalization
    normalize: bool = True
    normalization_mode: str = "per_slice"  # 'per_slice', 'per_stack', 'percentile'
    clip_percent: float = 0.0
    
    # Advanced options
    border_width: int = 0
    gaussian_sigma: float = 0.0
    min_variance_threshold: float = 0.0
    fft_cutoff: float = 0.15
    weights: Optional[List[float]] = None
    
    # Quality metrics
    compute_quality: bool = False
    step_distance: float = 1.0
    n_fitting_points: int = 5
    
    # Output options
    save_plot: bool = False
    output_path: Optional[Union[str, Path]] = None
    
    # Axis handling
    axes: Optional[str] = None
