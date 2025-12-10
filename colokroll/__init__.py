"""
Colok-Roll: Comprehensive confocal microscopy image analysis with colocalization.

A unified package for analyzing microscopy images with all dependencies included.
"""

__version__ = "0.1.0"
__author__ = "Gabriel Duarte"

# Core utilities
from .core import Config, Phase1Config
from .core import (
    validate_file_path,
    get_pixel_size_from_metadata,
    convert_pixels_to_microns,
    convert_microns_to_pixels,
    get_fluorophore_color,
    create_channel_color_mapping,
    get_colormap_from_fluorophore,
)

# Data processing
from .data_processing import ImageLoader, MIPCreator, SMEResult

# Imaging preprocessing
from .imaging_preprocessing import (
    BackgroundSubtractor,
    ZSliceSelectionResult,
    StrategyComparisonResult,
    aggregate_focus_scores,
    compute_focus_scores,
    detect_slices_to_keep,
    select_z_slices,
    compare_strategies,
)

# Visualization
from .visualization import Visualizer, plot_mip, plot_channels

# Analysis modules
from .analysis import (
    CellSegmenter,
    CellposeResult,
    NucleiDetector,
    compute_colocalization,
    export_colocalization_json,
    estimate_min_area_threshold,
)

# Export all public components
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    
    # Core utilities
    "Config",
    "Phase1Config",
    "validate_file_path",
    "get_pixel_size_from_metadata",
    "convert_pixels_to_microns",
    "convert_microns_to_pixels",
    "get_fluorophore_color",
    "create_channel_color_mapping",
    "get_colormap_from_fluorophore",
    
    # Data processing
    "ImageLoader",
    "MIPCreator",
    "SMEResult",
    
    # Imaging preprocessing
    "BackgroundSubtractor",
    "ZSliceSelectionResult",
    "StrategyComparisonResult",
    "aggregate_focus_scores",
    "compute_focus_scores",
    "detect_slices_to_keep",
    "select_z_slices",
    "compare_strategies",
    
    # Visualization
    "Visualizer",
    "plot_mip",
    "plot_channels",
    
    # Analysis
    "CellSegmenter",
    "CellposeResult",
    "NucleiDetector",
    "compute_colocalization",
    "export_colocalization_json",
    "estimate_min_area_threshold",
]
