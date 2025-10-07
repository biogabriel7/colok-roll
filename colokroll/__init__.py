"""Colokroll microscopy toolkit.

Import Strategy:
- Core modules (config, io, visualization, preprocessing) are imported directly
  as they have only standard dependencies (numpy, etc.)
- Analysis modules use lazy imports as they have optional dependencies:
  * Cellpose segmentation requires gradio_client
  * Colocalization may require cupy for GPU acceleration
"""

__version__ = "0.1.0"
__author__ = "Gabriel Duarte"

from .config import (
    RuntimeConfig,
    ImageIOConfig,
    ProjectionConfig,
    SegmentationConfig,
    RingAnalysisConfig,
    QuantificationConfig,
    PreprocessingConfig,
    create_runtime_config,
    load_config,
    save_config,
)
from .io import ImageLoader, MIPCreator, FormatConverter
from .visualization import Visualizer, plot_mip, plot_channels
from .analysis import get_cell_segmenter, get_colocalization_module, NucleiDetector
from .preprocessing import BackgroundSubtractor

__all__ = [
    "__version__",
    "__author__",
    "RuntimeConfig",
    "ImageIOConfig",
    "ProjectionConfig",
    "SegmentationConfig",
    "RingAnalysisConfig",
    "QuantificationConfig",
    "PreprocessingConfig",
    "create_runtime_config",
    "load_config",
    "save_config",
    "ImageLoader",
    "MIPCreator",
    "FormatConverter",
    "Visualizer",
    "plot_mip",
    "plot_channels",
    "get_cell_segmenter",
    "get_colocalization_module",
    "NucleiDetector",
    "BackgroundSubtractor",
]