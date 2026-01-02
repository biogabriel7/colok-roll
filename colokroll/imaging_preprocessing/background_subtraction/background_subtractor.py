"""
Background subtraction preprocessing for confocal microscopy images.

This module implements various background subtraction methods optimized for 3D ome.tiff
confocal z-stack images with different channel types. Follows the critical first step 
in the preprocessing pipeline: background subtraction → denoising → deconvolution.

Features:
- CUDA-only implementation (requires CuPy/CUDA)
- Memory-efficient processing with chunk-based operations (GPU-aware)
- Channel-specific parameter auto-selection
- Integration with existing ImageLoader workflow

Input: 3D numpy arrays (Z, Y, X) from ome.tiff files
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple, List
import gc

import numpy as np
from skimage import filters, morphology
from skimage.metrics import structural_similarity as skimage_ssim

from ...core.config import BackgroundSubtractionConfig, PreprocessingConfig
from ...core.utils import convert_microns_to_pixels
from ...data_processing.image_loader import ImageLoader
import json
from pathlib import Path

from typing import TYPE_CHECKING

from .backends import BackendAdapter, CpuAdapter, CudaAdapter, MpsAdapter
from .cpu_backend import subtract_background_cpu
from .cuda_backend import subtract_background_cuda
from .mps_backend import subtract_background_mps

# Optional backends are lazy-loaded to keep imports light
CUDA_AVAILABLE = False
cp = None

MPS_AVAILABLE = False
torch = None
kornia = None

# CPU fallback is always available via scipy
from scipy import ndimage as sp_ndimage
CPU_AVAILABLE = True

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

AUTO_CACHE_FILENAME = "auto_bg_cache.json"
# Heavier emphasis on background removal and contrast, lighter zero-penalty.
# We add SSIM as an additional term (separate constant below).
DEFAULT_AUTO_WEIGHTS: Tuple[float, float, float, float] = (0.7, 0.4, 0.2, 0.1)
SSIM_WEIGHT: float = 0.3
AUTO_METHODS: Tuple[str, ...] = ("rolling_ball", "gaussian", "two_stage", "morphological")
DEFAULT_COARSE_SEEDS: Dict[str, Dict[str, List[Any]]] = {
    "rolling_ball": {"radius": [16, 32, 64, 96, 128], "light_background": [False, True]},
    "gaussian": {"sigma": [5.0, 20.0, 50.0, 100.0, 200.0]},
    "two_stage": {
        "sigma_stage1": [5.0, 15.0, 30.0, 60.0],
        "radius_stage2": [16, 32, 64, 96],
        "light_background": [False, True],
    },
    "morphological": {"size": [5, 15, 30, 60, 90], "shape": ["disk", "square"]},
}
REFINE_POINTS: Dict[str, int] = {"rolling_ball": 7, "gaussian": 8, "two_stage": 5, "morphological": 7}
SECOND_REFINE_POINTS: Dict[str, int] = {"rolling_ball": 5, "gaussian": 6, "two_stage": 4, "morphological": 5}
IMPROVEMENT_EPS = 0.02
MAX_EVALS_PER_METHOD = 60
MIN_INT_STEP = 1
MIN_FLOAT_STEP = 0.5
TOPN_COARSE = 3
TESTED_VALUES_MAX = 20
MIN_STD_RATIO = 0.08
MIN_STD_ABS = 1e-3
FG_MIN_PIXELS = 50
FG_DILATE_RADIUS = 2
def _ensure_cuda() -> bool:
    """Lazy-load CUDA dependencies."""
    global cp, CUDA_AVAILABLE
    if CUDA_AVAILABLE:
        return True
    try:
        import cupy as _cp  # type: ignore
        cp = _cp
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False
    return CUDA_AVAILABLE


def _ensure_mps() -> bool:
    """Lazy-load MPS (PyTorch + kornia) dependencies."""
    global torch, kornia, MPS_AVAILABLE
    if MPS_AVAILABLE:
        return True
    try:
        import torch as _torch  # type: ignore

        if not (_torch.backends.mps.is_available() and _torch.backends.mps.is_built()):
            return False

        import kornia as _kornia  # type: ignore

        torch = _torch
        kornia = _kornia
        MPS_AVAILABLE = True
    except ImportError:
        MPS_AVAILABLE = False
    return MPS_AVAILABLE


class BackgroundSubtractor:
    """
    Background subtraction processor for 3D confocal z-stack ome.tiff images.
    
    Implements multiple background subtraction methods optimized for 3D data:
    - Rolling ball: Best for uneven illumination (slice-by-slice processing)
    - Gaussian: Fast approximation for gradual backgrounds
    - Morphological: For structured backgrounds
    
    Features:
    - CUDA acceleration for 10-50x speedup on compatible GPUs
    - Automatic CPU fallback when CUDA is not available
    - Memory-efficient processing with chunk-based operations
    - Channel-specific parameter auto-selection
    - Integration with existing ImageLoader workflow
    
    Input: 3D numpy arrays (Z, Y, X) from ome.tiff files
    """
    
    def __init__(self, config: Optional[BackgroundSubtractionConfig] = None, 
                 use_cuda: Optional[bool] = None,
                 use_mps: Optional[bool] = None,
                 auto_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the background subtractor.
        
        Args:
            config: Background subtraction configuration. If None, uses defaults.
            use_cuda: Force CUDA usage (True) or disable (False). If None, auto-detect.
            use_mps: Force MPS usage (True) or disable (False). If None, auto-detect.
            auto_config: Custom auto-configuration dictionary.
        """
        self.config = config or BackgroundSubtractionConfig()
        self.logger = logging.getLogger(__name__)
        self.adapter: BackendAdapter
        # auto_config is deprecated; zoom-search now uses built-in defaults plus cache
        self.auto_weights = DEFAULT_AUTO_WEIGHTS
        self.auto_cache_path = Path(__file__).with_name(AUTO_CACHE_FILENAME)
        
        # Determine backend: CUDA > MPS > CPU
        self.backend = self._select_backend(use_cuda, use_mps)
        self.use_cuda = (self.backend == 'cuda')
        self.use_mps = (self.backend == 'mps')
        
        if self.backend == 'cuda':
            self.logger.info("Initializing CUDA-accelerated background subtractor")
            self._initialize_cuda()
            self.adapter = CudaAdapter(self)
        elif self.backend == 'mps':
            self.logger.info("Initializing MPS-accelerated background subtractor (Apple Silicon)")
            self._initialize_mps()
            self.adapter = MpsAdapter(self)
        else:
            self.logger.info("Initializing CPU background subtractor (SciPy)")
            self._initialize_cpu()
            self.adapter = CpuAdapter(self)
    
    def _select_backend(self, use_cuda: Optional[bool], use_mps: Optional[bool]) -> str:
        """Select the best available backend."""
        # Explicit CUDA request
        if use_cuda is True:
            if _ensure_cuda():
                return 'cuda'
            else:
                raise RuntimeError("CUDA requested but CuPy is not available")
        
        # Explicit MPS request
        if use_mps is True:
            if _ensure_mps():
                return 'mps'
            else:
                raise RuntimeError("MPS requested but PyTorch MPS backend is not available")
        
        # Explicit disable
        if use_cuda is False and use_mps is False:
            return 'cpu'
        
        # Auto-detect: prefer CUDA > MPS > CPU
        if use_cuda is not False and _ensure_cuda():
            return 'cuda'
        if use_mps is not False and _ensure_mps():
            return 'mps'
        return 'cpu'
    
    def _initialize_mps(self) -> None:
        """Initialize MPS (Metal Performance Shaders) for Apple Silicon."""
        if not _ensure_mps():
            raise RuntimeError("MPS backend requested but dependencies are unavailable")
        self.device = torch.device('mps')
        self.logger.info(f"MPS device initialized: {self.device}")
        # MPS doesn't have memory query like CUDA, estimate conservatively
        self.gpu_memory_gb = 8.0  # Assume 8GB for Apple Silicon
        self.max_gpu_memory_gb = 6.0
    
    def _initialize_cpu(self) -> None:
        """Initialize CPU backend."""
        self.gpu_memory_gb = 0
        self.max_gpu_memory_gb = 0
        self.logger.info("CPU backend initialized (no GPU acceleration)")
    
    def _initialize_cuda(self) -> None:
        """Initialize CUDA context and check GPU memory."""
        if not _ensure_cuda():
            raise RuntimeError("CUDA backend requested but dependencies are unavailable")
        try:
            # Get GPU info
            mempool = cp.get_default_memory_pool()
            
            # Get GPU memory info
            gpu_mem = cp.cuda.Device().mem_info
            total_mem = gpu_mem[1] / (1024**3)  # Convert to GB
            free_mem = gpu_mem[0] / (1024**3)
            
            self.logger.info(f"GPU Memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total")
            
            # Set memory pool limits (use 80% of available memory)
            max_mem_gb = free_mem * 0.8
            mempool.set_limit(size=int(max_mem_gb * 1024**3))
            
            self.gpu_memory_gb = free_mem
            self.max_gpu_memory_gb = max_mem_gb
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CUDA: {e}")
            raise RuntimeError(f"CUDA initialization failed: {e}")
        
    def subtract_background(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
        channel_name: Optional[str] = None,
        pixel_size: Optional[float] = None,
        chunk_size: Optional[int] = None,
        is_negative_control: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Subtract background from a 3D ome.tiff z-stack image.
        
        Args:
            image: Input 3D image array (Z, Y, X) from ome.tiff
            method: Background subtraction method ('rolling_ball', 'gaussian', 'morphological')
                   If None, uses config default
            channel_name: Name of the channel for automatic parameter selection
            pixel_size: Pixel size in micrometers for micron-based parameters
            chunk_size: Number of z-slices to process at once for memory efficiency
            is_negative_control: If True, optimize for minimal residual signal instead of
                               preserving features. Use for channels where no signal is expected.
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple of (background_corrected_image, metadata_dict)
        """
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D (Z, Y, X), got {image.ndim}D with shape {image.shape}")
        
        method = method or 'auto'
        
        if method == 'auto':
            self.logger.info("Running AUTO background subtraction (channel=%s, negative_control=%s)", 
                           channel_name or "unknown", is_negative_control)
            return self._auto_subtract_background(image, channel_name, pixel_size, 
                                                  is_negative_control=is_negative_control, **kwargs)
        
        # Route to appropriate backend via adapter
        channel_label = channel_name or "unknown"
        self.logger.info(
            f"Applying {self.backend.upper()} {method} background subtraction to 3D image {image.shape} (channel={channel_label})"
        )
        return self.adapter.subtract(
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            chunk_size=chunk_size,
            **kwargs,
        )
    
    def _subtract_background_cuda(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """CUDA-accelerated background subtraction (delegates to cuda_backend)."""
        return subtract_background_cuda(
            self,
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            **kwargs,
        )
    
    def _subtract_background_cpu(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        chunk_size: Optional[int],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """CPU implementation of background subtraction."""
        return subtract_background_cpu(
            self,
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            chunk_size=chunk_size,
            **kwargs,
        )
    
    def _get_method_parameters(
        self, 
        method: str, 
        channel_name: Optional[str], 
        pixel_size: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Get parameters for the specified method and channel."""
        params = {}
        
        if method == 'rolling_ball':
            # Get radius based on channel type
            if channel_name:
                radius = self._get_rolling_ball_radius_for_channel(channel_name)
            else:
                radius = kwargs.get('radius', 30)  # Default radius
            params['radius'] = radius
            
        elif method == 'gaussian':
            # Get sigma based on channel type
            if channel_name:
                sigma = self._get_gaussian_sigma_for_channel(channel_name)
            else:
                sigma = kwargs.get('sigma', 15.0)  # Default sigma
            params['sigma'] = sigma
            
        elif method == 'morphological':
            # Get structuring element size
            if pixel_size and 'size_um' in kwargs:
                size_pixels = convert_microns_to_pixels(kwargs['size_um'], pixel_size)
            else:
                size_pixels = kwargs.get('size_pixels', 20)
            params['size'] = int(size_pixels)
            params['shape'] = kwargs.get('shape', 'disk')
        elif method in {'two_stage', 'gaussian_then_rolling_ball'}:
            sigma = kwargs.get('sigma_stage1', 14.0)
            if channel_name:
                default_radius = self._get_rolling_ball_radius_for_channel(channel_name)
            else:
                default_radius = 30
            radius = int(kwargs.get('radius_stage2', default_radius))
            params['sigma_stage1'] = float(sigma)
            params['radius_stage2'] = radius
            params['light_background'] = bool(kwargs.get('light_background', False))
        
        # Override with any explicit kwargs
        params.update({k: v for k, v in kwargs.items() if k not in ['channel_name', 'pixel_size']})
        
        return params
    
    def _get_rolling_ball_radius_for_channel(self, channel_name: str) -> int:
        """Get appropriate rolling ball radius for channel type."""
        channel_lower = channel_name.lower()
        
        if any(name in channel_lower for name in ['dapi', 'hoechst']):
            return self.config.rolling_ball_radius_dapi
        elif any(name in channel_lower for name in ['phalloidin', 'actin', 'af488']):
            return self.config.rolling_ball_radius_phalloidin
        else:
            return self.config.rolling_ball_radius_protein
    
    def _get_gaussian_sigma_for_channel(self, channel_name: str) -> float:
        """Get appropriate Gaussian sigma for channel type."""
        channel_lower = channel_name.lower()
        
        if any(name in channel_lower for name in ['dapi', 'hoechst']):
            return self.config.gaussian_sigma_dapi
        elif any(name in channel_lower for name in ['phalloidin', 'actin', 'af488']):
            return self.config.gaussian_sigma_phalloidin
        else:
            return self.config.gaussian_sigma_protein
    
    def _calculate_optimal_chunk_size(self, image_shape: Tuple[int, int, int], dtype: np.dtype) -> int:
        """
        Calculate optimal chunk size for memory-efficient processing on M3 Pro.
        
        Args:
            image_shape: Shape of the 3D image (Z, Y, X)
            dtype: Data type of the image
            
        Returns:
            Optimal number of z-slices to process at once
        """
        z_slices, height, width = image_shape
        bytes_per_element = np.dtype(dtype).itemsize
        
        # Estimate memory per slice (including intermediate arrays)
        slice_memory_mb = (height * width * bytes_per_element * 3) / (1024 * 1024)  # 3x for intermediate processing
        
        # Use conservative 2GB limit for chunk processing (from 14GB total available)
        max_chunk_memory_mb = 2048
        
        optimal_chunk_size = max(1, min(z_slices, int(max_chunk_memory_mb / slice_memory_mb)))
        
        self.logger.info(f"Calculated chunk size: {optimal_chunk_size} slices "
                        f"({slice_memory_mb:.1f} MB per slice)")
        
        return optimal_chunk_size

    def _rolling_ball_subtraction_3d(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any],
        chunk_size: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform rolling ball background subtraction on 3D image with chunked processing.
        
        This method processes z-slices in chunks to manage memory usage on M3 Pro.
        """
        radius = params['radius']
        z_slices, height, width = image.shape
        
        # Pre-allocate output array
        corrected_image = np.empty_like(image, dtype=np.float32)
        
        # Track background statistics
        background_stats_list = []
        
        # Process in chunks
        for start_z in range(0, z_slices, chunk_size):
            end_z = min(start_z + chunk_size, z_slices)
            chunk_slice = slice(start_z, end_z)
            
            self.logger.debug(f"Processing z-slices {start_z} to {end_z-1}")
            
            # Process each slice in the chunk
            for z in range(start_z, end_z):
                light_background = params.get('light_background', False)
                background = self._create_rolling_ball_background(image[z], radius, light_background)
                corrected_image[z] = image[z].astype(np.float32) - background
                
                # Collect background statistics
                background_stats_list.append({
                    'z_slice': z,
                    'mean': float(np.mean(background)),
                    'std': float(np.std(background))
                })
            
            # Force garbage collection after processing chunk
            gc.collect()
        
        # Aggregate background statistics
        all_means = [stats['mean'] for stats in background_stats_list]
        all_stds = [stats['std'] for stats in background_stats_list]
        
        metadata = {
            'background_method': 'rolling_ball_3d',
            'radius_pixels': radius,
            'z_slices_processed': z_slices,
            'chunk_size': chunk_size,
            'background_stats': {
                'mean_across_slices': float(np.mean(all_means)),
                'std_across_slices': float(np.mean(all_stds)),
                'mean_variation': float(np.std(all_means)),
                'per_slice_stats': background_stats_list[:5]  # Keep first 5 for debugging
            }
        }
        
        return corrected_image, metadata
    
    def _create_rolling_ball_background(self, image: np.ndarray, radius: int, light_background: bool = False) -> np.ndarray:
        """
        Create rolling ball background using proper rolling ball algorithm.
        
        Uses the opencv-rolling-ball package which is ported from ImageJ's Background Subtractor.
        This is the most accurate implementation of the rolling ball algorithm.
        """
        import time
        start_time = time.time()
        
        # Check if image is too large and suggest downsampling
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            self.logger.warning(f"Large image detected ({image.shape}). Rolling ball may be slow. Consider using smaller radius or downsampling.")
        
        try:
            from cv2_rolling_ball import subtract_background_rolling_ball
        except ImportError:
            # Fallback to scikit-image if opencv-rolling-ball is not available
            from skimage import restoration
            self.logger.warning("opencv-rolling-ball not available, falling back to scikit-image")
            
            if image.dtype != np.float64:
                image = image.astype(np.float64)
            background = restoration.rolling_ball(image, radius=radius, light_background=light_background)
            return background.astype(np.float32)
        
        # Convert to 8-bit if needed (opencv-rolling-ball requirement)
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image_8bit = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image_8bit = np.zeros_like(image, dtype=np.uint8)
        else:
            image_8bit = image
        
        self.logger.info(f"Starting rolling ball background subtraction (radius={radius}, size={image.shape})")
        
        # Use opencv-rolling-ball implementation
        corrected_img, background = subtract_background_rolling_ball(
            image_8bit, 
            radius, 
            light_background=light_background,
            use_paraboloid=False,  # Use rolling ball, not paraboloid
            do_presmooth=True
        )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Rolling ball completed in {elapsed_time:.2f}s")
        
        # Convert background back to original data type and range
        if image.dtype != np.uint8:
            background = background.astype(np.float32) / 255.0 * (img_max - img_min) + img_min
        
        return background.astype(np.float32)
    
    def _gaussian_subtraction_3d(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform 3D Gaussian background subtraction.
        
        This method is faster than rolling ball and works well for gradual
        background variations. Applies Gaussian filtering across all three dimensions.
        """
        sigma = params['sigma']
        
        self.logger.info(f"Applying 3D Gaussian background subtraction with sigma={sigma}")
        
        # Create 3D Gaussian background
        background = filters.gaussian(image, sigma=sigma, preserve_range=True)
        
        # Subtract background
        corrected_image = image.astype(np.float32) - background.astype(np.float32)
        
        metadata = {
            'background_method': 'gaussian_3d',
            'sigma': sigma,
            'background_stats': {
                'mean': float(np.mean(background)),
                'std': float(np.std(background)),
                'min': float(np.min(background)),
                'max': float(np.max(background))
            }
        }
        
        # Clean up memory
        del background
        gc.collect()
        
        return corrected_image, metadata
    
    def _morphological_subtraction_3d(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any],
        chunk_size: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform morphological background subtraction on 3D image with chunked processing.
        
        Uses morphological closing to estimate background, useful for
        images with structured backgrounds. Processes slice-by-slice for memory efficiency.
        """
        size = params['size']
        shape = params.get('shape', 'disk')
        z_slices = image.shape[0]
        
        # Create structuring element for 2D operations on each slice
        if shape == 'disk':
            selem = morphology.disk(size)
        elif shape == 'square':
            selem = morphology.rectangle(size, size)
        else:
            selem = morphology.disk(size)  # Default to disk
        
        # Pre-allocate output array
        corrected_image = np.empty_like(image, dtype=np.float32)
        
        # Track background statistics
        background_stats_list = []
        
        # Process in chunks
        for start_z in range(0, z_slices, chunk_size):
            end_z = min(start_z + chunk_size, z_slices)
            
            self.logger.debug(f"Processing morphological background for z-slices {start_z} to {end_z-1}")
            
            # Process each slice in the chunk
            for z in range(start_z, end_z):
                # Morphological closing to estimate background
                background = morphology.closing(image[z], selem)
                corrected_image[z] = image[z].astype(np.float32) - background
                
                # Collect background statistics
                background_stats_list.append({
                    'z_slice': z,
                    'mean': float(np.mean(background)),
                    'std': float(np.std(background))
                })
            
            # Force garbage collection after processing chunk
            gc.collect()
        
        # Aggregate background statistics
        all_means = [stats['mean'] for stats in background_stats_list]
        all_stds = [stats['std'] for stats in background_stats_list]
        
        metadata = {
            'background_method': 'morphological_3d',
            'size': size,
            'shape': shape,
            'z_slices_processed': z_slices,
            'chunk_size': chunk_size,
            'background_stats': {
                'mean_across_slices': float(np.mean(all_means)),
                'std_across_slices': float(np.mean(all_stds)),
                'mean_variation': float(np.std(all_means)),
                'per_slice_stats': background_stats_list[:5]  # Keep first 5 for debugging
            }
        }
        
        return corrected_image, metadata
    
    def _two_stage_subtraction_3d_cpu(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
        chunk_size: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Two-stage CPU background subtraction: Gaussian → Rolling Ball.
        
        Stage 1 removes diffuse haze; stage 2 removes local background while preserving
        puncta/structures.
        """
        sigma = float(params['sigma_stage1'])
        radius = int(params['radius_stage2'])
        light_background = bool(params.get('light_background', False))
        z_slices = image.shape[0]
        
        self.logger.info(f"Applying CPU two-stage background subtraction: Gaussian(σ={sigma}) → Rolling ball(r={radius})")
        
        # Stage 1: Gaussian (operates on full 3D volume)
        gauss_bg = filters.gaussian(image, sigma=sigma, preserve_range=True)
        gauss_corr = image.astype(np.float32) - gauss_bg.astype(np.float32)
        
        if getattr(self.config, 'clip_negative_values', False):
            gauss_corr = np.clip(gauss_corr, 0, None)
        
        # Stage 2: Rolling ball via morphological opening/closing (slice by slice)
        selem = morphology.disk(radius)
        corrected_image = np.empty_like(gauss_corr, dtype=np.float32)
        
        for start_z in range(0, z_slices, chunk_size):
            end_z = min(start_z + chunk_size, z_slices)
            
            for z in range(start_z, end_z):
                if light_background:
                    background = morphology.closing(gauss_corr[z], selem)
                else:
                    background = morphology.opening(gauss_corr[z], selem)
                corrected_image[z] = gauss_corr[z] - background.astype(np.float32)
            
            gc.collect()
        
        metadata = {
            'background_method': 'two_stage_cpu',
            'stage1': {'sigma': sigma},
            'stage2': {'radius': radius, 'light_background': light_background},
            'gpu_accelerated': False
        }
        
        del gauss_bg, gauss_corr
        gc.collect()
        
        return corrected_image, metadata
    
    # ==================== MPS BACKEND (Apple Silicon) ====================
    
    def _subtract_background_mps(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """MPS-accelerated background subtraction using PyTorch + kornia."""
        return subtract_background_mps(
            self,
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            **kwargs,
        )
    
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range while preserving relative intensities."""
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max > image_min:
            normalized = (image - image_min) / (image_max - image_min)
        else:
            normalized = np.zeros_like(image)
        
        return normalized
    
    # ==================== CUDA-ACCELERATED METHODS ====================
    # ==================== AUTO MODE (GRID SEARCH + SCORING) ====================

    def _slice_masks_otsu(self, slice_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        thr = filters.threshold_otsu(slice_np)
        fg = slice_np >= thr
        fg = morphology.remove_small_objects(fg, min_size=FG_MIN_PIXELS)
        if FG_DILATE_RADIUS > 0:
            fg = morphology.binary_dilation(fg, morphology.disk(FG_DILATE_RADIUS))
        fg = morphology.binary_closing(fg, morphology.disk(1))
        bg = ~fg
        return fg, bg

    def _slice_metrics(self, orig_slice: np.ndarray, corr_slice: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        fg, bg = self._slice_masks_otsu(orig_slice)
        bg0 = float(np.median(orig_slice[bg])) if np.any(bg) else float(np.median(orig_slice))
        bg1 = float(np.median(corr_slice[bg])) if np.any(bg) else float(np.median(corr_slice))
        c0 = float(np.mean(orig_slice[fg]) - np.median(orig_slice[bg])) if np.any(fg) and np.any(bg) else 0.0
        c1 = float(np.mean(corr_slice[fg]) - np.median(corr_slice[bg])) if np.any(fg) and np.any(bg) else 0.0
        g0 = float(np.mean(filters.sobel(orig_slice)[fg])) if np.any(fg) else 0.0
        g1 = float(np.mean(filters.sobel(corr_slice)[fg])) if np.any(fg) else 0.0
        zf = float(np.mean(corr_slice == 0))
        # Masked SSIM; zero background to avoid noise influence
        if np.any(fg):
            orig_fg = orig_slice * fg
            corr_fg = corr_slice * fg
            dr = float(orig_fg.max() - orig_fg.min() or 1.0)
            try:
                ssim_fg = float(skimage_ssim(orig_fg, corr_fg, data_range=dr))
            except Exception:
                ssim_fg = 0.0
        else:
            ssim_fg = 0.0
        orig_std = float(np.std(orig_slice))
        corr_std = float(np.std(corr_slice))
        return bg0, bg1, c0, c1, g0, g1, zf, ssim_fg, orig_std, corr_std

    def _score_volume(self, original: np.ndarray, corrected: np.ndarray, weights: Tuple[float, float, float, float], max_slices: int = 7) -> Tuple[float, Dict[str, float]]:
        w_bg, w_contrast, w_grad, w_zero = weights
        z = original.shape[0]
        idx = np.linspace(0, z - 1, num=min(z, max_slices), dtype=int)
        scores: List[float] = []
        metrics_accum = {
            "bg_improve": 0.0,
            "contrast_gain": 0.0,
            "grad_ratio": 0.0,
            "zero_frac": 0.0,
            "ssim_fg": 0.0,
            "orig_std": 0.0,
            "corr_std": 0.0,
        }
        for zi in idx:
            b0, b1, c0, c1, g0, g1, zf, ssim_fg, orig_std, corr_std = self._slice_metrics(original[zi], corrected[zi])
            bg_improve = (b0 - b1) / (b0 + 1e-6)
            contrast_gain = (c1 - c0) / (abs(c0) + 1e-6)
            grad_ratio = g1 / (g0 + 1e-6)
            score = (
                w_bg * bg_improve
                + w_contrast * contrast_gain
                + w_grad * min(1.0, grad_ratio)
                - w_zero * zf
                + SSIM_WEIGHT * ssim_fg
            )
            scores.append(score)
            metrics_accum["bg_improve"] += bg_improve
            metrics_accum["contrast_gain"] += contrast_gain
            metrics_accum["grad_ratio"] += grad_ratio
            metrics_accum["zero_frac"] += zf
            metrics_accum["ssim_fg"] += ssim_fg
            metrics_accum["orig_std"] += orig_std
            metrics_accum["corr_std"] += corr_std
        n = len(idx) or 1
        for k in metrics_accum:
            metrics_accum[k] /= n
        return float(np.mean(scores)), metrics_accum

    def _score_volume_negative_control(
        self, 
        original: np.ndarray, 
        corrected: np.ndarray, 
        max_slices: int = 7
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score background subtraction for negative control channels.
        
        For negative controls, we want to minimize residual signal while avoiding
        complete signal flattening (which makes data unusable for analysis).
        
        Higher scores indicate better background removal with preserved variance.
        
        Scoring formula:
            score = w_mean * (1 - normalized_mean) + w_std * std_score + w_zero * zero_fraction
        
        Where:
            - normalized_mean = corrected_mean / original_mean (lower is better)
            - std_score: penalizes both insufficient AND excessive std reduction
            - zero_fraction = fraction of pixels near-zero (higher is better, but not 100%)
        """
        # Weights balanced for negative control scoring
        w_mean = 0.5   # Weight for mean reduction
        w_std = 0.3    # Weight for std preservation (not too high, not too low)
        w_zero = 0.2   # Weight for zero fraction
        
        z = original.shape[0]
        idx = np.linspace(0, z - 1, num=min(z, max_slices), dtype=int)
        scores: List[float] = []
        
        metrics_accum = {
            "orig_mean": 0.0,
            "corr_mean": 0.0,
            "orig_std": 0.0,
            "corr_std": 0.0,
            "zero_frac": 0.0,
            "mean_reduction": 0.0,
            "std_score": 0.0,
            "normalized_std": 0.0,
        }
        
        for zi in idx:
            orig_slice = original[zi].astype(np.float32)
            corr_slice = corrected[zi].astype(np.float32)
            
            # Compute metrics
            orig_mean = float(np.mean(orig_slice))
            corr_mean = float(np.mean(corr_slice))
            orig_std = float(np.std(orig_slice))
            corr_std = float(np.std(corr_slice))
            
            # Zero fraction (pixels at or near zero)
            near_zero_threshold = 1.0  # Consider values <= 1 as "zero"
            zero_frac = float(np.mean(corr_slice <= near_zero_threshold))
            
            # Normalized metrics (how much we reduced relative to original)
            normalized_mean = corr_mean / (orig_mean + 1e-6)
            normalized_std = corr_std / (orig_std + 1e-6)
            
            # Mean reduction: reward lower means
            mean_reduction = 1.0 - min(1.0, normalized_mean)
            
            # Std score: We want SOME reduction but not complete flattening
            # Target: reduce std to 20-40% of original (not to near-zero)
            # Optimal normalized_std around 0.3 (30% of original)
            target_std_ratio = 0.3
            std_deviation = abs(normalized_std - target_std_ratio)
            std_score = max(0.0, 1.0 - (std_deviation / target_std_ratio))
            
            # Penalize complete signal removal (when corrected is essentially flat)
            if corr_std < 0.5:  # Absolute minimum variance threshold
                std_score = 0.0  # Heavily penalize flat images
            
            score = (
                w_mean * mean_reduction
                + w_std * std_score
                + w_zero * min(0.9, zero_frac)  # Cap zero_frac contribution at 0.9
            )
            scores.append(score)
            
            # Accumulate metrics
            metrics_accum["orig_mean"] += orig_mean
            metrics_accum["corr_mean"] += corr_mean
            metrics_accum["orig_std"] += orig_std
            metrics_accum["corr_std"] += corr_std
            metrics_accum["zero_frac"] += zero_frac
            metrics_accum["mean_reduction"] += mean_reduction
            metrics_accum["std_score"] += std_score
            metrics_accum["normalized_std"] += normalized_std
        
        n = len(idx) or 1
        for k in metrics_accum:
            metrics_accum[k] /= n
        
        return float(np.mean(scores)), metrics_accum

    def _compute_negative_control_metrics(self, corrected: np.ndarray) -> Dict[str, float]:
        """
        Compute validation metrics for negative control channels.
        
        These metrics help assess if the background subtraction achieved
        the expected result for a negative control (minimal residual signal).
        
        Returns:
            Dictionary with:
            - residual_mean: Mean intensity of corrected image
            - residual_std: Standard deviation of corrected image
            - residual_percentile_95: 95th percentile intensity
            - residual_percentile_99: 99th percentile intensity
            - zero_fraction: Fraction of pixels at or near zero
        """
        corrected_flat = corrected.flatten().astype(np.float32)
        
        near_zero_threshold = 1.0
        zero_fraction = float(np.mean(corrected_flat <= near_zero_threshold))
        
        return {
            'residual_mean': float(np.mean(corrected_flat)),
            'residual_std': float(np.std(corrected_flat)),
            'residual_percentile_95': float(np.percentile(corrected_flat, 95)),
            'residual_percentile_99': float(np.percentile(corrected_flat, 99)),
            'zero_fraction': zero_fraction,
        }
    
    def _normalize_channel_key(self, channel_name: Optional[str]) -> str:
        return channel_name.lower() if channel_name else "unknown"

    def _load_auto_cache(self) -> Dict[str, Any]:
        if self.auto_cache_path.exists():
            try:
                return json.loads(self.auto_cache_path.read_text())
            except Exception as exc:
                self.logger.warning("Failed to load auto cache %s: %s", self.auto_cache_path, exc)
        return {"channels": {}}

    def _save_auto_cache(self, cache: Dict[str, Any]) -> None:
        try:
            self.auto_cache_path.write_text(json.dumps(cache, indent=2))
        except Exception as exc:
            self.logger.warning("Failed to save auto cache %s: %s", self.auto_cache_path, exc)

    def _select_sample_indices(self, z: int, max_slices: int = 9) -> np.ndarray:
        if z <= max_slices:
            return np.arange(z, dtype=int)
        return np.linspace(0, z - 1, num=max_slices, dtype=int)

    def _merge_seed_ranges(self, method: str, channel_entry: Dict[str, Any]) -> Dict[str, List[Any]]:
        base = {k: list(v) for k, v in DEFAULT_COARSE_SEEDS.get(method, {}).items()}
        method_cache = channel_entry.get("methods", {}).get(method, {})
        tested_values = method_cache.get("tested_values", {})
        for param, values in tested_values.items():
            merged = list(set(base.get(param, []) + list(values)))
            # Keep deterministic ordering for reproducibility
            try:
                merged_sorted = sorted(merged)
            except Exception:
                merged_sorted = merged
            base[param] = merged_sorted
        return base

    def _refine_numeric_values(
        self,
        best_value: float,
        neighbor_values: List[Any],
        points: int,
        shrink: float = 0.75,
        is_int: bool = False,
        min_value: Optional[float] = None,
    ) -> List[Any]:
        numeric_values = sorted({float(v) for v in neighbor_values})
        lower_neighbors = [v for v in numeric_values if v < best_value]
        upper_neighbors = [v for v in numeric_values if v > best_value]
        lower_bound = max(lower_neighbors) if lower_neighbors else max(best_value * 0.5, 0.0)
        upper_bound = min(upper_neighbors) if upper_neighbors else best_value * 1.5
        span_lower = best_value - lower_bound
        span_upper = upper_bound - best_value
        window = max(span_lower, span_upper) * shrink
        # Apply parameter-specific minimum bounds
        absolute_min = min_value if min_value is not None else 0.0
        start = max(absolute_min, best_value - window)
        end = best_value + window
        if points <= 1 or end <= start:
            refined = [best_value]
        else:
            step = (end - start) / max(points - 1, 1)
            step = max(step, MIN_INT_STEP if is_int else MIN_FLOAT_STEP)
            refined = [start + i * step for i in range(points)]
        refined.append(best_value)
        refined_unique = sorted(set(refined))
        if is_int:
            refined_unique = sorted({int(round(v)) for v in refined_unique})
        return refined_unique

    def _build_refined_ranges(
        self,
        best_params: Dict[str, Any],
        coarse_ranges: Dict[str, List[Any]],
        method: str,
        points_map: Dict[str, int],
        shrink: float = 0.75,
    ) -> Dict[str, List[Any]]:
        # Define minimum bounds for specific parameters
        PARAM_MIN_VALUES = {
            'size': 3.0,  # Morphological size must be at least 3 for valid structuring element
            'radius': 5.0,  # Rolling ball radius minimum
            'radius_stage2': 5.0,  # Two-stage rolling ball minimum
            'sigma': 0.5,  # Gaussian sigma minimum
            'sigma_stage1': 0.5,  # Two-stage Gaussian minimum
        }
        
        refined: Dict[str, List[Any]] = {}
        points = points_map.get(method, 7)
        for param, values in coarse_ranges.items():
            if isinstance(values[0], bool) or isinstance(values[0], str):
                refined[param] = [best_params.get(param, values[0])]
            else:
                is_int = isinstance(values[0], int)
                min_value = PARAM_MIN_VALUES.get(param, None)
                refined[param] = self._refine_numeric_values(
                    float(best_params.get(param, values[0])),
                    values,
                    points,
                    shrink=shrink,
                    is_int=is_int,
                    min_value=min_value,
                )
        return refined

    def _expand_param_grid(self, grid_spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        from itertools import product
        keys = [k for k in grid_spec.keys()]
        values = [grid_spec[k] for k in keys]
        candidates: List[Dict[str, Any]] = []
        for combo in product(*values):
            p = {k: v for k, v in zip(keys, combo)}
            candidates.append(p)
        return candidates

    def _run_single_cuda(
        self, image: np.ndarray, method: str, params: Dict[str, Any], channel_name: Optional[str], pixel_size: Optional[float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        clean_params = {k: v for k, v in params.items() if k != 'method'}
        return self._subtract_background_cuda(image, method, channel_name, pixel_size, **clean_params)
    
    def _run_single_mps(
        self, image: np.ndarray, method: str, params: Dict[str, Any], channel_name: Optional[str], pixel_size: Optional[float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        clean_params = {k: v for k, v in params.items() if k != 'method'}
        return self._subtract_background_mps(image, method, channel_name, pixel_size, **clean_params)
    
    def _run_single_cpu(
        self, image: np.ndarray, method: str, params: Dict[str, Any], channel_name: Optional[str], pixel_size: Optional[float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        clean_params = {k: v for k, v in params.items() if k != 'method'}
        chunk_size = self._calculate_optimal_chunk_size(image.shape, image.dtype)
        return self._subtract_background_cpu(image, method, channel_name, pixel_size, chunk_size, **clean_params)
    
    def _run_single(
        self, image: np.ndarray, method: str, params: Dict[str, Any], channel_name: Optional[str], pixel_size: Optional[float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run a single background subtraction using the current backend."""
        if self.backend == 'cuda':
            return self._run_single_cuda(image, method, params, channel_name, pixel_size)
        elif self.backend == 'mps':
            return self._run_single_mps(image, method, params, channel_name, pixel_size)
        else:
            return self._run_single_cpu(image, method, params, channel_name, pixel_size)

    def _evaluate_candidates(
        self,
        image_sample: np.ndarray,
        method: str,
        candidates: List[Dict[str, Any]],
        weights: Tuple[float, float, float, float],
        max_slices: int,
        max_evals: int,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        is_negative_control: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        scores: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        for idx, params in enumerate(candidates):
            if idx >= max_evals:
                break
            corrected_sample, _ = self._run_single(image_sample, method, params, channel_name, pixel_size)
            self._sync_backend()
            
            # Use appropriate scoring function based on control type
            if is_negative_control:
                # For negative controls: optimize for minimal residual signal
                score, comps = self._score_volume_negative_control(
                    image_sample, corrected_sample, max_slices=max_slices
                )
                # No std_penalty for negative controls (low std is actually good)
                std_penalty = False
            else:
                # Standard scoring: preserve signal features
                score, comps = self._score_volume(image_sample, corrected_sample, weights, max_slices=max_slices)
                # Post-check: reject flattened outputs
                std_penalty = False
                if comps["corr_std"] < MIN_STD_ABS or comps["corr_std"] < MIN_STD_RATIO * max(comps["orig_std"], 1e-6):
                    score = -1e6
                    std_penalty = True
            
            entry = {'params': params, 'score': float(score), 'components': comps, 'std_penalty': std_penalty}
            scores.append(entry)
            if best is None or score > best['score']:
                best = {'params': params, 'score': float(score), 'components': comps, 'std_penalty': std_penalty}
        return scores, best

    def _zoom_search_method(
        self,
        image_sample: np.ndarray,
        method: str,
        seed_ranges: Dict[str, List[Any]],
        weights: Tuple[float, float, float, float],
        max_slices: int,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        is_negative_control: bool = False,
    ) -> Dict[str, Any]:
        all_scores: List[Dict[str, Any]] = []
        ranges_used: Dict[str, Dict[str, List[Any]]] = {"coarse": seed_ranges}

        coarse_candidates = self._expand_param_grid(seed_ranges)
        coarse_scores, best_coarse = self._evaluate_candidates(
            image_sample, method, coarse_candidates, weights, max_slices=max_slices, max_evals=MAX_EVALS_PER_METHOD,
            channel_name=channel_name, pixel_size=pixel_size, is_negative_control=is_negative_control,
        )
        all_scores.extend(coarse_scores)
        if best_coarse is None:
            return {
                "method": method,
                "best_params": None,
                "best_score": -np.inf,
                "scores": all_scores,
                "ranges_used": ranges_used,
            }

        # Prune to top-N coarse before refinement
        coarse_sorted = sorted(coarse_scores, key=lambda x: x['score'], reverse=True)
        top_coarse = coarse_sorted[:TOPN_COARSE]
        remaining_evals = max(0, MAX_EVALS_PER_METHOD - len(all_scores))
        per_candidate_budget = max(1, remaining_evals // max(len(top_coarse), 1))

        current_best = best_coarse
        refine_idx = 0
        for candidate in top_coarse:
            refine_ranges = self._build_refined_ranges(candidate["params"], seed_ranges, method, REFINE_POINTS)
            ranges_used[f"refine{refine_idx}"] = refine_ranges
            refine_idx += 1
            refined_candidates = self._expand_param_grid(refine_ranges)
            refined_scores, best_refined = self._evaluate_candidates(
                image_sample, method, refined_candidates, weights, max_slices=max_slices,
                max_evals=per_candidate_budget, channel_name=channel_name, pixel_size=pixel_size,
                is_negative_control=is_negative_control,
            )
            all_scores.extend(refined_scores)
            if best_refined and best_refined["score"] > current_best["score"]:
                current_best = best_refined

        return {
            "method": method,
            "best_params": current_best["params"],
            "best_score": current_best["score"],
            "scores": all_scores,
            "ranges_used": ranges_used,
        }

    def _collect_tested_values(self, scores: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        tested: Dict[str, List[Any]] = {}
        for entry in scores:
            for k, v in entry["params"].items():
                tested.setdefault(k, [])
                if v not in tested[k]:
                    tested[k].append(v)
                # Keep only the most recent TESTED_VALUES_MAX
                if len(tested[k]) > TESTED_VALUES_MAX:
                    tested[k] = tested[k][-TESTED_VALUES_MAX:]
        return tested

    def _auto_subtract_background(
        self,
        image: np.ndarray,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        is_negative_control: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Automatic background subtraction via per-method zoom search.

        Cache behavior:
        - On cache hit, by default we first try the cached best (method, params) on the sampled slices.
          If it validates (score close to cached score), we skip the zoom search and apply it to the full volume.
        - You can disable this fast path by passing `auto_use_cache_best=False`.

        Args:
            auto_use_cache_best: If True (default), attempt cached best params first and potentially skip search.
            auto_cache_score_tolerance: Accept cached best if sample score >= cached_best_score - tolerance (default: 0.05).
            is_negative_control: If True, use negative control scoring (minimize residual signal).
        """
        auto_use_cache_best = bool(kwargs.pop("auto_use_cache_best", True))
        auto_cache_score_tolerance = float(kwargs.pop("auto_cache_score_tolerance", 0.05))

        channel_key = self._normalize_channel_key(channel_name)
        cache = self._load_auto_cache()
        channel_entry = cache.get("channels", {}).get(channel_key, {})

        z = image.shape[0]
        sample_idx = self._select_sample_indices(z)
        image_sample = image[sample_idx]
        weights = self.auto_weights

        # Fast path: on cache hit, try cached best params first and skip zoom search if validated.
        cached_best_method: Optional[str] = None
        cached_best_params: Optional[Dict[str, Any]] = None
        cached_best_score: Optional[float] = None
        if auto_use_cache_best and channel_entry:
            methods_cache = channel_entry.get("methods", {})
            for m, entry in methods_cache.items():
                if m not in AUTO_METHODS:
                    continue
                p = entry.get("best_params")
                s = entry.get("best_score")
                if p is None or s is None:
                    continue
                try:
                    s_f = float(s)
                except Exception:
                    continue
                if cached_best_score is None or s_f > cached_best_score:
                    cached_best_score = s_f
                    cached_best_method = m
                    cached_best_params = p

        # For negative controls, skip cache fast path (use different scoring)
        if cached_best_method and cached_best_params and cached_best_score is not None and not is_negative_control:
            corrected_sample, _ = self._run_single(image_sample, cached_best_method, cached_best_params, channel_name, pixel_size)
            self._sync_backend()
            score, comps = self._score_volume(image_sample, corrected_sample, weights, max_slices=min(5, len(sample_idx)))
            std_penalty = False
            if comps["corr_std"] < MIN_STD_ABS or comps["corr_std"] < MIN_STD_RATIO * max(comps["orig_std"], 1e-6):
                score = -1e6
                std_penalty = True

            if (not std_penalty) and score >= (cached_best_score - auto_cache_score_tolerance):
                corrected_full, metadata = self._run_single(image, cached_best_method, cached_best_params, channel_name, pixel_size)
                method_suffix = f'_{self.backend}' if self.backend != 'cpu' else ''
                metadata.update(
                    {
                        "auto_selected": True,
                        "auto_mode": "cache_best_fastpath",
                        "auto_sample_indices": sample_idx.tolist(),
                        "auto_cache_path": str(self.auto_cache_path),
                        "auto_cache_hit": True,
                        "auto_cache_used_best": True,
                        "auto_cache_skip_search": True,
                        "auto_cache_cached_best_score": float(cached_best_score),
                        "auto_cache_validated_score": float(score),
                        "auto_cache_score_tolerance": float(auto_cache_score_tolerance),
                        "parameters_used": cached_best_params,
                        "method": (
                            f'gaussian+rolling_ball{method_suffix}'
                            if cached_best_method in {"two_stage", "gaussian_then_rolling_ball"}
                            else metadata.get("method", f"{cached_best_method}{method_suffix}")
                        ),
                    }
                )
                return corrected_full, metadata

        method_results: List[Dict[str, Any]] = []
        for method in AUTO_METHODS:
            seed_ranges = self._merge_seed_ranges(method, channel_entry)
            result = self._zoom_search_method(
                image_sample,
                method,
                seed_ranges,
                weights,
                max_slices=min(5, len(sample_idx)),
                channel_name=channel_name,
                pixel_size=pixel_size,
                is_negative_control=is_negative_control,
            )
            method_results.append(result)

        # Pick best method
        best_overall = max(method_results, key=lambda r: r.get("best_score", -np.inf))
        best_method = best_overall["method"]
        best_params = best_overall["best_params"]

        if best_params is None:
            raise RuntimeError("Auto background subtraction failed to find valid parameters")

        corrected_full, metadata = self._run_single(image, best_method, best_params, channel_name, pixel_size)
        method_suffix = f'_{self.backend}' if self.backend != 'cpu' else ''

        # Cache update
        cache.setdefault("channels", {})
        channel_cache = cache["channels"].setdefault(channel_key, {"methods": {}})
        for result in method_results:
            method_name = result["method"]
            tested_values = self._collect_tested_values(result["scores"])
            channel_cache["methods"][method_name] = {
                "best_params": result["best_params"],
                "best_score": result["best_score"],
                "tested_values": tested_values,
                "ranges_used": result["ranges_used"],
            }
        self._save_auto_cache(cache)

        top3 = sorted(
            [
                {
                    "method": r["method"],
                    "best_score": r.get("best_score"),
                    "best_params": r.get("best_params"),
                }
                for r in method_results
            ],
            key=lambda x: x.get("best_score", -np.inf),
            reverse=True,
        )[:3]

        metadata.update({
            'auto_selected': True,
            'auto_mode': 'zoom_search_v2',
            'auto_candidates_tested': sum(len(r["scores"]) for r in method_results),
            'auto_method_scores': [{'method': r['method'], 'best_score': r.get('best_score')} for r in method_results],
            'auto_top3_methods': top3,
            'auto_sample_indices': sample_idx.tolist(),
            'auto_cache_path': str(self.auto_cache_path),
            'auto_cache_hit': bool(channel_entry),
            'auto_ranges_used': best_overall.get("ranges_used", {}),
            'parameters_used': best_params,
            'method': f'gaussian+rolling_ball{method_suffix}' if best_method in {'two_stage', 'gaussian_then_rolling_ball'} else metadata.get('method', f'{best_method}{method_suffix}'),
            'is_negative_control': is_negative_control,
        })
        
        # Add validation metrics for negative controls
        if is_negative_control:
            validation_metrics = self._compute_negative_control_metrics(corrected_full)
            metadata['negative_control_validation'] = validation_metrics
        
        return corrected_full, metadata
    
    def _sync_backend(self) -> None:
        """Synchronize the current backend (wait for GPU operations to complete)."""
        if self.backend == 'cuda' and _ensure_cuda():
            cp.cuda.Stream.null.synchronize()
        elif self.backend == 'mps' and _ensure_mps():
            torch.mps.synchronize()
        # CPU doesn't need synchronization
    
    def process_from_loader(
        self,
        image_loader: ImageLoader,
        channel_index: Optional[int] = None,
        channel_name: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process image directly from ImageLoader instance.
        
        Args:
            image_loader: Initialized ImageLoader instance
            channel_index: Index of channel to process
            channel_name: Name of channel to process (alternative to index)
            method: Background subtraction method override
            **kwargs: Additional method parameters
            
        Returns:
            Tuple of (background_corrected_image, metadata_dict)
        """
        if not hasattr(image_loader, 'image_data') or image_loader.image_data is None:
            raise ValueError("ImageLoader must be loaded with image data")
        
        # Get the image data
        if channel_index is not None:
            if len(image_loader.image_data.shape) < 4:
                raise ValueError("Channel index specified but image is not multi-channel")
            image = image_loader.image_data[:, :, :, channel_index]
        elif channel_name is not None:
            if not hasattr(image_loader, 'metadata') or 'channels' not in image_loader.metadata:
                raise ValueError("Channel name specified but no channel metadata available")
            
            channels = image_loader.metadata['channels']
            try:
                channel_idx = channels.index(channel_name)
                image = image_loader.image_data[:, :, :, channel_idx]
            except ValueError:
                raise ValueError(f"Channel '{channel_name}' not found in {channels}")
        else:
            # Use entire image
            image = image_loader.image_data
        
        # Get pixel size from metadata if available
        pixel_size = None
        if hasattr(image_loader, 'metadata') and 'pixel_size' in image_loader.metadata:
            pixel_size = image_loader.metadata['pixel_size']
        
        # Process the image
        return self.subtract_background(
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            **kwargs
        )
    
    def batch_process(
        self,
        images: Dict[str, np.ndarray],
        channel_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        pixel_size: Optional[float] = None
    ) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process multiple images/channels in batch.
        
        Args:
            images: Dictionary mapping channel names to image arrays
            channel_configs: Optional per-channel configuration overrides
            pixel_size: Pixel size in micrometers
            
        Returns:
            Dictionary mapping channel names to (corrected_image, metadata) tuples
        """
        results = {}
        channel_configs = channel_configs or {}
        
        for channel_name, image in images.items():
            self.logger.info(f"Processing channel: {channel_name}")
            
            # Get channel-specific config
            channel_config = channel_configs.get(channel_name, {})
            
            # Process the image
            corrected_image, metadata = self.subtract_background(
                image=image,
                channel_name=channel_name,
                pixel_size=pixel_size,
                **channel_config
            )
            
            results[channel_name] = (corrected_image, metadata)
        
        return results
    
    @classmethod
    def from_preprocessing_config(cls, preprocessing_config: PreprocessingConfig) -> 'BackgroundSubtractor':
        """
        Create BackgroundSubtractor from a PreprocessingConfig.
        
        Args:
            preprocessing_config: Full preprocessing configuration
            
        Returns:
            BackgroundSubtractor instance
        """
        return cls(config=preprocessing_config.background_subtraction)
    
    def get_recommended_parameters(self, channel_name: str) -> Dict[str, Any]:
        """
        Get recommended parameters for a specific channel type.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Dictionary of recommended parameters
        """
        channel_lower = channel_name.lower()
        
        if any(name in channel_lower for name in ['dapi', 'hoechst']):
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_dapi,
                'description': 'DAPI/nuclear staining - rolling ball with large radius for nuclear regions'
            }
        elif any(name in channel_lower for name in ['phalloidin', 'actin']):
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_phalloidin,
                'description': 'Phalloidin/actin - smaller radius to preserve filament structures'
            }
        elif any(name in channel_lower for name in ['lamp1', 'lysosome']):
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_protein,
                'description': 'LAMP1 punctate structures - medium radius for vesicle preservation'
            }
        else:
            return {
                'method': 'rolling_ball',
                'radius': self.config.rolling_ball_radius_protein,
                'description': 'Protein marker - standard radius for punctate/diffuse patterns'
            }
    
    def plot_background_subtraction_comparison(self, 
                                             original_data: np.ndarray,
                                             corrected_results: Dict[str, Tuple[np.ndarray, Dict]],
                                             channel_names: List[str],
                                             z_slice: Optional[int] = None,
                                             figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Create a detailed comparison plot showing original vs background-subtracted images.
        
        This method generates a comprehensive visualization with:
        - Original and corrected images side by side
        - Intensity histograms for both original and corrected data
        - Statistics and method information for each channel
        
        Args:
            original_data: Original 3D image data (Z, Y, X, C) from ImageLoader
            corrected_results: Dictionary mapping channel names to (corrected_image, metadata) tuples.
                              Can come from batch_process(), individual subtract_background() calls,
                              or any custom processing results.
            channel_names: List of channel names corresponding to the data
            z_slice: Z-slice index to display. If None, uses middle slice
            figsize: Figure size tuple (width, height). If None, auto-calculates
            
        Returns:
            Figure: Matplotlib figure object for further customization
            
        Examples:
            # From batch processing:
            >>> results = bg_subtractor.batch_process(images_dict, pixel_size)
            >>> fig = bg_subtractor.plot_background_subtraction_comparison(
            ...     loaded_data, results, channel_names
            ... )
            
            # From individual processing:
            >>> corrected_img, metadata = bg_subtractor.subtract_background(
            ...     channel_data, channel_name="DAPI", pixel_size=pixel_size
            ... )
            >>> results = {"DAPI": (corrected_img, metadata)}
            >>> fig = bg_subtractor.plot_background_subtraction_comparison(
            ...     loaded_data, results, ["DAPI"]
            ... )
            
            # Compare different parameter settings:
            >>> results1 = {"DAPI": bg_subtractor.subtract_background(data, radius=50, ...)}
            >>> results2 = {"DAPI": bg_subtractor.subtract_background(data, radius=100, ...)}
            >>> # Plot each separately to compare
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            self.logger.warning("matplotlib not available; cannot generate comparison plot: %s", exc)
            return None  # type: ignore[return-value]

        if z_slice is None:
            z_slice = original_data.shape[0] // 2
        
        n_channels = len(channel_names)
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (5 * n_channels, 12)
        
        # Create subplots: 3 rows (original, corrected, histogram) x n_channels
        fig, axes = plt.subplots(3, n_channels, figsize=figsize)
        
        # Handle single channel case
        if n_channels == 1:
            axes = axes.reshape(-1, 1)
        
        for i, channel_name in enumerate(channel_names):
            # Get data for this channel
            original_channel = original_data[:, :, :, i]
            corrected_img, metadata = corrected_results[channel_name]
            
            # Calculate display ranges using percentiles for better visualization
            orig_p1, orig_p99 = np.percentile(original_channel, [1, 99])
            corr_p1, corr_p99 = np.percentile(corrected_img, [1, 99])
            
            # Row 1: Original images
            im_orig = axes[0, i].imshow(original_channel[z_slice], 
                                       cmap='viridis', 
                                       vmin=orig_p1, vmax=orig_p99)
            axes[0, i].set_title(f'Original {channel_name}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            plt.colorbar(im_orig, ax=axes[0, i], shrink=0.8, label='Intensity')
            
            # Row 2: Corrected images
            im_corr = axes[1, i].imshow(corrected_img[z_slice], 
                                       cmap='viridis', 
                                       vmin=corr_p1, vmax=corr_p99)
            
            # Create informative title with method and parameters
            method = metadata['method']
            params = metadata.get('parameters_used', {})
            if method == 'rolling_ball':
                radius = params.get('radius', 'N/A')
                title = f'Corrected {channel_name}\n{method} (radius={radius})'
            elif method == 'gaussian':
                sigma = params.get('sigma', 'N/A')
                title = f'Corrected {channel_name}\n{method} (σ={sigma})'
            else:
                title = f'Corrected {channel_name}\n{method}'
            
            axes[1, i].set_title(title, fontsize=12, fontweight='bold')
            axes[1, i].axis('off')
            plt.colorbar(im_corr, ax=axes[1, i], shrink=0.8, label='Intensity')
            
            # Row 3: Histograms
            # Sample data for histogram (every 10th pixel to speed up plotting)
            orig_sample = original_channel.flatten()[::10]
            corr_sample = corrected_img.flatten()[::10]
            
            # Calculate statistics for display
            orig_mean = np.mean(orig_sample)
            orig_std = np.std(orig_sample)
            corr_mean = np.mean(corr_sample)
            corr_std = np.std(corr_sample)
            
            axes[2, i].hist(orig_sample, bins=50, alpha=0.7, label='Original', 
                           density=True, color='blue', edgecolor='black', linewidth=0.5)
            axes[2, i].hist(corr_sample, bins=50, alpha=0.7, label='Corrected', 
                           density=True, color='red', edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            stats_text = (f'Original: {orig_mean:.1f} ± {orig_std:.1f}\n'
                         f'Corrected: {corr_mean:.1f} ± {corr_std:.1f}')
            axes[2, i].text(0.02, 0.98, stats_text, transform=axes[2, i].transAxes,
                           verticalalignment='top', fontsize=9, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[2, i].set_title(f'{channel_name} Intensity Distribution', fontsize=11)
            axes[2, i].set_yscale('log')
            axes[2, i].legend(fontsize=10)
            axes[2, i].set_xlabel('Intensity', fontsize=10)
            axes[2, i].set_ylabel('Density (log)', fontsize=10)
            axes[2, i].grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Background Subtraction Analysis - Z-slice {z_slice}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        
        return fig
    
    def compare_parameter_settings(self,
                                 original_data: np.ndarray,
                                 channel_name: str,
                                 channel_index: int,
                                 parameter_sets: List[Dict[str, Any]],
                                 pixel_size: float,
                                 z_slice: Optional[int] = None,
                                 figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Compare different parameter settings for background subtraction on a single channel.
        
        This is useful for parameter optimization - you can test different radius values,
        methods, or other parameters and see their effects side by side.
        
        Args:
            original_data: Original 3D image data (Z, Y, X, C) from ImageLoader
            channel_name: Name of the channel to process
            channel_index: Index of the channel in the original data
            parameter_sets: List of parameter dictionaries to test
            pixel_size: Pixel size in micrometers
            z_slice: Z-slice index to display. If None, uses middle slice
            figsize: Figure size tuple (width, height). If None, auto-calculates
            
        Returns:
            Figure: Matplotlib figure object showing comparison of all parameter sets
            
        Example:
            >>> parameter_sets = [
            ...     {'method': 'rolling_ball', 'radius': 50},
            ...     {'method': 'rolling_ball', 'radius': 100},
            ...     {'method': 'rolling_ball', 'radius': 150},
            ...     {'method': 'gaussian', 'sigma': 2.0}
            ... ]
            >>> fig = bg_subtractor.compare_parameter_settings(
            ...     loaded_data, "DAPI", 0, parameter_sets, pixel_size
            ... )
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            self.logger.warning("matplotlib not available; cannot generate comparison plot: %s", exc)
            return None  # type: ignore[return-value]

        if z_slice is None:
            z_slice = original_data.shape[0] // 2
        
        n_params = len(parameter_sets)
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (5 * n_params, 10)
        
        # Create subplots: 2 rows (original, corrected) x n_params
        fig, axes = plt.subplots(2, n_params, figsize=figsize)
        
        # Handle single parameter case
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        
        # Get original channel data
        original_channel = original_data[:, :, :, channel_index]
        orig_p1, orig_p99 = np.percentile(original_channel, [1, 99])
        
        # Process each parameter set
        for i, params in enumerate(parameter_sets):
            # Process with these parameters
            corrected_img, metadata = self.subtract_background(
                image=original_channel,
                channel_name=channel_name,
                pixel_size=pixel_size,
                **params
            )
            
            # Calculate display range for corrected image
            corr_p1, corr_p99 = np.percentile(corrected_img, [1, 99])
            
            # Row 1: Original image (show only once, in first column)
            if i == 0:
                im_orig = axes[0, i].imshow(original_channel[z_slice], 
                                           cmap='viridis', 
                                           vmin=orig_p1, vmax=orig_p99)
                axes[0, i].set_title(f'Original {channel_name}', fontsize=12, fontweight='bold')
                axes[0, i].axis('off')
                plt.colorbar(im_orig, ax=axes[0, i], shrink=0.8, label='Intensity')
            else:
                axes[0, i].axis('off')  # Hide other original images
            
            # Row 2: Corrected images
            im_corr = axes[1, i].imshow(corrected_img[z_slice], 
                                       cmap='viridis', 
                                       vmin=corr_p1, vmax=corr_p99)
            
            # Create informative title with parameters
            method = metadata['method']
            params_used = metadata.get('parameters_used', {})
            if method == 'rolling_ball':
                radius = params_used.get('radius', 'N/A')
                title = f'{method}\nradius={radius}'
            elif method == 'gaussian':
                sigma = params_used.get('sigma', 'N/A')
                title = f'{method}\nσ={sigma}'
            else:
                title = f'{method}'
            
            axes[1, i].set_title(title, fontsize=11, fontweight='bold')
            axes[1, i].axis('off')
            plt.colorbar(im_corr, ax=axes[1, i], shrink=0.8, label='Intensity')
        
        # Add overall title
        fig.suptitle(f'Parameter Comparison - {channel_name} (Z-slice {z_slice})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return fig
    
    # ==================== GPU UTILITY METHODS ====================
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about GPU availability and memory."""
        if not self.use_cuda or not _ensure_cuda():
            return {
                'cuda_available': False,
                'gpu_accelerated': False,
                'reason': 'CUDA libraries not available or disabled'
            }
        
        try:
            gpu_mem = cp.cuda.Device().mem_info
            total_mem = gpu_mem[1] / (1024**3)
            free_mem = gpu_mem[0] / (1024**3)
            
            return {
                'cuda_available': True,
                'gpu_accelerated': True,
                'gpu_name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                'total_memory_gb': total_mem,
                'free_memory_gb': free_mem,
                'max_usable_memory_gb': self.max_gpu_memory_gb,
                'cupy_version': cp.__version__
            }
        except Exception as e:
            return {
                'cuda_available': True,
                'gpu_accelerated': False,
                'error': str(e)
            }
    
    def benchmark_methods(self, image: np.ndarray, channel_name: str = "test") -> Dict[str, Any]:
        """Benchmark GPU performance for different methods (CUDA-only)."""
        import time
        
        results = {}
        methods = ['rolling_ball', 'gaussian', 'morphological']
        
        for method in methods:
            self.logger.info(f"Benchmarking {method} method...")
            
            # Test GPU performance only
            start_time = time.time()
            try:
                gpu_result, gpu_metadata = self._subtract_background_cuda(
                    image, method, channel_name, None
                )
                # Ensure timing includes GPU work
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.time() - start_time
                results[f'{method}_gpu'] = {
                    'time_seconds': gpu_time,
                    'success': True,
                    'metadata': gpu_metadata
                }
            except Exception as e:
                results[f'{method}_gpu'] = {
                    'time_seconds': None,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    