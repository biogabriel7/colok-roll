from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

CUDA_AVAILABLE = False
cp = None
cp_ndimage = None


def _ensure_cuda() -> bool:
    """Lazy-load CUDA dependencies for this backend module."""
    global cp, cp_ndimage, CUDA_AVAILABLE
    if CUDA_AVAILABLE:
        return True
    try:
        import cupy as _cp  # type: ignore
        import cupyx.scipy.ndimage as _cp_ndimage  # type: ignore

        cp = _cp
        cp_ndimage = _cp_ndimage
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False
    return CUDA_AVAILABLE


def subtract_background_cuda(
    owner: "BackgroundSubtractor",
    image: np.ndarray,
    method: str,
    channel_name: Optional[str],
    pixel_size: Optional[float],
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CUDA-accelerated background subtraction entrypoint."""
    if not _ensure_cuda():
        raise RuntimeError("CUDA backend selected but CuPy is unavailable")
    if cp_ndimage is None:
        raise RuntimeError("CuPy ndimage module not available; reinstall cupy/cupyx")
    channel_label = channel_name or "unknown"
    try:
        # Check if image fits in GPU memory
        image_memory_gb = image.nbytes / (1024**3)
        if image_memory_gb > owner.max_gpu_memory_gb:
            owner.logger.error(
                "Image too large for GPU memory (%.1fGB > %.1fGB)",
                image_memory_gb,
                owner.max_gpu_memory_gb,
            )
            raise MemoryError(
                "Input image exceeds available GPU memory. Consider reducing image size or parameters."
            )

        # Transfer image to GPU
        gpu_image = cp.asarray(image, dtype=cp.float32)

        # Get method-specific parameters
        params = owner._get_method_parameters(method, channel_name, pixel_size, **kwargs)

        # Apply background subtraction
        if method == "rolling_ball":
            owner.logger.info(
                "CUDA rolling_ball (channel=%s, radius=%s, z=%d)", channel_label, params.get("radius"), gpu_image.shape[0]
            )
            corrected_image, metadata = _rolling_ball_subtraction_3d_cuda(owner, gpu_image, params)
        elif method == "gaussian":
            owner.logger.info("CUDA gaussian (channel=%s, sigma=%s, z=%d)", channel_label, params.get("sigma"), gpu_image.shape[0])
            corrected_image, metadata = _gaussian_subtraction_3d_cuda(owner, gpu_image, params)
        elif method == "morphological":
            owner.logger.info(
                "CUDA morphological (channel=%s, size=%s, shape=%s, z=%d)",
                channel_label,
                params.get("size"),
                params.get("shape"),
                gpu_image.shape[0],
            )
            corrected_image, metadata = _morphological_subtraction_3d_cuda(owner, gpu_image, params)
        elif method in {"two_stage", "gaussian_then_rolling_ball"}:
            owner.logger.info(
                "CUDA two_stage (channel=%s, sigma1=%s, radius2=%s, light_bg=%s, z=%d)",
                channel_label,
                params.get("sigma_stage1"),
                params.get("radius_stage2"),
                params.get("light_background", False),
                gpu_image.shape[0],
            )
            corrected_image, metadata = _two_stage_subtraction_3d_cuda(owner, gpu_image, params)
        else:
            raise ValueError(f"Unknown background subtraction method: {method}")

        # Transfer result back to CPU
        result = cp.asnumpy(corrected_image)

        # Clean up GPU memory
        del gpu_image, corrected_image
        cp.get_default_memory_pool().free_all_blocks()

        # Post-processing
        if owner.config.clip_negative_values:
            result = np.clip(result, 0, None)

        if owner.config.normalize_output:
            result = owner._normalize_image(result)

        # Update metadata
        metadata.update(
            {
                "method": f"{method}_cuda",
                "original_shape": image.shape,
                "original_dtype": str(image.dtype),
                "clipped_negative": owner.config.clip_negative_values,
                "normalized": owner.config.normalize_output,
                "parameters_used": params,
                "gpu_accelerated": True,
                "gpu_memory_used_gb": image_memory_gb,
            }
        )

        return result, metadata

    except Exception as e:
        owner.logger.error("CUDA processing failed: %s", e)
        raise


def _rolling_ball_subtraction_3d_cuda(
    owner: "BackgroundSubtractor", gpu_image: "cp.ndarray", params: Dict[str, Any]
) -> Tuple["cp.ndarray", Dict[str, Any]]:
    """CUDA-accelerated rolling ball background subtraction (morphological approximation)."""
    radius = params["radius"]
    z_slices, _, _ = gpu_image.shape

    owner.logger.info("Processing %d z-slices with CUDA rolling ball approximation (radius=%s)", z_slices, radius)

    selem = _create_disk_selem_cuda(radius)
    corrected_image = cp.empty_like(gpu_image, dtype=cp.float32)

    background_means: list[float] = []
    background_stds: list[float] = []
    light_background = params.get("light_background", False)

    for z in range(z_slices):
        if light_background:
            background = cp_ndimage.grey_closing(gpu_image[z], footprint=selem)
        else:
            background = cp_ndimage.grey_opening(gpu_image[z], footprint=selem)
        corrected_image[z] = gpu_image[z].astype(cp.float32) - background.astype(cp.float32)
        background_means.append(float(cp.mean(background)))
        background_stds.append(float(cp.std(background)))

    metadata = {
        "background_method": "rolling_ball_3d_cuda",
        "radius_pixels": radius,
        "z_slices_processed": z_slices,
        "background_stats": {
            "mean_across_slices": float(np.mean(background_means)),
            "std_across_slices": float(np.mean(background_stds)),
            "gpu_processing": True,
        },
    }

    return corrected_image, metadata


def _create_rolling_ball_background_cuda(
    owner: "BackgroundSubtractor", gpu_slice: "cp.ndarray", radius: int, light_background: bool = False
) -> "cp.ndarray":
    """Create rolling ball background on GPU using proper rolling ball algorithm."""
    cpu_slice = cp.asnumpy(gpu_slice)
    cpu_background = owner._create_rolling_ball_background(cpu_slice, radius, light_background)
    return cp.asarray(cpu_background, dtype=cp.float32)


def _create_disk_selem_cuda(radius: int) -> "cp.ndarray":
    if not _ensure_cuda():
        raise RuntimeError("CUDA dependencies not available for structuring element creation")
    r = int(radius)
    y, x = cp.ogrid[-r : r + 1, -r : r + 1]
    mask = x * x + y * y <= r * r
    return mask.astype(cp.uint8)


def _create_rect_selem_cuda(h: int, w: Optional[int] = None) -> "cp.ndarray":
    if not _ensure_cuda():
        raise RuntimeError("CUDA dependencies not available for structuring element creation")
    w = h if w is None else w
    return cp.ones((int(h), int(w)), dtype=cp.uint8)


def _gaussian_subtraction_3d_cuda(
    owner: "BackgroundSubtractor", gpu_image: "cp.ndarray", params: Dict[str, Any]
) -> Tuple["cp.ndarray", Dict[str, Any]]:
    sigma = params["sigma"]
    owner.logger.info("Applying CUDA 3D Gaussian background subtraction with sigma=%s", sigma)

    background = cp_ndimage.gaussian_filter(gpu_image, sigma=sigma)
    corrected_image = gpu_image.astype(cp.float32) - background.astype(cp.float32)

    metadata = {
        "background_method": "gaussian_3d_cuda",
        "sigma": sigma,
        "background_stats": {
            "mean": float(cp.mean(background)),
            "std": float(cp.std(background)),
            "min": float(cp.min(background)),
            "max": float(cp.max(background)),
            "gpu_processing": True,
        },
    }

    del background
    cp.get_default_memory_pool().free_all_blocks()

    return corrected_image, metadata


def _two_stage_subtraction_3d_cuda(
    owner: "BackgroundSubtractor", gpu_image: "cp.ndarray", params: Dict[str, Any]
) -> Tuple["cp.ndarray", Dict[str, Any]]:
    """Two-stage CUDA background subtraction: Gaussian → Rolling Ball."""
    sigma = float(params["sigma_stage1"])
    radius = int(params["radius_stage2"])
    light_background = bool(params.get("light_background", False))

    gauss_bg = cp_ndimage.gaussian_filter(gpu_image, sigma=sigma)
    gauss_corr = gpu_image.astype(cp.float32) - gauss_bg.astype(cp.float32)
    if getattr(owner.config, "clip_negative_values", False):
        gauss_corr = cp.maximum(gauss_corr, 0)

    selem = _create_disk_selem_cuda(radius)
    z_slices = gauss_corr.shape[0]
    final_corr = cp.empty_like(gauss_corr, dtype=cp.float32)
    for z in range(z_slices):
        if light_background:
            background = cp_ndimage.grey_closing(gauss_corr[z], footprint=selem)
        else:
            background = cp_ndimage.grey_opening(gauss_corr[z], footprint=selem)
        final_corr[z] = gauss_corr[z] - background.astype(cp.float32)

    metadata = {
        "background_method": "two_stage_cuda",
        "stage1": {"sigma": sigma},
        "stage2": {"radius": radius, "light_background": light_background},
    }
    return final_corr, metadata


def _morphological_subtraction_3d_cuda(
    owner: "BackgroundSubtractor", gpu_image: "cp.ndarray", params: Dict[str, Any]
) -> Tuple["cp.ndarray", Dict[str, Any]]:
    size = params["size"]
    shape = params.get("shape", "disk")
    z_slices = gpu_image.shape[0]

    if shape == "disk":
        selem = _create_disk_selem_cuda(size)
    elif shape == "square":
        selem = _create_rect_selem_cuda(size, size)
    else:
        selem = _create_disk_selem_cuda(size)

    corrected_image = cp.empty_like(gpu_image, dtype=cp.float32)
    background_means: list[float] = []
    background_stds: list[float] = []

    for z in range(z_slices):
        background = cp_ndimage.grey_closing(gpu_image[z], footprint=selem)
        corrected_image[z] = gpu_image[z].astype(cp.float32) - background.astype(cp.float32)
        background_means.append(float(cp.mean(background)))
        background_stds.append(float(cp.std(background)))

    metadata = {
        "background_method": "morphological_3d_cuda",
        "size": size,
        "shape": shape,
        "z_slices_processed": z_slices,
        "background_stats": {
            "mean_across_slices": float(np.mean(background_means)),
            "std_across_slices": float(np.mean(background_stds)),
            "gpu_processing": True,
        },
    }

    return corrected_image, metadata

