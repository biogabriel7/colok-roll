from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def subtract_background_cpu(
    owner: "BackgroundSubtractor",
    image: np.ndarray,
    method: str,
    channel_name: Optional[str],
    pixel_size: Optional[float],
    chunk_size: Optional[int],
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CPU implementation of background subtraction (delegates heavy ops to owner)."""
    # Determine chunk size for memory efficiency
    if chunk_size is None:
        chunk_size = owner._calculate_optimal_chunk_size(image.shape, image.dtype)

    # Get method-specific parameters
    params = owner._get_method_parameters(method, channel_name, pixel_size, **kwargs)

    # Apply background subtraction with chunked processing
    if method == "rolling_ball":
        corrected_image, metadata = owner._rolling_ball_subtraction_3d(image, params, chunk_size)
    elif method == "gaussian":
        corrected_image, metadata = owner._gaussian_subtraction_3d(image, params)
    elif method == "morphological":
        corrected_image, metadata = owner._morphological_subtraction_3d(image, params, chunk_size)
    elif method in {"two_stage", "gaussian_then_rolling_ball"}:
        corrected_image, metadata = owner._two_stage_subtraction_3d_cpu(image, params, chunk_size)
    else:
        raise ValueError(f"Unknown background subtraction method: {method}")

    # Post-processing
    if owner.config.clip_negative_values:
        corrected_image = np.clip(corrected_image, 0, None)

    if owner.config.normalize_output:
        corrected_image = owner._normalize_image(corrected_image)

    # Update metadata
    metadata.update(
        {
            "method": method,
            "original_shape": image.shape,
            "original_dtype": str(image.dtype),
            "clipped_negative": owner.config.clip_negative_values,
            "normalized": owner.config.normalize_output,
            "parameters_used": params,
            "chunk_size_used": chunk_size,
            "memory_efficient": chunk_size < image.shape[0],
            "gpu_accelerated": False,
        }
    )

    return corrected_image, metadata

