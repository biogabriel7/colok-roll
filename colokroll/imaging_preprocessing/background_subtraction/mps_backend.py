from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

MPS_AVAILABLE = False
torch = None
kornia = None


def _ensure_mps() -> bool:
    """Lazy-load MPS (PyTorch + kornia) dependencies for this backend module."""
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


def subtract_background_mps(
    owner: "BackgroundSubtractor",
    image: np.ndarray,
    method: str,
    channel_name: Optional[str],
    pixel_size: Optional[float],
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """MPS-accelerated background subtraction using PyTorch + kornia."""
    if not _ensure_mps():
        raise RuntimeError("MPS backend selected but PyTorch/kornia are unavailable")
    if kornia is None or torch is None:
        raise RuntimeError("MPS dependencies missing; install torch with MPS and kornia")

    params = owner._get_method_parameters(method, channel_name, pixel_size, **kwargs)
    img_tensor = torch.from_numpy(image.astype(np.float32)).to(owner.device)

    if method == "gaussian":
        corrected_tensor, metadata = _gaussian_subtraction_mps(img_tensor, params)
    elif method in {"two_stage", "gaussian_then_rolling_ball"}:
        corrected_tensor, metadata = _two_stage_subtraction_mps(owner, img_tensor, params)
    elif method == "rolling_ball":
        corrected_tensor, metadata = _rolling_ball_subtraction_mps(owner, img_tensor, params)
    elif method == "morphological":
        corrected_tensor, metadata = _morphological_subtraction_mps(owner, img_tensor, params)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = corrected_tensor.cpu().numpy()

    if owner.config.clip_negative_values:
        result = np.clip(result, 0, None)
    if owner.config.normalize_output:
        result = owner._normalize_image(result)

    metadata.update(
        {
            "method": f"{method}_mps",
            "original_shape": image.shape,
            "parameters_used": params,
            "gpu_accelerated": True,
            "backend": "mps",
        }
    )

    return result, metadata


def _gaussian_subtraction_mps(
    img_tensor: "torch.Tensor", params: Dict[str, Any]
) -> Tuple["torch.Tensor", Dict[str, Any]]:
    """MPS Gaussian background subtraction using kornia."""
    sigma = params["sigma"]
    z_slices = img_tensor.shape[0]
    corrected = torch.empty_like(img_tensor)

    kernel_size = int(6 * sigma) | 1  # Make odd

    for z in range(z_slices):
        slice_4d = img_tensor[z : z + 1, None, :, :]  # (1, 1, H, W)
        background = kornia.filters.gaussian_blur2d(
            slice_4d, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma)
        )
        corrected[z] = (slice_4d - background).squeeze()

    metadata = {"background_method": "gaussian_mps", "sigma": sigma, "kernel_size": kernel_size}
    return corrected, metadata


def _two_stage_subtraction_mps(
    owner: "BackgroundSubtractor", img_tensor: "torch.Tensor", params: Dict[str, Any]
) -> Tuple["torch.Tensor", Dict[str, Any]]:
    """MPS two-stage subtraction: Gaussian → morphological opening."""
    sigma = float(params["sigma_stage1"])
    radius = int(params["radius_stage2"])
    light_background = bool(params.get("light_background", False))

    z_slices = img_tensor.shape[0]
    kernel_size = int(6 * sigma) | 1

    gauss_corr = torch.empty_like(img_tensor)
    for z in range(z_slices):
        slice_4d = img_tensor[z : z + 1, None, :, :]
        background = kornia.filters.gaussian_blur2d(
            slice_4d, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma)
        )
        gauss_corr[z] = (slice_4d - background).squeeze()

    if getattr(owner.config, "clip_negative_values", False):
        gauss_corr = torch.clamp(gauss_corr, min=0)

    morph_kernel = torch.ones(2 * radius + 1, 2 * radius + 1, device=owner.device)
    corrected = torch.empty_like(gauss_corr)

    for z in range(z_slices):
        slice_4d = gauss_corr[z : z + 1, None, :, :]
        if light_background:
            background = kornia.morphology.closing(slice_4d, morph_kernel)
        else:
            background = kornia.morphology.opening(slice_4d, morph_kernel)
        corrected[z] = (slice_4d - background).squeeze()

    metadata = {
        "background_method": "two_stage_mps",
        "stage1": {"sigma": sigma},
        "stage2": {"radius": radius, "light_background": light_background},
    }
    return corrected, metadata


def _rolling_ball_subtraction_mps(
    owner: "BackgroundSubtractor", img_tensor: "torch.Tensor", params: Dict[str, Any]
) -> Tuple["torch.Tensor", Dict[str, Any]]:
    """MPS rolling ball approximation via morphological opening."""
    radius = params["radius"]
    light_background = params.get("light_background", False)

    z_slices = img_tensor.shape[0]
    morph_kernel = torch.ones(2 * radius + 1, 2 * radius + 1, device=owner.device)
    corrected = torch.empty_like(img_tensor)

    for z in range(z_slices):
        slice_4d = img_tensor[z : z + 1, None, :, :]
        if light_background:
            background = kornia.morphology.closing(slice_4d, morph_kernel)
        else:
            background = kornia.morphology.opening(slice_4d, morph_kernel)
        corrected[z] = (slice_4d - background).squeeze()

    metadata = {
        "background_method": "rolling_ball_mps",
        "radius": radius,
        "light_background": light_background,
    }

    return corrected, metadata


def _morphological_subtraction_mps(
    owner: "BackgroundSubtractor", img_tensor: "torch.Tensor", params: Dict[str, Any]
) -> Tuple["torch.Tensor", Dict[str, Any]]:
    """MPS morphological background subtraction."""
    size = params["size"]
    z_slices = img_tensor.shape[0]
    morph_kernel = torch.ones(2 * size + 1, 2 * size + 1, device=owner.device)
    corrected = torch.empty_like(img_tensor)

    for z in range(z_slices):
        slice_4d = img_tensor[z : z + 1, None, :, :]
        background = kornia.morphology.closing(slice_4d, morph_kernel)
        corrected[z] = (slice_4d - background).squeeze()

    metadata = {"background_method": "morphological_mps", "size": size}
    return corrected, metadata

