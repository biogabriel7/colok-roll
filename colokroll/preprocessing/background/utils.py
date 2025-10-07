from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - optional GPU path
    import cupy as cp  # type: ignore
    HAS_CUDA = True
except Exception:  # pragma: no cover - CPU fallback
    cp = None  # type: ignore
    HAS_CUDA = False


def apply_bleedthrough_unmix(img: np.ndarray, mix_matrix: np.ndarray) -> np.ndarray:
    """
    Linear unmixing to correct spectral bleed-through.

    true = inv(M) @ measured

    - img shape: (Z, Y, X, C)
    - mix_matrix shape: (C, C), where measured = M @ true
    """
    if img.ndim != 4:
        raise ValueError(f"Expected 4D image (Z,Y,X,C); got {img.shape}")
    if mix_matrix.ndim != 2 or mix_matrix.shape[0] != mix_matrix.shape[1]:
        raise ValueError("mix_matrix must be square (C x C)")

    channels = img.shape[-1]
    if mix_matrix.shape[0] != channels:
        raise ValueError(f"mix_matrix size {mix_matrix.shape} != channels {channels}")

    if HAS_CUDA:
        mod = cp  # type: ignore[assignment]
    else:
        mod = np

    M = mod.asarray(mix_matrix, dtype=mod.float32)
    Minv = mod.linalg.inv(M)
    data = mod.asarray(img, dtype=mod.float32)

    # reshape to (N, C), unmix with right-multiply by Minv.T, then reshape back
    num_voxels = int(data.size // channels)
    data_2d = data.reshape(num_voxels, channels)
    unmixed = data_2d @ Minv.T
    out = unmixed.reshape(img.shape).astype(mod.float32, copy=False)

    # Preserve backend: return cp.ndarray when CUDA is active, otherwise numpy
    return out  # type: ignore[return-value]


def subtract_background_percentile_roi(
    img: np.ndarray,
    roi_union_2d: np.ndarray,
    *,
    percentile: float = 1.0,
) -> np.ndarray:
    """
    Subtract per-channel background estimated from outside the ROI union.

    - img shape: (Z, Y, X, C)
    - roi_union_2d shape: (Y, X) boolean mask of the analysis region (labels > 0)
    - percentile: background estimate percentile (e.g., 1.0)

    Falls back to global percentile if no outside-ROI pixels exist.
    """
    if img.ndim != 4:
        raise ValueError(f"Expected 4D image (Z,Y,X,C); got {img.shape}")
    if roi_union_2d.ndim != 2:
        raise ValueError(f"roi_union_2d must be 2D; got {roi_union_2d.shape}")

    z, h, w, c = img.shape
    bg_mask_2d = ~roi_union_2d.astype(bool)

    if HAS_CUDA:
        mod = cp  # type: ignore[assignment]
        bg_mask_zyx = cp.broadcast_to(cp.asarray(bg_mask_2d)[cp.newaxis, ...], (z, h, w))
        out = cp.asarray(img, dtype=cp.float32)
    else:
        mod = np
        bg_mask_zyx = np.broadcast_to(bg_mask_2d[np.newaxis, ...], (z, h, w))
        out = np.asarray(img, dtype=np.float32).copy()

    for ch in range(c):
        vals_out = out[..., ch][bg_mask_zyx]
        if int(vals_out.size) == 0:
            vals_all = out[..., ch].reshape(-1)
            bg_val = float(cp.asnumpy(cp.percentile(vals_all, percentile)) if HAS_CUDA else np.percentile(vals_all, percentile))
        else:
            bg_val = float(cp.asnumpy(cp.percentile(vals_out, percentile)) if HAS_CUDA else np.percentile(vals_out, percentile))
        out[..., ch] = mod.maximum(out[..., ch] - bg_val, 0.0)

    # Preserve backend: return cp.ndarray when CUDA is active, otherwise numpy
    return out  # type: ignore[return-value]


