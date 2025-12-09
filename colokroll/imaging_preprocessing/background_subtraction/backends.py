from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
from .cpu_backend import subtract_background_cpu
from .cuda_backend import subtract_background_cuda
from .mps_backend import subtract_background_mps


class BackendAdapter(Protocol):
    """Uniform interface for backend-specific subtraction implementations."""

    def subtract(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        chunk_size: Optional[int],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        ...


class CpuAdapter:
    """CPU backend adapter delegating to BackgroundSubtractor CPU methods."""

    def __init__(self, owner: "BackgroundSubtractor") -> None:
        self.owner = owner

    def subtract(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        chunk_size: Optional[int],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return subtract_background_cpu(
            self.owner,
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            chunk_size=chunk_size,
            **kwargs,
        )


class CudaAdapter:
    """CUDA backend adapter delegating to BackgroundSubtractor CUDA methods."""

    def __init__(self, owner: "BackgroundSubtractor") -> None:
        self.owner = owner

    def subtract(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        chunk_size: Optional[int],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # chunk_size unused on CUDA; kept for uniform signature
        return subtract_background_cuda(
            self.owner,
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            **kwargs,
        )


class MpsAdapter:
    """MPS backend adapter delegating to BackgroundSubtractor MPS methods."""

    def __init__(self, owner: "BackgroundSubtractor") -> None:
        self.owner = owner

    def subtract(
        self,
        image: np.ndarray,
        method: str,
        channel_name: Optional[str],
        pixel_size: Optional[float],
        chunk_size: Optional[int],
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # chunk_size unused on MPS; kept for uniform signature
        return subtract_background_mps(
            self.owner,
            image=image,
            method=method,
            channel_name=channel_name,
            pixel_size=pixel_size,
            **kwargs,
        )

