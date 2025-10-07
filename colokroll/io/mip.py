"""Maximum intensity projection utilities."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile

from ..config import ProjectionConfig
from ..core.utils import ensure_directory
from .loaders import ImageLoader

logger = logging.getLogger(__name__)

# Conversion factor from 16-bit to 8-bit: 65535 / 255 = 257
# Used for efficient downscaling: (uint16_value // 257) ≈ (uint16_value * 255 / 65535)
UINT16_TO_UINT8_DIVISOR = 257


class MIPCreator:
    """Create Maximum Intensity Projections from z-stack images."""

    def __init__(self, config: Optional[ProjectionConfig] = None):
        self.config = config or ProjectionConfig()
        self.quality_metrics: Dict[str, Any] = {}

    def create_mip(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
        z_range: Optional[Tuple[int, int]] = None,
        channel: Optional[Union[int, str]] = None,
    ) -> np.ndarray:
        if image.ndim not in [3, 4]:
            raise ValueError(f"Image must be 3D or 4D z-stack, got {image.ndim}D")

        if image.ndim == 3:
            image = image[..., np.newaxis]
            single_channel = True
        else:
            single_channel = False

        method = method or self.config.projection_method
        z_range = z_range or self.config.z_range

        if z_range is not None:
            start, end = z_range
            if start < 0 or end > image.shape[0]:
                raise ValueError(f"Invalid z_range {z_range}. Valid range: (0, {image.shape[0]})")
            image = image[start:end]
            logger.info("Using z-slices %s to %s", start, end)

        if channel is not None:
            if isinstance(channel, str):
                raise ValueError("Channel names require ImageLoader context")
            if channel < 0 or channel >= image.shape[-1]:
                raise ValueError(f"Invalid channel {channel}. Valid range: 0-{image.shape[-1] - 1}")
            image = image[..., channel : channel + 1]

        logger.info("Creating %s projection from shape %s", method, image.shape)

        if method == "max":
            projection = np.max(image, axis=0)
        elif method == "mean":
            projection = np.mean(image, axis=0)
        elif method == "sum":
            projection = np.sum(image, axis=0)
        elif method == "median":
            projection = np.median(image, axis=0)
        elif method == "std":
            projection = np.std(image, axis=0)
        else:
            raise ValueError(f"Unknown projection method: {method}")

        if single_channel and projection.shape[-1] == 1:
            projection = projection[..., 0]

        if self.config.calculate_quality_metrics:
            self._calculate_quality_metrics(image, projection, method)

        logger.info("Created %s projection with shape %s", method, projection.shape)
        return projection

    def create_color_mip(
        self,
        image: np.ndarray,
        channel_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
        method: str = "max",
    ) -> np.ndarray:
        if image.ndim == 3:
            image = image[..., np.newaxis]

        mip = self.create_mip(image, method=method)

        if mip.ndim == 2:
            mip = mip[..., np.newaxis]

        if channel_colors is None:
            default_colors = [
                (0, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 0),
            ]
            channel_colors = {i: default_colors[i % len(default_colors)] for i in range(mip.shape[-1])}

        height, width, n_channels = mip.shape
        color_image = np.zeros((height, width, 3), dtype=np.float32)

        for c in range(n_channels):
            color = channel_colors.get(c, (1, 1, 1))
            channel_data = mip[..., c].astype(np.float32)
            if channel_data.max() > 0:
                channel_data = channel_data / channel_data.max()
            for i, color_val in enumerate(color):
                color_image[..., i] += channel_data * color_val

        return np.clip(color_image, 0, 1)

    def create_multi_method_mip(
        self,
        image: np.ndarray,
        methods: List[str] = ["max", "mean", "std"],
    ) -> Dict[str, np.ndarray]:
        projections: Dict[str, np.ndarray] = {}
        for method in methods:
            try:
                projections[method] = self.create_mip(image, method=method)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to create %s projection: %s", method, exc)
        return projections

    def _calculate_quality_metrics(
        self,
        z_stack: np.ndarray,
        projection: np.ndarray,
        method: str,
    ) -> None:
        metrics: Dict[str, Any] = {
            "method": method,
            "z_slices": z_stack.shape[0],
            "shape": projection.shape,
            "min_intensity": float(projection.min()),
            "max_intensity": float(projection.max()),
            "mean_intensity": float(projection.mean()),
            "std_intensity": float(projection.std()),
        }

        if projection.max() > projection.min():
            metrics["dynamic_range"] = float((projection.max() - projection.min()) / projection.max())
        else:
            metrics["dynamic_range"] = 0.0

        if projection.std() > 0:
            metrics["snr_estimate"] = float(projection.mean() / projection.std())
        else:
            metrics["snr_estimate"] = 0.0

        z_std = np.std(z_stack, axis=0)
        metrics["focus_variation"] = float(z_std.mean())
        metrics["coverage"] = float(np.sum(projection > 0) / projection.size)

        if projection.max() > 0:
            normalized = projection / projection.max()
            hist, _ = np.histogram(normalized.flatten(), bins=256)
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist = hist / hist.sum()
                metrics["entropy"] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            else:
                metrics["entropy"] = 0.0
        else:
            metrics["entropy"] = 0.0

        quality_score = 0.0
        if metrics["dynamic_range"] > 0.3:
            quality_score += 0.25
        if metrics["snr_estimate"] > 5:
            quality_score += 0.25
        if metrics["coverage"] > 0.1:
            quality_score += 0.25
        if metrics["entropy"] > 2:
            quality_score += 0.25

        metrics["quality_score"] = quality_score
        self.quality_metrics = metrics
        if quality_score < self.config.quality_threshold:
            warnings.warn(
                f"Low quality projection (score: {quality_score:.2f}). "
                "Consider adjusting parameters or checking input data."
            )

    def get_quality_metrics(self) -> Dict[str, Any]:
        return self.quality_metrics.copy()

    def optimize_z_range(
        self,
        image: np.ndarray,
        metric: str = "focus",
        percentile: float = 90,
    ) -> Tuple[int, int]:
        if image.ndim == 3:
            image = image[..., np.newaxis]

        n_slices = image.shape[0]
        scores = np.zeros(n_slices)

        for z in range(n_slices):
            slice_img = image[z]
            if metric == "focus":
                from scipy.ndimage import laplace

                for c in range(slice_img.shape[-1]):
                    lap = laplace(slice_img[..., c])
                    scores[z] += lap.var()
            elif metric == "intensity":
                scores[z] = slice_img.mean()
            elif metric == "variance":
                scores[z] = slice_img.var()
            else:
                raise ValueError(f"Unknown metric: {metric}")

        threshold = np.percentile(scores, 100 - percentile)
        good_slices = np.where(scores >= threshold)[0]
        if len(good_slices) > 0:
            z_start = good_slices.min()
            z_end = good_slices.max() + 1
        else:
            z_start = 0
            z_end = n_slices

        logger.info("Optimized z-range: %s-%s (using %s metric)", z_start, z_end, metric)
        return (z_start, z_end)

    def create_depth_coded_mip(
        self,
        image: np.ndarray,
        colormap: str = "viridis",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if image.ndim == 3:
            image = image[..., np.newaxis]

        if image.shape[-1] == 1:
            max_z_indices = np.argmax(image[..., 0], axis=0)
            max_intensities = np.max(image[..., 0], axis=0)
        else:
            mean_image = np.mean(image, axis=-1)
            max_z_indices = np.argmax(mean_image, axis=0)
            max_intensities = np.max(mean_image, axis=0)

        n_slices = image.shape[0]
        depth_normalized = max_z_indices.astype(np.float32) / (n_slices - 1)

        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(colormap)
        depth_colored = cmap(depth_normalized)[..., :3]

        max_intensities_norm = max_intensities / (max_intensities.max() + 1e-10)
        depth_colored = depth_colored * max_intensities_norm[..., np.newaxis]

        return depth_colored.astype(np.float32), max_z_indices

    def create_extended_depth_of_field(
        self,
        image: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        if image.ndim == 3:
            image = image[..., np.newaxis]

        from scipy.ndimage import generic_filter

        n_slices, height, width, n_channels = image.shape
        result = np.zeros((height, width, n_channels))

        for c in range(n_channels):
            focus_stack = np.zeros((n_slices, height, width))
            for z in range(n_slices):
                focus_stack[z] = generic_filter(
                    image[z, ..., c],
                    np.var,
                    size=window_size,
                )

            best_focus_indices = np.argmax(focus_stack, axis=0)
            for y in range(height):
                for x in range(width):
                    best_z = best_focus_indices[y, x]
                    result[y, x, c] = image[best_z, y, x, c]

        if result.shape[-1] == 1:
            result = result[..., 0]

        return result

    def save_mip_tiff(
        self,
        mip: np.ndarray,
        filepath: Union[str, Path],
        dtype: str = "auto",
        normalize: bool = True,
        compress: Union[int, None] = 6,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> Path:
        if mip.ndim not in (2, 3):
            raise ValueError(f"MIP must be 2D or 3D (Y, X[, C]), got shape {mip.shape}")

        out_path = Path(filepath)
        ensure_directory(out_path.parent)

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {out_path}")

        if dtype == "auto":
            if np.issubdtype(mip.dtype, np.floating):
                target_dtype = "uint16"
            elif np.issubdtype(mip.dtype, np.integer):
                target_dtype = "uint8" if np.max(mip) <= 255 else "uint16"
            else:
                target_dtype = "uint16"
        else:
            target_dtype = dtype

        if target_dtype in ("uint8", "uint16"):
            arr = mip.astype(np.float32)
            if normalize:
                min_val = float(arr.min())
                max_val = float(arr.max())
                if max_val > min_val:
                    arr = (arr - min_val) / (max_val - min_val)
                else:
                    arr = np.zeros_like(arr)
            else:
                if np.issubdtype(mip.dtype, np.floating):
                    arr = np.clip(arr, 0, 1)
            if target_dtype == "uint8":
                array_to_save = (arr * 255.0 + 0.5).astype(np.uint8)
            else:
                array_to_save = (arr * 65535.0 + 0.5).astype(np.uint16)
        elif target_dtype == "float32":
            arr = mip.astype(np.float32)
            if normalize:
                min_val = float(arr.min())
                max_val = float(arr.max())
                if max_val > min_val:
                    arr = (arr - min_val) / (max_val - min_val)
            array_to_save = arr
        else:
            raise ValueError(f"Unsupported dtype: {target_dtype}")

        if array_to_save.ndim == 3:
            tifffile.imwrite(
                str(out_path),
                array_to_save,
                ome=True,
                metadata={
                    "axes": "YXC",
                    "Channel": {"Name": [f"Channel_{i}" for i in range(array_to_save.shape[-1])]},
                    **({k: v for k, v in (metadata or {}).items() if k != "axes"}),
                },
                compression=("deflate" if compress is not None else None),
                compressionargs=({"level": int(compress)} if compress is not None else None),
            )
        else:
            tifffile.imwrite(
                str(out_path),
                array_to_save,
                photometric="minisblack",
                compression=("deflate" if compress is not None else None),
                compressionargs=({"level": int(compress)} if compress is not None else None),
                description=json.dumps(metadata or {}),
            )

        logger.info(
            "Saved MIP to %s (%s, shape=%s)", out_path, array_to_save.dtype, array_to_save.shape
        )
        return out_path


