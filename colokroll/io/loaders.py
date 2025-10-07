"""Image loading utilities for ND2/TIFF microscopy data."""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nd2reader
import numpy as np
from PIL import Image
import tifffile
from skimage import io as skio

from ..config import ImageIOConfig
from ..core.format_converter import FormatConverter
from ..core.utils import get_pixel_size_from_metadata, validate_file_path

logger = logging.getLogger(__name__)


class ImageLoader:
    """Comprehensive image loader for microscopy data files."""

    def __init__(self, config: Optional[ImageIOConfig] = None, auto_convert: bool = True):
        """Initialize the image loader.

        Args:
            config: Image IO configuration. If None, uses defaults.
            auto_convert: If True, automatically converts non-TIFF formats to TIFF.
        """
        self.config = config or ImageIOConfig()
        self.metadata: Dict[str, Any] = {}
        self.pixel_size_um: Optional[float] = None
        self.channels: List[str] = []
        self.auto_convert = auto_convert
        self.converter = FormatConverter(preserve_original=True)
        self.converted_files: Dict[str, Path] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        if self.config.verbose:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

    def load_image(self, filepath: Union[str, Path], force_convert: bool = None) -> np.ndarray:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        should_convert = force_convert if force_convert is not None else self.auto_convert
        suffix = filepath.suffix.lower()

        if suffix == ".nd2" and should_convert:
            logger.info("Converting %s to OME-TIFF format for pipeline compatibility", filepath.name)
            ome_tiff_path = filepath.with_suffix(".ome.tiff")
            if ome_tiff_path.exists():
                metadata_path = ome_tiff_path.with_suffix(".json")
                if metadata_path.exists():
                    logger.info("Using existing converted file: %s", ome_tiff_path)
                    self.converted_files[str(filepath)] = ome_tiff_path
                    return self.load_tiff_with_metadata(ome_tiff_path)

            output_path, metadata = self.converter.nd2_to_ome_tiff(filepath, ome_tiff_path, save_metadata=True)
            self.converted_files[str(filepath)] = output_path
            self.metadata = metadata
            self.pixel_size_um = metadata.get("pixel_size_um")
            self.channels = metadata.get("channel_names", [])
            return self.load_tiff_with_metadata(output_path)

        if suffix == ".nd2" and not should_convert:
            return self.load_nd2(filepath)

        if suffix == ".oir" and should_convert:
            logger.info("Converting %s to OME-TIFF format for pipeline compatibility", filepath.name)
            ome_tiff_path = filepath.with_suffix(".ome.tiff")
            if ome_tiff_path.exists():
                metadata_path = ome_tiff_path.with_suffix(".json")
                if metadata_path.exists():
                    logger.info("Using existing converted file: %s", ome_tiff_path)
                    self.converted_files[str(filepath)] = ome_tiff_path
                    return self.load_tiff_with_metadata(ome_tiff_path)

            output_path, metadata = self.converter.oir_to_ome_tiff(filepath, ome_tiff_path, save_metadata=True)
            self.converted_files[str(filepath)] = output_path
            self.metadata = metadata
            self.pixel_size_um = metadata.get("pixel_size_um")
            self.channels = metadata.get("channel_names", [])
            return self.load_tiff_with_metadata(output_path)

        if suffix == ".oir" and not should_convert:
            raise ValueError(
                "Direct loading of .oir files is not supported. Enable auto_convert or pass force_convert=True."
            )

        if suffix in [".tif", ".tiff"]:
            if ".ome.tif" in str(filepath).lower():
                return self.load_tiff_with_metadata(filepath)
            metadata_path = filepath.with_suffix(".json")
            if metadata_path.exists():
                return self.load_tiff_with_metadata(filepath)
            return self.load_tif_image(filepath)

        raise ValueError(f"Unsupported file format: {suffix}")

    def load_tiff_with_metadata(self, filepath: Union[str, Path]) -> np.ndarray:
        filepath = Path(filepath)
        metadata = self.converter.load_metadata(filepath)
        if metadata:
            self.metadata = metadata
            self.pixel_size_um = metadata.get("pixel_size_um")
            self.channels = metadata.get("channel_names", [])
            logger.info("Loaded metadata from converted TIFF: pixel_size=%sμm", self.pixel_size_um)

        try:
            with tifffile.TiffFile(str(filepath)) as tif:
                axes = tif.series[0].axes
                image = tif.series[0].asarray()

            if "T" in axes:
                t_axis = axes.index("T")
                image = np.take(image, indices=0, axis=t_axis)
                axes = axes.replace("T", "")

            if axes == "ZYXC":
                pass
            elif axes == "CZYX":
                image = np.transpose(image, (1, 2, 3, 0))
                axes = "ZYXC"
            elif axes == "ZCYX":
                image = np.transpose(image, (0, 2, 3, 1))
                axes = "ZYXC"
            elif axes == "YXC":
                image = image[np.newaxis, ...]
                axes = "ZYXC"
            elif axes == "ZYX":
                image = image[..., np.newaxis]
                axes = "ZYXC"
            elif axes == "TCZYX":
                if image.ndim == 5:
                    image = image[0]
                image = np.transpose(image, (1, 2, 3, 0))
                axes = "ZYXC"
            else:
                desired = ["Z", "Y", "X"]
                perm = [axes.index(ax) if ax in axes else None for ax in desired]
                if None not in perm:
                    tmp = np.moveaxis(image, perm, [0, 1, 2])
                    if "C" in axes:
                        c_axis = axes.index("C")
                        tmp = np.moveaxis(tmp, c_axis if c_axis < tmp.ndim else -1, -1)
                    else:
                        tmp = tmp[..., np.newaxis]
                    image = tmp
                    axes = "ZYXC"
                else:
                    if image.ndim == 2:
                        image = image[np.newaxis, ..., np.newaxis]
                    elif image.ndim == 3:
                        image = image[np.newaxis, ...]
                    else:
                        raise ValueError(
                            f"Unhandled TIFF axes/order: {axes} with shape {image.shape}"
                        )

            if image.ndim > 4:
                original_shape = image.shape
                while image.ndim > 4 and 1 in image.shape:
                    axis_to_squeeze = next(i for i, s in enumerate(image.shape) if s == 1)
                    image = np.squeeze(image, axis=axis_to_squeeze)
                if image.ndim == 3:
                    image = image[..., np.newaxis]
                if image.ndim != 4:
                    raise ValueError(
                        "Unexpected TIFF dimensionality after squeezing singletons: "
                        f"{original_shape} -> {image.shape} (axes {axes})"
                    )
                logger.info("Squeezed singleton dim(s): %s -> %s", original_shape, image.shape)

            if image.ndim != 4:
                raise ValueError(f"Expected 4D ZYXC after reordering, got shape {image.shape} (axes {axes})")

            self._validate_image_dimensions(image)
            logger.info("Successfully loaded TIFF image with shape: %s", image.shape)
            return image
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load TIFF file: %s", exc)
            raise ValueError(f"Error loading TIFF file: {exc}") from exc

    def load_nd2(self, filepath: Union[str, Path]) -> np.ndarray:
        filepath = Path(filepath)
        validate_file_path(filepath, [".nd2"])
        logger.info("Loading ND2 file: %s", filepath)

        try:
            with nd2reader.ND2Reader(str(filepath)) as reader:
                self._extract_nd2_metadata(reader)
                z_levels = len(reader.metadata.get("z_levels", [1]))
                channels = len(reader.metadata.get("channels", [])) or 1

                if "z" in reader.axes and "c" in reader.axes:
                    reader.iter_axes = "zc"
                    reader.bundle_axes = "yx"
                    images = []
                    for z in range(z_levels):
                        channel_images = []
                        for c in range(channels):
                            reader.default_coords["z"] = z
                            reader.default_coords["c"] = c
                            channel_images.append(reader[0])
                        images.append(np.stack(channel_images, axis=-1))
                    image_array = np.stack(images, axis=0)
                elif "z" in reader.axes:
                    reader.iter_axes = "z"
                    reader.bundle_axes = "yx"
                    images = []
                    for z in range(z_levels):
                        reader.default_coords["z"] = z
                        img = reader[0]
                        if img.ndim == 2:
                            img = img[..., np.newaxis]
                        images.append(img)
                    image_array = np.stack(images, axis=0)
                elif "c" in reader.axes:
                    reader.iter_axes = "c"
                    reader.bundle_axes = "yx"
                    images = []
                    for c in range(channels):
                        reader.default_coords["c"] = c
                        images.append(reader[0])
                    image_array = np.stack(images, axis=-1)
                    if image_array.ndim == 3:
                        image_array = image_array[np.newaxis, ...]
                else:
                    image_array = np.array(reader[0])
                    if image_array.ndim == 2:
                        image_array = image_array[np.newaxis, ..., np.newaxis]
                    elif image_array.ndim == 3:
                        image_array = image_array[np.newaxis, ...]

                self._validate_image_dimensions(image_array)
                logger.info("Successfully loaded ND2 image with shape: %s", image_array.shape)
                logger.info("Pixel size: %s μm", self.pixel_size_um)
                logger.info("Channels: %s", self.channels)
                return image_array
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load ND2 file: %s", exc)
            raise ValueError(f"Error loading ND2 file: {exc}") from exc

    def _extract_nd2_metadata(self, reader: nd2reader.ND2Reader) -> None:
        metadata = reader.metadata
        if self.config.extract_all_metadata:
            self.metadata = dict(metadata)

        pixel_size = get_pixel_size_from_metadata(metadata)
        if pixel_size:
            self.pixel_size_um = pixel_size
        elif "calibration" in metadata:
            self.pixel_size_um = metadata["calibration"]
        else:
            raise ValueError(
                "No pixel size found in image metadata. Image files must contain calibration information."
            )

        if "channels" in metadata:
            self.channels = metadata["channels"]
        else:
            raise ValueError(
                "No channel information found in image metadata. Channel names are required for proper analysis."
            )

        self.metadata["axes"] = reader.axes
        self.metadata["shape"] = reader.frame_shape
        self.metadata["z_levels"] = len(metadata.get("z_levels", [1]))
        self.metadata["timepoints"] = reader.metadata.get("total_images_per_channel", 1)

    def load_tif_mask(self, filepath: Union[str, Path]) -> np.ndarray:
        filepath = Path(filepath)
        validate_file_path(filepath, [".tif", ".tiff"])
        logger.info("Loading TIF mask: %s", filepath)

        try:
            mask = skio.imread(str(filepath))
            if mask.ndim == 3 and mask.shape[2] in [3, 4]:
                mask = mask.mean(axis=2).astype(mask.dtype)
            self._validate_mask(mask)
            logger.info("Successfully loaded mask with shape: %s", mask.shape)
            return mask
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load TIF mask: %s", exc)
            raise ValueError(f"Error loading TIF mask: {exc}") from exc

    def load_tif_image(self, filepath: Union[str, Path]) -> np.ndarray:
        filepath = Path(filepath)
        validate_file_path(filepath, [".tif", ".tiff"])
        logger.info("Loading TIF image: %s", filepath)

        try:
            image = skio.imread(str(filepath))
            with Image.open(str(filepath)) as img:
                if hasattr(img, "tag"):
                    tags = dict(img.tag)
                    self.metadata["tif_tags"] = tags
                    if 282 in tags and 283 in tags:
                        x_res = tags[282][0] if isinstance(tags[282], tuple) else tags[282]
                        y_res = tags[283][0] if isinstance(tags[283], tuple) else tags[283]
                        if 296 in tags and tags[296] == 2:
                            self.pixel_size_um = 25400.0 / x_res

            if image.ndim == 2:
                image = image[np.newaxis, ..., np.newaxis]
            elif image.ndim == 3:
                if image.shape[2] <= 4:
                    image = image[np.newaxis, ...]
                else:
                    image = image[..., np.newaxis]

            self._validate_image_dimensions(image)
            logger.info("Successfully loaded TIF image with shape: %s", image.shape)
            return image
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load TIF image: %s", exc)
            raise ValueError(f"Error loading TIF image: {exc}") from exc

    def _validate_image_dimensions(self, image: np.ndarray) -> None:
        if image.ndim not in [3, 4]:
            raise ValueError(f"Image must be 3D or 4D, got {image.ndim}D")
        height, width = image.shape[-3:-1] if image.ndim == 4 else image.shape[-2:]
        if height < self.config.min_image_dimension or width < self.config.min_image_dimension:
            raise ValueError(
                f"Image dimensions too small: {height}x{width}. Minimum: {self.config.min_image_dimension}"
            )
        if height > self.config.max_image_dimension or width > self.config.max_image_dimension:
            raise ValueError(
                f"Image dimensions too large: {height}x{width}. Maximum: {self.config.max_image_dimension}"
            )

    def _validate_mask(self, mask: np.ndarray) -> None:
        if mask.ndim not in [2, 3]:
            raise ValueError(f"Mask must be 2D or 3D, got {mask.ndim}D")
        unique_values = np.unique(mask)
        if len(unique_values) == 2 and not (
            np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0, 255])
        ):
            warnings.warn(f"Unusual binary mask values: {unique_values}")
        logger.info("Mask contains %d unique values", len(unique_values))

    def get_pixel_size(self) -> float:
        if self.pixel_size_um is None:
            raise ValueError("No pixel size available. Image metadata must contain calibration information.")
        return self.pixel_size_um

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata.copy()

    def get_channel_names(self) -> List[str]:
        return self.channels.copy()

    def rename_channels(self, new_names: List[str]) -> None:
        if len(new_names) != len(self.channels):
            raise ValueError(
                f"Number of new names ({len(new_names)}) must match number of channels ({len(self.channels)})"
            )
        self.channels = new_names.copy()
        logger.info("Renamed channels to: %s", self.channels)

    def set_channel_name(self, channel_index: int, new_name: str) -> None:
        if channel_index < 0 or channel_index >= len(self.channels):
            raise IndexError(f"Channel index {channel_index} out of range (0-{len(self.channels) - 1})")
        old_name = self.channels[channel_index]
        self.channels[channel_index] = new_name
        logger.info("Renamed channel %d from '%s' to '%s'", channel_index, old_name, new_name)

    def extract_channel(self, image: np.ndarray, channel: Union[int, str]) -> np.ndarray:
        if isinstance(channel, str):
            if channel not in self.channels:
                raise ValueError(f"Unknown channel: {channel}. Available: {self.channels}")
            channel = self.channels.index(channel)
        if image.ndim in (3, 4):
            return image[..., channel]
        raise ValueError(f"Image must be 3D or 4D for channel extraction, got {image.ndim}D")

    def split_channels(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        channels: Dict[str, np.ndarray] = {}
        num_channels = image.shape[-1]
        for i in range(num_channels):
            channel_name = self.channels[i] if i < len(self.channels) else f"Channel_{i}"
            channels[channel_name] = self.extract_channel(image, i)
        return channels

    def get_z_stack(self, image: np.ndarray, z_index: int) -> np.ndarray:
        if image.ndim != 4:
            raise ValueError(f"Image must be 4D for z-stack extraction, got {image.ndim}D")
        if z_index < 0 or z_index >= image.shape[0]:
            raise ValueError(f"Invalid z-index: {z_index}. Valid range: 0-{image.shape[0] - 1}")
        return image[z_index]

    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        info = {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "ndim": image.ndim,
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": float(image.mean()),
            "std": float(image.std()),
        }
        if image.ndim == 4:
            info.update(
                {
                    "z_slices": image.shape[0],
                    "height": image.shape[1],
                    "width": image.shape[2],
                    "channels": image.shape[3],
                }
            )
        elif image.ndim == 3:
            info.update({"height": image.shape[0], "width": image.shape[1], "channels": image.shape[2]})
        info["pixel_size_um"] = self.get_pixel_size()
        info["channel_names"] = self.get_channel_names()
        return info

    def batch_process_directory(self, directory: Union[str, Path], pattern: str = "*.nd2") -> Dict[Path, Path]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        files = list(directory.glob(pattern))
        logger.info("Found %d files matching pattern %s", len(files), pattern)

        processed: Dict[Path, Path] = {}
        for file in files:
            try:
                self.load_image(file)
                processed[file] = self.converted_files.get(str(file), file)
                logger.info("Processed: %s", file.name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process %s: %s", file, exc)
                processed[file] = None  # type: ignore[assignment]
        return processed

    def get_converted_files(self) -> Dict[str, Path]:
        return self.converted_files.copy()


