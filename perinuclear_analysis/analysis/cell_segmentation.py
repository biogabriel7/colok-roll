"""
Cell segmentation using the public Cellpose Space via the Gradio client.

Workflow:
- Load OME-TIFF using ImageLoader (expects ZYXC or compatible).
- Select two channels and compute per-channel MIPs.
- Build weighted composite [0,1], write a temporary PNG.
- Call Cellpose API to obtain masks and outlines, save artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile as tiff

from ..data_processing.image_loader import ImageLoader
from ..data_processing.mip_creator import MIPCreator
from .segmentation_config import get_hf_token


try:
    from gradio_client import Client, handle_file
except Exception as _e:  # pragma: no cover
    Client = None  # type: ignore
    handle_file = None  # type: ignore


def _normalize_to_unit_interval(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


@dataclass
class CellposeResult:
    mask_path: Path
    outlines_path: Path
    mask_array: np.ndarray


class CellSegmenter:
    """Segment cells by invoking the Cellpose Space through its API."""

    def __init__(
        self,
        cellpose_space: str = "mouseland/cellpose",
        resize_candidates: Sequence[int] = (1000, 600, 400),
        max_iter: int = 250,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        tmp_dir: Union[str, Path] = "/tmp",
        output_dir: Optional[Union[str, Path]] = None,
        hf_token_env: str = "HUGGINGFACE_TOKEN",
        api_pause_s: float = 1.0,
    ) -> None:
        if Client is None or handle_file is None:
            raise ImportError(
                "gradio_client is required. Install with: pip install gradio_client"
            )
        self.cellpose_space = cellpose_space
        self.resize_candidates = list(resize_candidates)
        self.max_iter = int(max_iter)
        self.flow_threshold = float(flow_threshold)
        self.cellprob_threshold = float(cellprob_threshold)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.hf_token_env = hf_token_env
        self.api_pause_s = float(api_pause_s)
        self._logger = logging.getLogger(__name__)

    def _build_composite_png(
        self,
        image: np.ndarray,
        channel_indices: Tuple[int, int],
        channel_weights: Tuple[float, float] = (0.8, 0.2),
        png_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Create a weighted composite PNG from two channels' MIPs."""
        if image.ndim == 3:
            image = image[..., np.newaxis]

        if image.ndim != 4:
            raise ValueError(f"Expected 4D array (Z,Y,X,C) or (Y,X,C), got {image.shape}")

        ch_a, ch_b = channel_indices
        w_a, w_b = channel_weights

        # Create MIPs
        mip_creator = MIPCreator()
        mip_all = mip_creator.create_mip(image, method="max")  # (Y,X,C) or (Y,X)
        if mip_all.ndim == 2:
            raise ValueError("Input must be multichannel to build composite from two channels")

        a = _normalize_to_unit_interval(mip_all[..., ch_a])
        b = _normalize_to_unit_interval(mip_all[..., ch_b])
        composite = (w_a * a + w_b * b).astype(np.float32)
        composite = np.nan_to_num(composite, nan=0.0, posinf=1.0, neginf=0.0)
        composite = np.clip(composite, 0.0, 1.0)

        # Write PNG
        from imageio import imwrite

        out_png = Path(png_path) if png_path is not None else self.tmp_dir / "cellpose_composite.png"
        imwrite(str(out_png), (composite * 255.0).astype(np.uint8))
        return out_png

    def _client(self) -> Client:
        token = get_hf_token(self.hf_token_env)
        return Client(self.cellpose_space, hf_token=token)

    def _run_cellpose(self, client: Client, png_path: Path, resize: int, use_update: bool = True):
        # Optionally perform the update_button step first
        if use_update:
            self._logger.info("Calling /update_button with image %s", png_path)
            _ = client.predict(filepath=handle_file(str(png_path)), api_name="/update_button")
            time.sleep(self.api_pause_s)

        self._logger.info("Calling /cellpose_segment (resize=%s)", resize)
        return client.predict(
            filepath=[handle_file(str(png_path))],
            resize=float(resize),
            max_iter=float(self.max_iter),
            flow_threshold=float(self.flow_threshold),
            cellprob_threshold=float(self.cellprob_threshold),
            api_name="/cellpose_segment",
        )

    @staticmethod
    def _extract_output_path(output_entry) -> Path:
        # Support both dict and object with .path
        if isinstance(output_entry, dict) and "value" in output_entry:
            return Path(output_entry["value"])  # type: ignore[index]
        if hasattr(output_entry, "path"):
            return Path(getattr(output_entry, "path"))
        return Path(str(output_entry))

    def segment_from_image_array(
        self,
        image: np.ndarray,
        channel_indices: Tuple[int, int],
        channel_weights: Tuple[float, float] = (0.8, 0.2),
        save_basename: Optional[str] = None,
    ) -> CellposeResult:
        """Run segmentation given an image array.

        Args:
            image: Input image array (Z,Y,X,C) or (Y,X,C).
            channel_indices: Two channel indices to use for composite (A, B).
            channel_weights: Weights applied to channels (wA, wB).
            save_basename: If provided, writes outputs into output_dir with this stem.
        """
        png_path = self._build_composite_png(image, channel_indices, channel_weights)
        client = self._client()

        last_error: Optional[BaseException] = None
        result = None

        # First pass: with update_button
        for rs in self.resize_candidates:
            try:
                result = self._run_cellpose(client, png_path, rs, use_update=True)
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                self._logger.warning("Cellpose attempt failed at resize=%s with update_button: %s", rs, e)
                time.sleep(self.api_pause_s)
                continue

        # Second pass: try without update_button if first pass failed
        if result is None:
            for rs in self.resize_candidates:
                try:
                    result = self._run_cellpose(client, png_path, rs, use_update=False)
                    break
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    self._logger.warning("Cellpose attempt failed at resize=%s without update_button: %s", rs, e)
                    time.sleep(self.api_pause_s)
                    continue

        if result is None:
            detail = f"; last_error={last_error!r}" if last_error is not None else ""
            raise RuntimeError(f"Cellpose Space failed after retries{detail}")

        masks_tif = self._extract_output_path(result[2])
        outlines_png = self._extract_output_path(result[3])

        mask_array = tiff.imread(str(masks_tif)).astype(np.int32)

        # Optional save to output_dir
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stem = save_basename or png_path.stem
            dst_mask = self.output_dir / f"{stem}_masks.tif"
            dst_outl = self.output_dir / f"{stem}_outlines.png"
            dst_mask.write_bytes(masks_tif.read_bytes())
            dst_outl.write_bytes(outlines_png.read_bytes())
            masks_tif = dst_mask
            outlines_png = dst_outl

        return CellposeResult(mask_path=masks_tif, outlines_path=outlines_png, mask_array=mask_array)

    def segment_from_file(
        self,
        image_path: Union[str, Path],
        channel_names: Sequence[str],
        channel_a: Union[int, str],
        channel_b: Union[int, str],
        channel_weights: Tuple[float, float] = (0.8, 0.2),
    ) -> CellposeResult:
        """Load an OME-TIFF, build composite from selected channels, run Cellpose.

        channel_a/channel_b can be integers or names present in channel_names.
        """
        loader = ImageLoader()
        image = loader.load_image(image_path)

        # Resolve channel indices
        name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(channel_names)}
        def to_index(val: Union[int, str]) -> int:
            if isinstance(val, int):
                return val
            if val not in name_to_index:
                raise ValueError(f"Unknown channel '{val}'. Available: {list(name_to_index)}")
            return name_to_index[val]

        ch_a = to_index(channel_a)
        ch_b = to_index(channel_b)

        basename = Path(image_path).stem + f"_{channel_names[ch_a]}_{channel_names[ch_b]}"
        return self.segment_from_image_array(
            image=image,
            channel_indices=(ch_a, ch_b),
            channel_weights=channel_weights,
            save_basename=basename,
        )


__all__ = ["CellSegmenter", "CellposeResult"]


