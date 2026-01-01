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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile as tiff

from ..data_processing.image_loader import ImageLoader
from ..data_processing.projection import MIPCreator
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


def _to_numpy_array(a: Any) -> np.ndarray:
    """Convert possibly CuPy/array-like to a NumPy ndarray."""
    try:  # pragma: no cover - optional CuPy path
        import cupy as _cp  # type: ignore
        if isinstance(a, _cp.ndarray):
            return _cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def _image_from_results_dict(results: Dict[str, Any], channel_order: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Build a (Z,Y,X,C) image from a preprocessing results dict for the requested channel order.

    Each value in results[channel] can be either a 3D array (Z,Y,X) or a tuple/list where the
    first element is the array. Returns (image, names) where names == list(channel_order).
    """
    arrays: List[np.ndarray] = []
    for nm in channel_order:
        if nm not in results:
            raise ValueError(f"Channel '{nm}' not found in results keys: {list(results.keys())[:8]}...")
        val = results[nm]
        arr = val[0] if (isinstance(val, (tuple, list)) and len(val) >= 1) else val
        arr_np = _to_numpy_array(arr)
        if arr_np.ndim == 4 and arr_np.shape[-1] == 1:
            arr_np = arr_np[..., 0]
        if arr_np.ndim != 3:
            raise ValueError(f"Per-channel array must be 3D (Z,Y,X); got {arr_np.shape} for channel '{nm}'")
        arrays.append(arr_np)

    zyx_shapes = {a.shape for a in arrays}
    if len(zyx_shapes) != 1:
        raise ValueError(f"All channels must share the same Z,Y,X shape; got {zyx_shapes}")
    img = np.stack(arrays, axis=-1)
    return img, list(channel_order)


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
        resize_candidates: Sequence[int] = (600,),
        *,
        auto_resize: bool = False,
        max_resize_cap: int = 600,
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
        # If auto_resize, we'll compute candidates dynamically per image
        self.resize_candidates = list(resize_candidates)
        self.auto_resize = bool(auto_resize)
        self.max_resize_cap = int(max_resize_cap)
        self.max_iter = int(max_iter)
        self.flow_threshold = float(flow_threshold)
        self.cellprob_threshold = float(cellprob_threshold)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.hf_token_env = hf_token_env
        self.api_pause_s = float(api_pause_s)
        self._logger = logging.getLogger(__name__)

    def _to_u16_percentile(self, img01: np.ndarray, percentiles: Tuple[float, float], apply_clahe: bool) -> np.ndarray:
        """Scale float image in [0,1] to uint16 using robust percentiles with optional CLAHE."""
        p_lo, p_hi = percentiles
        vals = img01[np.isfinite(img01)]
        if vals.size == 0:
            vals = np.array([0.0, 1.0], dtype=np.float32)
        lo, hi = np.percentile(vals, [p_lo, p_hi])
        denom = max(float(hi - lo), 1e-6)
        x = np.clip((img01 - float(lo)) / denom, 0.0, 1.0).astype(np.float32)
        if apply_clahe:
            try:
                from skimage.exposure import equalize_adapthist as clahe
                x = clahe(x, clip_limit=0.01)
            except Exception:
                pass
        return (x * 65535.0).astype(np.uint16)

    def _build_composite_image(
        self,
        image: np.ndarray,
        channel_indices: Tuple[int, int],
        channel_weights: Tuple[float, float] = (0.8, 0.2),
        *,
        projection: str = "middle",  # 'middle' | 'mip'
        output_format: str = "tiff16",  # 'tiff16' | 'png8'
        percentiles: Tuple[float, float] = (1.0, 99.9),
        apply_clahe: bool = False,
        out_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Create a composite 2D image for Cellpose from selected channels.

        - projection: 'middle' uses the middle Z slice; 'mip' uses max-intensity projection.
        - output_format: 'tiff16' writes 16-bit TIFF; 'png8' writes 8-bit PNG.
        - percentiles/clahe: applied before writing to enhance contrast.
        """
        if image.ndim == 3:
            image = image[..., np.newaxis]
        if image.ndim != 4:
            raise ValueError(f"Expected 4D array (Z,Y,X,C) or (Y,X,C), got {image.shape}")

        ch_a, ch_b = channel_indices
        w_a, w_b = channel_weights

        if projection not in {"middle", "mip"}:
            raise ValueError("projection must be one of {'middle','mip'}")

        if projection == "mip":
            mip_creator = MIPCreator()
            mip_all = mip_creator.create_mip(image, method="max")
            if mip_all.ndim == 2:
                raise ValueError("Input must be multichannel to build composite from two channels")
            a2d = _normalize_to_unit_interval(mip_all[..., ch_a])
            b2d = _normalize_to_unit_interval(mip_all[..., ch_b])
        else:
            z_mid = int(image.shape[0] // 2)
            a2d = _normalize_to_unit_interval(image[z_mid, :, :, ch_a])
            b2d = _normalize_to_unit_interval(image[z_mid, :, :, ch_b])

        composite01 = (w_a * a2d + w_b * b2d).astype(np.float32)
        composite01 = np.nan_to_num(composite01, nan=0.0, posinf=1.0, neginf=0.0)
        composite01 = np.clip(composite01, 0.0, 1.0)

        if output_format == "tiff16":
            u16 = self._to_u16_percentile(composite01, percentiles=percentiles, apply_clahe=apply_clahe)
            path = Path(out_path) if out_path is not None else self.tmp_dir / "cellpose_composite.tif"
            tiff.imwrite(str(path), u16, compression="deflate", photometric="minisblack")
            return path
        elif output_format == "png8":
            from imageio import imwrite
            path = Path(out_path) if out_path is not None else self.tmp_dir / "cellpose_composite.png"
            # Simple conversion matching batch_whole_analysis.py approach
            imwrite(str(path), (composite01 * 255).astype(np.uint8))
            return path
        else:
            raise ValueError("output_format must be one of {'tiff16','png8'}")

    def _client(self) -> Client:
        token = get_hf_token(self.hf_token_env)
        return Client(self.cellpose_space, token=token)

    def _run_cellpose(self, client: Client, png_path: Path, resize: int, use_update: bool = True):
        # Optionally perform the update_button step first
        if use_update:
            self._logger.info("Calling /update_button with image %s", png_path)
            _ = client.predict(filepath=handle_file(str(png_path)), api_name="/update_button")
            time.sleep(self.api_pause_s)

        self._logger.info("Calling /cellpose_segment (resize=%s)", resize)
        return client.predict(
            filepath=[handle_file(str(png_path))],
            resize=int(resize),
            max_iter=int(self.max_iter),
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
        *,
        plot: bool = True,
        plot_save_path: Optional[Union[str, Path]] = None,
    ) -> CellposeResult:
        """Run segmentation given an image array.

        Args:
            image: Input image array (Z,Y,X,C) or (Y,X,C).
            channel_indices: Two channel indices to use for composite (A, B).
            channel_weights: Weights applied to channels (wA, wB).
            save_basename: If provided, writes outputs into output_dir with this stem.
            plot: If True, create a quicklook segmentation plot (composite, overlay, mask).
            plot_save_path: Optional path to save the plot. If None and output_dir is set,
                saves next to mask/outlines with suffix "_segmentation_preview.png".
        """
        png_path = self._build_composite_image(
            image,
            channel_indices,
            channel_weights,
            projection="mip",
            output_format="png8",
            percentiles=(1.0, 99.9),
            apply_clahe=False,
        )
        # Build resize candidates
        candidates: List[int]
        if self.auto_resize:
            try:
                from imageio.v2 import imread as _imread  # type: ignore
            except Exception:  # pragma: no cover - fallback import path
                from imageio import imread as _imread  # type: ignore

            try:
                hh, ww = _imread(str(png_path)).shape[:2]
            except Exception:
                hh, ww = 0, 0
            max_dim = max(hh, ww)
            start = max(0, min(max_dim, self.max_resize_cap)) or 1000
            # Try largest feasible size first, then fall back
            base = [start, 2200, 2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400]
            # Ensure strictly positive integers and unique, descending
            candidates = sorted({int(x) for x in base if int(x) > 0}, reverse=True)
        else:
            candidates = list(self.resize_candidates) or [600, 400]
        client = self._client()

        last_error: Optional[BaseException] = None
        result = None

        # First pass: with update_button
        for rs in candidates:
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
            for rs in candidates:
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

        # Optional quicklook plot
        if plot:
            try:
                self._plot_segmentation_quicklook(
                    image,
                    mask_array,
                    channel_indices=channel_indices,
                    channel_weights=channel_weights,
                    save_path=plot_save_path or (
                        (self.output_dir / f"{stem}_segmentation_preview.png") if (self.output_dir is not None) else None
                    ),
                )
            except Exception as e:  # noqa: BLE001
                self._logger.warning("Segmentation quicklook plot failed: %s", e)

        return CellposeResult(mask_path=masks_tif, outlines_path=outlines_png, mask_array=mask_array)

    def segment_from_file(
        self,
        image_path: Union[str, Path],
        channel_names: Optional[Sequence[str]] = None,
        channel_a: Union[int, str] = 0,
        channel_b: Union[int, str] = 1,
        channel_weights: Tuple[float, float] = (0.8, 0.2),
        *,
        projection: str = "mip",
        output_format: str = "png8",
        percentiles: Tuple[float, float] = (1.0, 99.9),
        apply_clahe: bool = False,
        plot: bool = True,
        plot_save_path: Optional[Union[str, Path]] = None,
    ) -> CellposeResult:
        """Load an OME-TIFF, build composite from selected channels, run Cellpose.

        channel_a/channel_b can be integers or names present in channel_names.
        """
        loader = ImageLoader()
        image = loader.load_image(image_path)

        # Resolve channel indices
        if channel_names is None:
            try:
                channel_names = loader.get_channel_names()
            except Exception:
                channel_names = [str(i) for i in range(image.shape[-1])]
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
        # Build composite and run with requested projection/format
        png_path = self._build_composite_image(
            image,
            (ch_a, ch_b),
            channel_weights,
            projection=projection,
            output_format=output_format,
            percentiles=percentiles,
            apply_clahe=apply_clahe,
        )
        client = self._client()
        last_error: Optional[BaseException] = None
        result = None
        # Build resize candidates
        candidates: List[int]
        if self.auto_resize:
            try:
                from imageio.v2 import imread as _imread  # type: ignore
            except Exception:
                from imageio import imread as _imread  # type: ignore
            try:
                hh, ww = _imread(str(png_path)).shape[:2]
            except Exception:
                hh, ww = 0, 0
            max_dim = max(hh, ww)
            start = max(0, min(max_dim, self.max_resize_cap)) or 1000
            base = [start, 2200, 2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400]
            candidates = sorted({int(x) for x in base if int(x) > 0}, reverse=True)
        else:
            candidates = list(self.resize_candidates) or [600, 400]

        # First pass: with update_button
        for rs in candidates:
            try:
                result = self._run_cellpose(client, png_path, rs, use_update=True)
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                self._logger.warning("Cellpose attempt failed at resize=%s with update_button: %s", rs, e)
                time.sleep(self.api_pause_s)
                continue
        # Second pass: without update_button
        if result is None:
            for rs in candidates:
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
            stem = basename
            dst_mask = self.output_dir / f"{stem}_masks.tif"
            dst_outl = self.output_dir / f"{stem}_outlines.png"
            dst_mask.write_bytes(masks_tif.read_bytes())
            dst_outl.write_bytes(outlines_png.read_bytes())
            masks_tif = dst_mask
            outlines_png = dst_outl

        # Optional quicklook plot
        if plot:
            try:
                self._plot_segmentation_quicklook(
                    image,
                    mask_array,
                    channel_indices=(ch_a, ch_b),
                    channel_weights=channel_weights,
                    save_path=plot_save_path or (
                        (self.output_dir / f"{basename}_segmentation_preview.png") if (self.output_dir is not None) else None
                    ),
                )
            except Exception as e:  # noqa: BLE001
                self._logger.warning("Segmentation quicklook plot failed: %s", e)

        return CellposeResult(mask_path=masks_tif, outlines_path=outlines_png, mask_array=mask_array)

    def _plot_segmentation_quicklook(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        channel_indices: Tuple[int, int],
        channel_weights: Tuple[float, float] = (0.8, 0.2),
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Lightweight segmentation visualization (composite, overlay, mask)."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            self._logger.debug("matplotlib not available; skipping segmentation plot")
            return

        if image.ndim == 3:
            image = image[..., np.newaxis]
        mip_creator = MIPCreator()
        mip = mip_creator.create_mip(image, method="max")

        ch_a, ch_b = channel_indices
        a2d = _normalize_to_unit_interval(mip[..., ch_a])
        b2d = _normalize_to_unit_interval(mip[..., ch_b])

        # Simple RGB composite: A->red, B->green (weights applied before clamp)
        comp = np.zeros((*a2d.shape, 3), dtype=np.float32)
        comp[..., 0] = np.clip(channel_weights[0] * a2d, 0.0, 1.0)
        comp[..., 1] = np.clip(channel_weights[1] * b2d, 0.0, 1.0)

        mask_bool = mask > 0
        overlay = comp.copy()
        overlay[mask_bool] = 0.6 * overlay[mask_bool] + 0.4 * np.array([1.0, 1.0, 0.0], dtype=np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(comp)
        axes[0].set_title("Composite (A/B)")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Overlay with mask")
        axes[1].axis("off")

        axes[2].imshow(mask, cmap="nipy_spectral")
        axes[2].set_title("Mask labels")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            self._logger.info("Segmentation preview saved to %s", str(save_path))
            plt.close(fig)
        else:
            try:
                plt.show(block=False)
            except Exception:
                pass

    def segment_from_results(
        self,
        results: Dict[str, Any],
        *,
        channel_a: str,
        channel_b: str,
        channel_weights: Tuple[float, float] = (0.8, 0.2),
        projection: str = "mip",
        output_format: str = "png8",
        percentiles: Tuple[float, float] = (1.0, 99.9),
        apply_clahe: bool = False,
        save_basename: Optional[str] = None,
    ) -> CellposeResult:
        """Run segmentation using a preprocessing results dict mapping channel names to arrays.

        results: dict like {"DAPI": (Z,Y,X), "Phalloidin": (Z,Y,X), ...}
        channel_a/channel_b: names present in results to build the composite.
        """
        # Build a (Z,Y,X,C) with the two requested channels in order (A,B)
        image, names = _image_from_results_dict(results, [channel_a, channel_b])

        # Build composite and run with requested projection/format
        png_path = self._build_composite_image(
            image,
            (0, 1),
            channel_weights,
            projection=projection,
            output_format=output_format,
            percentiles=percentiles,
            apply_clahe=apply_clahe,
        )
        client = self._client()
        last_error: Optional[BaseException] = None
        result = None

        # Build resize candidates
        candidates: List[int]
        if self.auto_resize:
            try:
                from imageio.v2 import imread as _imread  # type: ignore
            except Exception:
                from imageio import imread as _imread  # type: ignore
            try:
                hh, ww = _imread(str(png_path)).shape[:2]
            except Exception:
                hh, ww = 0, 0
            max_dim = max(hh, ww)
            start = max(0, min(max_dim, self.max_resize_cap)) or 1000
            base = [start, 2200, 2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400]
            candidates = sorted({int(x) for x in base if int(x) > 0}, reverse=True)
        else:
            candidates = list(self.resize_candidates) or [600, 400]

        # First pass: with update_button
        for rs in candidates:
            try:
                result = self._run_cellpose(client, png_path, rs, use_update=True)
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                self._logger.warning("Cellpose attempt failed at resize=%s with update_button: %s", rs, e)
                time.sleep(self.api_pause_s)
                continue
        # Second pass: without update_button
        if result is None:
            for rs in candidates:
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
            stem = save_basename or f"{channel_a}_{channel_b}"
            dst_mask = self.output_dir / f"{stem}_masks.tif"
            dst_outl = self.output_dir / f"{stem}_outlines.png"
            dst_mask.write_bytes(masks_tif.read_bytes())
            dst_outl.write_bytes(outlines_png.read_bytes())
            masks_tif = dst_mask
            outlines_png = dst_outl

        return CellposeResult(mask_path=masks_tif, outlines_path=outlines_png, mask_array=mask_array)


__all__ = ["CellSegmenter", "CellposeResult"]