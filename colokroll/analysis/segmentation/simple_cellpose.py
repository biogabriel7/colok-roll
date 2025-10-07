"""
Simplified Cellpose segmentation function based on working notebook code.

This module provides a straightforward wrapper around the Cellpose Gradio Space
with the exact flow that has been tested to work reliably.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    from gradio_client import Client, handle_file
    import imageio.v3 as iio
except ImportError as e:
    raise ImportError(
        "Required dependencies not installed. "
        "Install with: pip install gradio_client imageio"
    ) from e

from .config import get_hf_token


def segment_cells(
    phalloidin_mip: np.ndarray,
    dapi_mip: np.ndarray,
    output_dir: Union[str, Path],
    filename_stem: str,
    *,
    phalloidin_weight: float = 0.8,
    dapi_weight: float = 0.2,
    resize_values: Tuple[int, ...] = (600, 400),
    max_iter: int = 250,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    pause_seconds: float = 1.0,
    tmp_dir: str = "/tmp",
) -> Tuple[Path, Path, np.ndarray]:
    """
    Segment cells using Cellpose Gradio Space with the proven working workflow.
    
    This function implements the exact flow that has been tested to work:
    1. Create weighted composite from Phalloidin and DAPI MIPs
    2. Save as PNG to temp location
    3. Call /update_button endpoint
    4. Pause briefly
    5. Call /cellpose_segment endpoint
    6. Retry with smaller resize values if needed
    
    Args:
        phalloidin_mip: 2D MIP of phalloidin channel (Y, X)
        dapi_mip: 2D MIP of DAPI channel (Y, X)
        output_dir: Directory to save mask and outlines
        filename_stem: Base name for output files (e.g., "sample1")
        phalloidin_weight: Weight for phalloidin in composite (default 0.8)
        dapi_weight: Weight for DAPI in composite (default 0.2)
        resize_values: Tuple of resize values to try, in order (default (600, 400))
        max_iter: Maximum cellpose iterations (default 250)
        flow_threshold: Cellpose flow threshold (default 0.4)
        cellprob_threshold: Cellpose cell probability threshold (default 0.0)
        pause_seconds: Seconds to pause between API calls (default 1.0)
        tmp_dir: Temporary directory for composite PNG (default "/tmp")
        
    Returns:
        Tuple of (mask_path, outlines_path, mask_array) where:
        - mask_path: Path to saved TIFF mask file
        - outlines_path: Path to saved PNG outlines file
        - mask_array: 2D numpy array of integer labels
        
    Raises:
        RuntimeError: If all resize attempts fail
        ImportError: If required dependencies are not installed
        
    Example:
        >>> from colokroll.io import ImageLoader, MIPCreator
        >>> loader = ImageLoader()
        >>> image = loader.load_image("data.ome.tiff")
        >>> mip_creator = MIPCreator()
        >>> 
        >>> # Extract channels and create MIPs
        >>> phall_stack = loader.extract_channel(image, "Phalloidin")
        >>> dapi_stack = loader.extract_channel(image, "DAPI")
        >>> phall_mip = mip_creator.create_mip(phall_stack, method="max")
        >>> dapi_mip = mip_creator.create_mip(dapi_stack, method="max")
        >>> 
        >>> # Segment
        >>> mask_path, outlines_path, mask = segment_cells(
        ...     phall_mip, dapi_mip,
        ...     output_dir="results/segmentation",
        ...     filename_stem="sample1"
        ... )
    """
    # Validate inputs
    if phalloidin_mip.ndim != 2:
        raise ValueError(f"phalloidin_mip must be 2D, got {phalloidin_mip.ndim}D")
    if dapi_mip.ndim != 2:
        raise ValueError(f"dapi_mip must be 2D, got {dapi_mip.ndim}D")
    if phalloidin_mip.shape != dapi_mip.shape:
        raise ValueError(
            f"MIPs must have same shape: phalloidin={phalloidin_mip.shape}, "
            f"dapi={dapi_mip.shape}"
        )
    
    # Normalize helper
    def norm01(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        return np.zeros_like(arr) if mx <= mn else (arr - mn) / (mx - mn)
    
    # Create composite
    phall_norm = norm01(phalloidin_mip)
    dapi_norm = norm01(dapi_mip)
    composite = phalloidin_weight * phall_norm + dapi_weight * dapi_norm
    composite = np.clip(
        np.nan_to_num(composite, nan=0.0, posinf=1.0, neginf=0.0), 
        0, 1
    ).astype(np.float32)
    
    # Save temporary PNG
    tmp_png = Path(tmp_dir) / "cellpose_composite.png"
    tmp_png.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(tmp_png), (composite * 255).astype(np.uint8))
    
    # Get authenticated client
    token = get_hf_token()
    client = Client("mouseland/cellpose", hf_token=token)
    
    # Define segmentation function
    def run_segmentation(resize: int):
        """Run the two-step segmentation flow."""
        # Step 1: Update button
        _ = client.predict(
            filepath=handle_file(str(tmp_png)), 
            api_name="/update_button"
        )
        
        # Step 2: Brief pause
        time.sleep(pause_seconds)
        
        # Step 3: Segment
        return client.predict(
            filepath=[handle_file(str(tmp_png))],
            resize=float(resize),
            max_iter=float(max_iter),
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
            api_name="/cellpose_segment",
        )
    
    # Try segmentation with retry logic
    result = None
    last_error = None
    
    for resize_val in resize_values:
        try:
            result = run_segmentation(resize_val)
            break  # Success!
        except Exception as e:
            last_error = e
            print(f"Retry with resize={resize_val} failed: {e}")
            time.sleep(pause_seconds)
    
    if result is None:
        raise RuntimeError(
            f"Cellpose Space failed after trying all resize values {resize_values}. "
            f"Last error: {last_error}"
        )
    
    # Extract outputs (handle both dict and object with .path attribute)
    def extract_path(output_entry):
        if isinstance(output_entry, dict):
            return output_entry.get("value", output_entry)
        if hasattr(output_entry, "path"):
            return output_entry.path
        return str(output_entry)
    
    masks_tif = extract_path(result[2])
    outlines_png = extract_path(result[3])
    
    # Load mask array
    mask = iio.imread(str(masks_tif)).astype(np.int32)
    
    # Save to output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dst_mask = output_dir / f"{filename_stem}_masks.tif"
    dst_outlines = output_dir / f"{filename_stem}_outlines.png"
    
    dst_mask.write_bytes(Path(masks_tif).read_bytes())
    dst_outlines.write_bytes(Path(outlines_png).read_bytes())
    
    return dst_mask, dst_outlines, mask


def segment_from_loader(
    image_loader,
    phalloidin_channel: Union[int, str],
    dapi_channel: Union[int, str],
    output_dir: Union[str, Path],
    filename_stem: Optional[str] = None,
    **kwargs
) -> Tuple[Path, Path, np.ndarray]:
    """
    Convenience wrapper that works directly with ImageLoader.
    
    Args:
        image_loader: ImageLoader instance with loaded image
        phalloidin_channel: Index or name of phalloidin channel
        dapi_channel: Index or name of DAPI channel
        output_dir: Directory to save outputs
        filename_stem: Optional base name for outputs (defaults to "segmentation")
        **kwargs: Additional arguments passed to segment_cells()
        
    Returns:
        Tuple of (mask_path, outlines_path, mask_array)
        
    Example:
        >>> from colokroll.io import ImageLoader, MIPCreator
        >>> from colokroll.analysis.segmentation import segment_from_loader
        >>> 
        >>> loader = ImageLoader()
        >>> image = loader.load_image("data.ome.tiff")
        >>> 
        >>> mask_path, outlines_path, mask = segment_from_loader(
        ...     loader, "Phalloidin", "DAPI",
        ...     output_dir="results/segmentation"
        ... )
    """
    from ...io import MIPCreator
    
    if not hasattr(image_loader, "image_data") or image_loader.image_data is None:
        # Try to load if extract_channel is called
        pass
    
    # Extract channels
    phall_stack = image_loader.extract_channel(image_loader.image_data, phalloidin_channel)
    dapi_stack = image_loader.extract_channel(image_loader.image_data, dapi_channel)
    
    # Create MIPs
    mip_creator = MIPCreator()
    phall_mip = mip_creator.create_mip(phall_stack, method="max")
    dapi_mip = mip_creator.create_mip(dapi_stack, method="max")
    
    # Generate filename stem if not provided
    if filename_stem is None:
        filename_stem = "segmentation"
    
    return segment_cells(
        phall_mip, dapi_mip,
        output_dir=output_dir,
        filename_stem=filename_stem,
        **kwargs
    )


__all__ = ["segment_cells", "segment_from_loader"]

