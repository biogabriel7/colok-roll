#!/usr/bin/env python3
"""
Batch runner that reproduces the creating_figures_poster notebook EXACTLY, per image file.

Features:
- Processes files sequentially (one sample at a time) and does not start the next until done
- Performs CPU+GPU garbage collection between samples
- Z-slice selection to filter out low-quality slices
- Renames channels to ['DAPI', 'ALIX', 'Phalloidin', 'LAMP1']
- Background subtraction on filtered z-slices
- Builds Phalloidin+DAPI composite MIP and calls Cellpose Space for segmentation
- Runs colocalization (ALIX vs LAMP1) with Otsu thresholding and writes per-sample CSVs

Requirements:
- CUDA + CuPy available for GPU-accelerated steps
- colokroll installed (this repo)
- gradio_client, imageio[v3], tifffile, matplotlib, scikit-image

Usage example:
  python scripts/batch_whole_analysis.py \
    --input-dir /fs/scratch/PAS2598/duarte63/ALIX_confocal_data/Madi/Nov_5 \
    --output-dir /fs/scratch/PAS2598/duarte63/outputs \
    --log-level INFO
"""

from pathlib import Path
import argparse
import logging
import time
import sys
import gc

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio
from skimage.transform import resize as sk_resize

from gradio_client import Client, handle_file

from colokroll.data_processing import ImageLoader
from colokroll.imaging_preprocessing.background_subtraction.background_subtractor import BackgroundSubtractor
from colokroll.imaging_preprocessing.z_slice_detection import select_z_slices
from colokroll.analysis.segmentation_config import get_hf_token
from colokroll.analysis.colocalization import (
    compute_colocalization,
    export_colocalization_json,
    estimate_min_area_threshold,
)


def norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx <= mn:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


def run_cellpose_space_on_composite(tmp_png: str, original_size: tuple[int, int]) -> tuple[np.ndarray, str]:
    """
    Run Cellpose Space segmentation and return the resized mask (matching original_size) and outlines path.
    """
    token = get_hf_token()
    client = Client("mouseland/cellpose", hf_token=token)
    
    # Run segmentation
    _ = client.predict(filepath=handle_file(tmp_png), api_name="/update_button")
    time.sleep(1.0)
    
    result = client.predict(
        filepath=[handle_file(tmp_png)],
        resize=600,
        max_iter=250,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        api_name="/cellpose_segment",
    )
    
    # Extract outputs
    masks_tif = result[2]["value"] if isinstance(result[2], dict) else (result[2].path if hasattr(result[2], "path") else result[2])
    outlines_png = result[3]["value"] if isinstance(result[3], dict) else (result[3].path if hasattr(result[3], "path") else result[3])
    
    # Load mask
    mask = iio.imread(str(masks_tif)).astype(np.int32)
    logging.info(f"Mask from Cellpose: {mask.shape}")
    
    # Resize mask back to original image size if needed
    if mask.shape != original_size:
        logging.info(f"Resizing mask from {mask.shape} to {original_size}")
        mask = sk_resize(
            mask,
            original_size,
            order=0,  # nearest-neighbor (preserves labels)
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.int32)
        logging.info(f"Resized mask: {mask.shape}")
    
    return mask, str(outlines_png)


def is_already_processed(image_path: Path, out_root: Path) -> bool:
    """Check if this image has already been processed by checking for output files."""
    coloc_dir = out_root / "colocalization"
    per_cell_csv = coloc_dir / f"{image_path.stem}_per_cell.csv"
    total_csv = coloc_dir / f"{image_path.stem}_total.csv"
    
    # Check if both key output files exist
    return per_cell_csv.exists() and total_csv.exists()


def process_one_image(image_path: Path, out_root: Path, args) -> None:
    logging.info("Processing: %s", image_path)
    image_loader = ImageLoader()

    # Step 1: Load the image
    loaded_data = image_loader.load_image(image_path)
    pixel_size = image_loader.get_pixel_size()
    logging.info(f"Pixel size: {pixel_size} μm")
    logging.info(f"Data type: {loaded_data.dtype}")
    logging.info(f"Shape: {loaded_data.shape}")

    # Step 2: Z-slice selection
    result = select_z_slices(
        loaded_data,
        method="combined",
        aggregation='median',
        strategy="relative",
        threshold=0.6,
        smooth=3,
        clip_percent=1.0,
    )
    
    # Step 3: Filter to keep only selected slices
    filtered_image = loaded_data[result.indices_keep]
    logging.info(f"Kept {len(result.indices_keep)} slices out of {loaded_data.shape[0]}")

    # Step 4: Rename channels directly
    new_channel_names = ['DAPI', 'ALIX', 'Phalloidin', 'LAMP1']
    image_loader.rename_channels(new_channel_names)
    channel_names = image_loader.get_channel_names()
    logging.info(f"Channels: {channel_names}")

    # Step 5: Background subtraction on filtered_image (CUDA)
    bg_subtractor = BackgroundSubtractor()
    results: dict[str, tuple[np.ndarray, dict]] = {}

    for i, ch in enumerate(channel_names):
        ch_data = filtered_image[:, :, :, i]
        t0 = time.perf_counter()
        corrected, meta = bg_subtractor.subtract_background(
            image=ch_data,
            channel_name=ch,
            # method omitted -> auto search + full run (EXACTLY as notebook)
        )
        try:
            if cp is not None:
                cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
        dt = time.perf_counter() - t0
        logging.info("Background subtraction for %s took %.2fs", ch, dt)
        results[ch] = (corrected, meta)

    # Plot and save background subtraction comparison figure (use filtered_image)
    middle_slice_idx = filtered_image.shape[0] // 2
    fig = bg_subtractor.plot_background_subtraction_comparison(
        original_data=filtered_image,
        corrected_results=results,
        channel_names=channel_names,
        z_slice=middle_slice_idx,
        figsize=(5 * len(channel_names), 12),
    )
    bg_fig_dir = out_root / "background_subtraction"
    bg_fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = bg_fig_dir / f"{image_path.stem}_bg_subtraction_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Step 6: Composite for Cellpose (Phalloidin + DAPI MIP from ORIGINAL loaded_data)
    ph_idx = channel_names.index("Phalloidin")
    da_idx = channel_names.index("DAPI")
    ph_mip = loaded_data[..., ph_idx].max(axis=0).astype(np.float32)
    da_mip = loaded_data[..., da_idx].max(axis=0).astype(np.float32)
    composite = 0.8 * norm01(ph_mip) + 0.2 * norm01(da_mip)
    composite = np.clip(np.nan_to_num(composite, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).astype(np.float32)
    
    # Store original size for mask resize
    original_size = composite.shape
    logging.info(f"Original composite size: {original_size}")

    tmp_png = "/tmp/composite.png"
    iio.imwrite(tmp_png, (composite * 255).astype(np.uint8))

    # Cellpose Space with mask resize
    mask, outlines_png_tmp = run_cellpose_space_on_composite(tmp_png, original_size)

    cellpose_dir = out_root / "cellpose"
    cellpose_dir.mkdir(parents=True, exist_ok=True)
    dst_mask = cellpose_dir / f"{image_path.stem}_phall_dapi_masks.tif"
    dst_outl = cellpose_dir / f"{image_path.stem}_phall_dapi_outlines.png"
    
    # Save the resized mask
    iio.imwrite(dst_mask, mask.astype(np.uint16))
    Path(dst_outl).write_bytes(Path(outlines_png_tmp).read_bytes())
    logging.info(f"✓ Saved mask to: {dst_mask}")
    logging.info(f"✓ Mask shape: {mask.shape} (matches image: {original_size})")

    # Step 7: Colocalization (ALIX vs LAMP1) using the background-corrected results
    mask_path = dst_mask
    min_area = estimate_min_area_threshold(mask_path, fraction_of_median=0.70)

    res = compute_colocalization(
        image=results,  # dict[str, (array, meta)] EXACTLY as notebook usage
        mask=mask_path,
        channel_a="ALIX",
        channel_b="LAMP1",
        thresholding='otsu',  # Using Otsu as in the notebook
        max_border_fraction=0.10,
        min_area=int(min_area),
        border_margin_px=1,
        plot_mask=True,
    )

    # Save colocalization outputs (CSV per sample)
    import pandas as pd
    coloc_dir = out_root / "colocalization"
    coloc_dir.mkdir(parents=True, exist_ok=True)

    df_cells = pd.DataFrame(res["results"]["per_label"]).sort_values("label")
    df_cells.to_csv(coloc_dir / f"{image_path.stem}_per_cell.csv", index=False)

    df_total = pd.DataFrame([res["results"]["total_image"]])
    df_total.to_csv(coloc_dir / f"{image_path.stem}_total.csv", index=False)

    # Optional JSON for provenance
    try:
        export_colocalization_json(
            res,
            out_path=str(coloc_dir / f"{image_path.stem}_colocalization.json"),
        )
    except Exception as e:
        logging.warning("JSON export failed (continuing): %s", e)

    # Cleanup per-sample intermediates (CPU+GPU)
    try:
        del loaded_data, filtered_image, results, ph_mip, da_mip, composite, mask
    except Exception:
        pass
    gc.collect()
    try:
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    plt.close('all')


def find_inputs(input_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(input_dir.rglob(pat)))
    # de-duplicate preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch creating_figures_poster runner (sequential, with GC).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/fs/scratch/PAS2598/duarte63/ALIX_confocal_data/Madi/Nov_5"),
        help="Folder containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/fs/scratch/PAS2598/duarte63/outputs"),
        help="Root folder for outputs",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.ome.tif", "*.ome.tiff"],
        help="Filename patterns to include (recursive glob)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    out_root: Path = args.output_dir
    if not input_dir.exists():
        logging.error("Input dir not found: %s", input_dir)
        return 2
    out_root.mkdir(parents=True, exist_ok=True)

    files = find_inputs(input_dir, args.patterns)
    if not files:
        logging.error("No input files found in %s with patterns: %s", input_dir, args.patterns)
        return 3

    logging.info("Found %d files. Processing sequentially.", len(files))
    skipped_count = 0
    for idx, f in enumerate(files, 1):
        logging.info("=== [%d/%d] %s ===", idx, len(files), f.name)
        
        # Skip if already processed
        if is_already_processed(f, out_root):
            logging.info("⏭ Skipping (already processed): %s", f.name)
            skipped_count += 1
            continue
        
        t0 = time.perf_counter()
        process_one_image(f, out_root, args)
        dt = time.perf_counter() - t0
        logging.info("Finished %s in %.1fs", f.name, dt)

        # Inter-sample cleanup (extra safety)
        gc.collect()
        try:
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

    processed_count = len(files) - skipped_count
    logging.info("All done. Processed: %d, Skipped: %d, Total: %d", processed_count, skipped_count, len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())


