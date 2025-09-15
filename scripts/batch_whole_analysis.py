#!/usr/bin/env python3
"""
Batch runner that reproduces the whole_analysis notebook EXACTLY, per image file.

Features:
- Processes files sequentially (one sample at a time) and does not start the next until done
- Performs CPU+GPU garbage collection between samples
- Renames channels by matching fluor tags to canonical names:
  AF488 -> ALIX, AF647 -> LAMP1, AF568 -> Phalloidin, DAPI -> DAPI
- Builds Phalloidin+DAPI composite MIP and calls Cellpose Space for segmentation
- Runs colocalization (ALIX vs LAMP1) and writes per-sample CSVs (per-cell and total)

Requirements:
- CUDA + CuPy available for GPU-accelerated steps
- perinuclear_analysis installed (this repo)
- gradio_client, imageio[v3], tifffile, matplotlib

Usage example:
  python scripts/batch_whole_analysis.py \
    --input-dir /path/to/folder \
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

from gradio_client import Client, handle_file

from colokroll.data_processing import ImageLoader
from colokroll.imaging_preprocessing.background_subtraction.background_subtractor import BackgroundSubtractor
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


def derive_channel_names_from_fluor_tags(channel_names: list[str]) -> list[str]:
    """
    Build a renamed channel list of the same length as input, mapping by substrings:
      - contains 'af488' -> 'ALIX'
      - contains 'af647' -> 'LAMP1'
      - contains 'af568' -> 'Phalloidin'
      - contains 'dapi'  -> 'DAPI'
    Unmatched channels retain their original name.
    """
    renamed: list[str] = []
    for nm in channel_names:
        low = str(nm).lower()
        if "af488" in low:
            renamed.append("ALIX")
        elif "af647" in low:
            renamed.append("LAMP1")
        elif "af568" in low:
            renamed.append("Phalloidin")
        elif "dapi" in low:
            renamed.append("DAPI")
        else:
            renamed.append(nm)
    return renamed


def ensure_required_channels_present(renamed: list[str]) -> None:
    required = {"ALIX", "LAMP1", "Phalloidin", "DAPI"}
    present = set(renamed)
    missing = sorted(list(required - present))
    if missing:
        raise ValueError(
            f"Required channels missing after rename by AF tags: {missing}. "
            f"Original metadata may not include AF488/AF647/AF568/DAPI hints."
        )


def run_cellpose_space_on_composite(tmp_png: str) -> tuple[str, str]:
    token = get_hf_token()
    client = Client("mouseland/cellpose", hf_token=token)

    def run_seg(resize: int):
        _ = client.predict(filepath=handle_file(tmp_png), api_name="/update_button")
        time.sleep(1.0)
        return client.predict(
            filepath=[handle_file(tmp_png)],
            resize=resize,
            max_iter=250,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            api_name="/cellpose_segment",
        )

    result = None
    for rs in (600, 400):
        try:
            result = run_seg(rs)
            break
        except Exception as e:
            logging.warning("Cellpose retry with smaller resize due to: %s", e)
            time.sleep(1.0)

    if result is None:
        raise RuntimeError("Cellpose Space failed after retries")

    masks_tif = result[2]["value"] if isinstance(result[2], dict) else (result[2].path if hasattr(result[2], "path") else result[2])
    outlines_png = result[3]["value"] if isinstance(result[3], dict) else (result[3].path if hasattr(result[3], "path") else result[3])
    return str(masks_tif), str(outlines_png)


def process_one_image(image_path: Path, out_root: Path) -> None:
    logging.info("Processing: %s", image_path)
    image_loader = ImageLoader()

    loaded_data = image_loader.load_image(image_path)
    image_loader.get_pixel_size()  # same as notebook
    _ = loaded_data.dtype

    # Rename channels by AF tag mapping
    meta_channel_names = image_loader.get_channel_names()
    mapped_names = derive_channel_names_from_fluor_tags(meta_channel_names)
    image_loader.rename_channels(mapped_names)
    channel_names = image_loader.get_channel_names()
    ensure_required_channels_present(channel_names)

    # Background subtraction (CUDA)
    bg_subtractor = BackgroundSubtractor()
    results: dict[str, tuple[np.ndarray, dict]] = {}

    for i, ch in enumerate(channel_names):
        ch_data = loaded_data[:, :, :, i]
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

    # Plot and save background subtraction comparison figure
    middle_slice_idx = loaded_data.shape[0] // 2
    fig = bg_subtractor.plot_background_subtraction_comparison(
        original_data=loaded_data,
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

    # Composite for Cellpose (Phalloidin + DAPI MIP)
    ph_idx = channel_names.index("Phalloidin")
    da_idx = channel_names.index("DAPI")
    ph_mip = loaded_data[..., ph_idx].max(axis=0).astype(np.float32)
    da_mip = loaded_data[..., da_idx].max(axis=0).astype(np.float32)
    composite = 0.8 * norm01(ph_mip) + 0.2 * norm01(da_mip)
    composite = np.clip(np.nan_to_num(composite, nan=0.0, posinf=1.0, neginf=0.0), 0, 1).astype(np.float32)

    tmp_png = "/tmp/composite.png"
    iio.imwrite(tmp_png, (composite * 255).astype(np.uint8))

    # Cellpose Space
    masks_tif_tmp, outlines_png_tmp = run_cellpose_space_on_composite(tmp_png)

    cellpose_dir = out_root / "cellpose"
    cellpose_dir.mkdir(parents=True, exist_ok=True)
    dst_mask = cellpose_dir / f"{image_path.stem}_phall_dapi_masks.tif"
    dst_outl = cellpose_dir / f"{image_path.stem}_phall_dapi_outlines.png"
    Path(dst_mask).write_bytes(Path(masks_tif_tmp).read_bytes())
    Path(dst_outl).write_bytes(Path(outlines_png_tmp).read_bytes())

    # Colocalization (ALIX vs LAMP1) using the background-corrected results
    mask_path = dst_mask
    min_area = estimate_min_area_threshold(mask_path, fraction_of_median=0.30)

    res = compute_colocalization(
        image=results,  # dict[str, (array, meta)] EXACTLY as notebook usage
        mask=mask_path,
        channel_a="ALIX",
        channel_b="LAMP1",
        normalization_scope="mask",
        thresholding='costes',
        max_border_fraction=0.01,
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
        del loaded_data, results, ph_mip, da_mip, composite
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
    parser = argparse.ArgumentParser(description="Batch whole_analysis runner (sequential, with GC).")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/fs/scratch/PAS2598/duarte63/outputs"),
        help="Root folder for outputs (default aligns with notebook cellpose path)",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff", "*.nd2"],
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
    for idx, f in enumerate(files, 1):
        logging.info("=== [%d/%d] %s ===", idx, len(files), f.name)
        t0 = time.perf_counter()
        process_one_image(f, out_root)
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

    logging.info("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


