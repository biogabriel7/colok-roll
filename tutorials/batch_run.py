#!/usr/bin/env python3
"""
Batch tutorial: run exploratory steps across a folder of OME-TIFFs.

Example:
  python tutorials/batch_run.py \
    --input-dir /path/to/ome_tiffs \
    --output-dir /path/to/output \
    --channels "GM130,Phalloidin,Chlorotoxin,DAPI" \
    --seg-channels "Phalloidin,DAPI"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import colokroll as cr


logger = logging.getLogger(__name__)


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_images(input_dir: Path, pattern: str) -> List[Path]:
    return sorted(input_dir.rglob(pattern))


def run_single(
    image_path: Path,
    output_dir: Path,
    channel_names: Optional[List[str]],
    auto_keep_fraction: float,
    seg_channels: Tuple[str, str],
    seg_weights: Tuple[float, float],
    skip_segmentation: bool,
) -> Dict[str, object]:
    t0 = time.perf_counter()

    loader = cr.ImageLoader(auto_convert=False)
    image = loader.load_image(image_path)
    if channel_names:
        loader.rename_channels(channel_names)
    channel_names = loader.get_channel_names()

    z_result, _ = cr.select_z_slices_auto_method(
        image,
        axes="ZYXC",
        auto_keep_fraction=auto_keep_fraction,
    )
    filtered = image[z_result.indices_keep]

    bg_subtractor = cr.BackgroundSubtractor()
    bg_results: Dict[str, Tuple[np.ndarray, Dict[str, object]]] = {}
    bg_methods: Dict[str, str] = {}

    for i, ch in enumerate(channel_names):
        ch_data = filtered[:, :, :, i]
        corrected, meta = bg_subtractor.subtract_background(
            image=ch_data,
            channel_name=ch,
            auto_cache_score_tolerance=0.20,
        )
        bg_results[ch] = (corrected, meta)
        bg_methods[ch] = str(meta.get("method", "auto"))

    seg_info: Dict[str, object] = {}
    if not skip_segmentation:
        seg_dir = output_dir / "segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        segmenter = cr.CellSegmenter(output_dir=seg_dir)
        seg = segmenter.segment_from_results(
            results=bg_results,
            channel_a=seg_channels[0],
            channel_b=seg_channels[1],
            channel_weights=seg_weights,
            projection="mip",
            output_format="png8",
            save_basename=image_path.stem,
        )
        seg_info = {
            "mask_path": str(seg.mask_path),
            "outlines_path": str(seg.outlines_path),
            "n_labels": int(np.unique(seg.mask_array).size - 1),
        }

    elapsed = time.perf_counter() - t0

    summary = {
        "input": str(image_path),
        "shape": list(image.shape),
        "kept_slices": int(len(z_result.indices_keep)),
        "bg_methods": bg_methods,
        "segmentation": seg_info,
        "elapsed_s": round(elapsed, 2),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch tutorial runner")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.ome.tiff")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--auto-keep-fraction", type=float, default=0.8)
    parser.add_argument("--seg-channels", type=str, default="Phalloidin,DAPI")
    parser.add_argument("--seg-weights", type=str, default="0.8,0.2")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    images = discover_images(args.input_dir, args.pattern)
    if args.max_images:
        images = images[: args.max_images]

    if not images:
        raise SystemExit(f"No images found in {args.input_dir} with pattern {args.pattern}")

    channel_names = _parse_list(args.channels)
    seg_channels = tuple(_parse_list(args.seg_channels) or ["Phalloidin", "DAPI"])  # type: ignore[assignment]
    if len(seg_channels) != 2:
        raise SystemExit("--seg-channels must be two comma-separated names")

    seg_weights_raw = [float(x) for x in (args.seg_weights.split(",") if args.seg_weights else [0.8, 0.2])]
    if len(seg_weights_raw) != 2:
        raise SystemExit("--seg-weights must be two comma-separated floats")
    seg_weights = (seg_weights_raw[0], seg_weights_raw[1])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, object]] = []
    for image_path in images:
        logger.info("Processing %s", image_path.name)
        image_out_dir = args.output_dir / image_path.stem
        image_out_dir.mkdir(parents=True, exist_ok=True)
        summary = run_single(
            image_path=image_path,
            output_dir=image_out_dir,
            channel_names=channel_names,
            auto_keep_fraction=args.auto_keep_fraction,
            seg_channels=(seg_channels[0], seg_channels[1]),
            seg_weights=seg_weights,
            skip_segmentation=args.skip_segmentation,
        )
        summary_path = image_out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        all_summaries.append(summary)

    (args.output_dir / "batch_summary.json").write_text(json.dumps(all_summaries, indent=2))
    logger.info("Done. Wrote %d summaries to %s", len(all_summaries), args.output_dir)


if __name__ == "__main__":
    main()
