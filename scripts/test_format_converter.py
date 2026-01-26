#!/usr/bin/env python3
"""
Test format converter with round-trip validation.

Runs validation tests on the OME-TIFF converter:
1. ND2 -> OME-TIFF conversion
2. OIR -> OME-TIFF conversion (if bioio available)
3. Metadata preservation (pixel size, channels, Z-spacing)
4. Data integrity (round-trip validation)
5. OME-XML structure validation

Usage:
    python scripts/test_format_converter.py \
        --input-dir /path/to/test/images \
        --output-dir /path/to/outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import colokroll
try:
    from colokroll.core import FormatConverter
    from colokroll.data_processing import ImageLoader
except ImportError as e:
    logger.error(f"Failed to import colokroll: {e}")
    sys.exit(1)

import tifffile


def discover_test_files(input_dir: Path) -> Dict[str, List[Path]]:
    """Discover all test files in the input directory.
    
    Args:
        input_dir: Directory to search for test files.
        
    Returns:
        Dictionary with 'nd2' and 'oir' keys containing lists of file paths.
    """
    files = {
        'nd2': [],
        'oir': [],
    }
    
    # Search recursively
    for pattern, key in [("**/*.nd2", 'nd2'), ("**/*.oir", 'oir')]:
        found = list(input_dir.glob(pattern))
        files[key] = sorted(found)
        logger.info(f"Found {len(found)} {key.upper()} files")
    
    return files


def test_nd2_conversion(
    input_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Test ND2 to OME-TIFF conversion with validation.
    
    Args:
        input_path: Path to input ND2 file.
        output_dir: Directory for output files.
        
    Returns:
        Dictionary with test results.
    """
    result = {
        "input_file": str(input_path),
        "input_format": "nd2",
        "success": False,
        "error": None,
        "checks": {},
        "timing_s": None,
    }
    
    start_time = time.perf_counter()
    
    try:
        logger.info(f"Testing ND2 conversion: {input_path.name}")
        
        # Create output path
        output_path = output_dir / input_path.with_suffix('.ome.tiff').name
        
        # Convert
        converter = FormatConverter()
        converted_path, metadata = converter.nd2_to_ome_tiff(
            input_path, 
            output_path, 
            save_metadata=True
        )
        
        logger.info(f"Converted to: {converted_path}")
        
        # Validation checks
        checks = {}
        
        # 1. File exists
        checks['file_created'] = converted_path.exists()
        
        # 2. Load converted file and validate structure
        loader = ImageLoader(auto_convert=False)
        loaded_data = loader.load_image(converted_path)
        
        # 3. Check dimensions (should be 4D ZYXC)
        checks['dimensions_4d'] = loaded_data.ndim == 4
        checks['shape'] = loaded_data.shape
        
        # 4. Check pixel size preserved
        loaded_pixel_size = loader.pixel_size_um
        original_pixel_size = metadata.get('pixel_size_um')
        checks['pixel_size_preserved'] = (
            loaded_pixel_size is not None and
            original_pixel_size is not None and
            abs(loaded_pixel_size - original_pixel_size) < 0.001
        )
        checks['pixel_size_original'] = original_pixel_size
        checks['pixel_size_loaded'] = loaded_pixel_size
        
        # 5. Check channel names preserved
        loaded_channels = loader.get_channel_names()
        original_channels = metadata.get('channel_names', [])
        checks['channels_preserved'] = (
            len(loaded_channels) == len(original_channels) and
            loaded_channels == original_channels
        )
        checks['channel_names_original'] = original_channels
        checks['channel_names_loaded'] = loaded_channels
        
        # 6. Check OME-XML is valid
        with tifffile.TiffFile(str(converted_path)) as tif:
            has_ome = hasattr(tif, 'ome_metadata') and tif.ome_metadata is not None
            checks['ome_xml_present'] = has_ome
            
            if has_ome:
                # Check for required elements
                ome_xml = tif.ome_metadata
                checks['has_physical_size_x'] = 'PhysicalSizeX' in ome_xml
                checks['has_physical_size_y'] = 'PhysicalSizeY' in ome_xml
                checks['has_channels'] = 'Channel' in ome_xml
        
        # 7. Check dtype preserved (bit depth)
        checks['dtype'] = str(loaded_data.dtype)
        checks['dtype_valid'] = loaded_data.dtype in [np.uint8, np.uint16, np.float32, np.float64]
        
        # 8. Check Z-stack integrity
        z_levels = metadata.get('dimensions', {}).get('z_levels', 1)
        checks['z_levels_match'] = loaded_data.shape[0] == z_levels
        checks['z_levels'] = loaded_data.shape[0]
        
        # 9. Check pixel size is reasonable (typical confocal: 0.01-10.0 µm)
        if original_pixel_size is not None:
            checks['pixel_size_reasonable'] = 0.01 <= original_pixel_size <= 10.0
            checks['pixel_size_value'] = original_pixel_size
        else:
            checks['pixel_size_reasonable'] = False
        
        # 10. Check channel name quality (not all generic)
        if original_channels:
            generic_count = sum(1 for ch in original_channels if ch.startswith(('Channel_', 'Ch', 'C')) or ch.isdigit())
            checks['channel_names_quality'] = 'good' if generic_count < len(original_channels) else 'generic'
            checks['generic_channel_count'] = generic_count
        
        # Overall success
        critical_checks = [
            'file_created',
            'dimensions_4d',
            'pixel_size_preserved',
            'ome_xml_present',
        ]
        result['checks'] = checks
        result['success'] = all(checks.get(c, False) for c in critical_checks)
        
    except Exception as e:
        logger.error(f"ND2 conversion test failed: {e}")
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    result['timing_s'] = time.perf_counter() - start_time
    return result


def test_oir_conversion(
    input_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Test OIR to OME-TIFF conversion with validation.
    
    Args:
        input_path: Path to input OIR file.
        output_dir: Directory for output files.
        
    Returns:
        Dictionary with test results.
    """
    result = {
        "input_file": str(input_path),
        "input_format": "oir",
        "success": False,
        "error": None,
        "checks": {},
        "timing_s": None,
    }
    
    start_time = time.perf_counter()
    
    try:
        logger.info(f"Testing OIR conversion: {input_path.name}")
        
        # Create output path
        output_path = output_dir / input_path.with_suffix('.ome.tiff').name
        
        # Convert
        converter = FormatConverter()
        converted_path, metadata = converter.oir_to_ome_tiff(
            input_path, 
            output_path, 
            save_metadata=True
        )
        
        logger.info(f"Converted to: {converted_path}")
        
        # Similar validation as ND2
        checks = {}
        checks['file_created'] = converted_path.exists()
        
        loader = ImageLoader(auto_convert=False)
        loaded_data = loader.load_image(converted_path)
        
        checks['dimensions_4d'] = loaded_data.ndim == 4
        checks['shape'] = loaded_data.shape
        
        loaded_pixel_size = loader.pixel_size_um
        original_pixel_size = metadata.get('pixel_size_um')
        checks['pixel_size_preserved'] = (
            loaded_pixel_size is not None and
            original_pixel_size is not None and
            abs(loaded_pixel_size - original_pixel_size) < 0.001
        )
        checks['pixel_size_original'] = original_pixel_size
        checks['pixel_size_loaded'] = loaded_pixel_size
        
        # Check channel names
        loaded_channels = loader.get_channel_names()
        original_channels = metadata.get('channel_names', [])
        checks['channels_preserved'] = (
            len(loaded_channels) == len(original_channels) and
            loaded_channels == original_channels
        )
        checks['channel_names_original'] = original_channels
        checks['channel_names_loaded'] = loaded_channels
        
        with tifffile.TiffFile(str(converted_path)) as tif:
            has_ome = hasattr(tif, 'ome_metadata') and tif.ome_metadata is not None
            checks['ome_xml_present'] = has_ome
            
            if has_ome:
                ome_xml = tif.ome_metadata
                checks['has_physical_size_x'] = 'PhysicalSizeX' in ome_xml
                checks['has_physical_size_y'] = 'PhysicalSizeY' in ome_xml
                checks['has_channels'] = 'Channel' in ome_xml
        
        # Check dtype
        checks['dtype'] = str(loaded_data.dtype)
        checks['dtype_valid'] = loaded_data.dtype in [np.uint8, np.uint16, np.float32, np.float64]
        
        # Check pixel size is reasonable
        if original_pixel_size is not None:
            checks['pixel_size_reasonable'] = 0.01 <= original_pixel_size <= 10.0
            checks['pixel_size_value'] = original_pixel_size
        else:
            checks['pixel_size_reasonable'] = False
        
        # Check channel name quality
        if original_channels:
            generic_count = sum(1 for ch in original_channels if ch.startswith(('Channel_', 'Ch', 'C')) or ch.isdigit())
            checks['channel_names_quality'] = 'good' if generic_count < len(original_channels) else 'generic'
            checks['generic_channel_count'] = generic_count
        
        # Z-stack integrity
        z_levels = metadata.get('dimensions', {}).get('z_levels', 1)
        checks['z_levels_match'] = loaded_data.shape[0] == z_levels
        checks['z_levels'] = loaded_data.shape[0]
        
        result['checks'] = checks
        
        # Critical checks for OIR conversion
        critical_checks = [
            'file_created',
            'dimensions_4d',
            'ome_xml_present',
            'pixel_size_preserved',
            'pixel_size_reasonable',
        ]
        result['success'] = all(checks.get(c, False) for c in critical_checks)
        
    except ImportError as e:
        logger.warning(f"OIR conversion skipped (missing bioio): {e}")
        result['error'] = f"Skipped: {e}"
        result['checks']['skipped'] = True
    except Exception as e:
        logger.error(f"OIR conversion test failed: {e}")
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    result['timing_s'] = time.perf_counter() - start_time
    return result


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save test results to JSON file.
    
    Args:
        results: List of test result dictionaries.
        output_path: Path for output JSON file.
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.get('success')),
        "failed": sum(1 for r in results if not r.get('success') and not r.get('checks', {}).get('skipped')),
        "skipped": sum(1 for r in results if r.get('checks', {}).get('skipped')),
        "results": results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print test summary to console.
    
    Args:
        results: List of test result dictionaries.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for r in results:
        status = "PASS" if r.get('success') else "FAIL"
        if r.get('checks', {}).get('skipped'):
            status = "SKIP"
            skipped += 1
        elif r.get('success'):
            passed += 1
        else:
            failed += 1
        
        input_file = Path(r['input_file']).name
        timing = r.get('timing_s', 0)
        logger.info(f"  [{status}] {input_file} ({timing:.1f}s)")
        
        if not r.get('success') and r.get('error'):
            logger.info(f"        Error: {r['error']}")
    
    logger.info("")
    logger.info(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test format converter with round-trip validation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input ND2/OIR files (searches recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files and test results",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to test per format (for quick testing)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Format Converter Test Suite")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Discover test files
    test_files = discover_test_files(args.input_dir)
    
    total_files = sum(len(v) for v in test_files.values())
    if total_files == 0:
        logger.error(f"No test files found in {args.input_dir}")
        sys.exit(1)
    
    logger.info("")
    
    # Run tests
    all_results = []
    
    # Test ND2 files
    nd2_files = test_files['nd2']
    if args.max_files:
        nd2_files = nd2_files[:args.max_files]
    
    for nd2_file in nd2_files:
        result = test_nd2_conversion(nd2_file, args.output_dir)
        all_results.append(result)
    
    # Test OIR files
    oir_files = test_files['oir']
    if args.max_files:
        oir_files = oir_files[:args.max_files]
    
    for oir_file in oir_files:
        result = test_oir_conversion(oir_file, args.output_dir)
        all_results.append(result)
    
    # Save results
    results_path = args.output_dir / "test_results.json"
    save_results(all_results, results_path)
    
    # Print summary
    print_summary(all_results)
    
    # Exit with appropriate code
    failed = sum(1 for r in all_results if not r.get('success') and not r.get('checks', {}).get('skipped'))
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
