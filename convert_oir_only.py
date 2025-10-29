#!/usr/bin/env python3
"""
Batch convert .oir files to .ome.tiff format using bioformats2raw and raw2ometiff.
Standalone script - no nd2reader dependency required.

Usage:
    python convert_oir_only.py <directory>
    python convert_oir_only.py  # uses current directory
"""

import sys
import json
import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tools():
    """Check if required conversion tools are available."""
    if not shutil.which("bioformats2raw"):
        logger.error("bioformats2raw not found. Install with: conda install -c ome bioformats2raw")
        return False
    if not shutil.which("raw2ometiff"):
        logger.error("raw2ometiff not found. Install with: conda install -c ome raw2ometiff")
        return False
    return True


def convert_oir_to_ome_tiff(
    input_path: Path,
    output_path: Optional[Path] = None,
    save_metadata: bool = True
) -> Optional[Path]:
    """Convert single .oir file to .ome.tiff.
    
    Args:
        input_path: Path to input .oir file
        output_path: Optional output path (defaults to same name with .ome.tiff)
        save_metadata: If True, saves metadata JSON
        
    Returns:
        Path to converted file, or None on failure
    """
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return None
    
    if input_path.suffix.lower() != ".oir":
        logger.error(f"Not an .oir file: {input_path}")
        return None
    
    if output_path is None:
        output_path = input_path.with_suffix(".ome.tiff")
    
    logger.info(f"Converting: {input_path.name}")
    
    # Create temporary directory for intermediate zarr format
    with tempfile.TemporaryDirectory(prefix="oir_conv_") as temp_dir:
        temp_zarr = Path(temp_dir) / "temp.zarr"
        
        try:
            # Step 1: OIR to Zarr
            logger.debug(f"  Step 1/2: Converting to Zarr...")
            result = subprocess.run(
                ["bioformats2raw", str(input_path), str(temp_zarr)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Step 2: Zarr to OME-TIFF
            logger.debug(f"  Step 2/2: Converting to OME-TIFF...")
            result = subprocess.run(
                ["raw2ometiff", str(temp_zarr), str(output_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract basic metadata if requested
            if save_metadata:
                try:
                    import tifffile
                    
                    with tifffile.TiffFile(str(output_path)) as tif:
                        data = tif.series[0].asarray()
                        axes = tif.series[0].axes
                        
                        # Try to extract OME-XML metadata
                        metadata = {
                            "original_format": "oir",
                            "conversion_method": "bioformats2raw + raw2ometiff",
                            "axes": axes,
                            "shape": data.shape,
                        }
                        
                        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                            try:
                                import xml.etree.ElementTree as ET
                                root = ET.fromstring(tif.ome_metadata)
                                namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                                
                                # Extract pixel sizes
                                pixels = root.find('.//ome:Pixels', namespaces)
                                if pixels is not None:
                                    metadata['pixel_size_x'] = pixels.get('PhysicalSizeX')
                                    metadata['pixel_size_y'] = pixels.get('PhysicalSizeY')
                                    metadata['pixel_size_z'] = pixels.get('PhysicalSizeZ')
                                
                                # Extract channel names
                                channels = root.findall('.//ome:Channel', namespaces)
                                metadata['channels'] = [ch.get('Name', f'Channel_{i}') 
                                                       for i, ch in enumerate(channels)]
                            except Exception as e:
                                logger.debug(f"  Could not parse OME-XML: {e}")
                        
                        # Save metadata JSON
                        metadata_path = output_path.with_suffix('.json')
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2, default=str)
                        logger.debug(f"  Metadata saved: {metadata_path.name}")
                        
                except ImportError:
                    logger.debug("  tifffile not available, skipping metadata extraction")
                except Exception as e:
                    logger.debug(f"  Metadata extraction failed: {e}")
            
            logger.info(f"✓ Success: {output_path.name}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Conversion failed: {input_path.name}")
            logger.error(f"  Command: {' '.join(e.cmd)}")
            logger.error(f"  Return code: {e.returncode}")
            if e.stderr:
                logger.error(f"  Error output:")
                for line in e.stderr.strip().split('\n')[:10]:  # Show first 10 lines
                    logger.error(f"    {line}")
            return None
        except Exception as e:
            logger.error(f"✗ Error: {input_path.name} - {e}")
            return None


def batch_convert_oir(input_dir: Path) -> Dict[Path, Optional[Path]]:
    """Convert all .oir files in directory.
    
    Args:
        input_dir: Directory containing .oir files
        
    Returns:
        Dictionary mapping input paths to output paths (None on failure)
    """
    oir_files = sorted(input_dir.glob("*.oir"))
    
    if not oir_files:
        logger.warning(f"No .oir files found in {input_dir}")
        return {}
    
    logger.info(f"Found {len(oir_files)} .oir files")
    logger.info("=" * 80)
    
    conversions = {}
    for oir_file in oir_files:
        output_path = convert_oir_to_ome_tiff(oir_file)
        conversions[oir_file] = output_path
    
    return conversions


def main():
    """Main entry point."""
    
    # Get directory from command line or use current directory
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    else:
        input_dir = Path.cwd()
    
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Not a directory: {input_dir}")
        sys.exit(1)
    
    logger.info(f"OIR to OME-TIFF Batch Conversion")
    logger.info(f"Directory: {input_dir}")
    logger.info("=" * 80)
    
    # Check tools
    if not check_tools():
        logger.error("Required tools not found. Make sure you're in the oir-convert environment:")
        logger.error("  conda activate oir-convert")
        sys.exit(1)
    
    # Batch convert
    conversions = batch_convert_oir(input_dir)
    
    # Summary
    successful = sum(1 for path in conversions.values() if path is not None)
    failed = len(conversions) - successful
    
    logger.info("=" * 80)
    logger.info(f"Conversion complete: {successful} successful, {failed} failed")
    
    if failed > 0:
        logger.warning("Failed conversions:")
        for input_path, output_path in conversions.items():
            if output_path is None:
                logger.warning(f"  - {input_path.name}")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

