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
    
    # Create temporary directory for intermediate zarr format in the same directory as input
    # This uses the scratch filesystem instead of /tmp which may have space/access issues
    temp_base_dir = input_path.parent
    with tempfile.TemporaryDirectory(prefix="oir_conv_", dir=str(temp_base_dir)) as temp_dir:
        temp_zarr = Path(temp_dir) / "temp.zarr"
        
        try:
            # Step 1: OIR to Zarr (with options to preserve metadata)
            logger.info(f"  Step 1/2: Converting to Zarr...")
            # Standard bioformats2raw syntax: bioformats2raw [OPTIONS] input_image output_directory
            result1 = subprocess.run(
                [
                    "bioformats2raw",
                    "--memo-directory", str(temp_dir),  # Cache for faster processing
                    str(input_path),  # Input .oir file
                    str(temp_zarr)    # Output zarr directory
                ],
                capture_output=True,
                text=True,
                check=False  # Don't raise exception, handle errors manually
            )
            
            # Check if step 1 succeeded
            if result1.returncode != 0:
                logger.error(f"✗ Step 1 (bioformats2raw) failed: {input_path.name}")
                logger.error(f"  Command: bioformats2raw --memo-directory {temp_dir} {input_path} {temp_zarr}")
                logger.error(f"  Return code: {result1.returncode}")
                if result1.stderr:
                    logger.error(f"  Error output:")
                    for line in result1.stderr.strip().split('\n')[:20]:
                        logger.error(f"    {line}")
                if result1.stdout:
                    logger.error(f"  Standard output:")
                    for line in result1.stdout.strip().split('\n')[:20]:
                        logger.error(f"    {line}")
                return None
            
            # Show stdout and stderr even on success to understand what happened
            if result1.stdout and result1.stdout.strip():
                logger.info(f"  bioformats2raw stdout:")
                for line in result1.stdout.strip().split('\n')[:10]:
                    logger.info(f"    {line}")
            if result1.stderr and result1.stderr.strip():
                logger.info(f"  bioformats2raw stderr:")
                for line in result1.stderr.strip().split('\n')[:10]:
                    logger.info(f"    {line}")
            
            # List what was actually created in temp directory
            temp_dir_path = Path(temp_dir)
            created_items = list(temp_dir_path.iterdir())
            logger.info(f"  Items created in temp dir: {[item.name for item in created_items]}")
            
            # Verify zarr was created
            if not temp_zarr.exists():
                logger.error(f"✗ Step 1 succeeded but zarr directory not created: {temp_zarr}")
                logger.error(f"  Expected: {temp_zarr}")
                logger.error(f"  Temp dir contents: {[str(item) for item in created_items]}")
                return None
            
            logger.info(f"  Step 1 complete: Zarr created at {temp_zarr}")
            
            # Step 2: Zarr to OME-TIFF (with compression)
            logger.info(f"  Step 2/2: Converting to OME-TIFF...")
            result = subprocess.run(
                [
                    "raw2ometiff",
                    "--compression", "LZW",  # Use LZW compression
                    str(temp_zarr),
                    str(output_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract basic metadata if requested
            if save_metadata:
                try:
                    import tifffile
                    import xml.etree.ElementTree as ET
                    
                    with tifffile.TiffFile(str(output_path)) as tif:
                        # Try to extract OME-XML metadata
                        metadata = {
                            "original_format": "oir",
                            "conversion_method": "bioformats2raw + raw2ometiff",
                            "file_name": input_path.name,
                        }
                        
                        # Get basic image info
                        if tif.series:
                            series = tif.series[0]
                            metadata['axes'] = series.axes
                            metadata['shape'] = series.shape
                            metadata['dtype'] = str(series.dtype)
                        
                        # Extract OME-XML metadata
                        ome_xml = None
                        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                            ome_xml = tif.ome_metadata
                        elif hasattr(tif, 'pages') and len(tif.pages) > 0:
                            # Try to get from first page description
                            page = tif.pages[0]
                            if hasattr(page, 'description') and page.description:
                                ome_xml = page.description
                        
                        if ome_xml:
                            try:
                                root = ET.fromstring(ome_xml)
                                # Try multiple namespace versions
                                for ns_year in ['2016-06', '2015-01', '2013-06', '2012-06', '2011-06']:
                                    namespaces = {'ome': f'http://www.openmicroscopy.org/Schemas/OME/{ns_year}'}
                                    
                                    # Extract pixel sizes
                                    pixels = root.find('.//ome:Pixels', namespaces)
                                    if pixels is not None:
                                        px = pixels.get('PhysicalSizeX')
                                        py = pixels.get('PhysicalSizeY')
                                        pz = pixels.get('PhysicalSizeZ')
                                        pxu = pixels.get('PhysicalSizeXUnit', 'µm')
                                        pyu = pixels.get('PhysicalSizeYUnit', 'µm')
                                        pzu = pixels.get('PhysicalSizeZUnit', 'µm')
                                        
                                        if px: metadata['pixel_size_x'] = f"{px} {pxu}"
                                        if py: metadata['pixel_size_y'] = f"{py} {pyu}"
                                        if pz: metadata['pixel_size_z'] = f"{pz} {pzu}"
                                        
                                        # Get other pixel info
                                        metadata['size_x'] = pixels.get('SizeX')
                                        metadata['size_y'] = pixels.get('SizeY')
                                        metadata['size_z'] = pixels.get('SizeZ')
                                        metadata['size_c'] = pixels.get('SizeC')
                                        metadata['size_t'] = pixels.get('SizeT')
                                    
                                    # Extract channel information
                                    channels = root.findall('.//ome:Channel', namespaces)
                                    if channels:
                                        metadata['channels'] = []
                                        for i, ch in enumerate(channels):
                                            ch_info = {
                                                'name': ch.get('Name', f'Channel_{i}'),
                                                'id': ch.get('ID', ''),
                                                'samples_per_pixel': ch.get('SamplesPerPixel'),
                                            }
                                            # Get emission wavelength if available
                                            em_wave = ch.get('EmissionWavelength')
                                            ex_wave = ch.get('ExcitationWavelength')
                                            if em_wave: ch_info['emission_wavelength'] = f"{em_wave} nm"
                                            if ex_wave: ch_info['excitation_wavelength'] = f"{ex_wave} nm"
                                            
                                            metadata['channels'].append(ch_info)
                                        
                                        logger.info(f"  Found {len(channels)} channels with metadata")
                                        break
                                
                                if 'channels' not in metadata:
                                    logger.warning(f"  No channel metadata found in OME-XML")
                                if 'pixel_size_x' not in metadata:
                                    logger.warning(f"  No pixel size metadata found in OME-XML")
                                    
                            except Exception as e:
                                logger.warning(f"  Could not parse OME-XML: {e}")
                        else:
                            logger.warning(f"  No OME-XML metadata found in TIFF file")
                        
                        # Save metadata JSON
                        metadata_path = output_path.with_suffix('.json')
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2, default=str)
                        logger.info(f"  Metadata saved: {metadata_path.name}")
                        
                except ImportError as e:
                    logger.warning(f"  tifffile not available for metadata extraction: {e}")
                except Exception as e:
                    logger.warning(f"  Metadata extraction failed: {e}")
            
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

