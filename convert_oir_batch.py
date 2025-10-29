#!/usr/bin/env python3
"""
Batch convert .oir files to .ome.tiff format.

Usage:
    python convert_oir_batch.py <directory>
    python convert_oir_batch.py  # uses current directory
"""

import sys
import logging
from pathlib import Path
from colokroll.core.format_converter import FormatConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Convert all .oir files in specified directory to .ome.tiff."""
    
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
    
    logger.info(f"Scanning directory: {input_dir}")
    logger.info("=" * 80)
    
    # Count .oir files
    oir_files = list(input_dir.glob("*.oir"))
    if not oir_files:
        logger.warning(f"No .oir files found in {input_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(oir_files)} .oir files to convert")
    
    # Initialize converter
    converter = FormatConverter(preserve_original=True)
    
    # Batch convert
    try:
        conversions = converter.batch_convert_oir(
            input_dir=input_dir,
            output_dir=None,  # Save in same directory
            pattern="*.oir"
        )
        
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
        
    except Exception as e:
        logger.error(f"Batch conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

