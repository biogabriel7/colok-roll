#!/usr/bin/env python3
"""
Test script for OIR to OME-TIFF conversion using bioformats2raw.

This script demonstrates how the ImageLoader automatically converts .oir files.
"""

from pathlib import Path
from colokroll.data_processing.image_loader import ImageLoader

def test_oir_conversion():
    """Test OIR file conversion and loading."""
    
    # Example OIR file path
    image_path = Path("/fs/scratch/PAS2598/duarte63/ALIX_confocal_data/Madi/2025-09-18_U2OS_NTC_30 min_60X_DAPI_ALIX(488)_Phallodin(568)_LAMP1(647)_01.oir")
    
    print(f"Testing OIR conversion for: {image_path.name}")
    print("=" * 80)
    
    # Create ImageLoader instance with auto_convert enabled (default)
    image_loader = ImageLoader(auto_convert=True)
    print("✓ ImageLoader created with auto_convert=True")
    
    try:
        # Load the image - this will automatically convert .oir to .ome.tiff
        print(f"\nLoading image from {image_path}...")
        loaded_data = image_loader.load_image(image_path)
        
        print(f"\n✓ Image loaded successfully!")
        print(f"  Shape: {loaded_data.shape}")
        print(f"  Dtype: {loaded_data.dtype}")
        print(f"  Min value: {loaded_data.min()}")
        print(f"  Max value: {loaded_data.max()}")
        print(f"  Mean value: {loaded_data.mean():.2f}")
        
        # Get metadata
        pixel_size = image_loader.get_pixel_size()
        print(f"\n✓ Pixel size: {pixel_size} μm")
        
        channel_names = image_loader.get_channel_names()
        print(f"✓ Channels: {channel_names}")
        
        metadata = image_loader.get_metadata()
        print(f"\n✓ Metadata extracted:")
        print(f"  Original format: {metadata.get('original_format')}")
        print(f"  Conversion method: {metadata.get('conversion_method')}")
        print(f"  Axes: {metadata.get('axes')}")
        print(f"  Dimensions: {metadata.get('dimensions')}")
        
        # Check for converted file
        converted_files = image_loader.get_converted_files()
        if str(image_path) in converted_files:
            ome_tiff_path = converted_files[str(image_path)]
            print(f"\n✓ Converted file saved at: {ome_tiff_path}")
            print(f"  File size: {ome_tiff_path.stat().st_size / (1024**2):.2f} MB")
            
            # Check for metadata JSON
            metadata_json = ome_tiff_path.with_suffix('.json')
            if metadata_json.exists():
                print(f"✓ Metadata JSON saved at: {metadata_json}")
        
        print("\n" + "=" * 80)
        print("SUCCESS: OIR conversion and loading completed!")
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nMake sure bioformats2raw and raw2ometiff are installed:")
        print("  conda install -c ome bioformats2raw raw2ometiff")
        return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_oir_conversion()
    exit(0 if success else 1)

