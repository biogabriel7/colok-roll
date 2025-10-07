"""
Example: Simple Cell Segmentation using the Proven Working Method

This example demonstrates the simplest way to segment cells using the
Cellpose Gradio Space with the exact workflow that has been tested to work.
"""

from pathlib import Path
import matplotlib.pyplot as plt

from colokroll.io import ImageLoader, MIPCreator
from colokroll.analysis.segmentation import segment_cells, segment_from_loader


# Method 1: Using segment_cells directly (most control)
def example_direct_usage():
    """Example using segment_cells with explicit MIPs."""
    
    # Load image
    loader = ImageLoader()
    image = loader.load_image("path/to/your/image.ome.tiff")
    
    # Get channel names
    print("Available channels:", loader.get_channel_names())
    
    # Extract channels
    phall_stack = loader.extract_channel(image, "Phalloidin")  # or use index: 0
    dapi_stack = loader.extract_channel(image, "DAPI")  # or use index: 1
    
    # Create MIPs
    mip_creator = MIPCreator()
    phall_mip = mip_creator.create_mip(phall_stack, method="max")
    dapi_mip = mip_creator.create_mip(dapi_stack, method="max")
    
    # Segment cells
    mask_path, outlines_path, mask = segment_cells(
        phalloidin_mip=phall_mip,
        dapi_mip=dapi_mip,
        output_dir="results/segmentation",
        filename_stem="sample1",
        # Optional parameters with defaults shown:
        phalloidin_weight=0.8,  # Weight for phalloidin in composite
        dapi_weight=0.2,        # Weight for DAPI in composite
        resize_values=(600, 400),  # Try 600 first, then 400 if it fails
        max_iter=250,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        pause_seconds=1.0,
    )
    
    print(f"Segmentation complete!")
    print(f"Mask saved to: {mask_path}")
    print(f"Outlines saved to: {outlines_path}")
    print(f"Found {mask.max()} cells")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(phall_mip, cmap='gray')
    plt.title("Phalloidin MIP")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(dapi_mip, cmap='gray')
    plt.title("DAPI MIP")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='tab20')
    plt.title(f"Segmentation ({mask.max()} cells)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/segmentation/visualization.png", dpi=150)
    plt.show()


# Method 2: Using segment_from_loader (most convenient)
def example_loader_usage():
    """Example using segment_from_loader for simplicity."""
    
    # Load image
    loader = ImageLoader()
    image = loader.load_image("path/to/your/image.ome.tiff")
    
    # One-line segmentation!
    mask_path, outlines_path, mask = segment_from_loader(
        image_loader=loader,
        phalloidin_channel="Phalloidin",  # or use index: 0
        dapi_channel="DAPI",              # or use index: 1
        output_dir="results/segmentation",
        filename_stem="sample1",
    )
    
    print(f"Segmentation complete! Found {mask.max()} cells")
    print(f"Results saved to: {mask_path.parent}")


# Method 3: Batch processing multiple images
def example_batch_processing():
    """Example processing multiple images."""
    
    image_dir = Path("data/images")
    output_dir = Path("results/batch_segmentation")
    
    for image_path in image_dir.glob("*.ome.tiff"):
        print(f"\nProcessing {image_path.name}...")
        
        try:
            # Load image
            loader = ImageLoader()
            image = loader.load_image(image_path)
            
            # Segment
            mask_path, outlines_path, mask = segment_from_loader(
                image_loader=loader,
                phalloidin_channel="Phalloidin",
                dapi_channel="DAPI",
                output_dir=output_dir,
                filename_stem=image_path.stem,
            )
            
            print(f"  ✓ Found {mask.max()} cells")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


# Method 4: Working with preprocessing results
def example_with_preprocessing():
    """Example using segmentation after background subtraction."""
    
    from colokroll.preprocessing import BackgroundSubtractor
    
    # Load and preprocess
    loader = ImageLoader()
    image = loader.load_image("path/to/your/image.ome.tiff")
    
    # Background subtraction (optional but recommended)
    bg_subtractor = BackgroundSubtractor()
    
    phall_stack = loader.extract_channel(image, "Phalloidin")
    dapi_stack = loader.extract_channel(image, "DAPI")
    
    phall_corrected, _ = bg_subtractor.subtract_background(
        phall_stack, 
        channel_name="Phalloidin",
        method="auto"
    )
    
    dapi_corrected, _ = bg_subtractor.subtract_background(
        dapi_stack,
        channel_name="DAPI", 
        method="auto"
    )
    
    # Create MIPs from corrected data
    mip_creator = MIPCreator()
    phall_mip = mip_creator.create_mip(phall_corrected, method="max")
    dapi_mip = mip_creator.create_mip(dapi_corrected, method="max")
    
    # Segment
    mask_path, outlines_path, mask = segment_cells(
        phalloidin_mip=phall_mip,
        dapi_mip=dapi_mip,
        output_dir="results/segmentation",
        filename_stem="sample1_preprocessed",
    )
    
    print(f"Preprocessed segmentation complete! Found {mask.max()} cells")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_direct_usage()
    # example_loader_usage()
    # example_batch_processing()
    # example_with_preprocessing()
    
    print("\nAvailable examples:")
    print("1. example_direct_usage() - Most control over MIPs")
    print("2. example_loader_usage() - Most convenient")
    print("3. example_batch_processing() - Process multiple images")
    print("4. example_with_preprocessing() - With background subtraction")

