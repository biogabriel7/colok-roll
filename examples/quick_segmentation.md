# Quick Segmentation - Copy-Paste Ready Code

## One-Line Segmentation (Easiest)

```python
from colokroll.io import ImageLoader
from colokroll.analysis.segmentation import segment_from_loader

# Load and segment in one go
loader = ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")

mask_path, outlines_path, mask = segment_from_loader(
    loader, "Phalloidin", "DAPI",
    output_dir="results/segmentation",
    filename_stem="sample1"
)

print(f"✓ Found {mask.max()} cells")
print(f"✓ Mask: {mask_path}")
```

## With Preprocessing (Recommended)

```python
from colokroll.io import ImageLoader, MIPCreator
from colokroll.preprocessing import BackgroundSubtractor
from colokroll.analysis.segmentation import segment_cells
import numpy as np

# 1. Load image
loader = ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")
print(f"Loaded: {image.shape}, channels: {loader.get_channel_names()}")

# 2. Background subtraction
bg_sub = BackgroundSubtractor()

phall_stack = loader.extract_channel(image, "Phalloidin")
dapi_stack = loader.extract_channel(image, "DAPI")

phall_clean, _ = bg_sub.subtract_background(phall_stack, channel_name="Phalloidin", method="auto")
dapi_clean, _ = bg_sub.subtract_background(dapi_stack, channel_name="DAPI", method="auto")

# 3. Create MIPs
mip = MIPCreator()
phall_mip = mip.create_mip(phall_clean, method="max")
dapi_mip = mip.create_mip(dapi_clean, method="max")

# 4. Segment
mask_path, outlines_path, mask = segment_cells(
    phall_mip, dapi_mip,
    output_dir="results/segmentation",
    filename_stem="sample1",
    resize_values=(600, 400),  # Try 600 first, fallback to 400
)

print(f"✓ Segmented {mask.max()} cells")
```

## Batch Processing

```python
from pathlib import Path
from colokroll.io import ImageLoader
from colokroll.analysis.segmentation import segment_from_loader

# Process all images in a directory
image_dir = Path("data/raw")
output_dir = Path("results/segmentation")

for img_file in image_dir.glob("*.ome.tiff"):
    print(f"\n Processing: {img_file.name}")
    
    try:
        loader = ImageLoader()
        image = loader.load_image(img_file)
        
        mask_path, _, mask = segment_from_loader(
            loader, "Phalloidin", "DAPI",
            output_dir=output_dir,
            filename_stem=img_file.stem
        )
        
        print(f"  ✓ Found {mask.max()} cells -> {mask_path.name}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
```

## Direct Control (Your Working Code as Function)

```python
from colokroll.io import ImageLoader, MIPCreator  
from colokroll.analysis.segmentation import segment_cells

# If you already have MIPs from your workflow:
loader = ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")

# Extract channels
phall_stack = loader.extract_channel(image, "Phalloidin")
dapi_stack = loader.extract_channel(image, "DAPI")

# Create MIPs (this is what you had in your working code)
mip_creator = MIPCreator()
phall_mip = mip_creator.create_mip(phall_stack, method="max")
dapi_mip = mip_creator.create_mip(dapi_stack, method="max")

# This does exactly what your working code does!
mask_path, outlines_path, mask = segment_cells(
    phalloidin_mip=phall_mip,
    dapi_mip=dapi_mip,
    output_dir="/fs/scratch/PAS2598/duarte63/outputs/cellpose",
    filename_stem="my_sample",
    phalloidin_weight=0.8,  # Your weights
    dapi_weight=0.2,
    resize_values=(600, 400),  # Your resize values
    pause_seconds=1.0,  # Your pause
)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap="tab20")
plt.title(f"Mask ({mask.max()} labels)")
plt.axis("off")
plt.show()
```

## Troubleshooting Tips

### If segmentation fails:
```python
# Try with smaller resize values and longer pause
mask_path, outlines_path, mask = segment_cells(
    phall_mip, dapi_mip,
    output_dir="results",
    filename_stem="sample",
    resize_values=(400, 200, 100),  # More fallbacks
    pause_seconds=2.0,  # Longer pause
)
```

### If composite looks bad:
```python
# Adjust weights (more phalloidin, less DAPI)
mask_path, outlines_path, mask = segment_cells(
    phall_mip, dapi_mip,
    output_dir="results",
    filename_stem="sample",
    phalloidin_weight=0.9,  # Increase
    dapi_weight=0.1,        # Decrease
)
```

### Check composite before segmentation:
```python
import numpy as np
import matplotlib.pyplot as plt

def norm01(a):
    a = a.astype(np.float32)
    mn, mx = a.min(), a.max()
    return np.zeros_like(a) if mx <= mn else (a - mn) / (mx - mn)

# Preview the composite
composite = 0.8 * norm01(phall_mip) + 0.2 * norm01(dapi_mip)
plt.imshow(composite, cmap='gray')
plt.title("Composite (what Cellpose sees)")
plt.colorbar()
plt.show()
```

## Environment Setup

Required packages:
```bash
pip install gradio_client imageio
```

Set your Hugging Face token:
```bash
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

Or in Python:
```python
import os
os.environ["HUGGINGFACE_TOKEN"] = "hf_your_token_here"
```

