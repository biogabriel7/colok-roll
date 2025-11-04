#!/usr/bin/env python3
"""
Test script to verify colok-roll installation and all dependencies.

Run this after installing the package to ensure everything is working correctly.
"""

import sys
import importlib


def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    display_name = package_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: {e}")
        return False


def main():
    """Run all installation tests."""
    print("Testing colok-roll installation...\n")
    
    # Test core package
    print("Core Package:")
    print("-" * 50)
    success = test_import("colokroll")
    
    if not success:
        print("\n❌ Core package not installed correctly!")
        sys.exit(1)
    
    # Test colokroll submodules
    print("\nColok-Roll Modules:")
    print("-" * 50)
    modules = [
        ("colokroll.core", "Core utilities"),
        ("colokroll.data_processing", "Data processing"),
        ("colokroll.imaging_preprocessing", "Image preprocessing"),
        ("colokroll.analysis", "Analysis modules"),
        ("colokroll.visualization", "Visualization"),
    ]
    
    all_success = True
    for module, name in modules:
        if not test_import(module, name):
            all_success = False
    
    # Test critical dependencies
    print("\nCritical Dependencies:")
    print("-" * 50)
    dependencies = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("skimage", "scikit-image"),
        ("tifffile", "Tifffile"),
        ("imageio", "ImageIO"),
        ("cv2", "OpenCV"),
    ]
    
    for module, name in dependencies:
        if not test_import(module, name):
            all_success = False
    
    # Test deep learning dependencies
    print("\nDeep Learning:")
    print("-" * 50)
    dl_deps = [
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("cellpose", "Cellpose"),
        ("stardist", "StarDist"),
    ]
    
    for module, name in dl_deps:
        if not test_import(module, name):
            all_success = False
    
    # Test GPU acceleration
    print("\nGPU Acceleration:")
    print("-" * 50)
    try:
        import cupy as cp
        print(f"✓ CuPy")
        if cp.cuda.is_available():
            device = cp.cuda.Device()
            free_mem, total_mem = device.mem_info
            print(f"  GPU Memory: {free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total")
        else:
            print("  Warning: CUDA not available")
    except ImportError as e:
        print(f"✗ CuPy: {e}")
        all_success = False
    
    # Test optional dependencies
    print("\nOptional Dependencies:")
    print("-" * 50)
    optional = [
        ("nd2reader", "ND2 Reader"),
        ("openpyxl", "Excel export (openpyxl)"),
        ("xlsxwriter", "Excel export (xlsxwriter)"),
        ("seaborn", "Seaborn"),
        ("yaml", "PyYAML"),
        ("gradio_client", "Gradio Client"),
    ]
    
    for module, name in optional:
        test_import(module, name)
    
    # Test key classes
    print("\nKey Classes:")
    print("-" * 50)
    try:
        from colokroll import ImageLoader
        print("✓ ImageLoader")
        from colokroll import CellSegmenter
        print("✓ CellSegmenter")
        from colokroll import Visualizer
        print("✓ Visualizer")
        from colokroll.imaging_preprocessing.background_subtraction import BackgroundSubtractor
        print("✓ BackgroundSubtractor")
    except ImportError as e:
        print(f"✗ Error importing classes: {e}")
        all_success = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_success:
        print("✅ All critical components installed successfully!")
        print("\nYou can now use colok-roll for your analysis.")
        print("\nExample usage:")
        print("  from colokroll import ImageLoader")
        print("  loader = ImageLoader()")
        return 0
    else:
        print("⚠️  Some components are missing or failed to import.")
        print("\nPlease check the errors above and reinstall missing dependencies.")
        print("\nTo reinstall:")
        print("  conda env update --file environment.yml --prune")
        return 1


if __name__ == "__main__":
    try:
        import colokroll
        print(f"colok-roll version: {colokroll.__version__}\n")
    except ImportError:
        print("colok-roll is not installed!\n")
    
    sys.exit(main())

