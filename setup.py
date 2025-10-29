from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# All dependencies included by default
install_requires = [
    # Image I/O and processing
    "nd2reader>=3.3.0",
    "numpy>=1.20.0",
    "scikit-image>=0.19.0",
    "matplotlib>=3.5.0",
    "Pillow>=9.0.0",
    "tifffile>=2023.7.10",
    "imageio>=2.31.0",
    # Cell segmentation
    "cellpose>=2.0.0",
    "torch>=1.10.0",
    "opencv-python>=4.5.0",
    # Alternative segmentation backend
    "stardist>=0.8.0",
    "tensorflow>=2.8.0",
    # Analysis and statistics
    "scipy>=1.7.0",
    "pandas>=1.4.0",
    "seaborn>=0.11.0",
    # Excel export
    "openpyxl>=3.0.0",
    "xlsxwriter>=3.0.0",
    # GPU acceleration (auto-detects CUDA version)
    "cupy>=11.0.0",
    # Remote processing support
    "gradio_client>=0.15.0",
    "PyYAML>=6.0",
]

# Development tools only (optional)
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "sphinx>=4.5.0",
        "jupyterlab>=3.3.0",
        "ipykernel>=6.0.0",
    ],
}

setup(
    name="colokroll",
    version="0.1.0",
    author="Gabriel Duarte",
    author_email="gabriel.duarte@osumc.edu",
    description="A comprehensive module for confocal microscopy image analysis with colocalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheSaezAtienzarLab/colok-roll",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "colokroll-analyze=colokroll.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "colokroll": ["data/*.json", "templates/*"],
    },
)