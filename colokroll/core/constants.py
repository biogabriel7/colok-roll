"""
Core constants used throughout the colokroll package.

This module centralizes magic numbers and string constants to improve
code readability and maintainability.
"""

# =============================================================================
# Numeric Constants
# =============================================================================

# Small value to prevent division by zero
EPSILON = 1e-12

# Image dimension thresholds
MAX_CHANNEL_COUNT = 10  # Heuristic for channel detection in ambiguous arrays
MAX_CHANNEL_WARNING = 20  # Warn if channel count exceeds this
MIN_IMAGE_DIMENSION = 100  # Minimum valid image dimension
MAX_IMAGE_DIMENSION = 10000  # Maximum valid image dimension

# Image format constants
UINT8_MAX = 255
UINT16_MAX = 65535

# Normalization constants
NORMALIZATION_EPSILON = 1e-10  # Epsilon for image normalization to avoid div by zero

# =============================================================================
# String Templates
# =============================================================================

# Default channel name template
CHANNEL_NAME_TEMPLATE = 'Channel_{}'

# Standard axis order for all output images
STANDARD_AXES = 'ZYXC'
EXPECTED_NDIM = 4

# =============================================================================
# Conversion Factors
# =============================================================================

# DPI to micrometers conversion (25400 µm per inch)
DPI_TO_UM = 25400.0

# Bytes conversion factors
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 ** 2
BYTES_PER_GB = 1024 ** 3

# =============================================================================
# Synthetic Image Generation
# =============================================================================

DEFAULT_CROP_PADDING = 10
DEFAULT_SPOT_COUNT = 20
SPOT_RADIUS_SQUARED = 100  # Default radius squared for spot generation
STRIPE_FREQUENCY = 0.1  # Default frequency for stripe pattern
