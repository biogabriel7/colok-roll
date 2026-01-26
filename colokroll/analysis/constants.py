"""
Analysis module constants for colocalization and puncta detection.

This module centralizes magic numbers and constants used in
colocalization.py and puncta.py for improved maintainability.
"""

# =============================================================================
# Numeric Constants
# =============================================================================

# Small value to prevent division by zero
EPSILON = 1e-12

# =============================================================================
# Statistical Constants
# =============================================================================

# MAD (Median Absolute Deviation) to standard deviation conversion factor
# For a normal distribution: std ≈ 1.4826 * MAD
MAD_TO_STD = 1.4826

# Z-score for 95% confidence interval
Z_SCORE_95_CI = 1.96

# Fisher z-transform constants
FISHER_Z_TRANSFORM_MULTIPLIER = 0.5

# =============================================================================
# Histogram and Threshold Constants
# =============================================================================

# Default number of bins for Otsu thresholding
DEFAULT_HISTOGRAM_BINS = 256

# Minimum number of voxels for Costes threshold estimation
COSTES_MIN_POINTS = 100

# Number of steps for Costes threshold search
COSTES_STEPS = 256

# Percentile for winsorization in Costes
DEFAULT_WINSOR_CLIP = 0.5

# Coarse percentile grid for Costes threshold search
COSTES_COARSE_PERCENTILE_START = 5.0
COSTES_COARSE_PERCENTILE_STOP = 95.0
COSTES_COARSE_PERCENTILE_STEP = 10.0

# Percentile bounds for effective min/max
COSTES_PERCENTILE_LOWER = 0.1
COSTES_PERCENTILE_UPPER = 99.9

# Refinement steps for threshold search
COSTES_REFINE_STEPS = 8

# =============================================================================
# Correlation Constants
# =============================================================================

# Minimum samples for confidence interval calculation
MIN_SAMPLES_FOR_CI = 10

# Minimum absolute count for correlation
DEFAULT_MIN_ABS_COUNT = 20

# Minimum fraction for correlation
DEFAULT_MIN_FRACTION = 0.05

# Default percentile for Jaccard threshold
DEFAULT_JACCARD_PERCENTILE = 99.0

# Minimum union size for Jaccard
DEFAULT_MIN_UNION = 20

# Low support multiplier for flagging
LOW_SUPPORT_MULTIPLIER = 1.1

# =============================================================================
# Puncta Detection Constants
# =============================================================================

# Minimum distance between puncta seeds (pixels)
DEFAULT_MIN_DISTANCE_PX = 3

# Default SNR threshold for foreground detection
DEFAULT_SNR_THRESHOLD = 3.0

# Expected punctum diameter in micrometers
DEFAULT_EXPECTED_DIAMETER_UM = 0.5

# Minimum punctum diameter in micrometers
DEFAULT_MIN_DIAMETER_UM = 0.2

# Maximum punctum diameter in micrometers
DEFAULT_MAX_DIAMETER_UM = 2.0

# LoG sigma calculation factor: sigma = diameter / (2 * sqrt(2))
LOG_SIGMA_FACTOR = 2.0 * (2 ** 0.5)  # 2 * sqrt(2) ≈ 2.828

# Pixel-based fallback defaults when pixel_size_um not provided
FALLBACK_EXPECTED_DIAMETER_PX = 5.0
FALLBACK_MIN_DIAMETER_PX = 2.0
FALLBACK_MAX_DIAMETER_PX = 20.0

# Channel detection heuristic threshold
CHANNEL_DIM_HEURISTIC = 6

# Minimum background samples for MAD estimation
MIN_BACKGROUND_SAMPLES = 10

# =============================================================================
# Area Filtering Constants
# =============================================================================

# Default minimum area for puncta (pixels)
DEFAULT_MIN_AREA_PX = 4

# Safety margin multiplier for max area calculation
MAX_AREA_SAFETY_MARGIN = 2

# =============================================================================
# BigFISH Detection Constants
# =============================================================================

# Number of threshold samples for elbow curve
BIGFISH_THRESHOLD_SAMPLES = 100

# =============================================================================
# Cell Segmentation Constants
# =============================================================================

# CLAHE clip limit for contrast enhancement
CLAHE_CLIP_LIMIT = 0.01

# Resize candidates for API calls
RESIZE_CANDIDATES = [2200, 2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400]
RESIZE_FALLBACK = 1000

# Overlay blending constants for visualization
OVERLAY_ALPHA_IMAGE = 0.6
OVERLAY_ALPHA_MASK = 0.4
OVERLAY_RGB_MULTIPLIER = [1.0, 1.0, 0.0]  # Yellow tint for mask overlay
