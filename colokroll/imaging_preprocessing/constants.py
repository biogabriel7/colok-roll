"""
Imaging preprocessing constants for background subtraction and z-slice detection.

This module centralizes magic numbers and constants used in
background_subtraction and z_slice_detection modules.
"""

# =============================================================================
# Auto-mode Weights and Thresholds
# =============================================================================

# Weights for scoring background subtraction quality
# Order: (background_removal, contrast, detail_preservation, zero_penalty)
DEFAULT_AUTO_WEIGHTS = (0.7, 0.4, 0.2, 0.1)

# SSIM weight in composite score
SSIM_WEIGHT = 0.3

# =============================================================================
# Search Algorithm Parameters
# =============================================================================

# Improvement threshold to consider a result better
IMPROVEMENT_EPS = 0.02

# Maximum evaluations per method during auto-search
MAX_EVALS_PER_METHOD = 60

# Minimum step sizes for parameter refinement
MIN_INT_STEP = 1
MIN_FLOAT_STEP = 0.5

# Number of top candidates to refine from coarse search
TOPN_COARSE = 3

# Maximum tested parameter values to track
TESTED_VALUES_MAX = 20

# Shrink factor for parameter refinement
REFINEMENT_SHRINK_FACTOR = 0.75

# =============================================================================
# Quality Score Constants
# =============================================================================

# Minimum standard deviation ratio for valid scoring
MIN_STD_RATIO = 0.08

# Minimum absolute standard deviation
MIN_STD_ABS = 1e-3

# Minimum pixels for foreground detection
FG_MIN_PIXELS = 50

# Foreground dilation radius for mask expansion
FG_DILATE_RADIUS = 2

# =============================================================================
# Scoring Weights
# =============================================================================

# Weights for negative control scoring
SCORE_WEIGHT_MEAN = 0.5
SCORE_WEIGHT_STD = 0.3
SCORE_WEIGHT_ZERO_FRACTION = 0.2

# Near-zero threshold for intensity values
NEAR_ZERO_THRESHOLD = 1.0

# Target standard deviation ratio for good correction
TARGET_STD_RATIO = 0.3

# Correlation penalty threshold
CORRELATION_STD_THRESHOLD = 0.5

# =============================================================================
# Memory Management Constants
# =============================================================================

# Maximum memory fraction of free GPU memory to use
MAX_GPU_MEMORY_FRACTION = 0.8

# Maximum chunk memory for processing (MB)
MAX_CHUNK_MEMORY_MB = 2048

# Image size threshold for chunked processing
CHUNK_THRESHOLD_PIXELS = 1000

# =============================================================================
# GPU Configuration Constants
# =============================================================================

# Minimum GPU memory required (GB)
MIN_GPU_MEMORY_GB = 2.0

# Chunk processing threshold (GB)
CHUNK_PROCESSING_THRESHOLD_GB = 4.0

# Memory tiers for GPU performance estimation (GB)
GPU_MEMORY_TIER_HIGH = 16
GPU_MEMORY_TIER_MEDIUM = 8
GPU_MEMORY_TIER_LOW = 4

# Base speedup estimates by method
BASE_SPEEDUP_ROLLING_BALL = 15.0
BASE_SPEEDUP_GAUSSIAN = 8.0
BASE_SPEEDUP_TWO_STAGE = 20.0
BASE_SPEEDUP_MORPHOLOGICAL = 5.0

# Architecture multipliers
MULTIPLIER_AMPERE_ABOVE = 1.2
MULTIPLIER_TURING = 1.0
MULTIPLIER_OLDER = 0.8

# Memory multipliers
MULTIPLIER_HIGH_MEMORY = 1.1
MULTIPLIER_MEDIUM_MEMORY = 1.0
MULTIPLIER_LOW_MEMORY = 0.9

# Estimated memory bandwidth by architecture (GB/s)
BANDWIDTH_AMPERE = 900.0
BANDWIDTH_TURING = 700.0
BANDWIDTH_VOLTA = 500.0
BANDWIDTH_OLDER = 300.0

# CUDA compute capability threshold for modern features
CUDA_MIN_COMPUTE_CAPABILITY = 6

# =============================================================================
# Z-Slice Detection Constants
# =============================================================================

# Default threshold for relative strategy
DEFAULT_Z_THRESHOLD = 0.6

# Default smoothing window size
DEFAULT_SMOOTH_WINDOW = 3

# Default auto-keep fraction for closest_to_peak strategy
DEFAULT_AUTO_KEEP_FRACTION = 0.8

# Default FFT cutoff frequency
DEFAULT_FFT_CUTOFF = 0.15

# Default step distance for quality metrics
DEFAULT_STEP_DISTANCE = 1.0

# Default fitting points for quality metrics
DEFAULT_N_FITTING_POINTS = 5

# Maximum slices for sampling in scoring
MAX_SLICES_FOR_SCORING = 7

# Cache score tolerance for auto-selection
AUTO_CACHE_SCORE_TOLERANCE = 0.05

# =============================================================================
# Focus Measure Quality Constants
# =============================================================================

# Epsilon for slope comparison
SLOPE_EPSILON = 1e-10

# Fraction divisors for curve segmentation
LEFT_CRITICAL_POINT_DIVISOR = 3
RIGHT_CRITICAL_POINT_DIVISOR_NUMERATOR = 2
RIGHT_CRITICAL_POINT_DIVISOR_DENOMINATOR = 3

# FWHM threshold multiplier
FWHM_THRESHOLD_MULTIPLIER = 0.5

# Default tolerance for relative height comparison
DEFAULT_RELATIVE_HEIGHT_TOLERANCE = 0.1

# =============================================================================
# Visualization Constants
# =============================================================================

# Grid column count for strategy comparison plots
STRATEGY_PLOT_COLUMNS = 6

# Figure size multipliers
FIGURE_WIDTH_MULTIPLIER = 1.2
FIGURE_HEIGHT_MULTIPLIER = 0.25
FIGURE_MIN_WIDTH = 12
FIGURE_MIN_HEIGHT = 8

# Annotation thresholds
MAX_SLICES_FOR_ANNOTATION = 60
MAX_STRATEGIES_FOR_ANNOTATION = 14

# Default plot colors
PLOT_COLOR_REMOVED = '#ff6b6b'  # Red for removed slices
PLOT_COLOR_KEPT = '#51cf66'  # Green for kept slices

# Composite score formula weight for Rsg
COMPOSITE_RSG_WEIGHT = 2.0
