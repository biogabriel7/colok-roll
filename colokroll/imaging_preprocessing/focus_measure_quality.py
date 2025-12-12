"""Focus Measure Quality Metrics (Piao et al. 2025).

This module implements quantitative evaluation metrics for focus measure operators
based on the morphological characteristics of focus measure curves, as described in:

Piao, W., Han, Y., Hu, L., & Wang, C. (2025). Quantitative Evaluation of Focus
Measure Operators in Optical Microscopy. Sensors, 25, 3144.

Metrics implemented:
- Ws (steep slope region width): Sensitivity to focus changes
- Rsg (steep-to-gradual ratio): Ability to distinguish clear vs. blurry images
- Cp (curvature at peak): Sensitivity near the focal position
- RRMSE (relative root mean square error): Noise robustness

Integration with colokroll:
    These metrics can be added to ZSliceSelectionResult to help users
    objectively compare which focus strategy works best for their data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize_scalar


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class CurveSegmentation:
    """Results from multi-point linear fitting curve segmentation.
    
    Attributes:
        k_lcp: Left cutoff point index (float, interpolated)
        k_rcp: Right cutoff point index (float, interpolated)
        f_lcp: Focus measure value at left cutoff point
        f_rcp: Focus measure value at right cutoff point
        k_max: Index of the peak (maximum focus measure)
        f_max: Focus measure value at peak
        left_steep_fit: (slope, intercept) for left steep region line
        right_steep_fit: (slope, intercept) for right steep region line
        left_gradual_fit: (slope, intercept) for left gradual region line
        right_gradual_fit: (slope, intercept) for right gradual region line
    """
    k_lcp: float
    k_rcp: float
    f_lcp: float
    f_rcp: float
    k_max: int
    f_max: float
    left_steep_fit: Tuple[float, float]
    right_steep_fit: Tuple[float, float]
    left_gradual_fit: Tuple[float, float]
    right_gradual_fit: Tuple[float, float]


@dataclass(frozen=True)
class FocusMeasureQuality:
    """Container for focus measure operator quality metrics.
    
    Based on Piao et al. (2025) quantitative evaluation framework.
    
    Attributes:
        Ws: Steep slope region width (in z-indices or physical units if step_distance provided).
            Narrow Ws = high sensitivity to focus changes.
        Rsg: Steep-to-gradual ratio. Higher = better discrimination between
            focused and unfocused images.
        Cp: Curvature at peak. Higher = greater sensitivity to focal deviations.
        RRMSE: Relative root mean square error (noise robustness).
            Only computed if noisy scores are provided. Lower = more robust.
        FWHM: Full width at half maximum (traditional metric for comparison).
        Sp: Peak slope (traditional metric for comparison).
        segmentation: The underlying curve segmentation results.
        method: The focus measure method these metrics describe.
        is_unimodal: Whether the curve satisfies unimodality assumption.
    """
    Ws: float
    Rsg: float
    Cp: float
    RRMSE: Optional[float]
    FWHM: float
    Sp: float
    segmentation: CurveSegmentation
    method: str
    is_unimodal: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'Ws': self.Ws,
            'Rsg': self.Rsg,
            'Cp': self.Cp,
            'RRMSE': self.RRMSE,
            'FWHM': self.FWHM,
            'Sp': self.Sp,
            'method': self.method,
            'is_unimodal': self.is_unimodal,
            'k_lcp': self.segmentation.k_lcp,
            'k_rcp': self.segmentation.k_rcp,
            'k_max': self.segmentation.k_max,
        }
    
    def summary(self) -> str:
        """Return a human-readable summary of the metrics."""
        lines = [
            f"Focus Measure Quality Metrics ({self.method})",
            "=" * 50,
            f"  Ws (steep width):     {self.Ws:.2f} slices",
            f"  Rsg (steep/gradual):  {self.Rsg:.4f}",
            f"  Cp (peak curvature):  {self.Cp:.4f}",
            f"  FWHM:                 {self.FWHM:.2f} slices",
            f"  Sp (peak slope):      {self.Sp:.4f}",
        ]
        if self.RRMSE is not None:
            lines.append(f"  RRMSE:                {self.RRMSE:.4f}")
        lines.append(f"  Unimodal:             {self.is_unimodal}")
        return "\n".join(lines)


# =============================================================================
# Multi-Point Linear Fitting for Curve Segmentation
# =============================================================================

def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit a line y = slope * x + intercept using least squares.
    
    Returns:
        Tuple of (slope, intercept).
    """
    if len(x) < 2:
        return (0.0, y[0] if len(y) > 0 else 0.0)
    
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = result[0]
    return (float(slope), float(intercept))


def _compute_rss(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """Compute residual sum of squares for a fitted line."""
    predicted = slope * x + intercept
    return float(np.sum((y - predicted) ** 2))


def _find_intersection(
    slope1: float, intercept1: float,
    slope2: float, intercept2: float
) -> Optional[float]:
    """Find x-coordinate where two lines intersect."""
    if abs(slope1 - slope2) < 1e-10:
        return None  # Parallel lines
    x = (intercept2 - intercept1) / (slope1 - slope2)
    return float(x)


def _interpolate_value(scores: np.ndarray, k: float) -> float:
    """Linearly interpolate the focus score at a non-integer index."""
    if k <= 0:
        return float(scores[0])
    if k >= len(scores) - 1:
        return float(scores[-1])
    
    k_low = int(np.floor(k))
    k_high = int(np.ceil(k))
    
    if k_low == k_high:
        return float(scores[k_low])
    
    frac = k - k_low
    return float(scores[k_low] * (1 - frac) + scores[k_high] * frac)


def segment_focus_curve(
    scores: np.ndarray,
    n_fitting_points: int = 5,
    use_slope_criterion: bool = True,
) -> CurveSegmentation:
    """Segment focus curve into steep and gradual slope regions.
    
    Uses multi-point linear fitting as described in Piao et al. (2025).
    The method fits lines to sliding windows of points and finds where
    the steep slope region meets the gradual slope regions.
    
    Parameters
    ----------
    scores : np.ndarray
        1D array of focus measure scores along the z-axis.
    n_fitting_points : int, default=5
        Number of points to use for each linear fit. More points
        increases stability but requires more z-slices. Piao et al.
        recommend 3-7 points depending on step distance.
    use_slope_criterion : bool, default=True
        If True, use slope as criterion for steep region fitting
        (maximum slope). If False, use RSS (useful when n_fitting_points
        is large enough to avoid collinearity issues).
    
    Returns
    -------
    CurveSegmentation
        Object containing cutoff points and fitted line parameters.
    
    Raises
    ------
    ValueError
        If scores array is too short for the requested n_fitting_points.
    """
    n = len(scores)
    if n < 2 * n_fitting_points + 1:
        raise ValueError(
            f"Need at least {2 * n_fitting_points + 1} z-slices for "
            f"n_fitting_points={n_fitting_points}. Got {n}."
        )
    
    # Find peak
    k_max = int(np.argmax(scores))
    f_max = float(scores[k_max])
    
    # Ensure peak is not at the edges
    if k_max < n_fitting_points or k_max >= n - n_fitting_points:
        # Fallback: use simple threshold-based segmentation
        return _fallback_segmentation(scores, k_max, f_max)
    
    z = np.arange(n)
    
    # === Left side segmentation ===
    # Find best fit for left steep region (highest slope going up to peak)
    best_left_steep_slope = -np.inf
    best_left_steep_fit = (0.0, 0.0)
    best_left_steep_end = k_max
    
    for end_idx in range(n_fitting_points, k_max + 1):
        start_idx = end_idx - n_fitting_points
        x_window = z[start_idx:end_idx]
        y_window = scores[start_idx:end_idx]
        slope, intercept = _fit_line(x_window, y_window)
        
        if slope > best_left_steep_slope:
            best_left_steep_slope = slope
            best_left_steep_fit = (slope, intercept)
            best_left_steep_end = end_idx
    
    # Find best fit for left gradual region (flattest region on left)
    best_left_gradual_rss = np.inf
    best_left_gradual_slope = np.inf
    best_left_gradual_fit = (0.0, 0.0)
    
    left_search_end = min(best_left_steep_end - n_fitting_points, k_max - n_fitting_points)
    for start_idx in range(0, max(1, left_search_end)):
        end_idx = start_idx + n_fitting_points
        if end_idx > left_search_end:
            break
        x_window = z[start_idx:end_idx]
        y_window = scores[start_idx:end_idx]
        slope, intercept = _fit_line(x_window, y_window)
        
        if use_slope_criterion:
            # Use minimum absolute slope for gradual region
            if abs(slope) < abs(best_left_gradual_slope):
                best_left_gradual_slope = slope
                best_left_gradual_fit = (slope, intercept)
        else:
            # Use minimum RSS
            rss = _compute_rss(x_window, y_window, slope, intercept)
            if rss < best_left_gradual_rss:
                best_left_gradual_rss = rss
                best_left_gradual_fit = (slope, intercept)
    
    # Find left cutoff point (intersection of steep and gradual lines)
    k_lcp = _find_intersection(
        best_left_steep_fit[0], best_left_steep_fit[1],
        best_left_gradual_fit[0], best_left_gradual_fit[1]
    )
    if k_lcp is None or k_lcp < 0 or k_lcp >= k_max:
        k_lcp = float(k_max) / 3  # Fallback
    
    # === Right side segmentation ===
    # Find best fit for right steep region (steepest negative slope from peak)
    best_right_steep_slope = np.inf  # Looking for most negative
    best_right_steep_fit = (0.0, 0.0)
    best_right_steep_start = k_max
    
    for start_idx in range(k_max, n - n_fitting_points):
        end_idx = start_idx + n_fitting_points
        x_window = z[start_idx:end_idx]
        y_window = scores[start_idx:end_idx]
        slope, intercept = _fit_line(x_window, y_window)
        
        if slope < best_right_steep_slope:
            best_right_steep_slope = slope
            best_right_steep_fit = (slope, intercept)
            best_right_steep_start = start_idx
    
    # Find best fit for right gradual region
    best_right_gradual_rss = np.inf
    best_right_gradual_slope = -np.inf
    best_right_gradual_fit = (0.0, 0.0)
    
    right_search_start = max(best_right_steep_start + n_fitting_points, k_max + n_fitting_points)
    for start_idx in range(right_search_start, n - n_fitting_points + 1):
        end_idx = start_idx + n_fitting_points
        if end_idx > n:
            break
        x_window = z[start_idx:end_idx]
        y_window = scores[start_idx:end_idx]
        slope, intercept = _fit_line(x_window, y_window)
        
        if use_slope_criterion:
            if abs(slope) < abs(best_right_gradual_slope):
                best_right_gradual_slope = slope
                best_right_gradual_fit = (slope, intercept)
        else:
            rss = _compute_rss(x_window, y_window, slope, intercept)
            if rss < best_right_gradual_rss:
                best_right_gradual_rss = rss
                best_right_gradual_fit = (slope, intercept)
    
    # Find right cutoff point
    k_rcp = _find_intersection(
        best_right_steep_fit[0], best_right_steep_fit[1],
        best_right_gradual_fit[0], best_right_gradual_fit[1]
    )
    if k_rcp is None or k_rcp <= k_max or k_rcp >= n:
        k_rcp = k_max + (n - k_max) * 2 / 3  # Fallback
    
    # Interpolate F values at cutoff points
    f_lcp = _interpolate_value(scores, k_lcp)
    f_rcp = _interpolate_value(scores, k_rcp)
    
    return CurveSegmentation(
        k_lcp=k_lcp,
        k_rcp=k_rcp,
        f_lcp=f_lcp,
        f_rcp=f_rcp,
        k_max=k_max,
        f_max=f_max,
        left_steep_fit=best_left_steep_fit,
        right_steep_fit=best_right_steep_fit,
        left_gradual_fit=best_left_gradual_fit,
        right_gradual_fit=best_right_gradual_fit,
    )


def _fallback_segmentation(
    scores: np.ndarray,
    k_max: int,
    f_max: float
) -> CurveSegmentation:
    """Simple threshold-based fallback when curve is too short."""
    n = len(scores)
    
    # Use 50% of max as threshold to find steep region
    threshold = f_max * 0.5
    above_threshold = scores >= threshold
    
    # Find leftmost and rightmost points above threshold
    indices_above = np.where(above_threshold)[0]
    if len(indices_above) > 0:
        k_lcp = float(indices_above[0])
        k_rcp = float(indices_above[-1])
    else:
        k_lcp = float(k_max) * 0.5
        k_rcp = float(k_max) + (n - k_max) * 0.5
    
    return CurveSegmentation(
        k_lcp=k_lcp,
        k_rcp=k_rcp,
        f_lcp=_interpolate_value(scores, k_lcp),
        f_rcp=_interpolate_value(scores, k_rcp),
        k_max=k_max,
        f_max=f_max,
        left_steep_fit=(0.0, 0.0),
        right_steep_fit=(0.0, 0.0),
        left_gradual_fit=(0.0, 0.0),
        right_gradual_fit=(0.0, 0.0),
    )


# =============================================================================
# Metric Computations
# =============================================================================

def compute_Ws(
    segmentation: CurveSegmentation,
    step_distance: Optional[float] = None
) -> float:
    """Compute steep slope region width (Ws).
    
    Ws = d * (k_rcp - k_lcp)
    
    A narrow Ws indicates high sensitivity to focus changes.
    
    Parameters
    ----------
    segmentation : CurveSegmentation
        The curve segmentation result.
    step_distance : float, optional
        Physical distance between z-slices (e.g., in µm).
        If None, returns width in z-index units.
    
    Returns
    -------
    float
        Steep slope region width.
    """
    width = segmentation.k_rcp - segmentation.k_lcp
    if step_distance is not None:
        width *= step_distance
    return float(width)


def compute_Rsg(
    scores: np.ndarray,
    segmentation: CurveSegmentation
) -> float:
    """Compute steep-to-gradual ratio (Rsg).
    
    Rsg = [2*F(k_max) - F(k_lsm) - F(k_rsm)] / 
          [F(k_lmax) - F(k_lmin) + F(k_rmax) - F(k_rmin)]
    
    Higher Rsg indicates better ability to distinguish between
    focused and unfocused images.
    
    Parameters
    ----------
    scores : np.ndarray
        Focus measure scores along z-axis.
    segmentation : CurveSegmentation
        The curve segmentation result.
    
    Returns
    -------
    float
        Steep-to-gradual ratio.
    """
    k_max = segmentation.k_max
    k_lcp = int(np.floor(segmentation.k_lcp))
    k_rcp = int(np.ceil(segmentation.k_rcp))
    
    # Steep region: from k_lcp to k_rcp
    # Find lowest points in left and right steep regions
    left_steep = scores[max(0, k_lcp):k_max + 1]
    right_steep = scores[k_max:min(len(scores), k_rcp + 1)]
    
    f_lsm = float(np.min(left_steep)) if len(left_steep) > 0 else segmentation.f_lcp
    f_rsm = float(np.min(right_steep)) if len(right_steep) > 0 else segmentation.f_rcp
    
    # Gradual regions: outside the cutoff points
    left_gradual = scores[:max(1, k_lcp)]
    right_gradual = scores[min(len(scores) - 1, k_rcp):]
    
    f_lmax = float(np.max(left_gradual)) if len(left_gradual) > 0 else f_lsm
    f_lmin = float(np.min(left_gradual)) if len(left_gradual) > 0 else f_lsm
    f_rmax = float(np.max(right_gradual)) if len(right_gradual) > 0 else f_rsm
    f_rmin = float(np.min(right_gradual)) if len(right_gradual) > 0 else f_rsm
    
    numerator = 2 * segmentation.f_max - f_lsm - f_rsm
    denominator = (f_lmax - f_lmin) + (f_rmax - f_rmin)
    
    if denominator < 1e-10:
        # Avoid division by zero; perfectly flat gradual regions
        return float('inf')
    
    return float(numerator / denominator)


def compute_Cp(
    scores: np.ndarray,
    segmentation: CurveSegmentation,
    step_distance: float = 1.0
) -> float:
    """Compute curvature at peak (Cp).
    
    Cp = |F''(k_max)| / [1 + F'(k_max)^2]^(3/2)
    
    where:
        F'(k_max) = [F(k_max+1) - F(k_max-1)] / (2d)
        F''(k_max) = [F(k_max+1) - 2*F(k_max) + F(k_max-1)] / d
    
    Higher Cp indicates greater sensitivity to focal deviations.
    
    Parameters
    ----------
    scores : np.ndarray
        Focus measure scores along z-axis.
    segmentation : CurveSegmentation
        The curve segmentation result.
    step_distance : float, default=1.0
        Physical distance between z-slices.
    
    Returns
    -------
    float
        Curvature at peak.
    """
    k_max = segmentation.k_max
    d = step_distance
    
    if k_max <= 0 or k_max >= len(scores) - 1:
        return 0.0
    
    f_prev = float(scores[k_max - 1])
    f_curr = float(scores[k_max])
    f_next = float(scores[k_max + 1])
    
    # First derivative at peak
    f_prime = (f_next - f_prev) / (2 * d)
    
    # Second derivative at peak
    f_double_prime = (f_next - 2 * f_curr + f_prev) / d
    
    # Curvature formula
    denominator = (1 + f_prime ** 2) ** 1.5
    if denominator < 1e-10:
        return 0.0
    
    Cp = abs(f_double_prime) / denominator
    return float(Cp)


def compute_RRMSE(
    scores_clean: np.ndarray,
    scores_noisy: np.ndarray
) -> float:
    """Compute relative root mean square error (RRMSE).
    
    RRMSE measures the robustness of a focus measure operator to noise.
    Lower values indicate better noise robustness.
    
    Parameters
    ----------
    scores_clean : np.ndarray
        Focus scores from original (clean) images.
    scores_noisy : np.ndarray
        Focus scores from images with added noise (e.g., AWGN).
    
    Returns
    -------
    float
        Relative root mean square error.
    """
    if len(scores_clean) != len(scores_noisy):
        raise ValueError("Clean and noisy score arrays must have same length.")
    
    # Normalize both curves to [0, 1]
    clean_norm = _normalize_curve(scores_clean)
    noisy_norm = _normalize_curve(scores_noisy)
    
    # Compute RMSE
    mse = np.mean((clean_norm - noisy_norm) ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def _normalize_curve(scores: np.ndarray) -> np.ndarray:
    """Normalize curve to [0, 1] range."""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val < 1e-10:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def compute_FWHM(
    scores: np.ndarray,
    segmentation: CurveSegmentation
) -> float:
    """Compute full width at half maximum (FWHM).
    
    Traditional metric for comparison with Ws.
    
    Parameters
    ----------
    scores : np.ndarray
        Focus measure scores along z-axis.
    segmentation : CurveSegmentation
        The curve segmentation result.
    
    Returns
    -------
    float
        Full width at half maximum in z-index units.
        Returns NaN if FWHM cannot be determined.
    """
    f_max = segmentation.f_max
    f_min = float(np.min(scores))
    
    half_max = (f_max + f_min) / 2
    above_half = scores >= half_max
    
    indices_above = np.where(above_half)[0]
    if len(indices_above) < 2:
        return float('nan')
    
    # Find more precise boundaries using interpolation
    left_idx = indices_above[0]
    right_idx = indices_above[-1]
    
    # Interpolate left boundary
    if left_idx > 0:
        f_left = scores[left_idx - 1]
        f_right = scores[left_idx]
        if f_right - f_left > 1e-10:
            frac = (half_max - f_left) / (f_right - f_left)
            left_boundary = left_idx - 1 + frac
        else:
            left_boundary = float(left_idx)
    else:
        left_boundary = float(left_idx)
    
    # Interpolate right boundary
    if right_idx < len(scores) - 1:
        f_left = scores[right_idx]
        f_right = scores[right_idx + 1]
        if f_left - f_right > 1e-10:
            frac = (f_left - half_max) / (f_left - f_right)
            right_boundary = right_idx + frac
        else:
            right_boundary = float(right_idx)
    else:
        right_boundary = float(right_idx)
    
    return float(right_boundary - left_boundary)


def compute_Sp(
    scores: np.ndarray,
    segmentation: CurveSegmentation,
    step_distance: float = 1.0
) -> float:
    """Compute peak slope (Sp).
    
    Traditional metric for comparison with Cp.
    
    Sp = [2*F(k_max) - F(k_max+1) - F(k_max-1)] / (2*d)
    
    Parameters
    ----------
    scores : np.ndarray
        Focus measure scores along z-axis.
    segmentation : CurveSegmentation
        The curve segmentation result.
    step_distance : float, default=1.0
        Physical distance between z-slices.
    
    Returns
    -------
    float
        Peak slope.
    """
    k_max = segmentation.k_max
    d = step_distance
    
    if k_max <= 0 or k_max >= len(scores) - 1:
        return 0.0
    
    f_prev = float(scores[k_max - 1])
    f_curr = float(scores[k_max])
    f_next = float(scores[k_max + 1])
    
    Sp = (2 * f_curr - f_prev - f_next) / (2 * d)
    return float(Sp)


def check_unimodality(
    scores: np.ndarray,
    tolerance: float = 0.1
) -> bool:
    """Check if the focus curve is approximately unimodal.
    
    The Piao et al. metrics assume unimodality. This function
    checks for significant secondary peaks.
    
    Parameters
    ----------
    scores : np.ndarray
        Focus measure scores along z-axis.
    tolerance : float, default=0.1
        Secondary peaks must be at least this fraction of the
        primary peak height to be considered significant.
    
    Returns
    -------
    bool
        True if curve is approximately unimodal.
    """
    # Find all local maxima
    local_max_mask = np.zeros(len(scores), dtype=bool)
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
            local_max_mask[i] = True
    
    local_maxima = scores[local_max_mask]
    if len(local_maxima) <= 1:
        return True
    
    # Check if secondary peaks are significant
    primary_peak = np.max(local_maxima)
    secondary_peaks = np.sort(local_maxima)[-2]  # Second highest
    
    relative_height = secondary_peaks / primary_peak if primary_peak > 0 else 0
    
    # If secondary peak is more than (1-tolerance) of primary, not unimodal
    return relative_height < (1 - tolerance)


# =============================================================================
# Main API Function
# =============================================================================

def compute_focus_measure_quality(
    scores: np.ndarray,
    method: str = "unknown",
    step_distance: float = 1.0,
    n_fitting_points: int = 5,
    scores_noisy: Optional[np.ndarray] = None,
) -> FocusMeasureQuality:
    """Compute all focus measure quality metrics for a z-stack.
    
    This is the main entry point for evaluating the quality of a
    focus measure operator's performance on a given z-stack.
    
    Parameters
    ----------
    scores : np.ndarray
        1D array of focus measure scores along the z-axis.
        Should be the aggregated (e.g., median across channels)
        focus scores from compute_focus_scores().
    method : str, default="unknown"
        Name of the focus measure method (e.g., "laplacian", "tenengrad").
    step_distance : float, default=1.0
        Physical distance between z-slices in µm.
        Set to 1.0 if unknown (metrics will be in z-index units).
    n_fitting_points : int, default=5
        Number of points for multi-point linear fitting.
        Use 3 for small stacks, 5-7 for larger stacks.
    scores_noisy : np.ndarray, optional
        Focus scores computed on noise-added images.
        If provided, RRMSE will be computed.
    
    Returns
    -------
    FocusMeasureQuality
        Container with all computed metrics.
    
    Examples
    --------
    >>> from colokroll.imaging_preprocessing import compute_focus_scores
    >>> scores_zc, scores_agg = compute_focus_scores(z_stack, method="combined")
    >>> quality = compute_focus_measure_quality(
    ...     scores_agg, 
    ...     method="combined",
    ...     step_distance=0.5  # µm
    ... )
    >>> print(quality.summary())
    """
    # Validate input
    scores = np.asarray(scores).flatten()
    if len(scores) < 5:
        raise ValueError("Need at least 5 z-slices for quality metrics.")
    
    # Adjust n_fitting_points if necessary
    max_points = (len(scores) - 1) // 2
    n_fitting_points = min(n_fitting_points, max_points)
    n_fitting_points = max(n_fitting_points, 3)
    
    # Check unimodality
    is_unimodal = check_unimodality(scores)
    
    # Segment the curve
    segmentation = segment_focus_curve(
        scores, 
        n_fitting_points=n_fitting_points,
        use_slope_criterion=(n_fitting_points <= 4)
    )
    
    # Compute metrics
    Ws = compute_Ws(segmentation, step_distance)
    Rsg = compute_Rsg(scores, segmentation)
    Cp = compute_Cp(scores, segmentation, step_distance)
    FWHM = compute_FWHM(scores, segmentation)
    Sp = compute_Sp(scores, segmentation, step_distance)
    
    # Compute RRMSE if noisy scores provided
    RRMSE = None
    if scores_noisy is not None:
        RRMSE = compute_RRMSE(scores, scores_noisy)
    
    return FocusMeasureQuality(
        Ws=Ws,
        Rsg=Rsg,
        Cp=Cp,
        RRMSE=RRMSE,
        FWHM=FWHM,
        Sp=Sp,
        segmentation=segmentation,
        method=method,
        is_unimodal=is_unimodal,
    )


def compare_focus_methods(
    z_stack: np.ndarray,
    methods: List[str] = ["laplacian", "tenengrad", "combined"],
    step_distance: float = 1.0,
    **focus_kwargs
) -> Dict[str, FocusMeasureQuality]:
    """Compare multiple focus measure methods on the same z-stack.
    
    This is a convenience function for benchmarking different FMOs.
    
    Parameters
    ----------
    z_stack : np.ndarray
        Input z-stack in (Z, Y, X, C) or (Z, Y, X) format.
    methods : list of str
        Focus measure methods to compare.
    step_distance : float, default=1.0
        Physical distance between z-slices.
    **focus_kwargs
        Additional arguments passed to compute_focus_scores.
    
    Returns
    -------
    dict
        Mapping from method name to FocusMeasureQuality.
    
    Examples
    --------
    >>> results = compare_focus_methods(z_stack, step_distance=0.5)
    >>> for method, quality in results.items():
    ...     print(f"{method}: Ws={quality.Ws:.2f}, Cp={quality.Cp:.4f}")
    """
    # Import here to avoid circular imports
    try:
        from colokroll.imaging_preprocessing import compute_focus_scores
    except ImportError:
        raise ImportError(
            "This function requires colokroll.imaging_preprocessing. "
            "Use compute_focus_measure_quality() directly if you have "
            "pre-computed scores."
        )
    
    results = {}
    for method in methods:
        scores_zc, scores_agg = compute_focus_scores(
            z_stack, 
            method=method,
            **focus_kwargs
        )
        results[method] = compute_focus_measure_quality(
            scores_agg,
            method=method,
            step_distance=step_distance,
        )
    
    return results


# =============================================================================
# Integration Helper for ZSliceSelectionResult
# =============================================================================

def extend_z_slice_result_with_quality(
    result,  # ZSliceSelectionResult
    step_distance: float = 1.0,
    scores_noisy: Optional[np.ndarray] = None,
) -> FocusMeasureQuality:
    """Compute quality metrics from an existing ZSliceSelectionResult.
    
    This is the recommended way to integrate with the existing
    colokroll z-slice detection workflow.
    
    Parameters
    ----------
    result : ZSliceSelectionResult
        Result from select_z_slices() or detect_slices_to_keep().
    step_distance : float, default=1.0
        Physical distance between z-slices.
    scores_noisy : np.ndarray, optional
        Noisy scores for RRMSE computation.
    
    Returns
    -------
    FocusMeasureQuality
        Quality metrics for the focus measure method used.
    
    Examples
    --------
    >>> result = select_z_slices(z_stack, method="combined")
    >>> quality = extend_z_slice_result_with_quality(result, step_distance=0.5)
    >>> print(f"Ws={quality.Ws:.2f}, Rsg={quality.Rsg:.4f}")
    """
    return compute_focus_measure_quality(
        scores=result.scores_agg,
        method=result.method,
        step_distance=step_distance,
        scores_noisy=scores_noisy,
    )


# =============================================================================
# Visualization Helper
# =============================================================================

def plot_focus_curve_analysis(
    scores: np.ndarray,
    quality: FocusMeasureQuality,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """Visualize the focus curve with segmentation and metrics.
    
    Parameters
    ----------
    scores : np.ndarray
        Focus measure scores.
    quality : FocusMeasureQuality
        Computed quality metrics.
    output_path : str, optional
        Path to save the figure. If None, displays interactively.
    figsize : tuple, default=(12, 5)
        Figure size in inches.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    seg = quality.segmentation
    z = np.arange(len(scores))
    
    # Left plot: Focus curve with segmentation
    ax1.plot(z, scores, 'b-', linewidth=2, label='Focus curve')
    ax1.axvline(seg.k_lcp, color='g', linestyle='--', label=f'Left cutoff ({seg.k_lcp:.1f})')
    ax1.axvline(seg.k_rcp, color='r', linestyle='--', label=f'Right cutoff ({seg.k_rcp:.1f})')
    ax1.axvline(seg.k_max, color='orange', linestyle=':', label=f'Peak ({seg.k_max})')
    
    # Shade regions
    ax1.axvspan(0, seg.k_lcp, alpha=0.1, color='blue', label='Left gradual')
    ax1.axvspan(seg.k_lcp, seg.k_rcp, alpha=0.1, color='green', label='Steep')
    ax1.axvspan(seg.k_rcp, len(scores), alpha=0.1, color='red', label='Right gradual')
    
    ax1.set_xlabel('Z-index')
    ax1.set_ylabel('Focus score')
    ax1.set_title(f'Focus Curve Analysis ({quality.method})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Metrics comparison
    metrics = ['Ws', 'Rsg', 'Cp', 'FWHM', 'Sp']
    values = [quality.Ws, quality.Rsg, quality.Cp, quality.FWHM, quality.Sp]
    
    # Normalize for visualization
    max_vals = [max(abs(v), 1e-10) for v in values]
    norm_values = [v / m for v, m in zip(values, max_vals)]
    
    bars = ax2.barh(metrics, norm_values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'])
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    ax2.set_xlabel('Normalized value')
    ax2.set_title('Quality Metrics')
    ax2.set_xlim(0, 1.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Generate synthetic focus curve for demonstration
    np.random.seed(42)
    n_slices = 50
    z = np.arange(n_slices)
    
    # Simulate a focus curve: Gaussian peak with some noise
    peak_z = 25
    sigma = 8
    clean_scores = np.exp(-0.5 * ((z - peak_z) / sigma) ** 2)
    scores = clean_scores + np.random.normal(0, 0.02, n_slices)
    scores = np.maximum(scores, 0)  # Ensure non-negative
    
    # Add noise for RRMSE computation
    noisy_scores = scores + np.random.normal(0, 0.05, n_slices)
    noisy_scores = np.maximum(noisy_scores, 0)
    
    print("=" * 60)
    print("Focus Measure Quality Metrics Demo")
    print("=" * 60)
    
    # Compute quality metrics
    quality = compute_focus_measure_quality(
        scores,
        method="simulated_gaussian",
        step_distance=1.0,
        n_fitting_points=5,
        scores_noisy=noisy_scores,
    )
    
    print(quality.summary())
    print()
    
    # Show segmentation details
    seg = quality.segmentation
    print("Curve Segmentation Details:")
    print(f"  Peak location: z={seg.k_max}")
    print(f"  Left cutoff:   z={seg.k_lcp:.2f}")
    print(f"  Right cutoff:  z={seg.k_rcp:.2f}")
    print(f"  Steep width:   {seg.k_rcp - seg.k_lcp:.2f} slices")
    
    # Generate visualization
    print()
    print("Generating visualization...")
    plot_focus_curve_analysis(scores, quality, output_path="focus_curve_analysis.png")