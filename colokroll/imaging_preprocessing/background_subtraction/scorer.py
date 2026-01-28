"""
Background subtraction quality scoring.

This module provides scoring functions to evaluate the quality of
background subtraction results, supporting both standard and negative control cases.

The scorer evaluates:
- Background removal effectiveness (lower background mean after correction)
- Contrast improvement (better separation between foreground and background)
- Detail preservation (measured via edge detection and SSIM)
- Over-correction detection (avoiding excessive zeros or flattening)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from skimage import filters, morphology
from skimage.metrics import structural_similarity as skimage_ssim

from ..constants import (
    MIN_STD_RATIO,
    MIN_STD_ABS,
    FG_MIN_PIXELS,
    FG_DILATE_RADIUS,
    SSIM_WEIGHT,
    SCORE_WEIGHT_MEAN,
    SCORE_WEIGHT_STD,
    SCORE_WEIGHT_ZERO_FRACTION,
    NEAR_ZERO_THRESHOLD,
    TARGET_STD_RATIO,
    CORRELATION_STD_THRESHOLD,
    MAX_SLICES_FOR_SCORING,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BackgroundScorer:
    """
    Evaluates background subtraction quality.
    
    Provides methods for computing quality scores that measure:
    - Background removal effectiveness
    - Signal/feature preservation
    - Contrast improvement
    - Zero-pixel ratio (over-correction detection)
    
    Supports both standard images and negative control channels.
    """
    
    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (0.7, 0.4, 0.2, 0.1),
        ssim_weight: float = SSIM_WEIGHT,
    ):
        """
        Initialize the scorer.
        
        Args:
            weights: Tuple of (bg_removal, contrast, detail_preservation, zero_penalty)
            ssim_weight: Weight for SSIM component in scoring
        """
        self.weights = weights
        self.ssim_weight = ssim_weight
        self.logger = logging.getLogger(__name__)
    
    def slice_masks_otsu(self, slice_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute foreground and background masks using Otsu thresholding.
        
        The mask is post-processed with:
        - Small object removal (min_size=FG_MIN_PIXELS)
        - Dilation to capture signal boundaries
        - Closing to fill small holes
        
        Args:
            slice_np: 2D image slice
            
        Returns:
            Tuple of (foreground_mask, background_mask)
        """
        thr = filters.threshold_otsu(slice_np)
        fg = slice_np >= thr
        fg = morphology.remove_small_objects(fg, min_size=FG_MIN_PIXELS)
        if FG_DILATE_RADIUS > 0:
            fg = morphology.dilation(fg, morphology.disk(FG_DILATE_RADIUS))
        fg = morphology.closing(fg, morphology.disk(1))
        bg = ~fg
        return fg, bg
    
    def slice_metrics(
        self,
        orig_slice: np.ndarray,
        corr_slice: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Compute detailed metrics comparing original and corrected slices.
        
        Uses median for background (more robust to outliers) and Sobel gradient
        for edge detection quality.
        
        Returns:
            Tuple of (bg_median_orig, bg_median_corr, contrast_orig, contrast_corr,
                     grad_orig, grad_corr, zero_frac, ssim_fg, orig_std, corr_std)
        """
        fg, bg = self.slice_masks_otsu(orig_slice)
        
        # Background metrics (use median for robustness)
        bg0 = float(np.median(orig_slice[bg])) if np.any(bg) else float(np.median(orig_slice))
        bg1 = float(np.median(corr_slice[bg])) if np.any(bg) else float(np.median(corr_slice))
        
        # Contrast: foreground mean minus background median
        c0 = float(np.mean(orig_slice[fg]) - np.median(orig_slice[bg])) if np.any(fg) and np.any(bg) else 0.0
        c1 = float(np.mean(corr_slice[fg]) - np.median(corr_slice[bg])) if np.any(fg) and np.any(bg) else 0.0
        
        # Edge gradient (Sobel) in foreground region
        g0 = float(np.mean(filters.sobel(orig_slice)[fg])) if np.any(fg) else 0.0
        g1 = float(np.mean(filters.sobel(corr_slice)[fg])) if np.any(fg) else 0.0
        
        # Zero fraction (over-correction indicator)
        zf = float(np.mean(corr_slice == 0))
        
        # Masked SSIM in foreground region only
        if np.any(fg):
            orig_fg = orig_slice * fg
            corr_fg = corr_slice * fg
            dr = float(orig_fg.max() - orig_fg.min() or 1.0)
            try:
                ssim_fg = float(skimage_ssim(orig_fg, corr_fg, data_range=dr))
            except Exception:
                ssim_fg = 0.0
        else:
            ssim_fg = 0.0
        
        # Global standard deviations
        orig_std = float(np.std(orig_slice))
        corr_std = float(np.std(corr_slice))
        
        return bg0, bg1, c0, c1, g0, g1, zf, ssim_fg, orig_std, corr_std
    
    def score_volume(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        weights: Optional[Tuple[float, float, float, float]] = None,
        max_slices: int = MAX_SLICES_FOR_SCORING,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score the quality of background subtraction on a 3D volume.
        
        Higher scores indicate better background subtraction quality.
        The scoring formula is:
            score = w_bg * bg_improve + w_contrast * contrast_gain 
                  + w_grad * min(1, grad_ratio) - w_zero * zero_frac 
                  + ssim_weight * ssim_fg
        
        Args:
            original: Original 3D image
            corrected: Background-corrected 3D image
            weights: Tuple of (w_bg, w_contrast, w_grad, w_zero)
            max_slices: Maximum slices to sample for efficiency
            
        Returns:
            Tuple of (composite_score, component_scores_dict)
        """
        weights = weights or self.weights
        w_bg, w_contrast, w_grad, w_zero = weights
        
        z = original.shape[0]
        idx = np.linspace(0, z - 1, num=min(z, max_slices), dtype=int)
        
        scores: List[float] = []
        metrics_accum = {
            "bg_improve": 0.0,
            "contrast_gain": 0.0,
            "grad_ratio": 0.0,
            "zero_frac": 0.0,
            "ssim_fg": 0.0,
            "orig_std": 0.0,
            "corr_std": 0.0,
        }
        
        for zi in idx:
            b0, b1, c0, c1, g0, g1, zf, ssim_fg, orig_std, corr_std = self.slice_metrics(
                original[zi], corrected[zi]
            )
            
            # Background improvement (lower is better after correction)
            bg_improve = (b0 - b1) / (b0 + 1e-6)
            
            # Contrast gain (higher contrast after correction is better)
            contrast_gain = (c1 - c0) / (abs(c0) + 1e-6)
            
            # Gradient ratio (preserve edge sharpness)
            grad_ratio = g1 / (g0 + 1e-6)
            
            # Composite score for this slice
            score = (
                w_bg * bg_improve
                + w_contrast * contrast_gain
                + w_grad * min(1.0, grad_ratio)
                - w_zero * zf
                + self.ssim_weight * ssim_fg
            )
            scores.append(score)
            
            # Accumulate metrics
            metrics_accum["bg_improve"] += bg_improve
            metrics_accum["contrast_gain"] += contrast_gain
            metrics_accum["grad_ratio"] += grad_ratio
            metrics_accum["zero_frac"] += zf
            metrics_accum["ssim_fg"] += ssim_fg
            metrics_accum["orig_std"] += orig_std
            metrics_accum["corr_std"] += corr_std
        
        # Average the metrics
        n = len(idx) or 1
        for k in metrics_accum:
            metrics_accum[k] /= n
        
        return float(np.mean(scores)), metrics_accum
    
    def score_volume_negative_control(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        max_slices: int = MAX_SLICES_FOR_SCORING,
        debug: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score background subtraction for negative control channels.
        
        For negative controls, we want to minimize residual signal while avoiding
        complete signal flattening (which makes data unusable for analysis).
        
        Scoring formula:
            score = w_mean * (1 - normalized_mean) + w_std * std_score + w_zero * zero_fraction
        
        Where:
            - normalized_mean = corrected_mean / original_mean (lower is better)
            - std_score: penalizes both insufficient AND excessive std reduction
            - zero_fraction = fraction of pixels near-zero (higher is better, but capped)
        
        Args:
            original: Original 3D image
            corrected: Background-corrected 3D image
            max_slices: Maximum slices to sample
            debug: If True, log detailed debug info
            
        Returns:
            Tuple of (composite_score, component_scores_dict)
        """
        # Weights balanced for negative control scoring
        w_mean = 0.5   # Weight for mean reduction
        w_std = 0.3    # Weight for std preservation (not too high, not too low)
        w_zero = 0.2   # Weight for zero fraction
        
        z = original.shape[0]
        idx = np.linspace(0, z - 1, num=min(z, max_slices), dtype=int)
        scores: List[float] = []
        
        metrics_accum = {
            "orig_mean": 0.0,
            "corr_mean": 0.0,
            "orig_std": 0.0,
            "corr_std": 0.0,
            "zero_frac": 0.0,
            "mean_reduction": 0.0,
            "std_score": 0.0,
            "normalized_std": 0.0,
        }
        
        for zi in idx:
            orig_slice = original[zi].astype(np.float32)
            corr_slice = corrected[zi].astype(np.float32)
            
            # Compute metrics
            orig_mean = float(np.mean(orig_slice))
            corr_mean = float(np.mean(corr_slice))
            orig_std = float(np.std(orig_slice))
            corr_std = float(np.std(corr_slice))
            
            # Zero fraction (pixels at or near zero)
            near_zero_threshold = 1.0  # Consider values <= 1 as "zero"
            zero_frac = float(np.mean(corr_slice <= near_zero_threshold))
            
            # Normalized metrics (how much we reduced relative to original)
            normalized_mean = corr_mean / (orig_mean + 1e-6)
            normalized_std = corr_std / (orig_std + 1e-6)
            
            # Mean reduction: reward lower means
            mean_reduction = 1.0 - min(1.0, normalized_mean)
            
            # Std score: We want SOME reduction but not complete flattening
            # Target: reduce std to 20-40% of original (not to near-zero)
            # Optimal normalized_std around 0.3 (30% of original)
            target_std_ratio = 0.3
            std_deviation = abs(normalized_std - target_std_ratio)
            std_score = max(0.0, 1.0 - (std_deviation / target_std_ratio))
            
            # Penalize complete signal removal (when corrected is essentially flat)
            if corr_std < 0.5:  # Absolute minimum variance threshold
                std_score = 0.0  # Heavily penalize flat images
            
            score = (
                w_mean * mean_reduction
                + w_std * std_score
                + w_zero * min(0.9, zero_frac)  # Cap zero_frac contribution at 0.9
            )
            scores.append(score)
            
            # Accumulate metrics
            metrics_accum["orig_mean"] += orig_mean
            metrics_accum["corr_mean"] += corr_mean
            metrics_accum["orig_std"] += orig_std
            metrics_accum["corr_std"] += corr_std
            metrics_accum["zero_frac"] += zero_frac
            metrics_accum["mean_reduction"] += mean_reduction
            metrics_accum["std_score"] += std_score
            metrics_accum["normalized_std"] += normalized_std
        
        n = len(idx) or 1
        for k in metrics_accum:
            metrics_accum[k] /= n
        
        composite = float(np.mean(scores))
        
        if debug:
            logger.debug(f"Negative control score: {composite:.4f} (metrics: {metrics_accum})")
        
        return composite, metrics_accum
    
    def compute_negative_control_metrics(
        self, corrected: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute validation metrics for negative control channels.
        
        These metrics help assess if the background subtraction achieved
        the expected result for a negative control (minimal residual signal).
        
        Args:
            corrected: Background-corrected 3D image
            
        Returns:
            Dictionary with:
            - residual_mean: Mean intensity of corrected image
            - residual_std: Standard deviation of corrected image
            - residual_percentile_95: 95th percentile intensity
            - residual_percentile_99: 99th percentile intensity
            - zero_fraction: Fraction of pixels at or near zero
        """
        corrected_flat = corrected.flatten().astype(np.float32)
        
        near_zero_threshold = 1.0
        zero_fraction = float(np.mean(corrected_flat <= near_zero_threshold))
        
        return {
            'residual_mean': float(np.mean(corrected_flat)),
            'residual_std': float(np.std(corrected_flat)),
            'residual_percentile_95': float(np.percentile(corrected_flat, 95)),
            'residual_percentile_99': float(np.percentile(corrected_flat, 99)),
            'zero_fraction': zero_fraction,
        }
    
    def _select_sample_indices(self, z: int, max_slices: int = 9) -> np.ndarray:
        """Select evenly spaced slice indices for sampling."""
        if z <= max_slices:
            return np.arange(z, dtype=int)
        return np.linspace(0, z - 1, num=max_slices, dtype=int)
