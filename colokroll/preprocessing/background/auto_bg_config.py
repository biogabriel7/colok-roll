"""
Auto background subtraction configuration.

Simple per-channel configuration for automatic background subtraction.
Add your channels here with recommended methods and parameters.

Available methods:
- 'rolling_ball': Best for filaments and nuclear stains (radius 20-150)
- 'gaussian': Best for diffuse cytosolic markers (sigma 8-20)
- 'two_stage': Gaussian then rolling ball for complex backgrounds

If a channel is not configured, sensible defaults will be used.
"""

from typing import Dict, Any


# Helper function to create common configs quickly
def _rolling_ball(radius_range, explanation=""):
    """Quick config for rolling ball method."""
    return {
        "grid": {"rolling_ball": {"radius": radius_range, "light_background": [False]}},
        "weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        "explanation": explanation,
    }

def _gaussian(sigma_range, explanation=""):
    """Quick config for Gaussian method."""
    return {
        "grid": {"gaussian": {"sigma": sigma_range}},
        "weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        "explanation": explanation,
    }

def _two_stage(sigma_range, radius_range, explanation="", force_default=False):
    """Quick config for two-stage method."""
    config = {
        "grid": {
            "two_stage": {
                "sigma_stage1": sigma_range,
                "radius_stage2": radius_range,
                "light_background": [False],
            }
        },
        "weights": {"w_bg": 0.7, "w_contrast": 0.2, "w_grad": 0.1, "w_zero": 0.3},
        "explanation": explanation,
    }
    if force_default:
        config["force_default"] = True
        config["default"] = {
            "method": "two_stage",
            "sigma_stage1": sigma_range[len(sigma_range)//2],
            "radius_stage2": radius_range[len(radius_range)//2],
            "light_background": False,
        }
    return config


# =============================================================================
# CHANNEL CONFIGURATIONS - Add your channels here!
# =============================================================================

AUTO_BG_CONFIG: Dict[str, Dict[str, Any]] = {
    # Default for unknown channels
    "DEFAULT": {
        "grid": {
            "two_stage": {"sigma_stage1": [12, 14, 16], "radius_stage2": [24, 30, 36], "light_background": [False]},
            "gaussian": {"sigma": [8, 10, 12, 14]},
            "rolling_ball": {"radius": [20, 30, 40], "light_background": [False]},
        },
        "weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        "explanation": "General-purpose defaults. Add your channel below for optimized results!",
    },
    
    # Nuclear stains - large radius to avoid halos
    "DAPI": _rolling_ball([100, 110, 120], "Nuclear stain: large radius prevents halos"),
    "Hoechst": _rolling_ball([100, 110, 120], "Nuclear stain: large radius prevents halos"),
    
    # Filament markers - moderate radius preserves structure
    "Phalloidin": _rolling_ball([50, 55, 60, 65, 70], "Filament marker: moderate radius preserves structure"),
    "Actin": _rolling_ball([50, 55, 60, 65], "Filament marker: moderate radius preserves structure"),
    "Tubulin": _rolling_ball([50, 55, 60, 65], "Filament marker: moderate radius preserves structure"),
    
    # Punctate vesicles - two-stage for complex backgrounds
    "LAMP1": _two_stage([12, 14, 16], [24, 30, 36], "Punctate vesicles: two-stage removes haze", force_default=True),
    "LAMP2": _two_stage([12, 14, 16], [24, 30, 36], "Punctate vesicles: two-stage removes haze"),
    "RAB5": _two_stage([12, 14, 16], [24, 30, 36], "Punctate vesicles: two-stage removes haze"),
    "RAB7": _two_stage([12, 14, 16], [24, 30, 36], "Punctate vesicles: two-stage removes haze"),
    
    # Diffuse cytosolic markers - Gaussian for gradual backgrounds
    "ALIX": _gaussian([10, 12, 14, 16], "Diffuse marker: Gaussian removes gradual background"),
    "GFP": _gaussian([10, 12, 14, 16], "Diffuse marker: Gaussian removes gradual background"),
    "mCherry": _gaussian([10, 12, 14, 16], "Diffuse marker: Gaussian removes gradual background"),
    
    # Add your custom channels here! Examples:
    # "YourProtein": _rolling_ball([40, 50, 60], "Your explanation here"),
    # "YourMarker": _gaussian([12, 15, 18], "Your explanation here"),
    # "ComplexChannel": _two_stage([10, 12, 14], [30, 35, 40], "Your explanation here"),
}


