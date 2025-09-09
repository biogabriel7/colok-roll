"""
Auto background subtraction configuration.

Defines per-channel parameter grids and scoring weights for the automatic
background subtraction mode. If a channel is not present, the DEFAULT entry
is used so the system still runs a reasonable grid search.
"""

from typing import Dict, Any


AUTO_BG_CONFIG: Dict[str, Dict[str, Any]] = {
    "DEFAULT": {
        "grid": {
            # Two-stage option for general images: Gaussian (haze) → Rolling ball (local)
            "two_stage": {
                "sigma_stage1": [12, 14, 16],
                "radius_stage2": [24, 30, 36],
                "light_background": [False],
            },
            "gaussian": {"sigma": [8, 10, 12, 14]},
            "rolling_ball": {"radius": [20, 30, 40], "light_background": [False]},
        },
        "weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        "explanation": "General-purpose defaults when channel is unknown. Please update auto_bg_config.py to improve performance with your specific channel.",
    },

    # Punctate vesicles (e.g., lysosomes): emphasize background suppression
    "LAMP1": {
        # Default two-stage parameters preferred for LAMP1
        "default": {"method": "two_stage", "sigma_stage1": 14, "radius_stage2": 30, "light_background": False},
        # When True, skip grid search and use default directly
        "force_default": True,
        "grid": {
            "two_stage": {
                "sigma_stage1": [12, 14, 16],
                "radius_stage2": [24, 30, 36],
                "light_background": [False],
            },
            "rolling_ball": {"radius": [24, 27, 30, 33, 36], "light_background": [False]},
        },
        "weights": {"w_bg": 0.7, "w_contrast": 0.2, "w_grad": 0.1, "w_zero": 0.3},
        "explanation": "Punctate vesicles: prioritize diffuse haze removal; edges are high-frequency so gradient weight reduced.",
    },

    # Filament channel: moderate RB radii preserve filaments
    "Phalloidin": {
        "grid": {"rolling_ball": {"radius": [50, 55, 60, 65, 70], "light_background": [False]}},
        "weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        "explanation": "Filament preservation with moderate rolling ball radii.",
    },

    # Diffuse cytosolic marker: Gaussian to remove gradual background
    "ALIX": {
        "grid": {"gaussian": {"sigma": [10, 12, 14, 16]}},
        "weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        "explanation": "Diffuse background removal with 3D Gaussian.",
    },

    # Nuclear stain: large rolling ball to avoid halos
    "DAPI": {
        "grid": {"rolling_ball": {"radius": [100, 110, 120, 130, 140], "light_background": [False]}},
        "weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        "explanation": "Large rolling ball reduces nuclear background and halo artifacts.",
    },
        #"You_add_more_channels_here": {
        #"grid": {"rolling_ball": {"radius": [100, 110, 120, 130, 140], "light_background": [False]}},
        #"weights": {"w_bg": 0.5, "w_contrast": 0.3, "w_grad": 0.2, "w_zero": 0.3},
        #"explanation": "Explanation for the channel.",
}


