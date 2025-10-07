"""Preprocessing configuration templates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .base import PreprocessingConfig, RuntimeConfig


def create_preprocessing_templates() -> Dict[str, Dict]:
    """Return built-in preprocessing configuration templates."""

    standard_4channel = {
        "background_subtraction": {
            "method": "rolling_ball",
            "rolling_ball_radius_dapi": 50,
            "rolling_ball_radius_phalloidin": 20,
            "rolling_ball_radius_protein": 30,
            "clip_negative_values": True,
        },
        "denoising": {
            "dapi_method": "nl_means",
            "dapi_nl_means_h": 0.1,
            "phalloidin_method": "tv_chambolle",
            "phalloidin_tv_weight": 0.1,
            "protein_method": "bilateral",
            "protein_bilateral_sigma_color": 0.1,
            "protein_bilateral_sigma_spatial": 2.0,
        },
        "deconvolution": {
            "method": "richardson_lucy_tv",
            "max_iterations": 20,
            "tv_regularization_weight": 0.002,
            "auto_stopping": True,
        },
        "quality_control": {
            "snr_threshold": 10.0,
            "focus_threshold": 0.7,
            "uniformity_cv_threshold": 5.0,
            "enforce_quality_gates": True,
        },
        "channel_processing": {
            "dapi_channel_names": ["DAPI", "Hoechst", "DAPI - DAPI"],
            "phalloidin_channel_names": ["Phalloidin", "Actin", "AF488"],
            "lamp1_channel_names": ["LAMP1", "Lysosome"],
            "protein_channel_names": ["GAL3", "ALIX", "Protein"],
        },
    }

    colocalization_optimized = {
        "background_subtraction": {
            "method": "rolling_ball",
            "rolling_ball_radius_dapi": 50,
            "rolling_ball_radius_phalloidin": 20,
            "rolling_ball_radius_protein": 25,
            "clip_negative_values": True,
        },
        "denoising": {
            "dapi_method": "nl_means",
            "dapi_nl_means_h": 0.08,
            "phalloidin_method": "bilateral",
            "phalloidin_tv_weight": 0.05,
            "protein_method": "bilateral",
            "protein_bilateral_sigma_color": 0.08,
            "protein_bilateral_sigma_spatial": 1.5,
        },
        "deconvolution": {
            "method": "richardson_lucy_tv",
            "max_iterations": 15,
            "tv_regularization_weight": 0.001,
            "auto_stopping": True,
        },
        "quality_control": {
            "snr_threshold": 12.0,
            "focus_threshold": 0.8,
            "uniformity_cv_threshold": 3.0,
            "enforce_quality_gates": True,
            "fail_on_quality": True,
        },
    }

    high_throughput = {
        "background_subtraction": {
            "method": "gaussian",
            "gaussian_sigma_dapi": 25.0,
            "gaussian_sigma_phalloidin": 10.0,
            "gaussian_sigma_protein": 15.0,
            "clip_negative_values": True,
        },
        "denoising": {
            "dapi_method": "gaussian",
            "phalloidin_method": "bilateral",
            "protein_method": "bilateral",
            "protein_bilateral_sigma_color": 0.15,
            "protein_bilateral_sigma_spatial": 3.0,
        },
        "deconvolution": {
            "method": "richardson_lucy",
            "max_iterations": 10,
            "auto_stopping": False,
        },
        "quality_control": {
            "snr_threshold": 8.0,
            "focus_threshold": 0.6,
            "uniformity_cv_threshold": 8.0,
            "enforce_quality_gates": False,
        },
        "max_memory_usage_gb": 12.0,
        "chunk_processing": True,
        "parallel_channels": True,
        "max_parallel_channels": 4,
    }

    quality_assessment = {
        "background_subtraction": {
            "method": "rolling_ball",
            "rolling_ball_radius_dapi": 50,
            "rolling_ball_radius_phalloidin": 20,
            "rolling_ball_radius_protein": 30,
        },
        "denoising": {
            "dapi_method": "nl_means",
            "dapi_nl_means_h": 0.05,
            "phalloidin_method": "anisotropic",
            "protein_method": "nl_means",
        },
        "deconvolution": {
            "method": "richardson_lucy_tv",
            "max_iterations": 30,
            "tv_regularization_weight": 0.003,
            "calculate_metrics": True,
            "save_iterations": True,
        },
        "quality_control": {
            "snr_threshold": 15.0,
            "focus_threshold": 0.9,
            "uniformity_cv_threshold": 2.0,
            "check_photobleaching": True,
            "enforce_quality_gates": True,
            "fail_on_quality": True,
        },
        "run_quality_checks": True,
        "save_intermediate_steps": True,
    }

    return {
        "standard_4channel": standard_4channel,
        "colocalization_optimized": colocalization_optimized,
        "high_throughput": high_throughput,
        "quality_assessment": quality_assessment,
    }


def save_preprocessing_template(template_name: str, output_dir: Path = Path(".")) -> Path:
    templates = create_preprocessing_templates()
    if template_name not in templates:
        raise ValueError(f"Unknown template '{template_name}'. Available: {list(templates)}")

    template_config = PreprocessingConfig(**templates[template_name])
    runtime = RuntimeConfig(preprocessing=template_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{template_name}_preprocessing_config.yaml"
    runtime.save(path, format="yaml")
    return path


def create_all_preprocessing_templates(output_dir: Path = Path("config_templates")) -> None:
    templates = create_preprocessing_templates()
    for name in templates:
        save_preprocessing_template(name, output_dir)


def load_preprocessing_template(template_name: str, template_dir: Path = Path("config_templates")) -> PreprocessingConfig:
    path = template_dir / f"{template_name}_preprocessing_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Template '{template_name}' not found at {path}")
    runtime = RuntimeConfig.load(path)
    return runtime.preprocessing


