"""Core configuration dataclasses for colokroll."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import warnings

import yaml


# ---------------------------------------------------------------------------
# Image IO and processing configuration sections
# ---------------------------------------------------------------------------


@dataclass
class ImageIOConfig:
    """Configuration for image loading and metadata handling."""

    supported_image_formats: List[str] = field(default_factory=lambda: [".nd2", ".tif", ".tiff"])
    max_file_size_gb: float = 10.0
    min_image_dimension: int = 100
    max_image_dimension: int = 10000
    extract_all_metadata: bool = True
    required_metadata_fields: List[str] = field(default_factory=lambda: ["pixel_size", "channels"])
    verbose: bool = True
    log_level: str = "INFO"


@dataclass
class ProjectionConfig:
    """Configuration for maximum intensity projection generation."""

    projection_method: str = "max"
    z_range: Optional[Tuple[int, int]] = None
    calculate_quality_metrics: bool = True
    quality_threshold: float = 0.7
    figure_dpi: int = 100
    default_colormap: str = "viridis"
    save_intermediate_results: bool = True
    output_format: str = "png"


@dataclass
class SegmentationConfig:
    """Configuration for cell and nuclei segmentation."""

    cell_model_type: str = "cyto2"
    cell_diameter: Optional[float] = None
    cell_flow_threshold: float = 0.4
    cell_cellprob_threshold: float = 0.0
    min_cell_area_pixels: int = 100
    nuclei_model_type: str = "nuclei"
    nuclei_diameter: Optional[float] = None
    nuclei_flow_threshold: float = 0.4
    nuclei_cellprob_threshold: float = 0.0
    min_nuclei_area_pixels: int = 50
    max_nuclei_per_cell: int = 1
    require_nuclei_in_cell: bool = True
    use_gpu: bool = False
    gpu_device: int = 0


@dataclass
class RingAnalysisConfig:
    """Configuration for perinuclear ring generation."""

    inner_ring_distance_um: float = 5.0
    outer_ring_distance_um: float = 10.0
    exclusion_zone_start_um: float = 5.0
    exclusion_zone_end_um: float = 10.0
    use_distance_transform: bool = True
    smooth_ring_boundaries: bool = True
    smoothing_sigma: float = 1.0
    respect_cell_boundaries: bool = True
    handle_edge_cells: str = "exclude"
    min_ring_coverage: float = 0.5
    validate_ring_geometry: bool = True
    visualize_rings: bool = True


@dataclass
class QuantificationConfig:
    """Configuration for signal quantification and reporting."""

    background_correction_method: str = "local"
    background_percentile: float = 5.0
    measurements: List[str] = field(default_factory=lambda: ["mean", "median", "sum", "std", "min", "max"])
    normalize_intensities: bool = True
    normalization_method: str = "per_cell"
    calculate_ratios: bool = True
    ratio_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("perinuclear_5um", "peripheral_10um"),
        ("nuclear", "cytoplasmic"),
    ])
    batch_size: int = 10
    parallel_processing: bool = False
    n_workers: int = 4
    save_individual_results: bool = True
    save_summary_statistics: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["csv", "xlsx", "json"])
    generate_plots: bool = True
    plot_types: List[str] = field(default_factory=lambda: ["violin", "box", "scatter", "heatmap"])


# ---------------------------------------------------------------------------
# Preprocessing configuration hierarchy
# ---------------------------------------------------------------------------


@dataclass
class BackgroundSubtractionConfig:
    """Configuration for background subtraction preprocessing."""

    method: str = "rolling_ball"
    rolling_ball_radius_dapi: int = 50
    rolling_ball_radius_phalloidin: int = 20
    rolling_ball_radius_protein: int = 30
    gaussian_sigma_dapi: float = 25.0
    gaussian_sigma_phalloidin: float = 10.0
    gaussian_sigma_protein: float = 15.0
    clip_negative_values: bool = True
    normalize_output: bool = False


@dataclass
class DenoisingConfig:
    """Configuration for channel-specific denoising."""

    dapi_method: str = "nl_means"
    dapi_nl_means_patch_size: int = 7
    dapi_nl_means_patch_distance: int = 13
    dapi_nl_means_h: float = 0.1
    phalloidin_method: str = "tv_chambolle"
    phalloidin_tv_weight: float = 0.1
    phalloidin_tv_iterations: int = 100
    protein_method: str = "bilateral"
    protein_bilateral_sigma_color: float = 0.1
    protein_bilateral_sigma_spatial: float = 2.0
    preserve_edges: bool = True
    edge_threshold: float = 0.1


@dataclass
class DeconvolutionConfig:
    """Configuration for 3D deconvolution."""

    method: str = "richardson_lucy_tv"
    max_iterations: int = 20
    convergence_threshold: float = 1e-4
    tv_regularization_weight: float = 0.002
    auto_stopping: bool = True
    stopping_patience: int = 5
    use_gpu: bool = False
    chunk_size: Tuple[int, int, int] = (256, 256, 16)
    calculate_metrics: bool = True
    save_iterations: bool = False


@dataclass
class QualityControlConfig:
    """Configuration for quality control metrics and thresholds."""

    snr_threshold: float = 10.0
    snr_background_percentile: float = 5.0
    focus_method: str = "power_log_log"
    focus_threshold: float = 0.7
    uniformity_cv_threshold: float = 5.0
    uniformity_grid_size: int = 8
    max_saturated_pixels_percent: float = 1.0
    min_dynamic_range_bits: int = 8
    check_photobleaching: bool = True
    bleaching_threshold_percent: float = 10.0
    enforce_quality_gates: bool = True
    fail_on_quality: bool = False


@dataclass
class ChannelProcessingConfig:
    """Configuration for channel-specific processing parameters."""

    dapi_channel_names: List[str] = field(default_factory=lambda: ["DAPI", "Hoechst", "DAPI - DAPI"])
    phalloidin_channel_names: List[str] = field(default_factory=lambda: ["Phalloidin", "Actin", "AF488"])
    lamp1_channel_names: List[str] = field(default_factory=lambda: ["LAMP1", "Lysosome"])
    protein_channel_names: List[str] = field(default_factory=lambda: ["GAL3", "ALIX", "Protein"])
    dapi_gamma_correction: float = 1.0
    dapi_segmentation_blur_sigma: float = 3.0
    dapi_overexposure_strategy: bool = True
    phalloidin_edge_enhancement: bool = True
    phalloidin_clahe_clip_limit: float = 0.03
    phalloidin_clahe_tile_grid: Tuple[int, int] = (8, 8)
    phalloidin_unsharp_radius: float = 1.5
    phalloidin_unsharp_strength: float = 0.75
    lamp1_log_sigma_range: Tuple[float, float] = (0.5, 2.0)
    lamp1_vesicle_size_range: Tuple[float, float] = (0.5, 2.0)
    lamp1_percentile_threshold: float = 95.0
    lamp1_circularity_threshold: float = 0.7
    protein_scale_range: Tuple[float, float] = (0.5, 5.0)
    protein_vesicle_detection: bool = True
    protein_diffuse_detection: bool = True
    protein_normalization_method: str = "median"


@dataclass
class PreprocessingConfig:
    """Master configuration for image preprocessing."""

    background_subtraction: BackgroundSubtractionConfig = field(default_factory=BackgroundSubtractionConfig)
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    channel_processing: ChannelProcessingConfig = field(default_factory=ChannelProcessingConfig)
    processing_order: List[str] = field(default_factory=lambda: ["background_subtraction", "denoising", "deconvolution"])
    skip_steps: List[str] = field(default_factory=list)
    max_memory_usage_gb: float = 14.0
    chunk_processing: bool = True
    parallel_channels: bool = True
    max_parallel_channels: int = 2
    save_intermediate_steps: bool = False
    intermediate_format: str = "tiff"
    compression: Optional[str] = "lzw"
    run_quality_checks: bool = True
    quality_report_format: str = "json"

    def get_channel_config(self, channel_name: str) -> Dict[str, Any]:
        channel_name_lower = channel_name.lower()
        if any(name.lower() in channel_name_lower for name in self.channel_processing.dapi_channel_names):
            return self._get_dapi_config()
        if any(name.lower() in channel_name_lower for name in self.channel_processing.phalloidin_channel_names):
            return self._get_phalloidin_config()
        if any(name.lower() in channel_name_lower for name in self.channel_processing.lamp1_channel_names):
            return self._get_lamp1_config()
        return self._get_protein_config()

    def _get_dapi_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration optimized for DAPI/nuclear staining channels.
        
        Returns:
            Dict with background subtraction, denoising, and channel-specific parameters
            optimized for nuclear staining (larger radius, NL-means denoising).
        """
        return {
            "background_method": "rolling_ball",
            "background_radius": self.background_subtraction.rolling_ball_radius_dapi,
            "denoise_method": self.denoising.dapi_method,
            "denoise_params": {
                "patch_size": self.denoising.dapi_nl_means_patch_size,
                "patch_distance": self.denoising.dapi_nl_means_patch_distance,
                "h": self.denoising.dapi_nl_means_h,
            },
            "gamma_correction": self.channel_processing.dapi_gamma_correction,
            "segmentation_blur": self.channel_processing.dapi_segmentation_blur_sigma,
        }

    def _get_phalloidin_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration optimized for phalloidin/actin channels.
        
        Returns:
            Dict with background subtraction, denoising, and edge enhancement parameters
            optimized for cytoskeletal staining (smaller radius, TV denoising, CLAHE).
        """
        return {
            "background_method": "rolling_ball",
            "background_radius": self.background_subtraction.rolling_ball_radius_phalloidin,
            "denoise_method": self.denoising.phalloidin_method,
            "denoise_params": {
                "weight": self.denoising.phalloidin_tv_weight,
                "iterations": self.denoising.phalloidin_tv_iterations,
            },
            "edge_enhancement": self.channel_processing.phalloidin_edge_enhancement,
            "clahe_params": {
                "clip_limit": self.channel_processing.phalloidin_clahe_clip_limit,
                "tile_grid": self.channel_processing.phalloidin_clahe_tile_grid,
            },
            "unsharp_params": {
                "radius": self.channel_processing.phalloidin_unsharp_radius,
                "strength": self.channel_processing.phalloidin_unsharp_strength,
            },
        }

    def _get_lamp1_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration optimized for LAMP1/lysosomal marker channels.
        
        Returns:
            Dict with background subtraction, denoising, and vesicle detection parameters
            optimized for punctate lysosomal markers.
        """
        return {
            "background_method": "rolling_ball",
            "background_radius": self.background_subtraction.rolling_ball_radius_protein,
            "denoise_method": self.denoising.protein_method,
            "denoise_params": {
                "sigma_color": self.denoising.protein_bilateral_sigma_color,
                "sigma_spatial": self.denoising.protein_bilateral_sigma_spatial,
            },
            "log_sigma_range": self.channel_processing.lamp1_log_sigma_range,
            "vesicle_size_range": self.channel_processing.lamp1_vesicle_size_range,
            "percentile_threshold": self.channel_processing.lamp1_percentile_threshold,
            "circularity_threshold": self.channel_processing.lamp1_circularity_threshold,
        }

    def _get_protein_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration optimized for generic protein marker channels.
        
        Returns:
            Dict with background subtraction, denoising, and detection parameters
            optimized for protein markers (both punctate and diffuse patterns).
        """
        return {
            "background_method": "rolling_ball",
            "background_radius": self.background_subtraction.rolling_ball_radius_protein,
            "denoise_method": self.denoising.protein_method,
            "denoise_params": {
                "sigma_color": self.denoising.protein_bilateral_sigma_color,
                "sigma_spatial": self.denoising.protein_bilateral_sigma_spatial,
            },
            "scale_range": self.channel_processing.protein_scale_range,
            "vesicle_detection": self.channel_processing.protein_vesicle_detection,
            "diffuse_detection": self.channel_processing.protein_diffuse_detection,
            "normalization_method": self.channel_processing.protein_normalization_method,
        }


# ---------------------------------------------------------------------------
# Runtime configuration container
# ---------------------------------------------------------------------------


def _encode_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {f.name: _encode_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_value(item) for item in value)
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    return value


def _coerce_dataclass(cls, data: Dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        return data
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if hasattr(f.type, "__dataclass_fields__"):
            kwargs[f.name] = _coerce_dataclass(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


@dataclass
class RuntimeConfig:
    """Top-level configuration for the colokroll toolkit."""

    io: ImageIOConfig = field(default_factory=ImageIOConfig)
    projections: ProjectionConfig = field(default_factory=ProjectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    ring_analysis: RingAnalysisConfig = field(default_factory=RingAnalysisConfig)
    quantification: QuantificationConfig = field(default_factory=QuantificationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    project_name: str = "colokroll"
    output_dir: Path = field(default_factory=lambda: Path("results"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    random_seed: int = 42
    ensure_reproducibility: bool = True

    # ------------------------------------------------------------------
    # Compatibility helpers (legacy phase-based API)
    # ------------------------------------------------------------------
    def _phase_alias(self, name: str):
        warnings.warn(
            f"RuntimeConfig.{name} is deprecated; access config.{_PHASE_TO_ATTR[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, _PHASE_TO_ATTR[name])

    @property
    def phase1(self) -> ImageIOConfig:
        return self._phase_alias("phase1")

    @phase1.setter
    def phase1(self, value: ImageIOConfig) -> None:
        warnings.warn(
            "Setting RuntimeConfig.phase1 is deprecated; assign to config.io instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.io = value

    @property
    def phase2(self) -> ProjectionConfig:
        return self._phase_alias("phase2")

    @phase2.setter
    def phase2(self, value: ProjectionConfig) -> None:
        warnings.warn(
            "Setting RuntimeConfig.phase2 is deprecated; assign to config.projections instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.projections = value

    @property
    def phase3(self) -> SegmentationConfig:
        return self._phase_alias("phase3")

    @phase3.setter
    def phase3(self, value: SegmentationConfig) -> None:
        warnings.warn(
            "Setting RuntimeConfig.phase3 is deprecated; assign to config.segmentation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.segmentation = value

    @property
    def phase4(self) -> RingAnalysisConfig:
        return self._phase_alias("phase4")

    @phase4.setter
    def phase4(self, value: RingAnalysisConfig) -> None:
        warnings.warn(
            "Setting RuntimeConfig.phase4 is deprecated; assign to config.ring_analysis instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.ring_analysis = value

    @property
    def phase5(self) -> QuantificationConfig:
        return self._phase_alias("phase5")

    @phase5.setter
    def phase5(self, value: QuantificationConfig) -> None:
        warnings.warn(
            "Setting RuntimeConfig.phase5 is deprecated; assign to config.quantification instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.quantification = value

    def enable_phase(self, phase: str) -> None:  # pragma: no cover - legacy shim
        warnings.warn(
            "enable_phase() is deprecated; all sections are available by default.",
            DeprecationWarning,
            stacklevel=2,
        )

    def is_phase_enabled(self, phase: str) -> bool:  # pragma: no cover - legacy shim
        warnings.warn(
            "is_phase_enabled() is deprecated; all sections are always enabled.",
            DeprecationWarning,
            stacklevel=2,
        )
        return True

    def get_phase_config(self, phase: str) -> Any:  # pragma: no cover - legacy shim
        warnings.warn(
            "get_phase_config() is deprecated; access config sections directly (e.g., config.segmentation).",
            DeprecationWarning,
            stacklevel=2,
        )
        key = _PHASE_TO_ATTR.get(phase)
        if key is None:
            raise ValueError(f"Unknown phase '{phase}'.")
        return getattr(self, key)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "io": _encode_value(self.io),
            "projections": _encode_value(self.projections),
            "segmentation": _encode_value(self.segmentation),
            "ring_analysis": _encode_value(self.ring_analysis),
            "quantification": _encode_value(self.quantification),
            "preprocessing": _encode_value(self.preprocessing),
            "project_name": self.project_name,
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "random_seed": self.random_seed,
            "ensure_reproducibility": self.ensure_reproducibility,
        }

    def save(self, filepath: Union[str, Path], format: str = "auto") -> None:
        filepath = Path(filepath)
        format = _resolve_format(filepath, format)
        data = self.to_dict()
        with filepath.open("w", encoding="utf-8") as handle:
            if format == "yaml":
                yaml.safe_dump(data, handle, default_flow_style=False, indent=2)
            else:
                json.dump(data, handle, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RuntimeConfig":
        filepath = Path(filepath)
        with filepath.open("r", encoding="utf-8") as handle:
            if filepath.suffix.lower() in {".yml", ".yaml"}:
                raw = yaml.safe_load(handle) or {}
            else:
                raw = json.load(handle)

        # Handle legacy phase-based schema transparently
        mapped = _translate_legacy_schema(raw)

        config = cls()
        for key in ["io", "projections", "segmentation", "ring_analysis", "quantification", "preprocessing"]:
            if key in mapped:
                setattr(config, key, _coerce_dataclass(type(getattr(config, key)), mapped[key]))

        config.project_name = mapped.get("project_name", config.project_name)
        config.output_dir = Path(mapped.get("output_dir", config.output_dir))
        config.temp_dir = Path(mapped.get("temp_dir", config.temp_dir))
        config.random_seed = mapped.get("random_seed", config.random_seed)
        config.ensure_reproducibility = mapped.get("ensure_reproducibility", config.ensure_reproducibility)
        return config


def _resolve_format(filepath: Path, format_hint: str) -> str:
    if format_hint != "auto":
        return format_hint
    return "yaml" if filepath.suffix.lower() in {".yml", ".yaml"} else "json"


_PHASE_TO_ATTR = {
    "phase1": "io",
    "phase2": "projections",
    "phase3": "segmentation",
    "phase4": "ring_analysis",
    "phase5": "quantification",
}


def _translate_legacy_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    result = dict(data)
    # Map legacy phase keys to new attribute names
    for phase_key, attr in _PHASE_TO_ATTR.items():
        if phase_key in data and attr not in result:
            result[attr] = data[phase_key]
    # Provide defaults for missing sections
    for attr in ["io", "projections", "segmentation", "ring_analysis", "quantification"]:
        result.setdefault(attr, {})
    return result


