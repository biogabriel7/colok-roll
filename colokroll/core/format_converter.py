"""
Format converter for microscopy image files.
Supports converting proprietary formats (e.g., .nd2, .oir) to OME-TIFF while preserving metadata.

All output images are standardized to 4D arrays with ZYXC axis order:
- Z: Z-slices (depth)
- Y: Height
- X: Width  
- C: Channels
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import logging

import numpy as np
import tifffile
import nd2

logger = logging.getLogger(__name__)

# Standard axis order for all output images
STANDARD_AXES = 'ZYXC'
EXPECTED_NDIM = 4


class FormatConverter:
    """Convert microscopy image formats while preserving metadata.
    
    All output images are standardized to 4D arrays with ZYXC axis order.
    """
    
    def __init__(self, preserve_original: bool = True):
        """Initialize the format converter.
        
        Args:
            preserve_original: If True, keeps the original file after conversion.
        """
        self.preserve_original = preserve_original
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    def _ensure_4d_zyxc(self, data: np.ndarray, source_axes: str = None) -> np.ndarray:
        """Ensure data is 4D with ZYXC axis order.
        
        Args:
            data: Input array of any dimensionality.
            source_axes: Optional string indicating source axis order (e.g., 'YX', 'ZYX', 'CZYX').
            
        Returns:
            4D array with shape (Z, Y, X, C).
            
        Raises:
            ValueError: If data cannot be converted to 4D ZYXC format.
        """
        original_shape = data.shape
        original_ndim = data.ndim
        
        logger.info(f"Standardizing array: shape={original_shape}, source_axes={source_axes}")
        
        # Handle based on current dimensionality
        if data.ndim == 2:
            # YX -> ZYXC (add Z and C dimensions)
            data = data[np.newaxis, :, :, np.newaxis]
            logger.info(f"Expanded 2D (YX) to 4D (ZYXC): {original_shape} -> {data.shape}")
            
        elif data.ndim == 3:
            # Could be ZYX or YXC - need to determine based on source_axes or heuristics
            if source_axes:
                if source_axes.upper() == 'ZYX':
                    # ZYX -> ZYXC (add C dimension)
                    data = data[..., np.newaxis]
                elif source_axes.upper() == 'YXC':
                    # YXC -> ZYXC (add Z dimension)
                    data = data[np.newaxis, ...]
                elif source_axes.upper() == 'CYX':
                    # CYX -> ZYXC (transpose and add Z)
                    data = np.transpose(data, (1, 2, 0))  # CYX -> YXC
                    data = data[np.newaxis, ...]  # YXC -> ZYXC
                else:
                    # Default: assume ZYX, add C
                    data = data[..., np.newaxis]
            else:
                # Heuristic: if last dim is small (<=10), assume it's channels (YXC)
                if data.shape[-1] <= 10 and data.shape[-1] < data.shape[0] and data.shape[-1] < data.shape[1]:
                    # Likely YXC -> ZYXC
                    data = data[np.newaxis, ...]
                else:
                    # Assume ZYX -> ZYXC
                    data = data[..., np.newaxis]
            logger.info(f"Expanded 3D to 4D (ZYXC): {original_shape} -> {data.shape}")
            
        elif data.ndim == 4:
            # Already 4D, but may need reordering
            if source_axes and source_axes.upper() != 'ZYXC':
                # Reorder axes to ZYXC
                source_upper = source_axes.upper()
                if source_upper == 'CZYX':
                    data = np.transpose(data, (1, 2, 3, 0))  # CZYX -> ZYXC
                elif source_upper == 'ZCYX':
                    data = np.transpose(data, (0, 2, 3, 1))  # ZCYX -> ZYXC
                elif source_upper == 'YXZC':
                    data = np.transpose(data, (2, 0, 1, 3))  # YXZC -> ZYXC
                elif source_upper == 'YXCZ':
                    data = np.transpose(data, (3, 0, 1, 2))  # YXCZ -> ZYXC
                elif source_upper == 'CXYZ':
                    data = np.transpose(data, (3, 2, 1, 0))  # CXYZ -> ZYXC (reverse + channel last)
                elif source_upper == 'XYZC':
                    data = np.transpose(data, (2, 1, 0, 3))  # XYZC -> ZYXC
                else:
                    logger.warning(f"Unknown 4D axis order: {source_axes}, assuming ZYXC")
                logger.info(f"Reordered 4D from {source_axes} to ZYXC: {original_shape} -> {data.shape}")
            else:
                logger.info(f"Data already 4D ZYXC: {data.shape}")
                
        elif data.ndim == 5:
            # TCZYX, TZCYX, etc. - handle time dimension
            logger.warning(f"5D data detected: shape={original_shape}, axes={source_axes}")
            
            if source_axes:
                source_upper = source_axes.upper()
                # Find time dimension and remove it (take first timepoint)
                if 'T' in source_upper:
                    t_idx = source_upper.index('T')
                    data = np.take(data, 0, axis=t_idx)
                    # Remove T from axes string for recursive call
                    remaining_axes = source_upper.replace('T', '')
                    logger.info(f"Removed time dimension (axis {t_idx}), taking first timepoint")
                    return self._ensure_4d_zyxc(data, source_axes=remaining_axes)
                else:
                    # No T, just take first slice of first dimension
                    logger.warning("5D data without T axis, taking first slice of dim 0")
                    data = data[0]
                    remaining_axes = source_upper[1:] if len(source_upper) > 1 else None
                    return self._ensure_4d_zyxc(data, source_axes=remaining_axes)
            else:
                # No source_axes provided, assume first axis is T
                logger.warning(f"5D data detected without axis info, taking first timepoint: {original_shape}")
                data = data[0]
                return self._ensure_4d_zyxc(data, source_axes=None)
            
        else:
            raise ValueError(
                f"Cannot convert {data.ndim}D array to 4D ZYXC format. "
                f"Input shape: {original_shape}. Expected 2D, 3D, 4D, or 5D input."
            )
        
        # Final validation
        if data.ndim != 4:
            raise ValueError(
                f"Failed to convert to 4D ZYXC. "
                f"Input: {original_ndim}D {original_shape}, Output: {data.ndim}D {data.shape}"
            )
        
        return data
    
    def _validate_4d_zyxc(self, data: np.ndarray) -> None:
        """Validate that data is 4D with reasonable ZYXC dimensions.
        
        Args:
            data: Array to validate.
            
        Raises:
            ValueError: If data is not valid 4D ZYXC format.
        """
        if data.ndim != EXPECTED_NDIM:
            raise ValueError(
                f"Image must be 4D (ZYXC format). Got {data.ndim}D with shape {data.shape}. "
                f"Expected dimensions: Z (depth), Y (height), X (width), C (channels)."
            )
        
        z, y, x, c = data.shape
        
        # Sanity checks
        if z < 1:
            raise ValueError(f"Invalid Z dimension: {z}. Must have at least 1 Z-slice.")
        if y < 1 or x < 1:
            raise ValueError(f"Invalid spatial dimensions: Y={y}, X={x}. Must be positive.")
        if c < 1:
            raise ValueError(f"Invalid channel count: {c}. Must have at least 1 channel.")
        if c > 20:
            logger.warning(f"Unusually high channel count: {c}. Verify axis order is correct.")
        
        logger.info(f"Validated 4D ZYXC array: Z={z}, Y={y}, X={x}, C={c}")
    
    def _validate_colocalization_requirements(
        self, 
        data: np.ndarray, 
        metadata: Dict[str, Any],
        strict: bool = False
    ) -> Dict[str, Any]:
        """Validate that converted data meets requirements for colocalization analysis.
        
        Checks critical requirements for valid colocalization metrics (PCC, MCC):
        - Bit-depth preservation (no lossy compression or downsampling)
        - Spatial calibration (pixel size) for Costes randomization test
        - Z-stack integrity (not flattened)
        
        Args:
            data: Image array to validate.
            metadata: Associated metadata dictionary.
            strict: If True, raises errors instead of warnings for missing calibration.
            
        Returns:
            Dictionary with validation results and warnings.
            
        Raises:
            ValueError: If strict=True and critical metadata is missing.
        """
        validation = {
            'bit_depth_ok': True,
            'calibration_ok': True,
            'z_stack_ok': True,
            'warnings': [],
            'dtype': str(data.dtype),
            'shape': data.shape,
        }
        
        # Check bit-depth (should be uint16 or higher for microscopy)
        logger.info(f"Data dtype: {data.dtype}, shape: {data.shape}")
        
        if data.dtype == np.uint8:
            msg = (
                "WARNING: Image is 8-bit. Most microscopy acquisitions are 12-16 bit. "
                "8-bit data may indicate lossy downsampling which compromises "
                "colocalization metrics (PCC, MCC). Verify this matches original data."
            )
            logger.warning(msg)
            validation['warnings'].append(msg)
        
        # Check spatial calibration (critical for Costes randomization test)
        pixel_size = metadata.get('pixel_size_um')
        voxel_z = metadata.get('pixel_info', {}).get('voxel_size_z')
        
        if pixel_size is None or pixel_size == 0:
            msg = (
                "CRITICAL: Missing pixel size (PhysicalSizeX/Y). "
                "This is required for accurate Costes significance testing. "
                "Colocalization statistics may be unreliable without proper calibration."
            )
            validation['calibration_ok'] = False
            validation['warnings'].append(msg)
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info(f"Pixel size: {pixel_size} µm")
            validation['pixel_size_um'] = pixel_size
        
        if voxel_z is None or voxel_z == 0:
            msg = (
                "WARNING: Missing Z-spacing (PhysicalSizeZ). "
                "3D colocalization analysis requires Z calibration for accurate results."
            )
            validation['warnings'].append(msg)
            logger.warning(msg)
        else:
            logger.info(f"Z-spacing: {voxel_z} µm")
            validation['voxel_size_z'] = voxel_z
        
        # Check Z-stack integrity
        z_levels = data.shape[0]
        if z_levels == 1:
            msg = (
                "NOTE: Single Z-slice detected. For accurate colocalization, "
                "3D volumetric data (Z-stacks) is preferred over 2D images to avoid "
                "false-positive colocalization from depth projection artifacts."
            )
            validation['warnings'].append(msg)
            logger.info(msg)
        else:
            logger.info(f"Z-stack integrity preserved: {z_levels} slices")
            validation['z_stack_ok'] = True
        
        return validation
    
    def nd2_to_ome_tiff(
        self, 
        input_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        save_metadata: bool = True
    ) -> Tuple[Path, Dict[str, Any]]:
        """Convert .nd2 file to .ome.tiff format with metadata preservation.
        
        Uses the modern nd2 library for reliable metadata extraction and data reading.
        
        Args:
            input_path: Path to the input .nd2 file.
            output_path: Path for the output .ome.tiff file. If None, uses same name with .ome.tiff extension.
            save_metadata: If True, saves metadata to a separate JSON file.
            
        Returns:
            Tuple of (output_path, metadata_dict)
            
        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If conversion fails.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if not input_path.suffix.lower() == '.nd2':
            raise ValueError(f"Input file must be .nd2 format, got: {input_path.suffix}")
        
        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix('.ome.tiff')
        else:
            output_path = Path(output_path)
        
        logger.info(f"Converting {input_path} to {output_path}")
        
        try:
            # Read ND2 file using modern nd2 library
            with nd2.ND2File(str(input_path)) as f:
                # Get image data as numpy array
                image_data = f.asarray()
                
                # Get dimension sizes from the file
                sizes = f.sizes  # e.g., {'T': 1, 'C': 4, 'Z': 31, 'Y': 1800, 'X': 1800}
                logger.info(f"ND2 file sizes: {sizes}")
                logger.info(f"ND2 data shape: {image_data.shape}")
                
                # Build source axes string from sizes
                source_axes = ''.join(sizes.keys())
                logger.info(f"ND2 axes order: {source_axes}")
                
                # Extract metadata using nd2 library
                metadata = self._extract_nd2_metadata(f)
            
            # Ensure 4D ZYXC format
            image_data = self._ensure_4d_zyxc(image_data, source_axes=source_axes)
            self._validate_4d_zyxc(image_data)
            
            # Update metadata dimensions to match standardized array
            z, y, x, c = image_data.shape
            metadata['dimensions'] = {
                'z_levels': z,
                'height': y,
                'width': x,
                'channels': c,
                'timepoints': 1,
            }
            metadata['axes'] = STANDARD_AXES
            
            # Validate colocalization requirements
            validation = self._validate_colocalization_requirements(image_data, metadata)
            metadata['conversion_validation'] = validation
            
            if validation['warnings']:
                logger.warning(f"Conversion completed with {len(validation['warnings'])} warning(s)")
            
            # Save as TIFF with metadata
            self._save_as_tiff(image_data, output_path, metadata)
            
            # Save metadata to JSON if requested
            if save_metadata:
                metadata_path = output_path.with_suffix('.json')
                self._save_metadata_json(metadata, metadata_path)
                logger.info(f"Metadata saved to {metadata_path}")
            
            # Cache metadata for quick access
            self.metadata_cache[str(output_path)] = metadata
            
            logger.info(f"Successfully converted to {output_path}")
            return output_path, metadata
            
        except Exception as e:
            logger.error(f"Failed to convert {input_path}: {e}")
            raise ValueError(f"Conversion failed: {e}")

    def _parse_channel_names_from_filename(self, filename: str, num_channels: int) -> List[str]:
        """Extract channel names from Olympus-style filenames.
        
        Args:
            filename: Input filename to parse.
            num_channels: Expected number of channels.
            
        Returns:
            List of channel names extracted from filename, or generic names if parsing fails.
        """
        import re
        
        # Pattern to match channel names with optional wavelengths in parentheses
        # Example: DAPI_ALIX(488)_Phallodin(568)_LAMP1(647)
        
        # Common microscopy channel/marker names to look for (order matters - most specific first)
        known_markers = [
            'Phalloidin', 'Phallodin',  # Actin (check before other markers)
            'LAMP1', 'Lamp1',  # Endosomal markers
            'ALIX', 'CD63', 'TSG101',  # Other endosomal
            'DAPI', 'Hoechst', 'DRAQ5',  # Nuclear stains
            'GFP', 'RFP', 'CFP', 'YFP', 'mCherry',  # Fluorescent proteins
            'Alexa', 'Cy3', 'Cy5',  # Dyes
        ]
        
        found_channels = []
        found_positions = []
        found_markers_lower = set()  # Track found markers (case-insensitive)
        
        # Try to find known markers in filename (case-insensitive) and track their positions
        for marker in known_markers:
            # Skip if we already found this marker (case-insensitive)
            if marker.lower() in found_markers_lower:
                continue
            
            # Use looser pattern that allows underscores before/after
            pattern_obj = re.compile(rf'(?:^|_|\s){re.escape(marker)}(?:_|\s|\(|$)', re.IGNORECASE)
            match = pattern_obj.search(filename)
            if match:
                found_channels.append(marker)
                found_positions.append(match.start())
                found_markers_lower.add(marker.lower())
        
        # Sort by position in filename to maintain order
        if found_channels and found_positions:
            sorted_pairs = sorted(zip(found_positions, found_channels))
            found_channels = [ch for _, ch in sorted_pairs]
        
        # If we found the expected number of channels, return them
        if len(found_channels) == num_channels:
            logger.info(f"Extracted {len(found_channels)} channel names from filename: {found_channels}")
            return found_channels
        
        # If we found some but not all, pad with generic names
        if found_channels:
            while len(found_channels) < num_channels:
                found_channels.append(f'Channel_{len(found_channels)}')
            logger.info(f"Partially extracted channel names from filename: {found_channels}")
            return found_channels[:num_channels]
        
        # Fallback: return generic names
        logger.warning(f"Could not extract channel names from filename: {filename}")
        return [f'Channel_{i}' for i in range(num_channels)]

    def oir_to_ome_tiff(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_metadata: bool = True,
        strict_validation: bool = True,
        compression: Optional[str] = 'lzw',
    ) -> Tuple[Path, Dict[str, Any]]:
        """Convert .oir (Olympus) files to .ome.tiff using bioio.

        Args:
            input_path: Path to the input .oir file.
            output_path: Optional output path for the .ome.tiff result. Defaults to same name with .ome.tiff.
            save_metadata: If True, writes a sidecar JSON with extracted metadata.
            strict_validation: If True, raises error on missing critical metadata (pixel size). Default: True.
            compression: Compression method for TIFF ('lzw', 'zstd', None for no compression). Default: 'lzw'.

        Returns:
            Tuple of (output_path, metadata_dict).

        Raises:
            FileNotFoundError: If the input file is missing.
            ImportError: If bioio is not installed.
            ValueError: If conversion fails or critical metadata missing (when strict_validation=True).
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix.lower() != ".oir":
            raise ValueError(f"Input file must be .oir format, got: {input_path.suffix}")

        # Try to import bioio
        try:
            from bioio import BioImage
        except ImportError:
            raise ImportError(
                "bioio not found. Install with: pip install bioio bioio-ome-tiff\n"
                "For OIR support, also install: pip install bioio-bioformats\n"
                "Note: bioio-bioformats requires Java Runtime Environment (JRE)."
            )

        if output_path is None:
            output_path = input_path.with_suffix(".ome.tiff")
        else:
            output_path = Path(output_path)

        logger.info(f"Converting {input_path} to {output_path} using bioio")

        try:
            # Read OIR file using bioio
            logger.info("Reading OIR file with bioio...")
            img = BioImage(input_path)
            
            # Get image data - bioio returns data in TCZYX or similar order
            data = img.data  # This is a dask or numpy array
            if hasattr(data, 'compute'):
                data = data.compute()  # Convert dask to numpy if needed
            
            source_axes = img.dims.order if hasattr(img.dims, 'order') else str(img.dims)
            logger.info(f"OIR file loaded: shape={data.shape}, axes={source_axes}")
            
            # Extract metadata from bioio
            pixel_size_x = None
            pixel_size_y = None
            voxel_size_z = None
            pixel_size_um = None
            channel_names = []
            
            # Get physical pixel sizes
            if hasattr(img, 'physical_pixel_sizes'):
                pps = img.physical_pixel_sizes
                if pps.X is not None:
                    pixel_size_x = pps.X
                if pps.Y is not None:
                    pixel_size_y = pps.Y
                if pps.Z is not None:
                    voxel_size_z = pps.Z
                if pixel_size_x and pixel_size_y:
                    pixel_size_um = (pixel_size_x + pixel_size_y) / 2
            
            # Get channel names from bioio
            if hasattr(img, 'channel_names') and img.channel_names:
                channel_names = list(img.channel_names)
            
            # Try to extract additional metadata
            acquisition_info = {}
            try:
                if hasattr(img, 'metadata'):
                    img_meta = img.metadata
                    if hasattr(img_meta, 'get'):
                        # Try to get objective information
                        if 'objective' in img_meta:
                            acquisition_info['objective'] = str(img_meta.get('objective'))
                        if 'magnification' in img_meta:
                            acquisition_info['magnification'] = img_meta.get('magnification')
                        if 'numerical_aperture' in img_meta:
                            acquisition_info['numerical_aperture'] = img_meta.get('numerical_aperture')
            except Exception as e:
                logger.debug(f"Could not extract additional metadata: {e}")
            
            logger.info(f"Extracted pixel size: XY={pixel_size_um} µm, Z={voxel_size_z} µm")
            logger.info(f"Channel names from bioio: {channel_names}")
            if acquisition_info:
                logger.info(f"Additional metadata: {acquisition_info}")
            
            # Standardize to 4D ZYXC format
            # bioio typically returns TCZYX, we need to handle this
            data = self._ensure_4d_zyxc(data, source_axes=source_axes)
            self._validate_4d_zyxc(data)
            
            z, y, x, c = data.shape
            
            # Update channel names if needed
            # Check if bioio returned generic/poor quality names
            if not channel_names or len(channel_names) != c:
                # Try to parse from filename
                channel_names = self._parse_channel_names_from_filename(input_path.name, c)
            elif all(name.startswith(('Ch', 'Channel', 'C')) or name.isdigit() for name in channel_names):
                # bioio returned generic names like "Ch1", "Channel_0", etc.
                logger.info("bioio returned generic channel names, trying filename parsing...")
                parsed_names = self._parse_channel_names_from_filename(input_path.name, c)
                # Use parsed names if they look better than generic ones
                if any(not name.startswith('Channel_') for name in parsed_names):
                    channel_names = parsed_names
            
            # Build metadata dictionary with standardized dimensions
            metadata: Dict[str, Any] = {
                "original_format": "oir",
                "conversion_method": "bioio",
                "axes": STANDARD_AXES,
                "channel_names": channel_names,
                "pixel_size_um": pixel_size_um,
                "pixel_info": {
                    "pixel_microns": pixel_size_um,
                    "pixel_microns_x": pixel_size_x,
                    "pixel_microns_y": pixel_size_y,
                    "voxel_size_z": voxel_size_z,
                    "calibration": None,
                },
                "dimensions": {
                    "z_levels": z,
                    "height": y,
                    "width": x,
                    "channels": c,
                    "timepoints": 1,
                },
            }
            
            # Add acquisition info if available
            if acquisition_info:
                metadata['acquisition'] = acquisition_info
            
            # Validate colocalization requirements with strict validation
            validation = self._validate_colocalization_requirements(data, metadata, strict=strict_validation)
            metadata['conversion_validation'] = validation
            
            if validation['warnings']:
                logger.warning(f"Conversion completed with {len(validation['warnings'])} warning(s)")
            
            # Save in standardized ZYXC format
            logger.info(f"Saving in standardized ZYXC format: {data.shape}")
            self._save_as_tiff(data, output_path, metadata, compression=compression)

            if save_metadata:
                metadata_path = output_path.with_suffix(".json")
                self._save_metadata_json(metadata, metadata_path)
                logger.info(f"Metadata saved to {metadata_path}")

            self.metadata_cache[str(output_path)] = metadata
            logger.info(f"Successfully converted to {output_path}")
            return output_path, metadata

        except ImportError as e:
            logger.error(f"Missing bioio plugin: {e}")
            raise ImportError(
                f"Failed to read OIR file. You may need additional bioio plugins:\n"
                f"  pip install bioio-bioformats\n"
                f"Original error: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to convert {input_path}: {e}")
            raise ValueError(f"Conversion failed: {e}")
    
    def _extract_nd2_metadata(self, f: nd2.ND2File) -> Dict[str, Any]:
        """Extract comprehensive metadata from ND2 file using nd2 library.
        
        Args:
            f: nd2.ND2File instance.
            
        Returns:
            Dictionary containing all relevant metadata.
        """
        metadata = {}
        
        # Basic metadata
        metadata['original_format'] = 'nd2'
        metadata['conversion_method'] = 'nd2'
        metadata['axes'] = ''.join(f.sizes.keys())
        
        # Get voxel sizes (physical pixel sizes)
        voxel_size = f.voxel_size()
        pixel_size_x = voxel_size.x if voxel_size else None
        pixel_size_y = voxel_size.y if voxel_size else None
        pixel_size_z = voxel_size.z if voxel_size else None
        
        # Pixel/voxel information - CRITICAL for physical measurements
        metadata['pixel_info'] = {
            'pixel_microns': pixel_size_x,
            'pixel_microns_x': pixel_size_x,
            'pixel_microns_y': pixel_size_y,
            'voxel_size_z': pixel_size_z,
            'calibration': pixel_size_x,
        }
        
        # Calculate actual pixel size (average of x and y)
        if pixel_size_x and pixel_size_y:
            metadata['pixel_size_um'] = (pixel_size_x + pixel_size_y) / 2
        elif pixel_size_x:
            metadata['pixel_size_um'] = pixel_size_x
        else:
            metadata['pixel_size_um'] = None
            logger.warning("No pixel size information found in ND2 metadata")
        
        logger.info(f"Extracted pixel size: XY={metadata['pixel_size_um']} µm, Z={pixel_size_z} µm")
        
        # Extract channel names from metadata
        channel_names = []
        try:
            if hasattr(f, 'metadata') and f.metadata:
                channels_meta = f.metadata.channels
                if channels_meta:
                    for ch in channels_meta:
                        if hasattr(ch, 'channel') and hasattr(ch.channel, 'name'):
                            channel_names.append(ch.channel.name)
                        elif hasattr(ch, 'name'):
                            channel_names.append(ch.name)
        except Exception as e:
            logger.debug(f"Could not extract channel names from metadata: {e}")
        
        # Fallback: use sizes to determine number of channels
        num_channels = f.sizes.get('C', 1)
        if not channel_names or len(channel_names) != num_channels:
            logger.warning(f"Channel names not found or incomplete, using defaults")
            channel_names = [f'Channel_{i}' for i in range(num_channels)]
        
        metadata['channel_names'] = channel_names
        logger.info(f"Channel names: {channel_names}")
        
        # Dimensional information
        metadata['dimensions'] = {
            'width': f.sizes.get('X', 0),
            'height': f.sizes.get('Y', 0),
            'z_levels': f.sizes.get('Z', 1),
            'channels': f.sizes.get('C', 1),
            'timepoints': f.sizes.get('T', 1),
        }
        
        # Acquisition information
        metadata['acquisition'] = {}
        try:
            if hasattr(f, 'attributes') and f.attributes:
                attrs = f.attributes
                if hasattr(attrs, 'date'):
                    metadata['acquisition']['date'] = str(attrs.date) if attrs.date else None
        except Exception as e:
            logger.debug(f"Could not extract acquisition info: {e}")
        
        # Microscope settings
        metadata['microscope'] = {}
        try:
            if hasattr(f, 'metadata') and f.metadata:
                if hasattr(f.metadata, 'channels') and f.metadata.channels:
                    ch0 = f.metadata.channels[0]
                    if hasattr(ch0, 'microscope'):
                        mic = ch0.microscope
                        if hasattr(mic, 'objectiveName'):
                            metadata['microscope']['objective'] = mic.objectiveName
                        if hasattr(mic, 'objectiveMagnification'):
                            metadata['microscope']['magnification'] = mic.objectiveMagnification
                        if hasattr(mic, 'objectiveNumericalAperture'):
                            metadata['microscope']['numerical_aperture'] = mic.objectiveNumericalAperture
        except Exception as e:
            logger.debug(f"Could not extract microscope settings: {e}")
        
        return metadata
    
    
    def _save_as_tiff(
        self, 
        image_data: np.ndarray, 
        output_path: Path, 
        metadata: Dict[str, Any],
        compression: Optional[str] = 'lzw'
    ) -> None:
        """Save image data as OME-TIFF with metadata.
        
        Args:
            image_data: 4D image array with ZYXC axis order.
            output_path: Output file path.
            metadata: Metadata dictionary to embed.
            compression: Compression method ('lzw', 'zstd', None). Default: 'lzw'.
            
        Raises:
            ValueError: If image_data is not 4D ZYXC format.
        """
        # Validate input is 4D ZYXC
        self._validate_4d_zyxc(image_data)
        
        z, y, x, c = image_data.shape
        
        # Get channel names
        channel_names = metadata.get('channel_names', [])
        if not channel_names or len(channel_names) != c:
            channel_names = [f'Channel_{i}' for i in range(c)]
            logger.warning(f"Channel names missing or mismatched, using defaults: {channel_names}")
        
        # For proper multi-channel OME-TIFF, convert ZYXC to ZCYX
        # This ensures each channel is stored as a separate plane (not interleaved)
        image_zcyx = np.transpose(image_data, (0, 3, 1, 2))  # ZYXC -> ZCYX
        
        # Build complete OME metadata with all required fields
        ome_metadata = {
            # Axis order for the data array (now ZCYX)
            'axes': 'ZCYX',
            # Explicit dimension sizes (required for proper OME-XML)
            'SizeZ': z,
            'SizeY': y,
            'SizeX': x,
            'SizeC': c,
            'SizeT': 1,
            # OME dimension order (how dimensions are stored in file)
            'DimensionOrder': 'XYCZT',
        }
        
        # Add channel information with proper structure for multi-channel
        # tifffile expects a list of dicts for multiple channels
        ome_metadata['Channel'] = [{'Name': name} for name in channel_names]
        
        # Add physical pixel sizes
        if metadata.get('pixel_size_um'):
            ome_metadata['PhysicalSizeX'] = metadata['pixel_size_um']
            ome_metadata['PhysicalSizeY'] = metadata['pixel_size_um']
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
        
        # Add z-spacing if available
        if metadata.get('pixel_info', {}).get('voxel_size_z'):
            ome_metadata['PhysicalSizeZ'] = metadata['pixel_info']['voxel_size_z']
            ome_metadata['PhysicalSizeZUnit'] = 'µm'
        
        # Add bit depth information
        ome_metadata['SignificantBits'] = image_data.dtype.itemsize * 8
        
        logger.info(f"Saving OME-TIFF: input shape={image_data.shape} (ZYXC), output shape={image_zcyx.shape} (ZCYX)")
        logger.info(f"Dimensions: Z={z}, C={c}, Y={y}, X={x}")
        
        # Determine if BigTIFF is needed (for files > 2GB)
        use_bigtiff = image_data.nbytes > 2 * 1024**3
        if use_bigtiff:
            logger.info("Using BigTIFF format for large file")
        
        try:
            tifffile.imwrite(
                output_path,
                image_zcyx,
                ome=True,
                bigtiff=use_bigtiff,
                photometric='minisblack',
                metadata=ome_metadata,
                compression=compression if compression else None
            )
            logger.info(f"Successfully saved OME-TIFF: {output_path}")
        except Exception as e:
            logger.error(f"OME-TIFF save failed: {e}, trying fallback")
            # Fallback: save as regular TIFF with JSON description
            tifffile.imwrite(
                output_path,
                image_zcyx,
                bigtiff=use_bigtiff,
                photometric='minisblack',
                description=json.dumps(metadata, indent=2, default=str),
                compression=compression if compression else None
            )
    
    def _save_metadata_json(self, metadata: Dict[str, Any], output_path: Path) -> None:
        """Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary.
            output_path: Output JSON file path.
        """
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_image(self, tiff_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load image data from a converted TIFF file.
        
        Args:
            tiff_path: Path to the TIFF file.
            
        Returns:
            Tuple of (image_data, metadata) where image_data is 4D ZYXC array.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If image is not in 4D ZYXC format.
        """
        tiff_path = Path(tiff_path)
        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
        
        with tifffile.TiffFile(str(tiff_path)) as tif:
            data = tif.series[0].asarray()
            axes = tif.series[0].axes
        
        # Validate 4D ZYXC format
        if data.ndim != EXPECTED_NDIM:
            raise ValueError(
                f"Image must be 4D ZYXC format. Got {data.ndim}D with shape {data.shape}, axes={axes}. "
                f"Use the converter to standardize the image format first."
            )
        
        self._validate_4d_zyxc(data)
        metadata = self.load_metadata(tiff_path)
        
        return data, metadata
    
    def load_metadata(self, tiff_path: Union[str, Path]) -> Dict[str, Any]:
        """Load metadata from a converted TIFF file.
        
        Args:
            tiff_path: Path to the TIFF file.
            
        Returns:
            Metadata dictionary.
        """
        tiff_path = Path(tiff_path)
        
        # Check cache first
        if str(tiff_path) in self.metadata_cache:
            return self.metadata_cache[str(tiff_path)]
        
        # Try to load from JSON file
        json_path = tiff_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                self.metadata_cache[str(tiff_path)] = metadata
                return metadata
        
        # Extract from TIFF tags (JSON)
        try:
            with tifffile.TiffFile(tiff_path) as tif:
                if tif.pages[0].description:
                    metadata = json.loads(tif.pages[0].description)
                    self.metadata_cache[str(tiff_path)] = metadata
                    return metadata
        except Exception as e:
            logger.debug(f"Could not extract JSON metadata from TIFF: {e}")
        
        # Try to extract from OME-XML metadata (for OME-TIFF files)
        try:
            with tifffile.TiffFile(str(tiff_path)) as tif:
                data = tif.series[0].asarray()
                axes = tif.series[0].axes
                
                # Validate 4D ZYXC format
                if data.ndim != EXPECTED_NDIM:
                    logger.warning(
                        f"TIFF file is not in expected 4D ZYXC format: "
                        f"shape={data.shape}, axes={axes}"
                    )
                
                pixel_size_x = pixel_size_y = voxel_size_z = pixel_size_um = None
                channel_names = []
                
                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(tif.ome_metadata)
                    
                    namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels = root.find('.//ome:Pixels', namespaces)
                    
                    if pixels is not None:
                        pixel_size_x = float(pixels.get('PhysicalSizeX', 0) or 0)
                        pixel_size_y = float(pixels.get('PhysicalSizeY', 0) or 0)
                        voxel_size_z = float(pixels.get('PhysicalSizeZ', 0) or 0)
                        if pixel_size_x and pixel_size_y:
                            pixel_size_um = (pixel_size_x + pixel_size_y) / 2
                    
                    channels = root.findall('.//ome:Channel', namespaces)
                    channel_names = [ch.get('Name', f'Channel_{i}') for i, ch in enumerate(channels)]
                
                # For 4D ZYXC format, dimensions are straightforward
                if data.ndim == EXPECTED_NDIM:
                    z, y, x, c = data.shape
                else:
                    # Fallback for non-standard formats
                    z = data.shape[0] if data.ndim >= 4 else 1
                    y = data.shape[-3] if data.ndim >= 3 else data.shape[0]
                    x = data.shape[-2] if data.ndim >= 2 else 1
                    c = data.shape[-1] if data.ndim >= 4 else 1
                
                if not channel_names:
                    channel_names = [f'Channel_{i}' for i in range(c)]
                
                metadata = {
                    "original_format": "ome-tiff",
                    "axes": STANDARD_AXES if data.ndim == EXPECTED_NDIM else axes,
                    "channel_names": channel_names,
                    "pixel_size_um": pixel_size_um,
                    "pixel_info": {
                        "pixel_microns": pixel_size_um,
                        "pixel_microns_x": pixel_size_x,
                        "pixel_microns_y": pixel_size_y,
                        "voxel_size_z": voxel_size_z,
                    },
                    "dimensions": {
                        "z_levels": z,
                        "height": y,
                        "width": x,
                        "channels": c,
                        "timepoints": 1,
                    },
                }
                
                self.metadata_cache[str(tiff_path)] = metadata
                return metadata
                
        except Exception as e:
            logger.warning(f"Could not extract metadata from TIFF: {e}")
        
        return {}
    
    def batch_convert(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.nd2"
    ) -> Dict[Path, Path]:
        """Convert multiple ND2 files to TIFF format.
        
        Args:
            input_dir: Directory containing ND2 files.
            output_dir: Output directory for TIFF files. If None, uses input_dir.
            pattern: File pattern to match (default: "*.nd2").
            
        Returns:
            Dictionary mapping input paths to output paths.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all ND2 files
        nd2_files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(nd2_files)} files to convert")
        
        conversions = {}
        for nd2_file in nd2_files:
            try:
                output_file = output_dir / nd2_file.with_suffix('.ome.tiff').name
                output_path, _ = self.nd2_to_ome_tiff(nd2_file, output_file)
                conversions[nd2_file] = output_path
                logger.info(f"Converted: {nd2_file.name} -> {output_path.name}")
            except Exception as e:
                logger.error(f"Failed to convert {nd2_file}: {e}")
                conversions[nd2_file] = None
        
        return conversions

    def batch_convert_oir(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.oir",
    ) -> Dict[Path, Optional[Path]]:
        """Convert multiple OIR files to OME-TIFF format.

        Args:
            input_dir: Directory containing OIR files.
            output_dir: Output directory for OME-TIFF files. If None, uses input_dir.
            pattern: File pattern to match (default: "*.oir").

        Returns:
            Dictionary mapping input paths to output paths (or None on failure).
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        oir_files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(oir_files)} OIR files to convert")

        conversions: Dict[Path, Optional[Path]] = {}
        for oir_file in oir_files:
            try:
                output_file = output_dir / oir_file.with_suffix('.ome.tiff').name
                output_path, _ = self.oir_to_ome_tiff(oir_file, output_file)
                conversions[oir_file] = output_path
                logger.info(f"Converted: {oir_file.name} -> {output_path.name}")
            except ImportError as exc:
                logger.error(f"Skipping {oir_file}: {exc}")
                conversions[oir_file] = None
            except Exception as e:
                logger.error(f"Failed to convert {oir_file}: {e}")
                conversions[oir_file] = None

        return conversions