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
from typing import Dict, Any, Optional, Union, Tuple
import logging

import numpy as np
import tifffile
import nd2reader

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
            # Read ND2 file and extract metadata
            with nd2reader.ND2Reader(str(input_path)) as reader:
                metadata = self._extract_comprehensive_metadata(reader)
                image_data = self._read_nd2_as_array(reader)
            
            # Ensure 4D ZYXC format
            image_data = self._ensure_4d_zyxc(image_data, source_axes='ZYXC')
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

    def oir_to_ome_tiff(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_metadata: bool = True,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Convert .oir (Olympus) files to .ome.tiff using bioio.

        Args:
            input_path: Path to the input .oir file.
            output_path: Optional output path for the .ome.tiff result. Defaults to same name with .ome.tiff.
            save_metadata: If True, writes a sidecar JSON with extracted metadata.

        Returns:
            Tuple of (output_path, metadata_dict).

        Raises:
            FileNotFoundError: If the input file is missing.
            ImportError: If bioio is not installed.
            ValueError: If conversion fails.
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
            
            # Get channel names
            if hasattr(img, 'channel_names') and img.channel_names:
                channel_names = list(img.channel_names)
            
            logger.info(f"Extracted pixel size: XY={pixel_size_um} µm, Z={voxel_size_z} µm")
            logger.info(f"Channel names: {channel_names}")
            
            # Standardize to 4D ZYXC format
            # bioio typically returns TCZYX, we need to handle this
            data = self._ensure_4d_zyxc(data, source_axes=source_axes)
            self._validate_4d_zyxc(data)
            
            z, y, x, c = data.shape
            
            # Update channel names if needed
            if not channel_names or len(channel_names) != c:
                channel_names = [f'Channel_{i}' for i in range(c)]
            
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
            
            # Validate colocalization requirements
            validation = self._validate_colocalization_requirements(data, metadata)
            metadata['conversion_validation'] = validation
            
            if validation['warnings']:
                logger.warning(f"Conversion completed with {len(validation['warnings'])} warning(s)")
            
            # Save in standardized ZYXC format
            logger.info(f"Saving in standardized ZYXC format: {data.shape}")
            self._save_as_tiff(data, output_path, metadata)

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
    
    def _extract_comprehensive_metadata(self, reader: nd2reader.ND2Reader) -> Dict[str, Any]:
        """Extract comprehensive metadata from ND2 reader.
        
        Args:
            reader: ND2Reader instance.
            
        Returns:
            Dictionary containing all relevant metadata.
        """
        metadata = {}
        
        # Basic metadata
        metadata['original_format'] = 'nd2'
        metadata['axes'] = reader.axes
        metadata['frame_shape'] = reader.frame_shape
        
        # Dimensional information
        metadata['dimensions'] = {
            'width': reader.metadata.get('width', reader.frame_shape[1] if len(reader.frame_shape) > 1 else None),
            'height': reader.metadata.get('height', reader.frame_shape[0] if len(reader.frame_shape) > 0 else None),
            'z_levels': len(reader.metadata.get('z_levels', [1])),
            'channels': len(reader.metadata.get('channels', [])) or 1,
            'timepoints': reader.metadata.get('total_images_per_channel', 1)
        }
        
        # Pixel/voxel information - CRITICAL for physical measurements
        metadata['pixel_info'] = {
            'pixel_microns': reader.metadata.get('pixel_microns'),
            'pixel_microns_x': reader.metadata.get('pixel_microns_x'),
            'pixel_microns_y': reader.metadata.get('pixel_microns_y'),
            'voxel_size_z': reader.metadata.get('z_step', None),
            'calibration': reader.metadata.get('calibration'),
        }
        
        # Calculate actual pixel size
        if metadata['pixel_info']['pixel_microns']:
            metadata['pixel_size_um'] = metadata['pixel_info']['pixel_microns']
        elif metadata['pixel_info']['pixel_microns_x'] and metadata['pixel_info']['pixel_microns_y']:
            # Use average if x and y are different
            metadata['pixel_size_um'] = (
                metadata['pixel_info']['pixel_microns_x'] + 
                metadata['pixel_info']['pixel_microns_y']
            ) / 2
        elif metadata['pixel_info']['calibration']:
            metadata['pixel_size_um'] = metadata['pixel_info']['calibration']
        else:
            metadata['pixel_size_um'] = None
            logger.warning("No pixel size information found in ND2 metadata")
        
        # Channel information
        if 'channels' in reader.metadata:
            metadata['channel_names'] = reader.metadata['channels']
        else:
            raise ValueError(
                "No channel information found in image file. Channel names are "
                "required for proper analysis. Please ensure your image acquisition "
                "software saves proper channel metadata."
            )
        
        # Z-stack information
        if 'z_levels' in reader.metadata:
            metadata['z_levels'] = reader.metadata['z_levels']
            metadata['z_coordinates'] = reader.metadata.get('z_coordinates', [])
        
        # Acquisition information
        metadata['acquisition'] = {
            'date': reader.metadata.get('date'),
            'time': reader.metadata.get('time'),
            'experiment': reader.metadata.get('experiment', {}),
        }
        
        # Microscope settings
        metadata['microscope'] = {
            'objective': reader.metadata.get('objective'),
            'magnification': reader.metadata.get('magnification'),
            'numerical_aperture': reader.metadata.get('numerical_aperture'),
            'immersion': reader.metadata.get('immersion'),
        }
        
        # Additional raw metadata (for completeness)
        metadata['raw_metadata'] = {
            k: v for k, v in reader.metadata.items() 
            if k not in ['channels', 'z_levels', 'pixel_microns', 'width', 'height']
            and not isinstance(v, (list, dict)) or len(str(v)) < 1000  # Avoid huge data structures
        }
        
        return metadata
    
    def _read_nd2_as_array(self, reader: nd2reader.ND2Reader) -> np.ndarray:
        """Read ND2 file as numpy array.
        
        Args:
            reader: ND2Reader instance.
            
        Returns:
            Image array with dimensions (Z, Y, X, C) or appropriate subset.
        """
        # Get metadata for proper channel extraction
        z_levels = len(reader.metadata.get('z_levels', [1]))
        channels = len(reader.metadata.get('channels', [])) or 1
        
        logger.info(f"Expected Z levels: {z_levels}, Channels: {channels}")
        logger.info(f"Reader axes: {reader.axes}")
        logger.info(f"Reader sizes: {reader.sizes}")
        logger.info(f"Available frames: {len(reader)}")
        logger.info(f"Frame shape: {reader[0].shape}")
        
        # Check if reader has channel iteration capability
        original_iter_axes = getattr(reader, 'iter_axes', [])
        try:
            reader.iter_axes = 'c'
            logger.info(f"Can iterate over channels: True")
        except Exception as e:
            logger.info(f"Can iterate over channels: False ({e})")
            
        try:
            reader.iter_axes = 'z'
            logger.info(f"Can iterate over z: True")
        except Exception as e:
            logger.info(f"Can iterate over z: False ({e})")
            
        # Reset to original
        try:
            reader.iter_axes = original_iter_axes
        except:
            pass
        
        # Manually extract all channels and z-slices to ensure we get all data
        try:
            if 'z' in reader.axes and 'c' in reader.axes:
                # Multi-channel, multi-z - use proper iteration
                logger.info("Using nd2reader iteration approach for multi-channel z-stack")
                
                # Set up iteration over both z and c
                reader.iter_axes = ['z', 'c']
                reader.bundle_axes = ['y', 'x']
                
                logger.info(f"Set iter_axes to {reader.iter_axes}")
                logger.info(f"Reader now has {len(reader)} frames after iteration setup")
                
                try:
                    all_data = []
                    frame_count = 0
                    
                    # Now iterate - should give us all Z×C combinations
                    for frame in reader:
                        all_data.append(frame)
                        frame_count += 1
                        
                        # Debug first few frames
                        if frame_count <= 10:
                            logger.info(f"Iteration frame {frame_count}: shape={frame.shape}, mean={frame.mean():.2f}")
                        
                        # Safety check to avoid infinite loop
                        if frame_count >= z_levels * channels * 2:  # 2x safety margin
                            logger.warning(f"Safety break at {frame_count} frames")
                            break
                    
                    logger.info(f"Iteration collected {len(all_data)} frames")
                    
                    if len(all_data) == z_levels * channels:
                        # Perfect: we have all Z×C frames
                        frame_array = np.array(all_data)  # (Z*C, Y, X)
                        
                        # Reshape to (Z, C, Y, X) then transpose to (Z, Y, X, C)
                        reshaped = frame_array.reshape(z_levels, channels, frame_array.shape[1], frame_array.shape[2])
                        result = np.transpose(reshaped, (0, 2, 3, 1))
                        
                        logger.info(f"Perfect iteration result shape: {result.shape}")
                        
                        # Validate different z-slices and channels
                        if result.shape[0] > 1:
                            z_means = [result[z, :, :, 0].mean() for z in range(min(5, result.shape[0]))]
                            logger.info(f"Z-slice means (ch0): {z_means}")
                            
                        if result.shape[3] > 1:
                            c_means = [result[0, :, :, c].mean() for c in range(result.shape[3])]
                            logger.info(f"Channel means (z0): {c_means}")
                            
                        return result
                        
                    elif len(all_data) == z_levels:
                        # Got z-slices but no channel separation - this is what's happening
                        logger.warning("Got z-slices but no channel separation - trying channel extraction per z")
                        
                        # Try to extract channels for each z-slice manually
                        multichannel_data = []
                        for z in range(z_levels):
                            z_channels = []
                            for c in range(channels):
                                try:
                                    # Set specific coordinates
                                    reader.default_coords = {'z': z, 'c': c, 't': 0}
                                    frame = reader[0]  # Get first frame with these coords
                                    z_channels.append(frame)
                                    
                                    if z == 0:  # Debug first z-slice
                                        logger.info(f"Manual z={z}, c={c}: shape={frame.shape}, mean={frame.mean():.2f}")
                                        
                                except Exception as e:
                                    logger.error(f"Manual extraction failed z={z}, c={c}: {e}")
                                    # Use the first available frame as fallback
                                    z_channels.append(all_data[0] if all_data else np.zeros((1800, 1800)))
                            
                            multichannel_data.append(np.stack(z_channels, axis=-1))
                        
                        result = np.stack(multichannel_data, axis=0)
                        logger.info(f"Manual channel extraction result shape: {result.shape}")
                        
                        # Validate
                        if result.shape[3] > 1:
                            c_means = [result[0, :, :, c].mean() for c in range(result.shape[3])]
                            logger.info(f"Manual channel means (z0): {c_means}")
                            
                        return result
                        
                    else:
                        logger.error(f"Unexpected number of frames from iteration: {len(all_data)}")
                        return np.array(all_data)
                        
                except Exception as e:
                    logger.error(f"Iteration approach failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Last resort fallback
                    logger.info("Using basic fallback")
                    frame = reader[0]
                    return frame[np.newaxis, ..., np.newaxis]
                
            elif 'c' in reader.axes:
                # Multi-channel, single z
                all_data = []
                for c in range(channels):
                    reader.default_coords['c'] = c
                    frame = reader[0]
                    all_data.append(frame)
                
                result = np.stack(all_data, axis=-1)[np.newaxis, ...]  # Add Z dimension
                logger.info(f"Extracted shape: {result.shape}")
                return result
                
            elif 'z' in reader.axes:
                # Single channel, multi-z
                all_data = []
                for z in range(z_levels):
                    reader.default_coords['z'] = z
                    frame = reader[0]
                    all_data.append(frame)
                
                result = np.stack(all_data, axis=0)[..., np.newaxis]  # Add C dimension
                logger.info(f"Extracted shape: {result.shape}")
                return result
                
            else:
                # Single channel, single z
                frame = reader[0]
                result = frame[np.newaxis, ..., np.newaxis]
                logger.info(f"Extracted shape: {result.shape}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to read ND2 data: {e}")
            raise ValueError(f"Could not read ND2 data: {e}")
    
    def _save_as_tiff(
        self, 
        image_data: np.ndarray, 
        output_path: Path, 
        metadata: Dict[str, Any]
    ) -> None:
        """Save image data as OME-TIFF with metadata.
        
        Args:
            image_data: 4D image array with ZYXC axis order.
            output_path: Output file path.
            metadata: Metadata dictionary to embed.
            
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
        
        # Build OME metadata
        ome_metadata = {
            'axes': STANDARD_AXES,
        }
        
        # Add channel information
        if c > 1:
            ome_metadata['Channel'] = [{'Name': name} for name in channel_names]
        else:
            ome_metadata['Channel'] = {'Name': channel_names[0]}
        
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
        
        logger.info(f"Saving 4D ZYXC image: shape={image_data.shape}, axes={STANDARD_AXES}")
        logger.info(f"Dimensions: Z={z}, Y={y}, X={x}, C={c}")
        
        try:
            tifffile.imwrite(
                output_path,
                image_data,
                ome=True,
                metadata=ome_metadata,
                compression='lzw'
            )
            logger.info(f"Successfully saved OME-TIFF: {output_path}")
        except Exception as e:
            logger.error(f"OME-TIFF save failed: {e}, trying fallback")
            # Fallback: save as regular TIFF with JSON description
            tifffile.imwrite(
                output_path,
                image_data,
                description=json.dumps(metadata, indent=2, default=str),
                compression='lzw'
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