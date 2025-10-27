"""Image Stitcher Package.

This package provides tools for stitching microscope image tiles with support
for flatfield correction, tile registration, and z-layer selection.

Main functionality:
- Image stitching: Combine image tiles into panoramic images
- Tile registration: Automated and manual alignment of image tiles
- Flatfield correction: Correct illumination artifacts
- Z-layer selection: Optimal focal plane selection
- Multiple tensor backends for CPU and GPU acceleration

The package exposes key registration functions at the top level for convenience.
"""

from .registration.tile_registration import (
    register_tiles,
    RegistrationMode,
    register_tiles_batched,
    register_and_update_coordinates,
    process_multiple_timepoints,
    DEFAULT_FOV_RE,
    extract_tile_indices,
    get_tensor_backend,
    set_tensor_backend
)
from .registration._constrained_refinement import refine_translations
from .registration._global_optimization import compute_final_position, TileGrid
from .registration._tensor_backend import create_tensor_backend, TensorBackend
from .registration._translation_computation import compute_translation, TranslationStrategy

# Manual registration GUI is available as a submodule
# from . import manual

__all__ = [
    'register_tiles',
    'RegistrationMode',
    'register_tiles_batched',
    'register_and_update_coordinates', 
    'process_multiple_timepoints',
    'refine_translations',
    'compute_final_position',
    'TileGrid',
    'DEFAULT_FOV_RE',
    'extract_tile_indices',
    'get_tensor_backend',
    'set_tensor_backend',
    'create_tensor_backend',
    'TensorBackend',
    'compute_translation',
    'TranslationStrategy'
] 
