"""Registration module for image stitching.

This module provides functionality for registering microscope image tiles
without performing full stitching. Supports multiple tensor backends for
CPU and GPU acceleration.

Submodules:
- tile_registration: Core automated registration algorithms
- manual: GUI-based manual alignment tools
- _global_optimization: Global position optimization
- _tensor_backend: CPU/GPU tensor backend management
- _translation_computation: Translation computation strategies
"""

from .tile_registration import (
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
from ._constrained_refinement import refine_translations
from ._global_optimization import compute_final_position, TileGrid
from ._tensor_backend import create_tensor_backend, TensorBackend
from ._translation_computation import compute_translation, TranslationStrategy

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
