import logging
import random
from typing import Callable

import numpy as np
from basicpy import BaSiC

from .parameters import StitchingComputedParameters

MAX_FLATFIELD_IMAGES_PER_T = 32
MAX_FLATFIELD_IMAGES = 48


def _load_tile_image(tile_info) -> np.ndarray:
    """Load a single tile image using the appropriate loader.
    
    This handles different file formats (regular TIFF, multi-page TIFF, OME-TIFF)
    by checking for frame_idx metadata and using the appropriate loading method.
    
    Parameters
    ----------
    tile_info : AcquisitionMetadata
        Tile metadata containing filepath and frame_idx information
        
    Returns
    -------
    np.ndarray
        2D image array of shape (height, width)
    """
    if hasattr(tile_info, 'frame_idx') and tile_info.frame_idx is not None:
        # Check if it's an OME-TIFF (frame_idx is tuple) or multi-page TIFF (frame_idx is int)
        if isinstance(tile_info.frame_idx, tuple):
            # OME-TIFF: frame_idx is (channel_idx, z_idx)
            from .image_loaders import create_image_loader
            loader = create_image_loader(tile_info.filepath, format_hint='ome_tiff')
            channel_idx, z_idx = tile_info.frame_idx
            return loader.read_slice(channel=channel_idx, z=z_idx)
        elif tile_info.frame_idx > 0:
            # Multi-page TIFF file: frame_idx is page number
            import tifffile
            with tifffile.TiffFile(tile_info.filepath) as tif:
                return tif.pages[tile_info.frame_idx].asarray()
        else:
            # frame_idx is 0, treat as single file
            import skimage.io
            return skimage.io.imread(tile_info.filepath)
    else:
        # Single file without frame_idx
        import skimage.io
        return skimage.io.imread(tile_info.filepath)


def compute_flatfield_correction(
    computed_params: StitchingComputedParameters, progress_callback: Callable[[], None]
) -> dict[int, np.ndarray]:
    """Compute a flatfield correction across many images for each channel.

    This does not modify computed_params.flatfields; store the output there if
    desired external to this function.
    """

    flatfields = {}

    def process_images(images: np.ndarray, channel_name: str):
        if images.size == 0:
            logging.warning(f"No images found for channel {channel_name}")
            return

        if images.ndim != 3:
            raise ValueError(
                f"Images must be 3-dimensional array, with dimension of (N, Y, X). Got shape {images.shape}"
            )

        basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
        basic.fit(images)
        channel_index = computed_params.monochrome_channels.index(channel_name)
        flatfields[channel_index] = basic.flatfield
        if progress_callback:
            progress_callback()

    for channel in computed_params.channel_names:
        logging.info(f"Calculating {channel} flatfield...")

        all_tile_infos = []
        for t in computed_params.timepoints:
            # We might have a ton of images for this time point.  We only want to select up to MAX_FLATFIELD_IMAGES
            # to use for flatfield correction.  To do this, load all the metadata and pick random tile infos
            # to use for this timepoint.
            all_tile_infos.extend(
                [
                    tile
                    for key, tile in computed_params.acquisition_metadata.items()
                    if tile.channel == channel and key.t == int(t)
                ]
            )

        random.shuffle(all_tile_infos)
        selected_tile_infos = all_tile_infos[
            : min(MAX_FLATFIELD_IMAGES, len(all_tile_infos))
        ]

        if len(selected_tile_infos) < len(all_tile_infos):
            logging.warning(
                f"Limiting flatfield correction to {len(selected_tile_infos)} images instead of using all {len(all_tile_infos)}"
            )

        # Load images using the appropriate loader for each file type
        images = []
        for tile in selected_tile_infos:
            try:
                img = _load_tile_image(tile)
                # Ensure 2D images are consistently shaped
                if img.ndim == 2:
                    images.append(img)
                else:
                    logging.warning(
                        f"Loaded image from {tile.filepath} has unexpected shape {img.shape}, skipping"
                    )
            except Exception as e:
                logging.warning(f"Failed to load image from {tile.filepath}: {e}")
                continue

        if not images:
            logging.warning(f"No images found for channel {channel} for any timepoint")
            continue

        # Stack images into a 3D array (N, Y, X)
        images = np.array(images)

        if images.ndim == 3:
            # Images are in the expected shape (N, Y, X)
            process_images(images, channel)
        else:
            raise ValueError(
                f"Unexpected number of dimensions in images array: {images.ndim}. Expected 3D array of shape (N, Y, X)."
            )

    return flatfields
