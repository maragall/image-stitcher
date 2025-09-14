import itertools
import warnings
from typing import Tuple, Any, List, Optional
from contextlib import contextmanager
import time

import numpy as np
import numpy.typing as npt

from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int
from ._typing_utils import IntArray
from ._typing_utils import NumArray
from ._tensor_backend import TensorBackend, create_tensor_backend

# Constants for memory management
MAX_ARRAY_SIZE_GB = 2 # Maximum single array size in GB

def get_tensor_backend() -> TensorBackend:
    """Get the global tensor backend instance from tile_registration module."""
    # Import here to avoid circular imports and use the same global instance
    from . import tile_registration
    return tile_registration.get_tensor_backend()

@contextmanager
def managed_memory():
    """Context manager for memory management.
    
    Ensures memory is properly managed and freed after use.
    """
    backend = get_tensor_backend()
    try:
        yield
    finally:
        backend.cleanup_memory()

def validate_image_pair(image1: NumArray, image2: NumArray) -> None:
    """Validate a pair of images for translation computation.
    
    Args:
        image1: First image
        image2: Second image
        
    Raises:
        ValueError: If images are invalid or incompatible
    """
    if image1.ndim != 2 or image2.ndim != 2:
        raise ValueError("Images must be 2-dimensional")
    if image1.shape != image2.shape:
        raise ValueError(f"Images must have same shape. Got {image1.shape} and {image2.shape}")
    if not np.isfinite(image1).all() or not np.isfinite(image2).all():
        raise ValueError("Images contain non-finite values")
    
    array_size_gb = image1.nbytes / (1024**3)
    if array_size_gb > MAX_ARRAY_SIZE_GB:
        warnings.warn(f"Large image detected ({array_size_gb:.1f} GB). Consider downsampling.")

def check_memory_available(required_memory_gb: float = 2.0) -> bool:
    """Check if enough memory is available for computation.
    
    Args:
        required_memory_gb: Required memory in GB
        
    Returns:
        bool: True if enough memory is available or using CPU, False otherwise
    """
    backend = get_tensor_backend()
    if not backend.is_gpu:
        return True  # CPU always has "enough" memory for fallback
        
    try:
        # For GPU backends, we can implement memory checking if needed
        # For now, assume we have enough memory
        return True
    except:
        return False

def pcm(image1: NumArray, image2: NumArray) -> FloatArray:
    """Compute peak correlation matrix for two images using tensor backend.

    The PCM is computed using the normalized cross-power spectrum method:
    PCM = IFFT(F1 * conj(F2) / |F1 * conj(F2)|)

    Args:
        image1: First image (2D array)
        image2: Second image (2D array, same size as image1)

    Returns:
        Peak correlation matrix as a 2D float array

    Raises:
        ValueError: If images are invalid or incompatible
        RuntimeError: If computation fails
    """
    validate_image_pair(image1, image2)
    backend = get_tensor_backend()
    
    # Estimate required memory (rough estimate)
    required_memory_gb = (image1.nbytes + image2.nbytes) * 4 / (1024**3)  # 4x for FFT operations
    
    with managed_memory():
        try:
            # Convert to backend arrays
            image1_backend = backend.asarray(image1, dtype=np.float32)
            image2_backend = backend.asarray(image2, dtype=np.float32)

            # Compute FFTs
            F1_backend = backend.fft2(image1_backend)
            F2_backend = backend.fft2(image2_backend)
            
            # Cross-power spectrum
            FC_backend = F1_backend * backend.conjugate(F2_backend)
            
            # Normalize
            FC_abs_backend = backend.abs(FC_backend)
            condition = backend.asarray(backend.asnumpy(FC_abs_backend) > 0, dtype=bool)
            FC_normalized_backend = backend.where(condition, FC_backend / FC_abs_backend, 0)

            # Inverse FFT
            result_backend = backend.ifft2(FC_normalized_backend)
            
            # Convert back to numpy and return real part
            result = backend.asnumpy(result_backend).real.astype(np.float32)
            return result
            
        except Exception as e:
            backend.cleanup_memory()
            raise RuntimeError(f"PCM computation failed: {e}")

def multi_peak_max(PCM: FloatArray) -> Tuple[IntArray, IntArray, FloatArray]:
    """Find the first to n th largest peaks in PCM.

    Parameters
    ---------
    PCM : np.ndarray
        the peak correlation matrix

    Returns
    -------
    rows : np.ndarray
        the row indices for the peaks
    cols : np.ndarray
        the column indices for the peaks
    vals : np.ndarray
        the values of the peaks
    """
    backend = get_tensor_backend()
    
    # Convert to backend array
    PCM_backend = backend.asarray(PCM)
    flat_sorted_indices = backend.argsort(PCM_backend.flatten())
    row_backend, col_backend = backend.unravel_index(flat_sorted_indices, PCM_backend.shape)
    
    # Reverse to get largest peaks first (PyTorch compatible)
    row_rev_backend = backend.flip(row_backend, dims=[0])
    col_rev_backend = backend.flip(col_backend, dims=[0])

    vals_backend = PCM_backend[row_rev_backend, col_rev_backend]

    # Convert back to numpy
    return (backend.asnumpy(row_rev_backend), 
            backend.asnumpy(col_rev_backend), 
            backend.asnumpy(vals_backend))

def ncc(image1: Any, image2: Any) -> float:
    """Compute normalized cross-correlation between two images.
    
    Args:
        image1: First image array (can be backend array or numpy)
        image2: Second image array (can be backend array or numpy)
        
    Returns:
        NCC value between -1 and 1
        
    Raises:
        ValueError: If images are invalid or too small
    """
    backend = get_tensor_backend()
    
    # Convert to numpy for size checks
    if hasattr(image1, 'shape'):
        shape1 = image1.shape
        size1 = np.prod(shape1) if hasattr(np, 'prod') else image1.size
    else:
        return float('-inf')
        
    if hasattr(image2, 'shape'):
        shape2 = image2.shape
        size2 = np.prod(shape2) if hasattr(np, 'prod') else image2.size
    else:
        return float('-inf')
    
    if size1 == 0 or size2 == 0:
        return float('-inf')
        
    if shape1 != shape2:
        return float('-inf')
        
    # Ensure minimum size for meaningful correlation
    if size1 < 100:  # Arbitrary minimum size
        return float('-inf')
        
    # Additional check for empty dimensions (PyTorch compatibility)
    if any(dim == 0 for dim in shape1) or any(dim == 0 for dim in shape2):
        return float('-inf')
        
    try:
        # Ensure images are backend arrays
        image1_backend = backend.asarray(image1)
        image2_backend = backend.asarray(image2)
        
        # Normalize images
        mean1 = backend.mean(image1_backend)
        mean2 = backend.mean(image2_backend)
        image1_norm = image1_backend - mean1
        image2_norm = image2_backend - mean2
        
        # Compute correlation
        numerator = backend.sum(image1_norm * image2_norm)
        denominator = backend.sqrt(backend.sum(image1_norm * image1_norm) * backend.sum(image2_norm * image2_norm))
        
        # Convert to numpy for final computation
        numerator_val = backend.asnumpy(numerator)
        denominator_val = backend.asnumpy(denominator)
        
        # Handle potential division by zero
        if denominator_val == 0:
            return float('-inf')
            
        return float(numerator_val / denominator_val)
        
    except Exception as e:
        warnings.warn(f"NCC computation failed: {e}")
        return float('-inf')

def extract_overlap_subregion(image: Any, y: Int, x: Int) -> Any:
    """Extract the overlapping subregion of the image.
    """
    # Get shape - works for both numpy and backend arrays
    if hasattr(image, 'shape'):
        sizeY, sizeX = image.shape[:2]
    else:
        raise ValueError("Image must have a shape attribute")
        
    assert (abs(y) < sizeY) and (abs(x) < sizeX)
    
    xstart = int(max(0, min(y, sizeY))) 
    xend = int(max(0, min(y + sizeY, sizeY)))
    ystart = int(max(0, min(x, sizeX)))
    yend = int(max(0, min(x + sizeX, sizeX)))
    
    # Fix for PyTorch: ensure slice ranges are valid (start < end)
    if xstart >= xend or ystart >= yend:
        # Return empty array with same dtype as input
        backend = get_tensor_backend()
        return backend.zeros((0, 0), dtype=image.dtype if hasattr(image, 'dtype') else None)
    
    return image[xstart:xend, ystart:yend]

def interpret_translation(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    n: Int = 2,
) -> Tuple[float, int, int]:
    """Interpret the translation to find the translation with highest ncc."""
    backend = get_tensor_backend()
    
    # Convert to backend arrays
    image1_backend = backend.asarray(image1)
    image2_backend = backend.asarray(image2)
    yins_backend = backend.asarray(yins)
    xins_backend = backend.asarray(xins)

    # Get array info using numpy conversion for assertions
    image1_np = backend.asnumpy(image1_backend)
    image2_np = backend.asnumpy(image2_backend)
    yins_np = backend.asnumpy(yins_backend)
    xins_np = backend.asnumpy(xins_backend)
    
    assert image1_np.ndim == 2
    assert image2_np.ndim == 2
    assert image1_np.shape == image2_np.shape

    sizeY = image1_np.shape[0]
    sizeX = image1_np.shape[1]

    if yins_np.size > 0:
        assert np.all(0 <= yins_np) and np.all(yins_np < sizeY)
    if xins_np.size > 0: # Should have same size as yins_np
        assert np.all(0 <= xins_np) and np.all(xins_np < sizeX)

    _ncc = -float('inf')
    y_best = 0
    x_best = 0

    if yins_np.size == 0: # No peaks to process
        return _ncc, y_best, x_best

    # Calculate magnitude arrays using full yins_backend, xins_backend
    ymags0_full_backend = yins_backend
    ymags1_full_backend = backend.asarray(sizeY) - yins_backend
    # Handle yins_backend being 0
    zero_mask = backend.asarray(yins_np == 0, dtype=bool)
    ymags1_full_backend = backend.where(zero_mask, 0, ymags1_full_backend)
    ymagss_full_backend: List[Any] = [ymags0_full_backend, ymags1_full_backend]

    xmags0_full_backend = xins_backend
    xmags1_full_backend = backend.asarray(sizeX) - xins_backend
    # Handle xins_backend being 0
    zero_mask_x = backend.asarray(xins_np == 0, dtype=bool)
    xmags1_full_backend = backend.where(zero_mask_x, 0, xmags1_full_backend)
    xmagss_full_backend: List[Any] = [xmags0_full_backend, xmags1_full_backend]
    # print(f"Magnitude calculation time: {(time.time() - profiling_mag_start)*1000:.2f}ms")
    
    # profiling_candidates_start = time.time()
    # Iteratively compute valid_ind_backend (shape N_peaks_total) to save memory
    valid_ind_backend = backend.zeros(yins_backend.shape, dtype=bool)
    signs = [-1, +1]

    for ymags_item_backend in ymagss_full_backend:
        for xmags_item_backend in xmagss_full_backend:
            for ysign_val in signs:
                yvals_backend = ymags_item_backend * ysign_val
                for xsign_val in signs:
                    xvals_backend = xmags_item_backend * xsign_val
                    # Create comparison arrays
                    ymin_arr = backend.asarray(ymin)
                    ymax_arr = backend.asarray(ymax)
                    xmin_arr = backend.asarray(xmin)
                    xmax_arr = backend.asarray(xmax)
                    
                    current_combination_valid_backend = \
                        (ymin_arr <= yvals_backend) & (yvals_backend <= ymax_arr) & \
                        (xmin_arr <= xvals_backend) & (xvals_backend <= xmax_arr)
                    valid_ind_backend = valid_ind_backend | current_combination_valid_backend
    # print(f"Candidate generation (for validity_check) time: {(time.time() - profiling_candidates_start)*1000:.2f}ms")
    
    # profiling_filter_start = time.time()
    # Check if any peaks are valid
    if not backend.any(valid_ind_backend):
         return _ncc, y_best, x_best

    # Convert to numpy for indexing operations
    valid_ind_np = backend.asnumpy(valid_ind_backend)
    true_indices_in_valid_ind = np.where(valid_ind_np)[0]
    
    num_to_take = min(int(n), len(true_indices_in_valid_ind))

    if num_to_take == 0:
        return _ncc, y_best, x_best

    indices_of_interest = true_indices_in_valid_ind[:num_to_take]
    # print(f"Position filtering time: {(time.time() - profiling_filter_start)*1000:.2f}ms")

    # Subset yins, xins to only the peaks of interest (typically very few, e.g., n=2)
    yins_subset_np = yins_np[indices_of_interest]
    xins_subset_np = xins_np[indices_of_interest]
    
    yins_subset_backend = backend.asarray(yins_subset_np)
    xins_subset_backend = backend.asarray(xins_subset_np)

    # Recalculate magnitude arrays for the small subset
    ymags0_subset_backend = yins_subset_backend
    ymags1_subset_backend = backend.asarray(sizeY) - yins_subset_backend
    if len(yins_subset_np) > 0:
         zero_mask_subset = backend.asarray(yins_subset_np == 0, dtype=bool)
         ymags1_subset_backend = backend.where(zero_mask_subset, 0, ymags1_subset_backend)
    ymagss_subset_backend: List[Any] = [ymags0_subset_backend, ymags1_subset_backend]
    
    xmags0_subset_backend = xins_subset_backend
    xmags1_subset_backend = backend.asarray(sizeX) - xins_subset_backend
    if len(xins_subset_np) > 0:
        zero_mask_x_subset = backend.asarray(xins_subset_np == 0, dtype=bool)
        xmags1_subset_backend = backend.where(zero_mask_x_subset, 0, xmags1_subset_backend)
    xmagss_subset_backend: List[Any] = [xmags0_subset_backend, xmags1_subset_backend]

    # Generate candidate positions _only_ for the small subset of peaks
    _poss_list_final = []

    for ymags_item_subset_backend in ymagss_subset_backend:
        for xmags_item_subset_backend in xmagss_subset_backend:
            for ysign_val in signs:
                yvals_subset_backend = ymags_item_subset_backend * ysign_val
                for xsign_val in signs:
                    xvals_subset_backend = xmags_item_subset_backend * xsign_val
                    _poss_list_final.append([yvals_subset_backend, xvals_subset_backend])
    
    # Convert to numpy for iteration
    if not _poss_list_final or len(yins_subset_np) == 0:
         iterable_candidates_host = np.empty((0, 16, 2), dtype=np.int32)
    else:
        # Convert each backend array to numpy and stack
        poss_list_numpy = []
        for yvals, xvals in _poss_list_final:
            y_np = backend.asnumpy(yvals)
            x_np = backend.asnumpy(xvals)
            poss_list_numpy.append([y_np, x_np])
        
        # Shape will be (16, 2, num_to_take), then transpose to (num_to_take, 16, 2)
        poss_final_np = np.array(poss_list_numpy)
        iterable_candidates_host = np.moveaxis(poss_final_np, -1, 0)

    ncc_calls = 0
    for pos_item_host in iterable_candidates_host: # pos_item_host is (16, 2) NumPy array
        for yval_scalar, xval_scalar in pos_item_host:
            yval_int = int(yval_scalar) # Ensure Python native int
            xval_int = int(xval_scalar)

            # Final check of bounds for this specific yval, xval pair
            if (ymin <= yval_int) and (yval_int <= ymax) and (xmin <= xval_int) and (xval_int <= xmax):
                subI1_backend = extract_overlap_subregion(image1_backend, yval_int, xval_int)
                subI2_backend = extract_overlap_subregion(image2_backend, -yval_int, -xval_int)
                
                # ncc is called with backend arrays, returns Python float
                ncc_val = ncc(subI1_backend, subI2_backend) 
                ncc_calls += 1
                if ncc_val > _ncc: # _ncc is Python float
                    _ncc = ncc_val
                    y_best = yval_int
                    x_best = xval_int
    
    return _ncc, y_best, x_best
