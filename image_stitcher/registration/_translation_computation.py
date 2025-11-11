import itertools
import warnings
from enum import Enum
from typing import Tuple, Any, List, Optional, Callable
from contextlib import contextmanager
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares, OptimizeResult

try:
    from skimage.registration import phase_cross_correlation
    from skimage.measure import ransac
    from skimage.transform import EuclideanTransform, AffineTransform
    SKIMAGE_AVAILABLE = True
    RANSAC_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    RANSAC_AVAILABLE = False
    warnings.warn("scikit-image not available. Subpixel phase correlation and RANSAC disabled.", ImportWarning)

from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int
from ._typing_utils import IntArray
from ._typing_utils import NumArray
from ._tensor_backend import TensorBackend, create_tensor_backend

# Constants for memory management
MAX_ARRAY_SIZE_GB = 2 # Maximum single array size in GB


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class TranslationStrategy(Enum):
    """Strategy for computing translation between image pairs."""
    BASIC = "basic"           # Basic peak finding with NCC
    SUBPIXEL = "subpixel"     # Subpixel phase correlation
    OPTIMIZED = "optimized"   # Continuous optimization refinement
    RANSAC = "ransac"         # RANSAC outlier rejection


@dataclass
class SubpixelConfig:
    """Configuration for subpixel phase correlation refinement."""
    upsample_factor: int = 100
    use_subpixel: bool = True
    reference_mask: Optional[np.ndarray] = None
    moving_mask: Optional[np.ndarray] = None
    overlap_ratio: float = 0.3
    space: str = 'real'
    normalization: str = 'phase'


@dataclass
class OptimizationConfig:
    """Configuration for advanced optimization methods."""
    method: str = 'trf'
    max_nfev: int = 100
    ftol: float = 1e-6
    xtol: float = 1e-6
    gtol: float = 1e-6
    verbose: int = 0


@dataclass
class RANSACConfig:
    """Configuration for RANSAC outlier rejection."""
    use_ransac: bool = True
    min_samples: int = 3
    residual_threshold: float = 2.0
    max_trials: int = 1000
    stop_probability: float = 0.99
    min_inlier_ratio: float = 0.3
    transform_type: str = 'euclidean'
    return_inlier_mask: bool = True


@dataclass
class _TranslationBounds:
    """Encapsulates translation bounds for cleaner parameter passing."""
    ymin: int
    ymax: int
    xmin: int
    xmax: int
    
    def contains(self, y: int, x: int) -> bool:
        """Check if translation (y, x) is within bounds."""
        return (self.ymin <= y <= self.ymax) and (self.xmin <= x <= self.xmax)


class _MagnitudeCalculator:
    """Handles magnitude calculations with proper broadcasting."""
    
    def __init__(self, backend, size_y: int, size_x: int):
        self.backend = backend
        self.size_y = size_y
        self.size_x = size_x
        
    def calculate_y_magnitudes(self, yins_backend) -> Tuple[Any, Any]:
        """Calculate Y magnitudes with proper broadcasting."""
        # Forward magnitudes: distance from top edge
        ymags0 = yins_backend
        
        # Backward magnitudes: distance from bottom edge
        # Create proper broadcast-compatible array using zeros + scalar addition
        size_y_array = self.backend.zeros(yins_backend.shape, dtype=yins_backend.dtype) + self.size_y
        ymags1 = size_y_array - yins_backend
        
        return ymags0, ymags1
    
    def calculate_x_magnitudes(self, xins_backend) -> Tuple[Any, Any]:
        """Calculate X magnitudes with proper broadcasting."""
        # Forward magnitudes: distance from left edge  
        xmags0 = xins_backend
        
        # Backward magnitudes: distance from right edge
        size_x_array = self.backend.zeros(xins_backend.shape, dtype=xins_backend.dtype) + self.size_x
        xmags1 = size_x_array - xins_backend
        
        return xmags0, xmags1


class _CandidateGenerator:
    """Generates translation candidates efficiently."""
    
    def __init__(self, backend):
        self.backend = backend
        self.signs = [-1, +1]
    
    def generate_validity_mask(self, yins_backend, xins_backend, 
                             mag_calc: _MagnitudeCalculator, 
                             bounds: _TranslationBounds) -> Any:
        """Generate boolean mask for valid translation candidates."""
        # Pre-convert bounds to backend arrays (avoid repeated conversions)
        bounds_arrays = self._convert_bounds_to_arrays(bounds, yins_backend)
        
        # Get magnitude arrays
        ymags0, ymags1 = mag_calc.calculate_y_magnitudes(yins_backend)
        xmags0, xmags1 = mag_calc.calculate_x_magnitudes(xins_backend)
        
        ymagss = [ymags0, ymags1]
        xmagss = [xmags0, xmags1]
        
        # Initialize validity mask
        valid_mask = self.backend.zeros(yins_backend.shape, dtype=bool)
        
        # Check all sign combinations
        for ymags in ymagss:
            for xmags in xmagss:
                for ysign in self.signs:
                    yvals = ymags * ysign
                    for xsign in self.signs:
                        xvals = xmags * xsign
                        
                        # Check bounds efficiently
                        current_valid = (
                            (bounds_arrays.ymin <= yvals) & (yvals <= bounds_arrays.ymax) &
                            (bounds_arrays.xmin <= xvals) & (xvals <= bounds_arrays.xmax)
                        )
                        valid_mask = valid_mask | current_valid
                        
        return valid_mask
    
    def _convert_bounds_to_arrays(self, bounds: _TranslationBounds, reference_array) -> Any:
        """Convert bounds to backend arrays for efficient comparison."""
        @dataclass
        class _BoundsArrays:
            ymin: Any
            ymax: Any  
            xmin: Any
            xmax: Any
            
        return _BoundsArrays(
            ymin=self.backend.asarray(bounds.ymin),
            ymax=self.backend.asarray(bounds.ymax),
            xmin=self.backend.asarray(bounds.xmin),
            xmax=self.backend.asarray(bounds.xmax)
        )
    
    def generate_candidate_positions(self, yins_subset, xins_subset, 
                                   mag_calc: _MagnitudeCalculator) -> np.ndarray:
        """Generate candidate positions with correct array shapes."""
        ymags0, ymags1 = mag_calc.calculate_y_magnitudes(yins_subset)
        xmags0, xmags1 = mag_calc.calculate_x_magnitudes(xins_subset)
        
        ymagss = [ymags0, ymags1]
        xmagss = [xmags0, xmags1]
        
        candidates = []
        
        for ymags in ymagss:
            for xmags in xmagss:
                for ysign in self.signs:
                    yvals = ymags * ysign
                    for xsign in self.signs:
                        xvals = xmags * xsign
                        # Convert to numpy for final processing
                        y_np = self.backend.asnumpy(yvals)
                        x_np = self.backend.asnumpy(xvals)
                        candidates.append(np.column_stack([y_np, x_np]))
        
        if not candidates:
            return np.empty((0, 2), dtype=np.int32)
        
        # Stack all candidates: shape will be (n_peaks, n_combinations * 2)
        # where n_combinations = 16 (2*2*2*2)
        all_candidates = np.concatenate(candidates, axis=1)
        
        # Reshape to (n_peaks, 16, 2) for iteration
        n_peaks = all_candidates.shape[0]
        return all_candidates.reshape(n_peaks, -1, 2)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    required_memory_gb = (image1.nbytes + image2.nbytes) * 8 / (1024**3)  # 8x for float64 FFT operations
    
    with managed_memory():
        try:
            # Convert to backend arrays (float64 for sub-pixel accuracy)
            image1_backend = backend.asarray(image1, dtype=np.float64)
            image2_backend = backend.asarray(image2, dtype=np.float64)

            # Compute FFTs
            F1_backend = backend.fft2(image1_backend)
            F2_backend = backend.fft2(image2_backend)
            
            # Cross-power spectrum
            FC_backend = F1_backend * backend.conjugate(F2_backend)
            
            # Normalize with epsilon for numerical stability
            FC_abs_backend = backend.abs(FC_backend)
            epsilon = np.finfo(np.float64).eps * 100  # ~2.22e-14
            FC_normalized_backend = FC_backend / (FC_abs_backend + epsilon)

            # Inverse FFT
            result_backend = backend.ifft2(FC_normalized_backend)
            
            # Convert back to numpy and validate result
            result_np = backend.asnumpy(result_backend)
            
            # Validate imaginary part is negligible
            max_imag = np.max(np.abs(result_np.imag))
            if max_imag > np.finfo(np.float64).eps * 1000:
                warnings.warn(f"Large imaginary component in PCM result: {max_imag:.2e}")
            
            # Return real part as float64 for sub-pixel accuracy
            result = result_np.real
            return result
            
        except Exception as e:
            backend.cleanup_memory()
            raise RuntimeError(f"PCM computation failed: {e}")

def multi_peak_max(
    PCM: FloatArray, 
    max_peaks: Optional[int] = None,
    validate_input: bool = True
) -> Tuple[IntArray, IntArray, FloatArray]:
    """Find the largest peaks in a Peak Correlation Matrix (PCM).
    
    This function identifies correlation peaks in descending order of magnitude,
    returning their coordinates and values. Supports multiple tensor backends
    for CPU/GPU acceleration with automatic memory management.

    Parameters
    ----------
    PCM : FloatArray
        Peak correlation matrix as a 2D floating-point array. Must be finite
        and non-empty. Typically the output of phase correlation or similar
        correlation analysis.
    max_peaks : int, optional
        Maximum number of peaks to return. If None, returns all peaks sorted
        by magnitude. Must be positive if specified. Default: None.
    validate_input : bool, optional
        Whether to perform comprehensive input validation. Disable only for
        performance-critical code where inputs are guaranteed valid. 
        Default: True.

    Returns
    -------
    rows : IntArray
        Row indices of peaks in descending order of magnitude. Shape: (n_peaks,)
    cols : IntArray  
        Column indices of peaks in descending order of magnitude. Shape: (n_peaks,)
    vals : FloatArray
        Peak values in descending order of magnitude. Shape: (n_peaks,)
        
    Raises
    ------
    ValueError
        If PCM is invalid (wrong dimensions, empty, non-numeric, etc.)
    TypeError
        If PCM is not array-like or max_peaks is not integer
    RuntimeError
        If backend operations fail or memory allocation fails
        
    Notes
    -----
    - Uses the globally configured tensor backend (numpy/cupy/torch)
    - Automatically manages GPU memory if using CUDA backends
    - Preserves maximum numerical precision (float64 internally)
    - Peak ordering is stable for equal values
    """
    # Input validation with detailed error messages
    if validate_input:
        _validate_pcm_input(PCM)
        if max_peaks is not None:
            _validate_max_peaks(max_peaks, PCM.size)
    
    # Get backend with error handling
    try:
        backend = get_tensor_backend()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize tensor backend: {e}") from e
    
    # Use memory management context for automatic cleanup
    with managed_memory():
        try:
            # Convert to backend array with explicit dtype for precision
            PCM_backend = backend.asarray(PCM, dtype=np.float64)
            
            # Use numpy for utilities (works with all array types)
            pcm_np = backend.asnumpy(PCM_backend)
            pcm_shape = pcm_np.shape
            pcm_flat = pcm_np.flatten()
            
            # Numpy operations for sorting and indexing
            flat_sorted_indices = np.argsort(pcm_flat)
            row_np, col_np = np.unravel_index(flat_sorted_indices, pcm_shape)
            
            # Reverse to get largest peaks first (descending order)
            # Make copies to avoid negative stride issues
            row_rev_np = np.flip(row_np).copy().astype(np.int64)
            col_rev_np = np.flip(col_np).copy().astype(np.int64)

            # Apply peak limit if specified (before indexing for efficiency)
            if max_peaks is not None and max_peaks > 0:
                effective_max = min(max_peaks, PCM.size)
                row_rev_np = row_rev_np[:effective_max]
                col_rev_np = col_rev_np[:effective_max]

            # Extract peak values using numpy indexing (works with all array types)
            pcm_array_for_indexing = backend.asnumpy(PCM_backend)
            vals_np = pcm_array_for_indexing[row_rev_np, col_rev_np].astype(np.float64)
            
            rows_np = row_rev_np.astype(np.int64, copy=False)
            cols_np = col_rev_np.astype(np.int64, copy=False)
            
            return rows_np, cols_np, vals_np
            
        except Exception as e:
            # Memory cleanup is handled by managed_memory() context
            raise RuntimeError(f"Peak finding computation failed: {e}") from e


def _validate_pcm_input(PCM: Any) -> None:
    """Validate PCM input with comprehensive error checking.
    
    Parameters
    ----------
    PCM : Any
        Input to validate as a valid PCM array
        
    Raises
    ------
    TypeError
        If PCM is not array-like
    ValueError
        If PCM has invalid properties (shape, dtype, values, etc.)
    """
    # Check for None
    if PCM is None:
        raise TypeError("PCM cannot be None")
    
    # Check for array-like interface
    if not hasattr(PCM, 'shape') or not hasattr(PCM, 'dtype'):
        raise TypeError(f"PCM must be array-like with 'shape' and 'dtype' attributes, got {type(PCM)}")
    
    # Check dimensions
    if PCM.ndim != 2:
        raise ValueError(f"PCM must be 2-dimensional, got {PCM.ndim}D array with shape {PCM.shape}")
    
    # Check size
    if PCM.size == 0:
        raise ValueError("PCM cannot be empty")
    
    # Check for minimum size (correlation matrices should be at least 2x2)
    if min(PCM.shape) < 2:
        raise ValueError(f"PCM dimensions too small: {PCM.shape}. Minimum size is 2x2 for meaningful correlation analysis")
    
    # Check dtype compatibility
    if not np.issubdtype(PCM.dtype, np.number):
        raise ValueError(f"PCM must be numeric, got dtype {PCM.dtype}")
    
    # Warn about non-floating point types (precision concern)
    if not np.issubdtype(PCM.dtype, np.floating):
        warnings.warn(
            f"PCM has integer dtype {PCM.dtype}. Using floating-point types "
            f"(float32/float64) is recommended for correlation analysis precision.",
            UserWarning,
            stacklevel=3
        )
    
    # Check for finite values
    try:
        if not np.all(np.isfinite(PCM)):
            raise ValueError("PCM contains non-finite values (NaN or infinity)")
    except (TypeError, ValueError) as e:
        # Handle cases where isfinite fails (e.g., complex dtypes)
        warnings.warn(f"Could not validate finite values in PCM: {e}", UserWarning, stacklevel=3)


def _validate_max_peaks(max_peaks: Any, pcm_size: int) -> None:
    """Validate max_peaks parameter.
    
    Parameters
    ----------
    max_peaks : Any
        Value to validate as max_peaks parameter
    pcm_size : int
        Size of the PCM array for bounds checking
        
    Raises
    ------
    TypeError
        If max_peaks is not integer-like
    ValueError
        If max_peaks has invalid value
    """
    # Type check
    if not isinstance(max_peaks, (int, np.integer)):
        raise TypeError(f"max_peaks must be an integer, got {type(max_peaks)}")
    
    # Value check
    if max_peaks <= 0:
        raise ValueError(f"max_peaks must be positive, got {max_peaks}")
    
    # Reasonable bounds check
    if max_peaks > pcm_size:
        warnings.warn(
            f"max_peaks ({max_peaks}) exceeds PCM size ({pcm_size}). "
            f"Will return all {pcm_size} peaks.",
            UserWarning,
            stacklevel=3
        )

def ncc(image1: Any, image2: Any, min_overlap_pixels: int = 25) -> float:
    """Compute normalized cross-correlation between two images.
    
    Uses the standard NCC formula: NCC = Σ((I1 - μ1)(I2 - μ2)) / √(Σ(I1 - μ1)² × Σ(I2 - μ2)²)
    where μ1, μ2 are the means of the images.
    
    Args:
        image1: First image array (can be backend array or numpy)
        image2: Second image array (can be backend array or numpy)  
        min_overlap_pixels: Minimum number of pixels required for meaningful correlation
        
    Returns:
        NCC value between -1 and 1, or -inf for invalid inputs
        
    Raises:
        ValueError: If images have incompatible shapes or invalid data
    """
    backend = get_tensor_backend()
    
    # Validate inputs exist and have shape attribute
    if not _has_valid_shape(image1) or not _has_valid_shape(image2):
        return float('-inf')
    
    # Check shape compatibility
    if image1.shape != image2.shape:
        return float('-inf')
    
    # Check for empty arrays
    size = int(np.prod(image1.shape))
    if size == 0:
        return float('-inf')
    
    # Check minimum overlap requirement
    if size < min_overlap_pixels:
        return float('-inf')
    
    try:
        # Convert to backend arrays with consistent dtype
        image1_backend = backend.asarray(image1, dtype=np.float64)
        image2_backend = backend.asarray(image2, dtype=np.float64)
        
        # Compute means
        mean1 = backend.mean(image1_backend)
        mean2 = backend.mean(image2_backend)
        
        # Center the images (subtract means)
        centered1 = image1_backend - mean1
        centered2 = image2_backend - mean2
        
        # Compute correlation components
        numerator = backend.sum(centered1 * centered2)
        var1 = backend.sum(centered1 * centered1)
        var2 = backend.sum(centered2 * centered2)
        
        # Convert to numpy for final computation and validation
        num_val = float(backend.asnumpy(numerator))
        var1_val = float(backend.asnumpy(var1))
        var2_val = float(backend.asnumpy(var2))
        
        # Compute final NCC with division by zero protection
        denominator = np.sqrt(var1_val * var2_val)
        
        # Handle division by zero (constant images)
        if denominator == 0.0 or not np.isfinite(denominator):
            # If both images are constant and identical, perfect correlation
            if var1_val == 0.0 and var2_val == 0.0 and np.isclose(num_val, 0.0):
                return 1.0
            # Otherwise, no meaningful correlation
            return float('-inf')
            
        ncc_value = num_val / denominator
        
        # Handle NaN or infinite results
        if not np.isfinite(ncc_value):
            return float('-inf')
        
        # Clamp to valid range (handle numerical precision issues)
        return float(np.clip(ncc_value, -1.0, 1.0))
        
    except Exception as e:
        warnings.warn(f"NCC computation failed: {e}", RuntimeWarning)
        return float('-inf')


def _has_valid_shape(array: Any) -> bool:
    """Check if array has a valid shape attribute."""
    return hasattr(array, 'shape') and hasattr(array, 'size')

def extract_overlap_subregion(image: Any, y: Int, x: Int) -> Any:
    """Extract the overlapping subregion of the image.
    
    Args:
        image: Input image array
        y: Y translation offset
        x: X translation offset
        
    Returns:
        Overlapping subregion, or empty array if no valid overlap exists
    """
    # Get shape - works for both numpy and backend arrays
    if hasattr(image, 'shape'):
        sizeY, sizeX = image.shape[:2]
    else:
        raise ValueError("Image must have a shape attribute")
    
    # Handle extreme translation values gracefully instead of asserting
    if abs(y) >= sizeY or abs(x) >= sizeX:
        # No possible overlap - return empty array
        backend = get_tensor_backend()
        return backend.zeros((0, 0), dtype=image.dtype if hasattr(image, 'dtype') else None)
    
    ystart = int(max(0, min(y, sizeY))) 
    yend = int(max(0, min(y + sizeY, sizeY)))
    xstart = int(max(0, min(x, sizeX)))
    xend = int(max(0, min(x + sizeX, sizeX)))
    
    # Fix for PyTorch: ensure slice ranges are valid (start < end)
    if xstart >= xend or ystart >= yend:
        # Return empty array with same dtype as input
        backend = get_tensor_backend()
        return backend.zeros((0, 0), dtype=image.dtype if hasattr(image, 'dtype') else None)
    
    return image[ystart:yend, xstart:xend]


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
    """
    Interpret the translation to find the translation with highest ncc.
    
    Fixed implementation with proper error handling and broadcasting.
    
    Args:
        image1: First image array
        image2: Second image array
        yins: Y coordinates of peaks
        xins: X coordinates of peaks
        ymin: Minimum Y translation bound
        ymax: Maximum Y translation bound
        xmin: Minimum X translation bound
        xmax: Maximum X translation bound
        n: Maximum number of peaks to evaluate
        
    Returns:
        Tuple of (best_ncc, best_y_translation, best_x_translation)
    """
    backend = get_tensor_backend()
    
    # Convert to backend arrays
    image1_backend = backend.asarray(image1)
    image2_backend = backend.asarray(image2)
    yins_backend = backend.asarray(yins)
    xins_backend = backend.asarray(xins)

    # Basic validation
    image1_np = backend.asnumpy(image1_backend)
    image2_np = backend.asnumpy(image2_backend)
    yins_np = backend.asnumpy(yins_backend)
    xins_np = backend.asnumpy(xins_backend)
    
    assert image1_np.ndim == 2, "Images must be 2D"
    assert image2_np.ndim == 2, "Images must be 2D"
    assert image1_np.shape == image2_np.shape, "Images must have same shape"

    size_y, size_x = image1_np.shape
    bounds = _TranslationBounds(int(ymin), int(ymax), int(xmin), int(xmax))

    # Validate peak coordinates
    if yins_np.size > 0:
        assert np.all(0 <= yins_np) and np.all(yins_np < size_y), "Y peaks out of bounds"
    if xins_np.size > 0:
        assert np.all(0 <= xins_np) and np.all(xins_np < size_x), "X peaks out of bounds"

    # Early return for empty input
    if yins_np.size == 0:
        return -float('inf'), 0, 0

    # Initialize helper classes
    mag_calc = _MagnitudeCalculator(backend, size_y, size_x)
    candidate_gen = _CandidateGenerator(backend)

    # Generate validity mask for all peaks
    valid_mask = candidate_gen.generate_validity_mask(
        yins_backend, xins_backend, mag_calc, bounds
    )

    # Check if any peaks are valid (use numpy for utility)
    valid_mask_np = backend.asnumpy(valid_mask)
    if not np.any(valid_mask_np):
        return -float('inf'), 0, 0

    # Select top n valid peaks
    # Convert to numpy to find valid indices (backend.where only supports conditional assignment)
    valid_mask_np = backend.asnumpy(valid_mask)
    valid_indices = np.where(valid_mask_np)[0]
    num_to_take = min(int(n), len(valid_indices))

    if num_to_take == 0:
        return -float('inf'), 0, 0

    selected_indices = valid_indices[:num_to_take]

    # Generate candidates for selected peaks
    yins_subset = backend.asarray(yins_np[selected_indices])
    xins_subset = backend.asarray(xins_np[selected_indices])

    candidates = candidate_gen.generate_candidate_positions(
        yins_subset, xins_subset, mag_calc
    )

    # Find best translation by evaluating NCC for all valid candidates
    best_ncc = -float('inf')
    best_y = 0
    best_x = 0

    # Flatten candidates for efficient iteration
    flat_candidates = candidates.reshape(-1, 2)

    # Early return if no candidates
    if flat_candidates.size == 0:
        return -float('inf'), 0, 0

    for y_val, x_val in flat_candidates:
        y_int, x_int = int(y_val), int(x_val)

        # Final bounds check
        if not bounds.contains(y_int, x_int):
            continue

        # Extract overlapping regions
        subI1 = extract_overlap_subregion(image1_backend, y_int, x_int)
        subI2 = extract_overlap_subregion(image2_backend, -y_int, -x_int)

        # Skip if either subregion is empty
        if (hasattr(subI1, 'shape') and np.prod(subI1.shape) == 0) or \
           (hasattr(subI2, 'shape') and np.prod(subI2.shape) == 0):
            continue

        # Compute NCC
        ncc_val = ncc(subI1, subI2)

        # Skip invalid NCC values (NaN only, -inf is a valid "no correlation" result)
        if np.isnan(ncc_val):
            continue

        if ncc_val > best_ncc:
            best_ncc = ncc_val
            best_y = y_int
            best_x = x_int

    return best_ncc, best_y, best_x


def compute_translation(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    strategy: TranslationStrategy = TranslationStrategy.BASIC,
    n: Int = 10,
    subpixel_config: Optional['SubpixelConfig'] = None,
    ransac_config: Optional['RANSACConfig'] = None,
    optimization_config: Optional['OptimizationConfig'] = None,
) -> Tuple[float, float, float]:
    """Unified translation computation with configurable strategy.
    
    Args:
        image1: First image array
        image2: Second image array
        yins: Y coordinates of peaks from PCM
        xins: X coordinates of peaks from PCM
        ymin: Minimum Y translation bound
        ymax: Maximum Y translation bound
        xmin: Minimum X translation bound
        xmax: Maximum X translation bound
        strategy: Translation computation strategy
        n: Maximum number of peaks to evaluate
        subpixel_config: Configuration for subpixel phase correlation
        ransac_config: Configuration for RANSAC outlier rejection
        optimization_config: Configuration for continuous optimization
        
    Returns:
        Tuple of (best_ncc, best_y_translation, best_x_translation)
    """
    # Initialize configs
    if subpixel_config is None:
        subpixel_config = SubpixelConfig()
    if ransac_config is None:
        ransac_config = RANSACConfig()
    if optimization_config is None:
        optimization_config = OptimizationConfig()
    
    if strategy == TranslationStrategy.BASIC:
        return interpret_translation(
            image1, image2, yins, xins, ymin, ymax, xmin, xmax, min(n, 2)
        )
    
    elif strategy == TranslationStrategy.SUBPIXEL:
        return interpret_translation_subpixel(
            image1, image2, yins, xins, ymin, ymax, xmin, xmax,
            n=min(n, 2),
            subpixel_config=subpixel_config,
            use_continuous_optimization=False,
            optimization_config=optimization_config
        )
    
    elif strategy == TranslationStrategy.OPTIMIZED:
        return interpret_translation_optimized(
            image1, image2, yins, xins, ymin, ymax, xmin, xmax,
            n=min(n, 2),
            use_continuous_optimization=True,
            optimization_config=optimization_config
        )
    
    elif strategy == TranslationStrategy.RANSAC:
        return interpret_translation_ransac(
            image1, image2, yins, xins, ymin, ymax, xmin, xmax,
            n=n,
            ransac_config=ransac_config,
            subpixel_config=subpixel_config,
            use_continuous_optimization=True,
            optimization_config=optimization_config
        )
    
    else:
        raise ValueError(f"Unknown translation strategy: {strategy}")


def generate_translation_candidates(
    image1: NumArray,
    image2: NumArray, 
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    n: Int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate translation candidates from PCM peaks for RANSAC.
    
    This function generates a set of translation candidates by evaluating
    multiple PCM peaks and their various sign combinations, creating a dataset
    suitable for RANSAC outlier rejection.
    
    Args:
        image1: First image array
        image2: Second image array  
        yins: Y coordinates of peaks
        xins: X coordinates of peaks
        ymin: Minimum Y translation bound
        ymax: Maximum Y translation bound
        xmin: Minimum X translation bound
        xmax: Maximum X translation bound
        n: Maximum number of peaks to evaluate
        
    Returns:
        Tuple of (translations, ncc_scores) where:
        - translations: Array of shape (N, 2) with (y, x) translations
        - ncc_scores: Array of shape (N,) with corresponding NCC scores
    """
    backend = get_tensor_backend()
    
    # Convert to backend arrays
    image1_backend = backend.asarray(image1)
    image2_backend = backend.asarray(image2)
    yins_backend = backend.asarray(yins)
    xins_backend = backend.asarray(xins)
    
    # Convert to numpy for processing
    image1_np = backend.asnumpy(image1_backend)
    image2_np = backend.asnumpy(image2_backend)
    yins_np = backend.asnumpy(yins_backend)
    xins_np = backend.asnumpy(xins_backend)
    
    if yins_np.size == 0:
        return np.empty((0, 2)), np.empty((0,))
    
    size_y, size_x = image1_np.shape
    bounds = _TranslationBounds(int(ymin), int(ymax), int(xmin), int(xmax))
    
    # Initialize helper classes
    mag_calc = _MagnitudeCalculator(backend, size_y, size_x)
    candidate_gen = _CandidateGenerator(backend)
    
    # Generate validity mask for all peaks
    valid_mask = candidate_gen.generate_validity_mask(
        yins_backend, xins_backend, mag_calc, bounds
    )
    
    # Select top n valid peaks
    valid_mask_np = backend.asnumpy(valid_mask)
    valid_indices = np.where(valid_mask_np)[0]
    num_to_take = min(int(n), len(valid_indices))
    
    if num_to_take == 0:
        return np.empty((0, 2)), np.empty((0,))
    
    selected_indices = valid_indices[:num_to_take]
    
    # Generate candidates for selected peaks
    yins_subset = backend.asarray(yins_np[selected_indices])
    xins_subset = backend.asarray(xins_np[selected_indices])
    
    candidates = candidate_gen.generate_candidate_positions(
        yins_subset, xins_subset, mag_calc
    )
    
    # Flatten candidates and evaluate NCC for each
    flat_candidates = candidates.reshape(-1, 2)
    
    if flat_candidates.size == 0:
        return np.empty((0, 2)), np.empty((0,))
    
    translations = []
    ncc_scores = []
    
    for y_val, x_val in flat_candidates:
        y_int, x_int = int(y_val), int(x_val)
        
        # Check bounds
        if not bounds.contains(y_int, x_int):
            continue
            
        # Extract overlapping regions
        subI1 = extract_overlap_subregion(image1_backend, y_int, x_int)
        subI2 = extract_overlap_subregion(image2_backend, -y_int, -x_int)
        
        # Skip if either subregion is empty
        if (hasattr(subI1, 'shape') and np.prod(subI1.shape) == 0) or \
           (hasattr(subI2, 'shape') and np.prod(subI2.shape) == 0):
            continue
        
        # Compute NCC
        ncc_val = ncc(subI1, subI2)
        
        # Skip invalid NCC values
        if np.isnan(ncc_val) or ncc_val == -float('inf'):
            continue
            
        translations.append([float(y_val), float(x_val)])
        ncc_scores.append(ncc_val)
    
    if not translations:
        return np.empty((0, 2)), np.empty((0,))
        
    return np.array(translations), np.array(ncc_scores)


def ransac_translation_estimation(
    translations: np.ndarray,
    ncc_scores: np.ndarray, 
    config: RANSACConfig = RANSACConfig()
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """Estimate robust translation using RANSAC.
    
    Uses RANSAC to find the best translation estimate while rejecting outliers.
    The approach treats translation candidates as a point cloud and fits a
    robust model to find the consensus translation.
    
    Args:
        translations: Array of shape (N, 2) with (y, x) translation candidates
        ncc_scores: Array of shape (N,) with corresponding NCC scores
        config: RANSAC configuration parameters
        
    Returns:
        Tuple of (best_translation, inlier_mask, consensus_score) where:
        - best_translation: Best (y, x) translation or None if failed
        - inlier_mask: Boolean mask indicating inlier translations
        - consensus_score: Quality score of the consensus (mean NCC of inliers)
    """
    if not config.use_ransac or not RANSAC_AVAILABLE:
        # Fallback: return best translation by NCC score
        if len(translations) == 0:
            return None, np.array([]), 0.0
            
        best_idx = np.argmax(ncc_scores)
        best_translation = translations[best_idx]
        inlier_mask = np.zeros(len(translations), dtype=bool)
        inlier_mask[best_idx] = True
        return best_translation, inlier_mask, ncc_scores[best_idx]
    
    if len(translations) < config.min_samples:
        # Not enough samples for RANSAC
        if len(translations) == 0:
            return None, np.array([]), 0.0
        best_idx = np.argmax(ncc_scores)
        best_translation = translations[best_idx]
        inlier_mask = np.zeros(len(translations), dtype=bool)
        inlier_mask[best_idx] = True
        return best_translation, inlier_mask, ncc_scores[best_idx]
    
    try:
        # Create dummy source and destination points for RANSAC
        # We treat this as a degenerate case where all points should map to the same translation
        # Source points: just indices
        src_points = np.arange(len(translations)).reshape(-1, 1).astype(np.float64)
        # Destination points: the translation vectors
        dst_points = translations.astype(np.float64)
        
        # Custom model class for translation consensus
        class TranslationConsensusModel:
            def __init__(self):
                self.params = None
                
            def estimate(self, data, *args):
                """Estimate model from minimal sample."""
                src, dst = data
                # For translation, we just take the median of the sample
                if len(dst) >= config.min_samples:
                    self.params = np.median(dst, axis=0)
                    return True
                return False
                
            def residuals(self, data, *args):
                """Compute residuals for all data points."""
                if self.params is None:
                    return np.inf * np.ones(len(data[1]))
                src, dst = data
                # Residual is distance from consensus translation
                residuals = np.linalg.norm(dst - self.params, axis=1)
                return residuals
                
            def is_model_valid(self, model, *args):
                """Check if model is valid."""
                return model is not None
        
        # Run RANSAC
        try:
            model_robust, inliers = ransac(
                (src_points, dst_points),
                TranslationConsensusModel,
                min_samples=config.min_samples,
                residual_threshold=config.residual_threshold,
                max_trials=config.max_trials,
                stop_probability=config.stop_probability
            )
        except Exception as ransac_error:
            # Try alternative RANSAC approach - simple consensus finding
            # Find the most common translation (cluster analysis)
            from scipy.spatial.distance import pdist, squareform
            
            # Compute pairwise distances between translations
            if len(translations) >= 2:
                distances = squareform(pdist(translations))
                
                # Find translation with most neighbors within threshold
                neighbor_counts = np.sum(distances <= config.residual_threshold, axis=1)
                best_idx = np.argmax(neighbor_counts)
                
                # Get inliers (neighbors of best translation)
                inliers = distances[best_idx] <= config.residual_threshold
                
                # Create a simple consensus model
                model_robust = type('SimpleModel', (), {
                    'params': translations[best_idx]
                })()
            else:
                # Not enough data, fall back
                raise ransac_error
        
        # Check if we have enough inliers
        inlier_ratio = np.sum(inliers) / len(inliers) if len(inliers) > 0 else 0.0
        
        if inlier_ratio < config.min_inlier_ratio or model_robust is None:
            # RANSAC failed, fall back to best single translation
            best_idx = np.argmax(ncc_scores)
            best_translation = translations[best_idx]
            inlier_mask = np.zeros(len(translations), dtype=bool)
            inlier_mask[best_idx] = True
            return best_translation, inlier_mask, ncc_scores[best_idx]
        
        # Get consensus translation
        best_translation = model_robust.params
        
        # Compute consensus score as mean NCC of inliers
        inlier_scores = ncc_scores[inliers]
        consensus_score = np.mean(inlier_scores) if len(inlier_scores) > 0 else 0.0
        
        return best_translation, inliers, consensus_score
        
    except Exception as e:
        warnings.warn(f"RANSAC failed: {e}, falling back to best single translation")
        
        # Fallback to best single translation
        if len(translations) == 0:
            return None, np.array([]), 0.0
            
        best_idx = np.argmax(ncc_scores)
        best_translation = translations[best_idx]
        inlier_mask = np.zeros(len(translations), dtype=bool)
        inlier_mask[best_idx] = True
        return best_translation, inlier_mask, ncc_scores[best_idx]


def interpret_translation_ransac(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    n: Int = 10,
    ransac_config: RANSACConfig = None,
    subpixel_config: Optional['SubpixelConfig'] = None,
    use_continuous_optimization: bool = True,
    optimization_config: Optional['OptimizationConfig'] = None
) -> Tuple[float, float, float]:
    """[DEPRECATED] Use compute_translation(strategy=TranslationStrategy.RANSAC) instead.
    
    Ultimate translation interpretation with RANSAC outlier rejection.
    
    This function combines all three methods with RANSAC for maximum robustness:
    1. Generate multiple translation candidates from PCM peaks
    2. Use RANSAC to find consensus translation and reject outliers
    3. Optionally apply subpixel phase correlation
    4. Optionally apply continuous optimization
    5. Return the best result among all methods
    
    Args:
        image1: First image array
        image2: Second image array
        yins: Y coordinates of peaks from PCM
        xins: X coordinates of peaks from PCM
        ymin: Minimum Y translation bound
        ymax: Maximum Y translation bound
        xmin: Minimum X translation bound
        xmax: Maximum X translation bound
        n: Maximum number of peaks to evaluate for RANSAC
        ransac_config: Configuration for RANSAC outlier rejection
        subpixel_config: Configuration for subpixel phase correlation
        use_continuous_optimization: Whether to apply continuous optimization
        optimization_config: Configuration for continuous optimization
        
    Returns:
        Tuple of (best_ncc, best_y_translation, best_x_translation)
        Note: translations are sub-pixel accurate
    """
    if ransac_config is None:
        ransac_config = RANSACConfig()
    if subpixel_config is None:
        subpixel_config = SubpixelConfig()
    if optimization_config is None:
        optimization_config = OptimizationConfig()
    
    results = []  # Store (ncc, y, x, method_name) tuples
    
    # Method 1: RANSAC-based robust estimation
    if ransac_config.use_ransac and RANSAC_AVAILABLE:
        try:
            # Generate translation candidates
            translations, ncc_scores = generate_translation_candidates(
                image1, image2, yins, xins, ymin, ymax, xmin, xmax, n
            )
            
            if len(translations) > 0:
                # Apply RANSAC
                best_translation, inlier_mask, consensus_score = ransac_translation_estimation(
                    translations, ncc_scores, ransac_config
                )
                
                if best_translation is not None:
                    ransac_y, ransac_x = best_translation
                    
                    # Validate RANSAC result is within bounds
                    if (ymin <= ransac_y <= ymax and xmin <= ransac_x <= xmax):
                        # Compute actual NCC at RANSAC position
                        y_int, x_int = int(np.round(ransac_y)), int(np.round(ransac_x))
                        
                        subI1 = extract_overlap_subregion(image1, y_int, x_int)
                        subI2 = extract_overlap_subregion(image2, -y_int, -x_int)
                        
                        if (hasattr(subI1, 'shape') and np.prod(subI1.shape) > 0 and
                            hasattr(subI2, 'shape') and np.prod(subI2.shape) > 0):
                            
                            ncc_val = ncc(subI1, subI2)
                            if np.isfinite(ncc_val):
                                results.append((ncc_val, ransac_y, ransac_x, "ransac"))
                                
        except Exception as e:
            warnings.warn(f"RANSAC method failed: {e}")
    
    # Method 2: Direct subpixel phase correlation
    if subpixel_config.use_subpixel and SKIMAGE_AVAILABLE:
        try:
            # Use scikit-image's phase correlation directly
            img1_np = np.asarray(image1, dtype=np.float64)
            img2_np = np.asarray(image2, dtype=np.float64)
            
            result = phase_cross_correlation(
                img1_np, img2_np,
                upsample_factor=subpixel_config.upsample_factor,
                reference_mask=subpixel_config.reference_mask,
                moving_mask=subpixel_config.moving_mask,
                overlap_ratio=subpixel_config.overlap_ratio,
                space=subpixel_config.space,
                normalization=subpixel_config.normalization
            )
            
            # Extract shift from result (handle different scikit-image versions)
            if len(result) >= 2:
                shift = result[0]
            else:
                shift = result
            subpixel_y, subpixel_x = float(shift[0]), float(shift[1])
            
            # Check if subpixel result is within bounds
            if (ymin <= subpixel_y <= ymax and xmin <= subpixel_x <= xmax):
                # Compute NCC at subpixel position
                y_int, x_int = int(np.round(subpixel_y)), int(np.round(subpixel_x))
                
                subI1 = extract_overlap_subregion(image1, y_int, x_int)
                subI2 = extract_overlap_subregion(image2, -y_int, -x_int)
                
                if (hasattr(subI1, 'shape') and np.prod(subI1.shape) > 0 and
                    hasattr(subI2, 'shape') and np.prod(subI2.shape) > 0):
                    
                    ncc_val = ncc(subI1, subI2)
                    if np.isfinite(ncc_val):
                        results.append((ncc_val, subpixel_y, subpixel_x, "subpixel_direct"))
                        
        except Exception as e:
            warnings.warn(f"Direct subpixel method failed: {e}")
    
    # Method 3: Discrete peak finding (existing method)
    try:
        discrete_ncc, discrete_y, discrete_x = interpret_translation(
            image1, image2, yins, xins, ymin, ymax, xmin, xmax, min(n, 2)
        )
        if discrete_ncc != -float('inf'):
            results.append((discrete_ncc, float(discrete_y), float(discrete_x), "discrete"))
    except Exception as e:
        warnings.warn(f"Discrete method failed: {e}")
    
    # Method 4: Continuous optimization (if enabled and we have a good starting point)
    if use_continuous_optimization and results:
        # Use the best result so far as starting point
        best_so_far = max(results, key=lambda x: x[0])
        _, start_y, start_x, _ = best_so_far
        
        try:
            continuous_ncc, continuous_y, continuous_x, _ = optimize_translation_continuous(
                image1=image1,
                image2=image2,
                initial_y=start_y,
                initial_x=start_x,
                ymin=float(ymin),
                ymax=float(ymax),
                xmin=float(xmin),
                xmax=float(xmax),
                config=optimization_config
            )
            
            if continuous_ncc != -float('inf'):
                results.append((continuous_ncc, continuous_y, continuous_x, "continuous"))
                
        except Exception as e:
            warnings.warn(f"Continuous optimization failed: {e}")
    
    # Return the best result
    if not results:
        return -float('inf'), 0.0, 0.0
    
    best_ncc, best_y, best_x, best_method = max(results, key=lambda x: x[0])
    
    # Optional: Log which method won (for debugging)
    if len(results) > 1:
        method_scores = {method: ncc for ncc, _, _, method in results}
        # Uncomment for debugging: print(f"Method comparison: {method_scores}, winner: {best_method}")
    
    return best_ncc, best_y, best_x


def interpret_translation_subpixel(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    n: Int = 2,
    subpixel_config: SubpixelConfig = None,
    use_continuous_optimization: bool = True,
    optimization_config: Optional['OptimizationConfig'] = None
) -> Tuple[float, float, float]:
    """[DEPRECATED] Use compute_translation(strategy=TranslationStrategy.SUBPIXEL) instead.
    
    Enhanced translation interpretation with subpixel phase correlation.
    
    This function combines subpixel phase correlation with the existing
    discrete peak finding and optional continuous optimization for maximum accuracy.
    
    Workflow:
    1. Compute subpixel phase correlation for direct sub-pixel estimate
    2. Use discrete peak finding as backup/validation
    3. Optionally apply continuous optimization for final refinement
    4. Return the best result among all methods
    
    Args:
        image1: First image array
        image2: Second image array
        yins: Y coordinates of peaks from standard PCM
        xins: X coordinates of peaks from standard PCM
        ymin: Minimum Y translation bound
        ymax: Maximum Y translation bound
        xmin: Minimum X translation bound
        xmax: Maximum X translation bound
        n: Maximum number of peaks to evaluate for discrete method
        subpixel_config: Configuration for subpixel phase correlation
        use_continuous_optimization: Whether to apply continuous optimization
        optimization_config: Configuration for continuous optimization
        
    Returns:
        Tuple of (best_ncc, best_y_translation, best_x_translation)
        Note: translations are sub-pixel accurate
    """
    if subpixel_config is None:
        subpixel_config = SubpixelConfig()
    if optimization_config is None:
        optimization_config = OptimizationConfig()
    
    results = []  # Store (ncc, y, x, method_name) tuples
    
    # Method 1: Direct subpixel phase correlation
    if subpixel_config.use_subpixel and SKIMAGE_AVAILABLE:
        try:
            # Use scikit-image's phase correlation directly
            img1_np = np.asarray(image1, dtype=np.float64)
            img2_np = np.asarray(image2, dtype=np.float64)
            
            result = phase_cross_correlation(
                img1_np, img2_np,
                upsample_factor=subpixel_config.upsample_factor,
                reference_mask=subpixel_config.reference_mask,
                moving_mask=subpixel_config.moving_mask,
                overlap_ratio=subpixel_config.overlap_ratio,
                space=subpixel_config.space,
                normalization=subpixel_config.normalization
            )
            
            # Extract shift from result (handle different scikit-image versions)
            if len(result) >= 2:
                shift = result[0]
            else:
                shift = result
            subpixel_y, subpixel_x = float(shift[0]), float(shift[1])
            
            # Check if subpixel result is within bounds
            if (ymin <= subpixel_y <= ymax and xmin <= subpixel_x <= xmax):
                # Compute NCC at subpixel position (rounded for overlap extraction)
                y_int, x_int = int(np.round(subpixel_y)), int(np.round(subpixel_x))
                
                subI1 = extract_overlap_subregion(image1, y_int, x_int)
                subI2 = extract_overlap_subregion(image2, -y_int, -x_int)
                
                if (hasattr(subI1, 'shape') and np.prod(subI1.shape) > 0 and
                    hasattr(subI2, 'shape') and np.prod(subI2.shape) > 0):
                    
                    ncc_val = ncc(subI1, subI2)
                    if np.isfinite(ncc_val):
                        results.append((ncc_val, subpixel_y, subpixel_x, "subpixel_direct"))
                        
        except Exception as e:
            warnings.warn(f"Direct subpixel method failed: {e}")
    
    # Method 2: Discrete peak finding (existing method)
    try:
        discrete_ncc, discrete_y, discrete_x = interpret_translation(
            image1, image2, yins, xins, ymin, ymax, xmin, xmax, n
        )
        if discrete_ncc != -float('inf'):
            results.append((discrete_ncc, float(discrete_y), float(discrete_x), "discrete"))
    except Exception as e:
        warnings.warn(f"Discrete method failed: {e}")
    
    # Method 3: Continuous optimization (if enabled and we have a good starting point)
    if use_continuous_optimization and results:
        # Use the best result so far as starting point
        best_so_far = max(results, key=lambda x: x[0])
        _, start_y, start_x, _ = best_so_far
        
        try:
            continuous_ncc, continuous_y, continuous_x, _ = optimize_translation_continuous(
                image1=image1,
                image2=image2,
                initial_y=start_y,
                initial_x=start_x,
                ymin=float(ymin),
                ymax=float(ymax),
                xmin=float(xmin),
                xmax=float(xmax),
                config=optimization_config
            )
            
            if continuous_ncc != -float('inf'):
                results.append((continuous_ncc, continuous_y, continuous_x, "continuous"))
                
        except Exception as e:
            warnings.warn(f"Continuous optimization failed: {e}")
    
    # Return the best result
    if not results:
        return -float('inf'), 0.0, 0.0
    
    best_ncc, best_y, best_x, best_method = max(results, key=lambda x: x[0])
    
    # Optional: Log which method won (for debugging)
    if len(results) > 1:
        method_scores = {method: ncc for ncc, _, _, method in results}
        # Uncomment for debugging: print(f"Method comparison: {method_scores}, winner: {best_method}")
    
    return best_ncc, best_y, best_x


def optimize_translation_continuous(
    image1: NumArray,
    image2: NumArray,
    initial_y: float,
    initial_x: float,
    ymin: float,
    ymax: float,
    xmin: float,
    xmax: float,
    config: OptimizationConfig = OptimizationConfig()
) -> Tuple[float, float, float, OptimizeResult]:
    """Optimize translation using advanced continuous optimization.
    
    This function uses Trust Region Reflective or Levenberg-Marquardt optimization
    to refine translation estimates with sub-integer precision.
    
    Args:
        image1: First image array
        image2: Second image array
        initial_y: Initial Y translation estimate
        initial_x: Initial X translation estimate
        ymin: Minimum Y translation bound
        ymax: Maximum Y translation bound
        xmin: Minimum X translation bound
        xmax: Maximum X translation bound
        config: Optimization configuration
        
    Returns:
        Tuple of (best_ncc, best_y_translation, best_x_translation, optimization_result)
        
    Raises:
        ValueError: If optimization bounds or initial values are invalid
        RuntimeError: If optimization fails completely
    """
    # Validate inputs
    if not (ymin <= initial_y <= ymax):
        raise ValueError(f"Initial Y ({initial_y}) must be within bounds [{ymin}, {ymax}]")
    if not (xmin <= initial_x <= xmax):
        raise ValueError(f"Initial X ({initial_x}) must be within bounds [{xmin}, {xmax}]")
    
    backend = get_tensor_backend()
    
    # Convert to backend arrays
    image1_backend = backend.asarray(image1)
    image2_backend = backend.asarray(image2)
    
    # Initial parameter vector
    x0 = np.array([initial_y, initial_x], dtype=np.float64)
    
    # Handle bounds based on method
    bounds = None
    if config.method in ['trf', 'dogbox']:
        # Trust Region methods support bounds
        bounds = ([ymin, xmin], [ymax, xmax])
    elif config.method == 'lm':
        # Levenberg-Marquardt doesn't support bounds - warn user
        if not (ymin <= initial_y <= ymax and xmin <= initial_x <= xmax):
            warnings.warn("Levenberg-Marquardt method doesn't support bounds. Initial values should be well within valid range.")
    
    # Inline residual function for least_squares
    def residual_fn(params: np.ndarray) -> np.ndarray:
        """Compute residual: sqrt(1 - NCC) for optimization."""
        y_trans, x_trans = params
        y_int, x_int = int(np.round(y_trans)), int(np.round(x_trans))
        try:
            subI1 = extract_overlap_subregion(image1_backend, y_int, x_int)
            subI2 = extract_overlap_subregion(image2_backend, -y_int, -x_int)
            if (hasattr(subI1, 'shape') and np.prod(subI1.shape) == 0) or \
               (hasattr(subI2, 'shape') and np.prod(subI2.shape) == 0):
                return np.array([1.0])  # Worst residual
            ncc_val = ncc(subI1, subI2)
            if np.isnan(ncc_val) or np.isinf(ncc_val):
                return np.array([1.0])
            return np.array([np.sqrt(np.maximum(0.0, 1.0 - ncc_val))])
        except Exception:
            return np.array([1.0])
    
    with managed_memory():
        try:
            # Run optimization
            result = least_squares(
                fun=residual_fn,
                x0=x0,
                bounds=bounds,
                method=config.method,
                max_nfev=config.max_nfev,
                ftol=config.ftol,
                xtol=config.xtol,
                gtol=config.gtol,
                verbose=config.verbose
            )
            
            # Extract optimized parameters
            opt_y, opt_x = result.x
            
            # For unbounded methods, manually check bounds
            if bounds is None:
                if not (ymin <= opt_y <= ymax and xmin <= opt_x <= xmax):
                    # Clamp to bounds if optimization went outside
                    opt_y = np.clip(opt_y, ymin, ymax)
                    opt_x = np.clip(opt_x, xmin, xmax)
                    warnings.warn(f"Optimization result clamped to bounds: ({opt_y:.3f}, {opt_x:.3f})")
            
            # Compute final NCC at optimized position (inline objective function)
            y_int, x_int = int(np.round(opt_y)), int(np.round(opt_x))
            subI1 = extract_overlap_subregion(image1_backend, y_int, x_int)
            subI2 = extract_overlap_subregion(image2_backend, -y_int, -x_int)
            if (hasattr(subI1, 'shape') and np.prod(subI1.shape) == 0) or \
               (hasattr(subI2, 'shape') and np.prod(subI2.shape) == 0):
                final_ncc = 0.0
            else:
                final_ncc = ncc(subI1, subI2)
                if np.isnan(final_ncc) or np.isinf(final_ncc):
                    final_ncc = 0.0
            
            # Validate result
            if not result.success:
                warnings.warn(f"Optimization did not converge: {result.message}")
            
            return final_ncc, opt_y, opt_x, result
            
        except Exception as e:
            raise RuntimeError(f"Continuous optimization failed: {e}")


def interpret_translation_optimized(
    image1: NumArray,
    image2: npt.NDArray,
    yins: IntArray,
    xins: IntArray,
    ymin: Int,
    ymax: Int,
    xmin: Int,
    xmax: Int,
    n: Int = 2,
    use_continuous_optimization: bool = True,
    optimization_config: OptimizationConfig = OptimizationConfig()
) -> Tuple[float, float, float]:
    """[DEPRECATED] Use compute_translation(strategy=TranslationStrategy.OPTIMIZED) instead.
    
    Enhanced translation interpretation with optional continuous optimization.
    
    This function extends the original interpret_translation with advanced
    optimization capabilities. It first uses the discrete grid search to find
    good initial estimates, then optionally refines them using continuous
    optimization methods.
    
    Args:
        image1: First image array
        image2: Second image array
        yins: Y coordinates of peaks
        xins: X coordinates of peaks
        ymin: Minimum Y translation bound
        ymax: Maximum Y translation bound
        xmin: Minimum X translation bound
        xmax: Maximum X translation bound
        n: Maximum number of peaks to evaluate for initial estimates
        use_continuous_optimization: Whether to apply continuous optimization
        optimization_config: Configuration for continuous optimization
        
    Returns:
        Tuple of (best_ncc, best_y_translation, best_x_translation)
        Note: translations may be non-integer if continuous optimization is used
    """
    # First, run the original discrete optimization to get initial estimate
    discrete_ncc, discrete_y, discrete_x = interpret_translation(
        image1, image2, yins, xins, ymin, ymax, xmin, xmax, n
    )
    
    # If discrete optimization failed or continuous optimization is disabled
    if discrete_ncc == -float('inf') or not use_continuous_optimization:
        return discrete_ncc, float(discrete_y), float(discrete_x)
    
    try:
        # Apply continuous optimization starting from discrete result
        continuous_ncc, continuous_y, continuous_x, opt_result = optimize_translation_continuous(
            image1=image1,
            image2=image2,
            initial_y=float(discrete_y),
            initial_x=float(discrete_x),
            ymin=float(ymin),
            ymax=float(ymax),
            xmin=float(xmin),
            xmax=float(xmax),
            config=optimization_config
        )
        
        # Use continuous result if it's better than discrete
        if continuous_ncc > discrete_ncc:
            return continuous_ncc, continuous_y, continuous_x
        else:
            # Fall back to discrete result if continuous optimization didn't improve
            return discrete_ncc, float(discrete_y), float(discrete_x)
            
    except Exception as e:
        # Fall back to discrete result on any optimization failure
        warnings.warn(f"Continuous optimization failed, using discrete result: {e}")
        return discrete_ncc, float(discrete_y), float(discrete_x)
