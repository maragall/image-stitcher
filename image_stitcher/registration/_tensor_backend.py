"""Tensor backend abstraction for supporting cupy, torch, and numpy."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Union, List
import numpy as np
import warnings

class TensorBackend(ABC):
    """Abstract base class for tensor backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def is_gpu(self) -> bool:
        pass
    
    @abstractmethod
    def asarray(self, array: Any, dtype: Any = None) -> Any:
        pass
    
    @abstractmethod
    def asnumpy(self, array: Any) -> np.ndarray:
        pass
    
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        pass
    
    @abstractmethod
    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        pass
    
    @abstractmethod
    def fft2(self, array: Any) -> Any:
        pass
    
    @abstractmethod
    def ifft2(self, array: Any) -> Any:
        pass
    
    @abstractmethod
    def mean(self, array: Any) -> Any:
        pass
    
    @abstractmethod
    def sum(self, array: Any) -> Any:
        pass
    
    @abstractmethod
    def sqrt(self, array: Any) -> Any:
        pass
    
    @abstractmethod
    def abs(self, array: Any) -> Any:
        pass
    
    @abstractmethod
    def conjugate(self, array: Any) -> Any:
        pass
    
    @abstractmethod
    def cleanup_memory(self) -> None:
        pass

class CupyBackend(TensorBackend):
    """CuPy tensor backend."""
    
    def __init__(self):
        try:
            import cupy as cp
            self.cp = cp
            # Test GPU functionality with a simple operation
            test_array = cp.array([1.0, 2.0, 3.0])
            _ = cp.sum(test_array)
            cp.cuda.Device().synchronize()
            self.available = True
        except ImportError:
            self.available = False
            raise ImportError("CuPy not available")
        except Exception as e:
            self.available = False
            raise RuntimeError(f"CuPy available but CUDA operations failed: {e}")
    
    @property
    def name(self) -> str:
        return "cupy"
    
    @property
    def is_gpu(self) -> bool:
        return True
    
    def asarray(self, array: Any, dtype: Any = None) -> Any:
        return self.cp.asarray(array, dtype=dtype)
    
    def asnumpy(self, array: Any) -> np.ndarray:
        return self.cp.asnumpy(array)
    
    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return self.cp.zeros(shape, dtype=dtype)
    
    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return self.cp.empty(shape, dtype=dtype)
    
    def fft2(self, array: Any) -> Any:
        return self.cp.fft.fft2(array)
    
    def ifft2(self, array: Any) -> Any:
        return self.cp.fft.ifft2(array)
    
    def mean(self, array: Any) -> Any:
        return self.cp.mean(array)
    
    def sum(self, array: Any) -> Any:
        return self.cp.sum(array)
    
    def sqrt(self, array: Any) -> Any:
        return self.cp.sqrt(array)
    
    def abs(self, array: Any) -> Any:
        return self.cp.abs(array)
    
    def conjugate(self, array: Any) -> Any:
        return self.cp.conjugate(array)
    
    def cleanup_memory(self) -> None:
        try:
            mempool = self.cp.get_default_memory_pool()
            pinned_mempool = self.cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            self.cp.cuda.Device().synchronize()
        except:
            pass

class TorchBackend(TensorBackend):
    """PyTorch tensor backend."""
    
    def __init__(self):
        try:
            import torch
            self.torch = torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Test GPU functionality if CUDA is reported as available
            if self.device == 'cuda':
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                _ = torch.sum(test_tensor)
                torch.cuda.synchronize()
            
            self.available = True
        except ImportError:
            self.available = False
            raise ImportError("PyTorch not available")
        except Exception as e:
            self.available = False
            raise RuntimeError(f"PyTorch available but CUDA operations failed: {e}")
    
    @property
    def name(self) -> str:
        return "torch"
    
    @property
    def is_gpu(self) -> bool:
        return self.device == 'cuda'
    
    def _convert_dtype(self, dtype: Any) -> Any:
        """Convert numpy dtype to torch dtype."""
        if dtype is None:
            return self.torch.float32
        
        # Convert numpy dtypes to torch dtypes
        if dtype == np.float32 or str(dtype) == 'float32':
            return self.torch.float32
        elif dtype == np.float64 or str(dtype) == 'float64':
            return self.torch.float64
        elif dtype == np.int32 or str(dtype) == 'int32':
            return self.torch.int32
        elif dtype == np.int64 or str(dtype) == 'int64':
            return self.torch.int64
        elif dtype == np.bool_ or str(dtype) == 'bool' or dtype == bool:
            return self.torch.bool
        else:
            # If already torch dtype or unknown, return as-is
            return dtype
    
    def asarray(self, array: Any, dtype: Any = None) -> Any:
        dtype = self._convert_dtype(dtype)
        return self.torch.tensor(array, device=self.device, dtype=dtype)
    
    def asnumpy(self, array: Any) -> np.ndarray:
        return array.detach().cpu().numpy()
    
    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        dtype = self._convert_dtype(dtype)
        return self.torch.zeros(shape, device=self.device, dtype=dtype)
    
    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        dtype = self._convert_dtype(dtype)
        return self.torch.empty(shape, device=self.device, dtype=dtype)
    
    def fft2(self, array: Any) -> Any:
        return self.torch.fft.fft2(array)
    
    def ifft2(self, array: Any) -> Any:
        return self.torch.fft.ifft2(array)
    
    def mean(self, array: Any) -> Any:
        return self.torch.mean(array)
    
    def sum(self, array: Any) -> Any:
        return self.torch.sum(array)
    
    def sqrt(self, array: Any) -> Any:
        return self.torch.sqrt(array)
    
    def abs(self, array: Any) -> Any:
        return self.torch.abs(array)
    
    def conjugate(self, array: Any) -> Any:
        return self.torch.conj(array)
    
    def cleanup_memory(self) -> None:
        if self.is_gpu:
            try:
                self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize()
            except:
                pass

class NumpyBackend(TensorBackend):
    """NumPy tensor backend (CPU only)."""
    
    def __init__(self):
        self.available = True
    
    @property
    def name(self) -> str:
        return "numpy"
    
    @property
    def is_gpu(self) -> bool:
        return False
    
    def asarray(self, array: Any, dtype: Any = None) -> Any:
        return np.asarray(array, dtype=dtype)
    
    def asnumpy(self, array: Any) -> np.ndarray:
        return np.asarray(array)
    
    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return np.zeros(shape, dtype=dtype)
    
    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any:
        return np.empty(shape, dtype=dtype)
    
    def fft2(self, array: Any) -> Any:
        return np.fft.fft2(array)
    
    def ifft2(self, array: Any) -> Any:
        return np.fft.ifft2(array)
    
    def mean(self, array: Any) -> Any:
        return np.mean(array)
    
    def sum(self, array: Any) -> Any:
        return np.sum(array)
    
    def sqrt(self, array: Any) -> Any:
        return np.sqrt(array)
    
    def abs(self, array: Any) -> Any:
        return np.abs(array)
    
    def conjugate(self, array: Any) -> Any:
        return np.conjugate(array)
    
    def cleanup_memory(self) -> None:
        pass  # No GPU memory to clean up

def create_tensor_backend(engine: Optional[str] = None, allow_fallback: bool = True) -> TensorBackend:
    """Create a tensor backend with optional automatic fallback.
    
    Args:
        engine: Preferred engine ('cupy', 'torch', 'numpy'), None for auto
        allow_fallback: Whether to try fallback engines if preferred engine fails
        
    Returns:
        TensorBackend instance
        
    Raises:
        RuntimeError: If no backends are available
    """
    engines_to_try = []
    
    if engine:
        engines_to_try.append(engine)
    else:
        engines_to_try = ['cupy', 'torch', 'numpy']
    
    # Add remaining engines as fallbacks only if allowed
    if allow_fallback:
        for fallback in ['cupy', 'torch', 'numpy']:
            if fallback not in engines_to_try:
                engines_to_try.append(fallback)
    
    for engine_name in engines_to_try:
        try:
            if engine_name == 'cupy':
                backend = CupyBackend()
            elif engine_name == 'torch':
                backend = TorchBackend()
            elif engine_name == 'numpy':
                backend = NumpyBackend()
            else:
                continue
            
            # If backend initialization succeeded, use it
            # Backend classes handle their own initialization checks
            print(f"Using tensor backend: {backend.name} ({'GPU' if backend.is_gpu else 'CPU'})")
            return backend
            
        except Exception as e:
            warnings.warn(f"Failed to initialize {engine_name} backend: {e}")
            continue
    
    raise RuntimeError("No tensor backends available")
