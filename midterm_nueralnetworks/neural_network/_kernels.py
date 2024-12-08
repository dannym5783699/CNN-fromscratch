import numpy as np
from typing import Tuple

def _kernel_op_size(
        input_size : Tuple[int, int],
        kernel_size : Tuple[int, int],
        stride: int,
        padding: int):
    """Calculate the output size of a kernel operation."""
    
    height = (input_size[0] + 2 * padding - kernel_size[0]) // stride + 1
    width = (input_size[1] + 2 * padding - kernel_size[1]) // stride + 1
    return height, width

def _2dconvolve(kernel: np.ndarray, X: np.ndarray, stride: int, padding: int):
    """Perform a 2D convolution operation on a 3D input tensor"""

    k, n, m = X.shape
    kern_k, kern_n, kern_m = kernel.shape

    if padding > 0:
        X = np.pad(X, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # Check channel compatibility
    if kern_k != k:
        raise ValueError("Number of channels in kernel and input must match.")
    
    height, width = _kernel_op_size((n, m), (kern_n, kern_m), stride, padding)

    # Initialize result array
    res = np.empty((height, width))

    for i, j in np.ndindex(res.shape):
        patch = X[:, i * stride:i * stride + kern_n, j * stride:j * stride + kern_m]
        res[i, j] = np.sum(patch * kernel)

    return res