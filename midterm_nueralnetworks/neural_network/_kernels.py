import numpy as np
from typing import Tuple
from numpy.lib.stride_tricks import as_strided

def _kernel_op_size(
        input_size: Tuple[int, int],
        kernel_size: Tuple[int, int],
        stride: int,
        padding: int):
    """Calculate the output size of a kernel operation."""
    height = (input_size[0] + 2 * padding - kernel_size[0]) // stride + 1
    width = (input_size[1] + 2 * padding - kernel_size[1]) // stride + 1
    return height, width

def pad_input(X: np.ndarray, padding: int):
    """Pad the input with zeros on all sides."""
    if padding == 0:
        return X
    else:
        return np.pad(
            X,
            ((0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )

def extract_patches(X: np.ndarray, kernel_size: Tuple[int, int], stride: int):
    """Extract sliding windows (patches) from the input tensor."""
    k, n, m = X.shape
    kern_n, kern_m = kernel_size
    out_height = (n - kern_n) // stride + 1
    out_width = (m - kern_m) // stride + 1

    # Calculate the strides for sliding windows
    s0, s1, s2 = X.strides
    new_shape = (k, out_height, out_width, kern_n, kern_m)
    new_strides = (s0, s1 * stride, s2 * stride, s1, s2)

    # Use as_strided to extract patches
    patches = as_strided(X, shape=new_shape, strides=new_strides)
    return patches

def _2dconvolve(kernels: np.ndarray, X: np.ndarray, stride: int, padding: int):
    """Perform a 2D convolution operation on a 3D input tensor with 3D kernels.
    Apply the convolution operation to each channel in the input tensor and sum the results.

    Args:
        kernels (np.ndarray): Kernels for the convolution operation (out_channels, in_channels, kern_n, kern_m)
        X (np.ndarray): Single sample input tensor for the convolution operation (in_channels, n, m)
        stride (int): Stride for the convolution operation
        padding (int): Padding for the convolution operation

    Returns:
        np.ndarray: Result of the convolution operation (out_channels, height, width)
    """

    k, n, m = X.shape
    out_channels, in_channels, kern_n, kern_m = kernels.shape

    if in_channels != k:
        raise ValueError("The number of input channels in kernels and X must match.")

    X_padded = pad_input(X, padding)
    height, width = _kernel_op_size((n, m), (kern_n, kern_m), stride, padding)

    patches = extract_patches(X_padded, (kern_n, kern_m), stride)

    # Perform convolution
    # Reshape patches for batch multiplication
    patches = patches.transpose(1, 2, 0, 3, 4).reshape(height * width, in_channels, kern_n, kern_m)
    kernels = kernels.reshape(out_channels, in_channels * kern_n * kern_m)

    # Compute dot product for convolution
    res = np.dot(patches.reshape(height * width, -1), kernels.T)
    res = res.reshape(height, width, out_channels).transpose(2, 0, 1)

    return res

def _2dmaxpool(kernel_size: Tuple[int, int], X: np.ndarray, stride: int, padding: int):
    """Perform a 2D max pooling operation on a 3D input tensor based on a kernel size.

    Args:
        kernel_size (Tuple[int, int]): Pooling kernel size 
        X (np.ndarray): Input tensor for the pooling operation (in_channels, n, m)
        stride (int): Stride for the pooling operation
        padding (int): Padding for the pooling operation

    Returns:
        np.ndarray: Result of the pooling operation (in_channels, height, width)
    """

    k, n, m = X.shape
    kern_n, kern_m = kernel_size

    X_padded = pad_input(X, padding)

    height, width = _kernel_op_size((n, m), kernel_size, stride, padding)

    s0, s1, s2 = X.strides
    new_shape = (k, height, width, kern_n, kern_m)
    new_strides = (s0, s1 * stride, s2 * stride, s1, s2)

    patches = as_strided(X, shape=new_shape, strides=new_strides)

    res = np.max(patches, axis=(3, 4))  # Max over kernel dimensions

    return res