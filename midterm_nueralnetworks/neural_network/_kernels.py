import numpy as np
from typing import Tuple
from numpy.lib.stride_tricks import as_strided, sliding_window_view


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
    """Pad the input with zeros on all sides of the last two dimensions."""
    # Get the number of dimensions in the input
    if X.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    
    num_dims = X.ndim

    # Create a padding specification for all dimensions
    pad_widths = [(0, 0)] * (num_dims - 2) + [(padding, padding), (padding, padding)]

    # Apply padding
    padded_X = np.pad(X, pad_width=pad_widths, mode='constant', constant_values=0)

    return padded_X


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

def _convolve(kernel: np.ndarray, X: np.ndarray, stride: int, padding: int):
    """Perform a convolution given a kernel and an input"""
    X_padded = pad_input(X, padding)
    out_height, out_width = _kernel_op_size(X.shape, kernel.shape, stride, padding)

    patches = sliding_window_view(X_padded, kernel.shape) [::stride, ::stride]
    
    res = np.zeros((out_height, out_width))
    for i, j in np.ndindex(out_height, out_width):
        res[i, j] = np.sum(patches[i, j] * kernel)
    
    return res

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
    if X.ndim == 3:
        X = np.expand_dims(X, axis=0)
    batch_size, in_channels, height, width = X.shape
    out_channels, in_channels_k, kern_n, kern_m = kernels.shape

    if in_channels != in_channels_k:
        print(X.shape)
        print(kernels.shape)
        raise ValueError("The number of input channels in kernels and X must match.")

    X_padded = pad_input(X, padding)
    out_height, out_width = _kernel_op_size((height, width), (kern_n, kern_m), stride, padding)

    patches_list = []
    for batch in range(batch_size):
        sample_patches = extract_patches(X_padded[batch], (kern_n, kern_m), stride)
        patches_list.append(sample_patches)

    patches = np.stack(patches_list, axis=0)

    # Perform convolution
    # Reshape patches for batch multiplication
    patches = patches.reshape(batch_size, -1, kern_n * kern_m * in_channels)
    kernels = kernels.reshape(out_channels, -1)

    # Compute dot product for convolution
    res = np.matmul(patches, kernels.T)
    res = res.reshape(batch_size, out_height, out_width, out_channels).transpose(0, 3, 1, 2)

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
        np.ndarray: Position of max values (in_channels, height, width, 2) where the last dimension is (i, j) position of max
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

    # Find the position of the max values
    # Note: Numpy does not have an argmax function that operates on multiple dimensions
    # So we reshape the patches to (k, height, width, kern_n * kern_m) then find the argmax
    # and convert the flattened index back to (k, height, width)
    flat_index = np.argmax(patches.reshape(k, height, width, -1), axis=-1)
    pos = np.stack(np.unravel_index(flat_index, (kern_n, kern_m)), axis=-1)

    return res, pos
