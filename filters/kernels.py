"""
Kernel Definitions
=================

Định nghĩa các kernel chuẩn cho image processing
- Edge detection kernels
- Blur kernels
- Sharpen kernels
- Custom kernels

Author: Image Processing Team
"""

import numpy as np


def create_gaussian_kernel(size, sigma=1.0):
    """
    Tạo Gaussian kernel - TỰ CODE
    
    Thuật toán:
    - G(x,y) = (1/(2π*σ²)) * exp(-(x²+y²)/(2σ²))
    - Normalize để tổng = 1
    
    Args:
        size: Kích thước kernel (phải lẻ)
        sigma: Standard deviation
        
    Returns:
        kernel: Gaussian kernel đã normalize
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    # Tính Gaussian values
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            # Công thức Gaussian 2D
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    # Normalize để tổng = 1
    kernel = kernel / np.sum(kernel)
    
    return kernel


def get_sobel_kernels():
    """
    Sobel edge detection kernels
    
    Returns:
        sobel_x, sobel_y: Sobel kernels cho X và Y direction
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)
    
    return sobel_x, sobel_y


def get_prewitt_kernels():
    """
    Prewitt edge detection kernels
    
    Returns:
        prewitt_x, prewitt_y: Prewitt kernels
    """
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    prewitt_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32)
    
    return prewitt_x, prewitt_y


def get_laplacian_kernel():
    """
    Laplacian edge detection kernel
    
    Returns:
        laplacian: Laplacian kernel
    """
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    return laplacian


def get_laplacian_8_kernel():
    """
    Laplacian 8-connected kernel
    
    Returns:
        laplacian_8: Laplacian 8-connected kernel
    """
    laplacian_8 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    return laplacian_8


def get_box_filter_kernel(size):
    """
    Box filter (average) kernel
    
    Args:
        size: Kích thước kernel
        
    Returns:
        box_kernel: Box filter kernel
    """
    box_kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return box_kernel


def get_sharpen_kernel():
    """
    Sharpen kernel
    
    Returns:
        sharpen: Sharpen kernel
    """
    sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    return sharpen


def get_unsharp_mask_kernel():
    """
    Unsharp mask kernel
    
    Returns:
        unsharp: Unsharp mask kernel
    """
    unsharp = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    return unsharp


def get_emboss_kernel():
    """
    Emboss effect kernel
    
    Returns:
        emboss: Emboss kernel
    """
    emboss = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
    
    return emboss


def get_identity_kernel():
    """
    Identity kernel (no change)
    
    Returns:
        identity: Identity kernel
    """
    identity = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    
    return identity


def get_all_kernels():
    """
    Lấy tất cả kernels có sẵn
    
    Returns:
        kernels: Dictionary chứa tất cả kernels
    """
    sobel_x, sobel_y = get_sobel_kernels()
    prewitt_x, prewitt_y = get_prewitt_kernels()
    
    kernels = {
        # Edge Detection
        'sobel_x': sobel_x,
        'sobel_y': sobel_y,
        'prewitt_x': prewitt_x,
        'prewitt_y': prewitt_y,
        'laplacian': get_laplacian_kernel(),
        'laplacian_8': get_laplacian_8_kernel(),
        
        # Blur
        'gaussian_3x3': create_gaussian_kernel(3, 0.8),
        'gaussian_5x5': create_gaussian_kernel(5, 1.2),
        'gaussian_7x7': create_gaussian_kernel(7, 1.5),
        'box_3x3': get_box_filter_kernel(3),
        'box_5x5': get_box_filter_kernel(5),
        
        # Sharpen
        'sharpen': get_sharpen_kernel(),
        'unsharp_mask': get_unsharp_mask_kernel(),
        
        # Effects
        'emboss': get_emboss_kernel(),
        'identity': get_identity_kernel()
    }
    
    return kernels


def get_kernel_info(kernel_name):
    """
    Lấy thông tin về kernel
    
    Args:
        kernel_name: Tên kernel
        
    Returns:
        info: Dictionary chứa thông tin kernel
    """
    kernel_descriptions = {
        'sobel_x': 'Sobel X - Detect vertical edges',
        'sobel_y': 'Sobel Y - Detect horizontal edges',
        'prewitt_x': 'Prewitt X - Detect vertical edges',
        'prewitt_y': 'Prewitt Y - Detect horizontal edges',
        'laplacian': 'Laplacian - Detect edges (all directions)',
        'laplacian_8': 'Laplacian 8-connected - Enhanced edge detection',
        'gaussian_3x3': 'Gaussian 3x3 - Light blur',
        'gaussian_5x5': 'Gaussian 5x5 - Medium blur',
        'gaussian_7x7': 'Gaussian 7x7 - Heavy blur',
        'box_3x3': 'Box filter 3x3 - Simple average blur',
        'box_5x5': 'Box filter 5x5 - Strong average blur',
        'sharpen': 'Sharpen - Enhance edges and details',
        'unsharp_mask': 'Unsharp mask - Advanced sharpening',
        'emboss': 'Emboss - 3D effect',
        'identity': 'Identity - No change (pass-through)'
    }
    
    kernels = get_all_kernels()
    
    if kernel_name not in kernels:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    
    kernel = kernels[kernel_name]
    
    info = {
        'name': kernel_name,
        'description': kernel_descriptions.get(kernel_name, 'Unknown kernel'),
        'size': kernel.shape,
        'sum': float(np.sum(kernel)),
        'min': float(np.min(kernel)),
        'max': float(np.max(kernel)),
        'kernel': kernel
    }
    
    return info
