"""
Blur Filters
============

Các bộ lọc làm mờ ảnh
- Gaussian blur (manual & library)
- Box filter (manual & library)
- Motion blur

Author: Image Processing Team
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.kernels import create_gaussian_kernel, get_box_filter_kernel
from core.convolution import convolution_2d_manual


def gaussian_blur_manual(image, kernel_size=5, sigma=1.0):
    """
    Gaussian blur - TỰ CODE
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel (phải lẻ)
        sigma: Standard deviation
        
    Returns:
        blurred: Ảnh đã blur
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Tạo Gaussian kernel
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Apply convolution
    padding = kernel_size // 2
    blurred = convolution_2d_manual(image, gaussian_kernel, padding=padding)
    
    return blurred


def gaussian_blur_library(image, kernel_size=5, sigma=1.0):
    """
    Gaussian blur - DÙNG THƯ VIỆN
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel (phải lẻ)
        sigma: Standard deviation
        
    Returns:
        blurred: Ảnh đã blur
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Sử dụng OpenCV
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return blurred


def box_blur_manual(image, kernel_size=5):
    """
    Box blur (average filter) - TỰ CODE
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel
        
    Returns:
        blurred: Ảnh đã blur
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Tạo box filter kernel
    box_kernel = get_box_filter_kernel(kernel_size)
    
    # Apply convolution
    padding = kernel_size // 2
    blurred = convolution_2d_manual(image, box_kernel, padding=padding)
    
    return blurred


def box_blur_library(image, kernel_size=5):
    """
    Box blur - DÙNG THƯ VIỆN
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel
        
    Returns:
        blurred: Ảnh đã blur
    """
    # Sử dụng OpenCV
    blurred = cv2.blur(image, (kernel_size, kernel_size))
    
    return blurred


def motion_blur_manual(image, kernel_size=15, angle=0):
    """
    Motion blur - TỰ CODE
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel
        angle: Góc motion (degrees)
        
    Returns:
        blurred: Ảnh đã motion blur
    """
    # Tạo motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Tính direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Fill kernel along motion direction
    center = kernel_size // 2
    for i in range(kernel_size):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    
    # Normalize kernel
    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)
    
    # Apply convolution
    padding = kernel_size // 2
    blurred = convolution_2d_manual(image, kernel, padding=padding)
    
    return blurred


def bilateral_blur_library(image, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral blur - DÙNG THƯ VIỆN
    
    Preserves edges while smoothing
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        blurred: Ảnh đã bilateral blur
    """
    # Sử dụng OpenCV
    blurred = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    return blurred


def compare_blur_methods(image, kernel_size=5):
    """
    So sánh các phương pháp blur
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel
        
    Returns:
        results: Dictionary chứa kết quả các phương pháp
    """
    results = {}
    
    # Gaussian blur
    results['gaussian_manual'] = gaussian_blur_manual(image, kernel_size, sigma=1.0)
    results['gaussian_library'] = gaussian_blur_library(image, kernel_size, sigma=1.0)
    
    # Box blur
    results['box_manual'] = box_blur_manual(image, kernel_size)
    results['box_library'] = box_blur_library(image, kernel_size)
    
    # Motion blur
    results['motion_horizontal'] = motion_blur_manual(image, kernel_size, angle=0)
    results['motion_diagonal'] = motion_blur_manual(image, kernel_size, angle=45)
    
    # Bilateral blur (library only)
    results['bilateral_library'] = bilateral_blur_library(image)
    
    return results


def analyze_blur_quality(original, blurred):
    """
    Phân tích chất lượng blur
    
    Args:
        original: Ảnh gốc
        blurred: Ảnh đã blur
        
    Returns:
        analysis: Dictionary chứa phân tích
    """
    # Tính MSE
    mse = np.mean((original.astype(float) - blurred.astype(float)) ** 2)
    
    # Tính PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Tính variance (measure of blur)
    original_var = np.var(original.astype(float))
    blurred_var = np.var(blurred.astype(float))
    variance_reduction = (original_var - blurred_var) / original_var
    
    analysis = {
        'mse': float(mse),
        'psnr': float(psnr),
        'original_variance': float(original_var),
        'blurred_variance': float(blurred_var),
        'variance_reduction': float(variance_reduction),
        'blur_strength': float(variance_reduction)
    }
    
    return analysis
