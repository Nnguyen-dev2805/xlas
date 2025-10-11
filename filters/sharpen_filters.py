"""
Sharpen Filters
===============

Các bộ lọc làm nét ảnh
- Basic sharpen (manual & library)
- Unsharp mask (manual & library)
- High-pass filters

Author: Image Processing Team
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.kernels import get_sharpen_kernel, get_unsharp_mask_kernel
from filters.blur_filters import gaussian_blur_manual, gaussian_blur_library
from core.convolution import convolution_2d_manual


def sharpen_manual(image):
    """
    Basic sharpen filter - TỰ CODE
    
    Args:
        image: Input image
        
    Returns:
        sharpened: Ảnh đã sharpen
    """
    sharpen_kernel = get_sharpen_kernel()
    
    # Apply convolution
    sharpened = convolution_2d_manual(image, sharpen_kernel, padding=1)
    
    return sharpened


def sharpen_library(image):
    """
    Basic sharpen filter - DÙNG THƯ VIỆN
    
    Args:
        image: Input image
        
    Returns:
        sharpened: Ảnh đã sharpen
    """
    sharpen_kernel = get_sharpen_kernel()
    
    # Sử dụng OpenCV filter2D
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    
    return sharpened


def unsharp_mask_manual(image, blur_kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    """
    Unsharp mask sharpening - TỰ CODE
    
    Thuật toán:
    1. Tạo ảnh blur
    2. Tạo mask = original - blurred
    3. Sharpened = original + amount * mask
    
    Args:
        image: Input image
        blur_kernel_size: Kích thước kernel cho blur
        sigma: Sigma cho Gaussian blur
        amount: Strength của sharpening
        threshold: Threshold cho mask
        
    Returns:
        sharpened: Ảnh đã unsharp mask
    """
    # Step 1: Tạo ảnh blur
    blurred = gaussian_blur_manual(image, blur_kernel_size, sigma)
    
    # Step 2: Tạo mask
    mask = image.astype(float) - blurred.astype(float)
    
    # Step 3: Apply threshold
    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0, mask)
    
    # Step 4: Apply unsharp mask
    sharpened = image.astype(float) + amount * mask
    
    # Clip về [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def unsharp_mask_library(image, blur_kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    """
    Unsharp mask sharpening - DÙNG THƯ VIỆN
    
    Args:
        image: Input image
        blur_kernel_size: Kích thước kernel cho blur
        sigma: Sigma cho Gaussian blur
        amount: Strength của sharpening
        threshold: Threshold cho mask
        
    Returns:
        sharpened: Ảnh đã unsharp mask
    """
    # Step 1: Tạo ảnh blur bằng library
    blurred = gaussian_blur_library(image, blur_kernel_size, sigma)
    
    # Step 2: Tạo mask
    mask = image.astype(float) - blurred.astype(float)
    
    # Step 3: Apply threshold
    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0, mask)
    
    # Step 4: Apply unsharp mask
    sharpened = image.astype(float) + amount * mask
    
    # Clip về [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def high_pass_filter_manual(image, cutoff_freq=0.1):
    """
    High-pass filter - TỰ CODE
    
    Args:
        cutoff_freq: Cutoff frequency (0-1)
        
    Returns:
        filtered: Ảnh sau high-pass filter
    """
    # Tạo high-pass kernel
    # High-pass = Identity - Low-pass
    
    # Low-pass kernel (Gaussian)
    kernel_size = 5
    sigma = 1.0 / cutoff_freq
    
    from filters.kernels import create_gaussian_kernel
    low_pass = create_gaussian_kernel(kernel_size, sigma)
    
    # Tạo high-pass kernel
    high_pass = np.zeros_like(low_pass)
    center = kernel_size // 2
    high_pass[center, center] = 1.0
    high_pass = high_pass - low_pass
    
    # Apply convolution
    filtered = convolution_2d_manual(image, high_pass, padding=kernel_size//2)
    
    return filtered


def laplacian_sharpen_manual(image, alpha=0.2):
    """
    Laplacian sharpening - TỰ CODE
    
    Sharpened = Original - alpha * Laplacian(Original)
    
    Args:
        image: Input image
        alpha: Sharpening strength
        
    Returns:
        sharpened: Ảnh đã sharpen
    """
    from filters.kernels import get_laplacian_kernel
    
    laplacian_kernel = get_laplacian_kernel()
    
    # Tính Laplacian
    laplacian = convolution_2d_manual(image, laplacian_kernel, padding=1)
    
    # Apply sharpening
    sharpened = image.astype(float) - alpha * laplacian.astype(float)
    
    # Clip về [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def adaptive_sharpen_manual(image, local_variance_threshold=100):
    """
    Adaptive sharpening - TỰ CODE
    
    Sharpen nhiều hơn ở vùng có variance cao (edges)
    
    Args:
        image: Input image
        local_variance_threshold: Threshold cho local variance
        
    Returns:
        sharpened: Ảnh đã adaptive sharpen
    """
    height, width = image.shape
    sharpened = image.copy().astype(float)
    
    # Tính local variance
    kernel_size = 5
    pad_size = kernel_size // 2
    
    # Add padding
    from core.convolution import add_padding
    padded = add_padding(image, pad_size)
    
    for i in range(height):
        for j in range(width):
            # Extract local neighborhood
            neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
            local_var = np.var(neighborhood.astype(float))
            
            # Adaptive sharpening strength
            if local_var > local_variance_threshold:
                # High variance area - apply strong sharpening
                strength = 0.5
            else:
                # Low variance area - apply weak sharpening
                strength = 0.1
            
            # Apply Laplacian sharpening với adaptive strength
            if i > 0 and i < height-1 and j > 0 and j < width-1:
                laplacian_val = (4 * image[i, j] - 
                               image[i-1, j] - image[i+1, j] - 
                               image[i, j-1] - image[i, j+1])
                
                sharpened[i, j] = image[i, j] - strength * laplacian_val
    
    # Clip về [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


def compare_sharpen_methods(image):
    """
    So sánh các phương pháp sharpen
    
    Args:
        image: Input image
        
    Returns:
        results: Dictionary chứa kết quả các phương pháp
    """
    results = {}
    
    # Basic sharpen
    results['sharpen_manual'] = sharpen_manual(image)
    results['sharpen_library'] = sharpen_library(image)
    
    # Unsharp mask
    results['unsharp_manual'] = unsharp_mask_manual(image, amount=1.5)
    results['unsharp_library'] = unsharp_mask_library(image, amount=1.5)
    
    # High-pass filter
    results['high_pass_manual'] = high_pass_filter_manual(image)
    
    # Laplacian sharpen
    results['laplacian_sharpen'] = laplacian_sharpen_manual(image)
    
    # Adaptive sharpen
    results['adaptive_sharpen'] = adaptive_sharpen_manual(image)
    
    return results


def analyze_sharpness(image):
    """
    Phân tích độ nét của ảnh
    
    Args:
        image: Input image
        
    Returns:
        analysis: Dictionary chứa phân tích
    """
    # Tính Laplacian variance (measure of sharpness)
    from .kernels import get_laplacian_kernel
    laplacian_kernel = get_laplacian_kernel()
    
    laplacian = convolution_2d_manual(image, laplacian_kernel, padding=1)
    laplacian_var = np.var(laplacian.astype(float))
    
    # Tính gradient magnitude
    from .edge_detection import sobel_edge_detection_manual
    magnitude, _, _, _ = sobel_edge_detection_manual(image)
    gradient_mean = np.mean(magnitude.astype(float))
    
    # Tính Tenengrad (sum of squared gradients)
    tenengrad = np.sum(magnitude.astype(float) ** 2)
    
    analysis = {
        'laplacian_variance': float(laplacian_var),
        'gradient_mean': float(gradient_mean),
        'tenengrad': float(tenengrad),
        'sharpness_score': float(laplacian_var)  # Primary sharpness metric
    }
    
    return analysis
