"""
Edge Detection Filters
======================

Các bộ lọc phát hiện biên
- Manual và library implementations
- Sobel, Prewitt, Laplacian, Canny

Author: Image Processing Team
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.kernels import get_sobel_kernels, get_prewitt_kernels, get_laplacian_kernel
from core.convolution import convolution_2d_manual, convolution_2d_library


def sobel_edge_detection_manual(image):
    """
    Sobel edge detection - TỰ CODE
    
    Args:
        image: Grayscale image
        
    Returns:
        magnitude: Edge magnitude
        direction: Edge direction
        grad_x: X gradient
        grad_y: Y gradient
    """
    sobel_x, sobel_y = get_sobel_kernels()
    
    # Tính gradient X và Y
    grad_x = convolution_2d_manual(image, sobel_x, padding=1)
    grad_y = convolution_2d_manual(image, sobel_y, padding=1)
    
    # Tính magnitude và direction
    magnitude = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    direction = np.arctan2(grad_y.astype(float), grad_x.astype(float))
    
    return magnitude, direction, grad_x, grad_y


def sobel_edge_detection_library(image):
    """
    Sobel edge detection - DÙNG THƯ VIỆN
    
    Args:
        image: Grayscale image
        
    Returns:
        magnitude: Edge magnitude
        direction: Edge direction
        grad_x: X gradient
        grad_y: Y gradient
    """
    # Sử dụng OpenCV
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Tính magnitude và direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    direction = np.arctan2(grad_y, grad_x)
    
    grad_x = np.clip(np.abs(grad_x), 0, 255).astype(np.uint8)
    grad_y = np.clip(np.abs(grad_y), 0, 255).astype(np.uint8)
    
    return magnitude, direction, grad_x, grad_y


def prewitt_edge_detection_manual(image):
    """
    Prewitt edge detection - TỰ CODE
    
    Args:
        image: Grayscale image
        
    Returns:
        magnitude: Edge magnitude
        grad_x: X gradient
        grad_y: Y gradient
    """
    prewitt_x, prewitt_y = get_prewitt_kernels()
    
    # Tính gradient X và Y
    grad_x = convolution_2d_manual(image, prewitt_x, padding=1)
    grad_y = convolution_2d_manual(image, prewitt_y, padding=1)
    
    # Tính magnitude
    magnitude = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude, grad_x, grad_y


def laplacian_edge_detection_manual(image):
    """
    Laplacian edge detection - TỰ CODE
    
    Args:
        image: Grayscale image
        
    Returns:
        edges: Edge image
    """
    laplacian = get_laplacian_kernel()
    
    # Apply Laplacian
    edges = convolution_2d_manual(image, laplacian, padding=1)
    
    return edges


def laplacian_edge_detection_library(image):
    """
    Laplacian edge detection - DÙNG THƯ VIỆN
    
    Args:
        image: Grayscale image
        
    Returns:
        edges: Edge image
    """
    # Sử dụng OpenCV
    edges = cv2.Laplacian(image, cv2.CV_64F)
    edges = np.clip(np.abs(edges), 0, 255).astype(np.uint8)
    
    return edges


def canny_edge_detection_library(image, low_threshold=50, high_threshold=150):
    """
    Canny edge detection - DÙNG THƯ VIỆN
    
    Args:
        image: Grayscale image
        low_threshold: Low threshold for edge linking
        high_threshold: High threshold for edge linking
        
    Returns:
        edges: Binary edge image
    """
    # Sử dụng OpenCV
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    return edges


def compare_edge_detectors(image):
    """
    So sánh các phương pháp phát hiện biên
    
    Args:
        image: Grayscale image
        
    Returns:
        results: Dictionary chứa kết quả các phương pháp
    """
    results = {}
    
    # Sobel
    sobel_mag, sobel_dir, sobel_x, sobel_y = sobel_edge_detection_manual(image)
    results['sobel_manual'] = {
        'magnitude': sobel_mag,
        'direction': sobel_dir,
        'grad_x': sobel_x,
        'grad_y': sobel_y
    }
    
    sobel_mag_lib, sobel_dir_lib, sobel_x_lib, sobel_y_lib = sobel_edge_detection_library(image)
    results['sobel_library'] = {
        'magnitude': sobel_mag_lib,
        'direction': sobel_dir_lib,
        'grad_x': sobel_x_lib,
        'grad_y': sobel_y_lib
    }
    
    # Prewitt
    prewitt_mag, prewitt_x, prewitt_y = prewitt_edge_detection_manual(image)
    results['prewitt_manual'] = {
        'magnitude': prewitt_mag,
        'grad_x': prewitt_x,
        'grad_y': prewitt_y
    }
    
    # Laplacian
    laplacian_manual = laplacian_edge_detection_manual(image)
    laplacian_library = laplacian_edge_detection_library(image)
    results['laplacian_manual'] = laplacian_manual
    results['laplacian_library'] = laplacian_library
    
    # Canny
    canny_edges = canny_edge_detection_library(image)
    results['canny_library'] = canny_edges
    
    return results
