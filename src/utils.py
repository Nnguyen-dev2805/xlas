"""
Utils Module - Các hàm tiện ích cho xử lý ảnh
=====================================

Chức năng chính:
- Load và save ảnh
- Chuyển đổi RGB sang Grayscale
- Padding và resize ảnh
- Tạo convolution kernels
- Normalize ảnh

Author: Image Processing Team
"""

import numpy as np
import cv2
from PIL import Image
import os


def load_image(image_path):
    """
    Load ảnh từ file path hoặc uploaded file
    
    Args:
        image_path: Đường dẫn đến file ảnh hoặc file object
        
    Returns:
        numpy array: Ảnh đã load (RGB format)
    """
    try:
        if hasattr(image_path, 'read'):  # Streamlit uploaded file
            img = Image.open(image_path)
        else:  # File path
            img = Image.open(image_path)
            
        img_array = np.array(img)
        
        # Convert sang RGB nếu cần
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        return img_array
    except Exception as e:
        raise Exception(f"Không thể load ảnh: {str(e)}")


def rgb_to_grayscale(image):
    """
    Chuyển ảnh RGB sang Grayscale sử dụng công thức chuẩn ITU-R BT.601
    Gray = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: Ảnh RGB (numpy array) shape (H, W, 3)
        
    Returns:
        numpy array: Ảnh grayscale shape (H, W)
    """
    if len(image.shape) == 2:
        return image
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input phải là ảnh RGB với shape (H, W, 3)")
    
    # Công thức weighted average theo chuẩn ITU-R BT.601
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return gray.astype(np.uint8)


def normalize_image(image):
    """
    Normalize ảnh về range [0, 255]
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        numpy array: Normalized image
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max - img_min == 0:
        return image.astype(np.uint8)
    
    normalized = ((image - img_min) / (img_max - img_min) * 255)
    return normalized.astype(np.uint8)


def pad_image(image, padding, pad_value=0):
    """
    Thêm padding vào ảnh
    
    Args:
        image: Input image (numpy array)
        padding: Số pixel padding (int)
        pad_value: Giá trị để fill (default=0)
        
    Returns:
        numpy array: Padded image
    """
    if len(image.shape) == 2:
        pad_width = ((padding, padding), (padding, padding))
    else:
        pad_width = ((padding, padding), (padding, padding), (0, 0))
    
    return np.pad(image, pad_width, mode='constant', constant_values=pad_value)


def resize_to_match(image1, image2):
    """
    Resize image1 để match kích thước của image2 bằng cách padding hoặc cropping
    
    Args:
        image1: Ảnh cần resize
        image2: Ảnh reference
        
    Returns:
        numpy array: image1 đã được resize
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    if h1 == h2 and w1 == w2:
        return image1
    
    # Nếu image1 nhỏ hơn, pad nó
    if h1 < h2 or w1 < w2:
        pad_h = max(0, h2 - h1)
        pad_w = max(0, w2 - w1)
        
        if len(image1.shape) == 2:
            padded = np.pad(image1, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        else:
            padded = np.pad(image1, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        
        return padded[:h2, :w2]  # Crop nếu cần
    else:
        # Nếu image1 lớn hơn, crop nó
        return image1[:h2, :w2]


def create_kernel(size, kernel_type='average'):
    """
    Tạo convolution kernel
    
    Args:
        size: Kích thước kernel (int, phải là số lẻ)
        kernel_type: Loại kernel ('average', 'gaussian', 'sharpen', 'edge')
        
    Returns:
        numpy array: Kernel matrix
    """
    if size % 2 == 0:
        raise ValueError("Kernel size phải là số lẻ")
    
    if kernel_type == 'average':
        # Average/Mean filter - làm mờ ảnh
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        
    elif kernel_type == 'gaussian':
        # Gaussian filter - làm mờ tự nhiên hơn
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        sigma = size / 6.0
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        kernel /= kernel.sum()
        
    elif kernel_type == 'sharpen':
        # Sharpening filter - làm sắc nét
        if size == 3:
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)
        else:
            # Tạo sharpen kernel tổng quát
            kernel = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            kernel[center, center] = size * size
            kernel = kernel - 1
            kernel[center, center] = size * size
            
    elif kernel_type == 'edge':
        # Edge detection filter
        if size == 3:
            kernel = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]], dtype=np.float32)
        else:
            # Tạo edge kernel tổng quát
            kernel = np.full((size, size), -1, dtype=np.float32)
            center = size // 2
            kernel[center, center] = size * size - 1
    else:
        # Default: average
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    
    return kernel


def save_image(image, output_path):
    """
    Lưu ảnh ra file
    
    Args:
        image: Ảnh cần lưu (numpy array)
        output_path: Đường dẫn output
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if len(image.shape) == 2:  # Grayscale
        img = Image.fromarray(image.astype(np.uint8), mode='L')
    else:  # RGB
        img = Image.fromarray(image.astype(np.uint8), mode='RGB')
    
    img.save(output_path)


def calculate_image_stats(image):
    """
    Tính các thống kê cơ bản của ảnh
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        dict: Dictionary chứa các thống kê
    """
    return {
        'shape': image.shape,
        'min': np.min(image),
        'max': np.max(image),
        'mean': np.mean(image),
        'std': np.std(image),
        'dtype': str(image.dtype)
    }


def clip_image(image, min_val=0, max_val=255):
    """
    Clip giá trị ảnh về range [min_val, max_val]
    
    Args:
        image: Input image
        min_val: Giá trị minimum
        max_val: Giá trị maximum
        
    Returns:
        numpy array: Clipped image
    """
    return np.clip(image, min_val, max_val).astype(np.uint8)
