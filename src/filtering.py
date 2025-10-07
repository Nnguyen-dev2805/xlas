"""
Filtering Module - Xử lý Filtering cho Bài 2
==========================================

Chức năng chính:
1. Convolution với các kernel khác nhau (I1, I2, I3)
2. Median filtering (I4)
3. Min filtering (I5)
4. Thresholding operation (I6)

Các phép tích chập:
- I1: Kernel 3x3, padding=1
- I2: Kernel 5x5, padding=2  
- I3: Kernel 7x7, padding=3, stride=2
- I4: Median filter 3x3 trên I3
- I5: Min filter 5x5 trên I1
- I6: Thresholding I4 vs I5

Author: Image Processing Team
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter, minimum_filter
import cv2
from .utils import create_kernel, pad_image, resize_to_match


def convolution_2d(image, kernel, padding=0, stride=1):
    """
    Thực hiện phép tích chập 2D (convolution) từ scratch
    
    Args:
        image: Input image (numpy array)
        kernel: Convolution kernel (numpy array)
        padding: Số pixel padding
        stride: Bước nhảy (default=1)
        
    Returns:
        numpy array: Convolved image
    """
    # Kiểm tra input
    if len(image.shape) != 2:
        raise ValueError("Image phải là grayscale")
    
    kernel_h, kernel_w = kernel.shape
    
    # Thêm padding nếu cần
    if padding > 0:
        padded_image = pad_image(image, padding)
    else:
        padded_image = image.copy()
    
    img_h, img_w = padded_image.shape
    
    # Tính kích thước output
    out_h = (img_h - kernel_h) // stride + 1
    out_w = (img_w - kernel_w) // stride + 1
    
    # Khởi tạo output
    output = np.zeros((out_h, out_w), dtype=np.float32)
    
    # Thực hiện convolution
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Tính vị trí trong ảnh gốc
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kernel_h
            end_j = start_j + kernel_w
            
            # Lấy region of interest
            roi = padded_image[start_i:end_i, start_j:end_j]
            
            # Tính tích chập (element-wise multiply và sum)
            conv_sum = np.sum(roi * kernel)
            output[i, j] = conv_sum
    
    # Clip về [0, 255] và convert về uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output


def apply_median_filter(image, kernel_size=3):
    """
    Áp dụng median filter
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel (default=3)
        
    Returns:
        numpy array: Filtered image
    """
    return median_filter(image, size=kernel_size)


def apply_min_filter(image, kernel_size=5):
    """
    Áp dụng min filter (erosion-like operation)
    
    Args:
        image: Input image
        kernel_size: Kích thước kernel (default=5)
        
    Returns:
        numpy array: Filtered image
    """
    return minimum_filter(image, size=kernel_size)


def threshold_operation(image1, image2):
    """
    Thực hiện thresholding: Nếu image1(x,y) > image2(x,y) thì output=0, ngược lại output=image2(x,y)
    
    Args:
        image1: Ảnh thứ nhất (I4)
        image2: Ảnh thứ hai (I5)
        
    Returns:
        numpy array: Thresholded image
    """
    # Đảm bảo hai ảnh có cùng kích thước
    if image1.shape != image2.shape:
        print(f"Resizing images: {image1.shape} vs {image2.shape}")
        image1 = resize_to_match(image1, image2)
    
    # Thực hiện thresholding
    result = np.where(image1 > image2, 0, image2)
    
    return result.astype(np.uint8)


def process_task2(image):
    """
    Xử lý đầy đủ Bài 2 - Filtering Operations
    
    Args:
        image: Ảnh grayscale
        
    Returns:
        dict: Dictionary chứa tất cả kết quả
    """
    results = {}
    
    print("🔄 Đang xử lý Bài 2 - Filtering Operations...")
    
    # Ảnh gốc
    results['original_image'] = image
    print(f"✓ Ảnh gốc: {image.shape}")
    
    # I1: Convolution với kernel 3x3, padding=1
    print("🔄 Tạo I1: Kernel 3x3, padding=1...")
    kernel_3x3 = create_kernel(3, 'average')
    i1 = convolution_2d(image, kernel_3x3, padding=1, stride=1)
    results['i1'] = i1
    results['kernel_3x3'] = kernel_3x3
    print(f"✓ I1 hoàn thành: {i1.shape}")
    
    # I2: Convolution với kernel 5x5, padding=2
    print("🔄 Tạo I2: Kernel 5x5, padding=2...")
    kernel_5x5 = create_kernel(5, 'average')
    i2 = convolution_2d(image, kernel_5x5, padding=2, stride=1)
    results['i2'] = i2
    results['kernel_5x5'] = kernel_5x5
    print(f"✓ I2 hoàn thành: {i2.shape}")
    
    # I3: Convolution với kernel 7x7, padding=3, stride=2
    print("🔄 Tạo I3: Kernel 7x7, padding=3, stride=2...")
    kernel_7x7 = create_kernel(7, 'average')
    i3 = convolution_2d(image, kernel_7x7, padding=3, stride=2)
    results['i3'] = i3
    results['kernel_7x7'] = kernel_7x7
    print(f"✓ I3 hoàn thành: {i3.shape}")
    
    # I4: Median filter 3x3 trên I3
    print("🔄 Tạo I4: Median filter 3x3 trên I3...")
    i4 = apply_median_filter(i3, kernel_size=3)
    results['i4'] = i4
    print(f"✓ I4 hoàn thành: {i4.shape}")
    
    # I5: Min filter 5x5 trên I1
    print("🔄 Tạo I5: Min filter 5x5 trên I1...")
    i5 = apply_min_filter(i1, kernel_size=5)
    results['i5'] = i5
    print(f"✓ I5 hoàn thành: {i5.shape}")
    
    # I6: Thresholding I4 vs I5
    print("🔄 Tạo I6: Thresholding I4 vs I5...")
    i6 = threshold_operation(i4, i5)
    results['i6'] = i6
    print(f"✓ I6 hoàn thành: {i6.shape}")
    
    print("✅ Hoàn thành Bài 2!")
    return results


def create_kernel_visualization():
    """
    Tạo visualization cho các kernels
    
    Returns:
        dict: Dictionary chứa kernels để hiển thị
    """
    kernels = {}
    
    # Tạo các kernels
    kernels['3x3_average'] = create_kernel(3, 'average')
    kernels['5x5_average'] = create_kernel(5, 'average')
    kernels['7x7_average'] = create_kernel(7, 'average')
    kernels['3x3_gaussian'] = create_kernel(3, 'gaussian')
    kernels['3x3_sharpen'] = create_kernel(3, 'sharpen')
    kernels['3x3_edge'] = create_kernel(3, 'edge')
    
    return kernels


def compare_filtering_methods(image):
    """
    So sánh các phương pháp filtering khác nhau
    
    Args:
        image: Input grayscale image
        
    Returns:
        dict: Kết quả các phương pháp filtering
    """
    results = {}
    
    # Original
    results['original'] = image
    
    # Average filters
    results['avg_3x3'] = convolution_2d(image, create_kernel(3, 'average'), padding=1)
    results['avg_5x5'] = convolution_2d(image, create_kernel(5, 'average'), padding=2)
    
    # Gaussian filters
    results['gauss_3x3'] = convolution_2d(image, create_kernel(3, 'gaussian'), padding=1)
    results['gauss_5x5'] = convolution_2d(image, create_kernel(5, 'gaussian'), padding=2)
    
    # Sharpen filter
    results['sharpen'] = convolution_2d(image, create_kernel(3, 'sharpen'), padding=1)
    
    # Edge detection
    results['edge'] = convolution_2d(image, create_kernel(3, 'edge'), padding=1)
    
    # Median filter
    results['median_3x3'] = apply_median_filter(image, 3)
    results['median_5x5'] = apply_median_filter(image, 5)
    
    # Min filter
    results['min_3x3'] = apply_min_filter(image, 3)
    results['min_5x5'] = apply_min_filter(image, 5)
    
    return results


def analyze_filter_effects(original, filtered, filter_name):
    """
    Phân tích hiệu ứng của filter
    
    Args:
        original: Ảnh gốc
        filtered: Ảnh sau filter
        filter_name: Tên filter
        
    Returns:
        dict: Thống kê so sánh
    """
    # Tính MSE (Mean Squared Error)
    mse = np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
    
    # Tính PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Tính độ tương quan
    correlation = np.corrcoef(original.flatten(), filtered.flatten())[0, 1]
    
    # Thống kê cơ bản
    orig_stats = {
        'mean': np.mean(original),
        'std': np.std(original),
        'min': np.min(original),
        'max': np.max(original)
    }
    
    filt_stats = {
        'mean': np.mean(filtered),
        'std': np.std(filtered),
        'min': np.min(filtered),
        'max': np.max(filtered)
    }
    
    return {
        'filter_name': filter_name,
        'mse': float(mse),
        'psnr': float(psnr),
        'correlation': float(correlation),
        'original_stats': orig_stats,
        'filtered_stats': filt_stats
    }


def custom_convolution_with_opencv_comparison(image, kernel):
    """
    So sánh kết quả convolution tự implement với OpenCV
    
    Args:
        image: Input image
        kernel: Convolution kernel
        
    Returns:
        dict: Kết quả so sánh
    """
    # Custom implementation
    custom_result = convolution_2d(image, kernel, padding=kernel.shape[0]//2)
    
    # OpenCV implementation
    opencv_result = cv2.filter2D(image, -1, kernel)
    
    # Tính độ khác biệt
    diff = np.abs(custom_result.astype(np.float32) - opencv_result.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    return {
        'custom_result': custom_result,
        'opencv_result': opencv_result,
        'difference': diff.astype(np.uint8),
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'are_similar': max_diff < 5  # Threshold for similarity
    }
