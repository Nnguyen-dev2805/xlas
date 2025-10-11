import numpy as np
import cv2

"""
Median Filter
=============

Median filter thay thế mỗi pixel bằng median của các pixels trong window
- RẤT TỐT cho Salt & Pepper noise
- Giữ edges tốt (không blur edges như Gaussian)
- Non-linear filter

Thuật toán:
1. Slide window (3x3, 5x5, 7x7, ...)
2. Sort các pixel values trong window
3. Lấy giá trị ở giữa (median)
4. Thay thế pixel center
"""

class MedianKernel:
    
    # median filter tự code
    @staticmethod
    def median_filter(image, kernel_size=3):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size phải là số lẻ")
        
        height, width = image.shape
        pad_size = kernel_size // 2
        
        # Padding với edge values (replicate)
        padded = np.pad(image, pad_size, mode='edge')
        
        filtered = np.zeros((height, width), dtype=np.uint8)
        
        # Sliding window
        for i in range(height):
            for j in range(width):
                # Extract window
                window = padded[i:i+kernel_size, j:j+kernel_size]
                
                # Flatten và tìm median
                flat = window.flatten()
                median_val = np.median(flat)
                
                filtered[i, j] = int(median_val)
        
        return filtered
    
    # median filter với method nhanh hơn
    @staticmethod
    def median_filter_fast(image, kernel_size=3):
        """Dùng np.percentile thay vì sort"""
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size phải là số lẻ")
        
        height, width = image.shape
        pad_size = kernel_size // 2
        
        padded = np.pad(image, pad_size, mode='edge')
        
        filtered = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                median_val = np.percentile(window, 50)
                filtered[i, j] = int(median_val)
        
        return filtered


class MedianProcessor:
    
    # xử lý với nhiều kernel sizes
    @staticmethod
    def process_with_multiple_kernels(image, kernel_configs):
        results = {}
        
        for config in kernel_configs:
            size = config.get('size', 3)
            name = config.get('name', f'Median_{size}x{size}')
            
            # Thực hiện median filter
            filtered = MedianKernel.median_filter(image, kernel_size=size)
            
            results[name] = {
                'filtered': filtered,
                'config': config
            }
        
        return results
    
    # tạo configs chuẩn
    @staticmethod
    def create_standard_configs():
        return [
            {
                'size': 3,
                'name': 'Median_3x3'
            },
            {
                'size': 5,
                'name': 'Median_5x5'
            },
            {
                'size': 7,
                'name': 'Median_7x7'
            }
        ]


class MedianLibrary:
    
    @staticmethod
    def median_filter_opencv(image, kernel_size=3):
        """Median filter sử dụng cv2.medianBlur()"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        filtered = cv2.medianBlur(image, kernel_size)
        
        return filtered


class MedianComparison:
    
    @staticmethod
    def compare_manual_vs_library(image, kernel_size=3):
        results = {}
        
        # Manual
        print("Đang xử lý với Manual implementation...")
        filtered_manual = MedianKernel.median_filter(image, kernel_size=kernel_size)
        
        results['manual'] = {
            'filtered': filtered_manual,
            'method': 'Manual (Tự code)'
        }
        
        # OpenCV
        print("Đang xử lý với OpenCV library...")
        filtered_lib = MedianLibrary.median_filter_opencv(image, kernel_size=kernel_size)
        
        results['library'] = {
            'filtered': filtered_lib,
            'method': 'OpenCV (Thư viện)'
        }
        
        # Difference
        diff = np.abs(
            results['manual']['filtered'].astype(np.float32) -
            results['library']['filtered'].astype(np.float32)
        )
        
        results['difference'] = {
            'filtered': diff.astype(np.uint8),
            'mean_diff': np.mean(diff),
            'max_diff': np.max(diff),
            'method': 'Difference'
        }
        
        print(f"\nSo sánh:")
        print(f"  Mean difference: {results['difference']['mean_diff']:.2f}")
        print(f"  Max difference: {results['difference']['max_diff']:.2f}")
        
        return results
