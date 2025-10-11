import numpy as np
import cv2
from core.convolution import convolution_2d_manual

"""
Gaussian kernel làm mượt ảnh bằng cách convolve với Gaussian kernel
- Giảm nhiễu (noise reduction)
- Làm mờ ảnh (blur)
- Chuẩn bị cho các bước tiếp theo (edge detection, etc.)
"""

class GaussianKernel:
    
    # tạo Gaussian 1D
    @staticmethod
    def create_gaussian_1d(size, sigma):
        if size % 2 == 0:
            raise ValueError("Kernel size phải là số lẻ")
        
        center = size // 2
        x = np.arange(size) - center
        
        # Gaussian formula
        gaussian = np.exp(-(x**2) / (2 * sigma**2))
        
        # Normalize để tổng = 1
        gaussian = gaussian / np.sum(gaussian)
        
        return gaussian.astype(np.float32)
    
    # tạo Gaussian 2D
    @staticmethod
    def create_gaussian_2d(size, sigma):
        if size % 2 == 0:
            raise ValueError("Kernel size phải là số lẻ")
        
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
        
        # Gaussian 2D formula
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize
        gaussian = gaussian / np.sum(gaussian)
        
        return gaussian.astype(np.float32)
    
    # tạo Gaussian 2D từ separable (efficient)
    @staticmethod
    def create_gaussian_2d_separable(size, sigma):
        """Tạo từ outer product của 2 Gaussian 1D"""
        gaussian_1d = GaussianKernel.create_gaussian_1d(size, sigma)
        gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
        return gaussian_2d.astype(np.float32)
    
    # áp dụng convolution
    @staticmethod
    def apply_convolution(image, kernel, padding=0, stride=1):
        image = image.astype(np.float32)
        kernel = kernel.astype(np.float32)
        
        output = convolution_2d_manual(image, kernel, padding=padding, stride=stride)
        
        return output
    
    # clip về uint8
    @staticmethod
    def clip_to_uint8(image):
        clipped = np.clip(image, 0, 255)
        return clipped.astype(np.uint8)
    
    # Gaussian blur filter
    @staticmethod
    def gaussian_blur(image, kernel_size=3, sigma=1.0, padding=None, stride=1):
        if padding is None:
            padding = kernel_size // 2
        
        # Tạo Gaussian kernel
        kernel = GaussianKernel.create_gaussian_2d_separable(kernel_size, sigma)
        
        # Áp dụng convolution
        blurred = GaussianKernel.apply_convolution(image, kernel, padding, stride)
        
        # Clip về uint8
        blurred = GaussianKernel.clip_to_uint8(blurred)
        
        return blurred


class GaussianProcessor:
    
    # xử lý với nhiều cấu hình
    @staticmethod
    def process_with_multiple_kernels(image, kernel_configs):
        results = {}
        
        for config in kernel_configs:
            size = config.get('size', 3)
            sigma = config.get('sigma', 1.0)
            padding = config.get('padding', size // 2)
            stride = config.get('stride', 1)
            name = config.get('name', f'Gaussian_{size}x{size}_σ={sigma}')
            
            # Thực hiện Gaussian blur
            blurred = GaussianKernel.gaussian_blur(
                image,
                kernel_size=size,
                sigma=sigma,
                padding=padding,
                stride=stride
            )
            
            results[name] = {
                'blurred': blurred,
                'config': config,
                'kernel': GaussianKernel.create_gaussian_2d_separable(size, sigma)
            }
        
        return results
    
    # tạo configs chuẩn
    @staticmethod
    def create_standard_configs():
        return [
            {
                'size': 3,
                'sigma': 0.5,
                'padding': 1,
                'stride': 1,
                'name': 'Gaussian_3x3_Light'
            },
            {
                'size': 3,
                'sigma': 1.0,
                'padding': 1,
                'stride': 1,
                'name': 'Gaussian_3x3_Medium'
            },
            {
                'size': 5,
                'sigma': 1.5,
                'padding': 2,
                'stride': 1,
                'name': 'Gaussian_5x5'
            },
            {
                'size': 7,
                'sigma': 2.0,
                'padding': 3,
                'stride': 1,
                'name': 'Gaussian_7x7'
            }
        ]


class GaussianLibrary:
    
    @staticmethod
    def gaussian_blur_opencv(image, kernel_size=3, sigma=1.0):
        """Gaussian blur sử dụng cv2.GaussianBlur()"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        return blurred


class GaussianComparison:
    
    @staticmethod
    def compare_manual_vs_library(image, kernel_size=3, sigma=1.0):
        results = {}
        
        # Manual
        print("Đang xử lý với Manual implementation...")
        blurred_manual = GaussianKernel.gaussian_blur(
            image,
            kernel_size=kernel_size,
            sigma=sigma
        )
        
        results['manual'] = {
            'blurred': blurred_manual,
            'method': 'Manual (Tự code)'
        }
        
        # OpenCV
        print("Đang xử lý với OpenCV library...")
        blurred_lib = GaussianLibrary.gaussian_blur_opencv(
            image,
            kernel_size=kernel_size,
            sigma=sigma
        )
        
        results['library'] = {
            'blurred': blurred_lib,
            'method': 'OpenCV (Thư viện)'
        }
        
        # Difference
        diff = np.abs(
            results['manual']['blurred'].astype(np.float32) -
            results['library']['blurred'].astype(np.float32)
        )
        
        results['difference'] = {
            'blurred': diff.astype(np.uint8),
            'mean_diff': np.mean(diff),
            'max_diff': np.max(diff),
            'method': 'Difference'
        }
        
        print(f"\nSo sánh:")
        print(f"  Mean difference: {results['difference']['mean_diff']:.2f}")
        print(f"  Max difference: {results['difference']['max_diff']:.2f}")
        
        return results
