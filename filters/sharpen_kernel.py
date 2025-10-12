import numpy as np
import cv2
from core.convolution import convolution_2d_manual

"""
Sharpen làm nét ảnh bằng cách tăng cường high-frequency components (edges, details)

Công thức: Sharpened = Original + α * Laplacian
Trong đó:
- α (alpha): Cường độ sharpen (0.5 = nhẹ, 1.0 = trung bình, 2.0 = mạnh)
- Laplacian: Second derivative (phát hiện edges)

Sharpen kernel = Identity + α * Laplacian

Ví dụ (α=1.0):
3x3:                    5x5:
[ 0 -1  0]              [-1 -1 -1 -1 -1]
[-1  5 -1]              [-1 -1 -1 -1 -1]
[ 0 -1  0]              [-1 -1 25 -1 -1]
                        [-1 -1 -1 -1 -1]
                        [-1 -1 -1 -1 -1]
"""

class SharpenKernel:
    
    # tạo sharpen kernel 3x3
    @staticmethod
    def create_sharpen_3x3(alpha=1.0, diagonal=False):
        """
        Alpha: cường độ sharpen (0.5=nhẹ, 1.0=trung bình, 2.0=mạnh)
        Diagonal: include diagonal neighbors
        """
        if diagonal:
            # 8-connected Laplacian
            laplacian = np.array([
                [1,  1,  1],
                [1, -8,  1],
                [1,  1,  1]
            ], dtype=np.float32)
        else:
            # 4-connected Laplacian
            laplacian = np.array([
                [0,  1,  0],
                [1, -4,  1],
                [0,  1,  0]
            ], dtype=np.float32)
        
        # Identity matrix
        identity = np.zeros((3, 3), dtype=np.float32)
        identity[1, 1] = 1.0
        
        # Sharpen = Identity - alpha * Laplacian
        sharpen = identity - alpha * laplacian
        
        return sharpen
    
    # tạo sharpen kernel 5x5
    @staticmethod
    def create_sharpen_5x5(alpha=1.0, diagonal=False):
        if diagonal:
            laplacian = np.ones((5, 5), dtype=np.float32)
            laplacian[2, 2] = -24
        else:
            laplacian = np.array([
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 24, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ], dtype=np.float32) / 3.0
        
        identity = np.zeros((5, 5), dtype=np.float32)
        identity[2, 2] = 1.0
        
        sharpen = identity - alpha * laplacian
        
        return sharpen
    
    # tạo sharpen kernel 7x7
    @staticmethod
    def create_sharpen_7x7(alpha=1.0, diagonal=False):
        if diagonal:
            laplacian = np.ones((7, 7), dtype=np.float32)
            laplacian[3, 3] = -48
        else:
            laplacian = np.zeros((7, 7), dtype=np.float32)
            laplacian[3, :] = 1
            laplacian[:, 3] = 1
            laplacian[3, 3] = -12
        
        identity = np.zeros((7, 7), dtype=np.float32)
        identity[3, 3] = 1.0
        
        sharpen = identity - alpha * laplacian
        
        return sharpen
    
    # tạo sharpen kernel dựa trên size
    @staticmethod
    def create_sharpen_kernel(size=3, alpha=1.0, diagonal=False):
        if size == 3:
            return SharpenKernel.create_sharpen_3x3(alpha, diagonal)
        elif size == 5:
            return SharpenKernel.create_sharpen_5x5(alpha, diagonal)
        elif size == 7:
            return SharpenKernel.create_sharpen_7x7(alpha, diagonal)
        else:
            raise ValueError(f"Size {size} chưa được support. Dùng 3, 5, hoặc 7.")
    
    # áp dụng convolution
    @staticmethod
    def apply_convolution(image, kernel, padding=0, stride=1):
        image = image.astype(np.float32)
        kernel = kernel.astype(np.float32)
        
        output = convolution_2d_manual(image, kernel, padding=padding, stride=stride)
        
        return output
    
    # clip về [0, 255]
    @staticmethod
    def clip_to_uint8(image):
        clipped = np.clip(image, 0, 255)
        return clipped.astype(np.uint8)
    
    # sharpen edge detection
    @staticmethod
    def sharpen_filter(image, kernel_size=3, alpha=1.0, diagonal=False, padding=None, stride=1):
        if padding is None:
            padding = kernel_size // 2
        
        # Tạo sharpen kernel
        kernel = SharpenKernel.create_sharpen_kernel(kernel_size, alpha, diagonal)
        
        # Áp dụng convolution
        sharpened = SharpenKernel.apply_convolution(image, kernel, padding, stride)
        
        # Clip về [0, 255]
        sharpened = SharpenKernel.clip_to_uint8(sharpened)
        
        return sharpened
    
    # unsharp masking (method 2)
    @staticmethod
    def unsharp_masking(image, kernel_size=3, alpha=1.0):
        """
        Unsharp Masking: Image + alpha * (Image - Blurred)
        
        Steps:
        1. Blur image (Gaussian)
        2. Subtract: mask = Image - Blurred
        3. Add back: Sharpened = Image + alpha * mask
        """
        image = image.astype(np.float32)
        
        # Blur
        if kernel_size == 3:
            blur_kernel = np.ones((3, 3), dtype=np.float32) / 9
        elif kernel_size == 5:
            blur_kernel = np.ones((5, 5), dtype=np.float32) / 25
        else:
            blur_kernel = np.ones((7, 7), dtype=np.float32) / 49
        
        padding = kernel_size // 2
        blurred = convolution_2d_manual(image, blur_kernel, padding=padding, stride=1)
        
        # Mask
        mask = image - blurred
        
        # Sharpen
        sharpened = image + alpha * mask
        
        # Clip
        sharpened = SharpenKernel.clip_to_uint8(sharpened)
        
        return sharpened


class SharpenProcessor:
    
    # xử lý với nhiều cấu hình kernel
    @staticmethod
    def process_with_multiple_kernels(image, kernel_configs):
        results = {}
        
        for config in kernel_configs:
            size = config.get('size', 3)
            alpha = config.get('alpha', 1.0)
            diagonal = config.get('diagonal', False)
            padding = config.get('padding', size // 2)
            stride = config.get('stride', 1)
            name = config.get('name', f'Sharpen_{size}x{size}')
            
            # Thực hiện sharpen
            sharpened = SharpenKernel.sharpen_filter(
                image,
                kernel_size=size,
                alpha=alpha,
                diagonal=diagonal,
                padding=padding,
                stride=stride
            )
            
            results[name] = {
                'sharpened': sharpened,
                'config': config,
                'kernel': SharpenKernel.create_sharpen_kernel(size, alpha, diagonal)
            }
        
        return results
    
    # tạo configs chuẩn
    @staticmethod
    def create_standard_configs():
        return [
            {
                'size': 3,
                'alpha': 0.5,
                'diagonal': False,
                'padding': 1,
                'stride': 1,
                'name': 'Sharpen_3x3_Light'
            },
            {
                'size': 3,
                'alpha': 1.0,
                'diagonal': False,
                'padding': 1,
                'stride': 1,
                'name': 'Sharpen_3x3_Medium'
            },
            {
                'size': 3,
                'alpha': 2.0,
                'diagonal': False,
                'padding': 1,
                'stride': 1,
                'name': 'Sharpen_3x3_Strong'
            },
            {
                'size': 5,
                'alpha': 1.0,
                'diagonal': False,
                'padding': 2,
                'stride': 1,
                'name': 'Sharpen_5x5'
            }
        ]


class SharpenLibrary:
    
    @staticmethod
    def sharpen_filter_opencv(image, kernel_size=3, alpha=1.0):
        """
        Sharpen filter sử dụng OpenCV
        """
        image = image.astype(np.uint8)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
        
        return sharpened
    
    @staticmethod
    def sharpen_filter2d(image, kernel_size=3, alpha=1.0, diagonal=False):
        """
        Sharpen sử dụng cv2.filter2D
        """
        from filters.sharpen_kernel import SharpenKernel
        
        kernel = SharpenKernel.create_sharpen_kernel(kernel_size, alpha, diagonal)
        sharpened = cv2.filter2D(image, -1, kernel)
        
        return sharpened


class SharpenComparison:
    
    @staticmethod
    def compare_manual_vs_library(image, kernel_size=3, alpha=1.0, diagonal=False):
        results = {}
        
        # Manual
        print("Đang xử lý với Manual implementation...")
        sharpened_manual = SharpenKernel.sharpen_filter(
            image,
            kernel_size=kernel_size,
            alpha=alpha,
            diagonal=diagonal
        )
        
        results['manual'] = {
            'sharpened': sharpened_manual,
            'method': 'Manual (Tự code)'
        }
        
        # OpenCV
        print("Đang xử lý với OpenCV library...")
        sharpened_lib = SharpenLibrary.sharpen_filter_opencv(
            image,
            kernel_size=kernel_size,
            alpha=alpha
        )
        
        results['library'] = {
            'sharpened': sharpened_lib,
            'method': 'OpenCV (Thư viện)'
        }
        
        # Difference
        diff = np.abs(
            results['manual']['sharpened'].astype(np.float32) -
            results['library']['sharpened'].astype(np.float32)
        )
        
        results['difference'] = {
            'sharpened': diff.astype(np.uint8),
            'mean_diff': np.mean(diff),
            'max_diff': np.max(diff),
            'method': 'Difference'
        }
        
        print(f"\nSo sánh:")
        print(f"  Mean difference: {results['difference']['mean_diff']:.2f}")
        print(f"  Max difference: {results['difference']['max_diff']:.2f}")
        
        return results
