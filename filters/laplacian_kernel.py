import numpy as np
import cv2
from core.convolution import convolution_2d_manual

"""
Laplacian là second derivative operator:
- Tính đạo hàm bậc 2: ∇²f = ∂²f/∂x² + ∂²f/∂y²
- Isotropic: không có hướng (phát hiện edges theo mọi hướng)
- Sensitive với noise → cần blur trước
- Zero-crossing: edges nằm ở nơi Laplacian đổi dấu
"""

class LaplacianKernel:
    
    # kết nối 8 hướng và kết nối 4 hướng
    @staticmethod
    def create_laplacian_3x3(diagonal=False):
        if diagonal:
            kernel = np.array([
                [1,  1,  1],
                [1, -8,  1],
                [1,  1,  1]
            ], dtype=np.float32)
        else:
            kernel = np.array([
                [0,  1,  0],
                [1, -4,  1],
                [0,  1,  0]
            ], dtype=np.float32)
        
        return kernel
    
    # kernel 5x5 tự config
    @staticmethod
    def create_laplacian_5x5(diagonal=False):
        if diagonal:
            kernel = np.ones((5, 5), dtype=np.float32)
            kernel[2, 2] = -24
        else:
            kernel = np.array([
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 24, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1]
            ], dtype=np.float32) / 3.0
        
        return kernel
    
    # kernel 7x7 tự config
    @staticmethod
    def create_laplacian_7x7(diagonal=False):
        if diagonal:
            kernel = np.ones((7, 7), dtype=np.float32)
            kernel[3, 3] = -48
        else:
            kernel = np.zeros((7, 7), dtype=np.float32)
            kernel[3, :] = 1 
            kernel[:, 3] = 1  
            kernel[3, 3] = -12
        
        return kernel
    
    # tao laplacian kernel dựa trên size
    @staticmethod
    def create_laplacian_kernel(size=3, diagonal=False):
        if size == 3:
            return LaplacianKernel.create_laplacian_3x3(diagonal)
        elif size == 5:
            return LaplacianKernel.create_laplacian_5x5(diagonal)
        elif size == 7:
            return LaplacianKernel.create_laplacian_7x7(diagonal)
        else:
            raise ValueError(f"Size {size} chưa được support. Dùng 3, 5, hoặc 7.")
    
    # áp dụng convolution
    @staticmethod
    def apply_convolution(image, kernel, padding=0, stride=1):
        image = image.astype(np.float32)
        kernel = kernel.astype(np.float32)
        
        output = convolution_2d_manual(image, kernel, padding=padding, stride=stride)
        
        return output
    
    # normalize về uint8 để hiển thị
    @staticmethod
    def normalize_to_uint8(image):
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max - image_min == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        normalized = (image - image_min) / (image_max - image_min) * 255
        
        return normalized.astype(np.uint8)
    
    # hàm chính để thực hiện laplacian edge detection
    @staticmethod
    def laplacian_edge_detection(image, kernel_size=3, diagonal=False, padding=None, stride=1):
        if padding is None:
            padding = kernel_size // 2
        
        # Tạo Laplacian kernel
        kernel = LaplacianKernel.create_laplacian_kernel(kernel_size, diagonal)
        
        # Áp dụng convolution
        laplacian = LaplacianKernel.apply_convolution(image, kernel, padding, stride)
        
        return laplacian


class LaplacianProcessor:
    
    # xử lý với nhiều cấu hình kernel khác nhau
    @staticmethod
    def process_with_multiple_kernels(image, kernel_configs):
        results = {}
        
        for config in kernel_configs:
            size = config.get('size', 3)
            diagonal = config.get('diagonal', False)
            padding = config.get('padding', size // 2)
            stride = config.get('stride', 1)
            name = config.get('name', f'Laplacian_{size}x{size}')
            
            # Thực hiện Laplacian edge detection
            laplacian = LaplacianKernel.laplacian_edge_detection(
                image,
                kernel_size=size,
                diagonal=diagonal,
                padding=padding,
                stride=stride
            )
            
            # Normalize về uint8 cho hiển thị
            laplacian_uint8 = LaplacianKernel.normalize_to_uint8(np.abs(laplacian))
            
            results[name] = {
                'laplacian': laplacian,
                'laplacian_uint8': laplacian_uint8,
                'laplacian_abs': np.abs(laplacian),
                'config': config,
                'kernel': LaplacianKernel.create_laplacian_kernel(size, diagonal)
            }
        
        return results
    
    @staticmethod
    def create_standard_configs():
        """
        Tạo configs chuẩn cho Laplacian
        """
        return [
            {
                'size': 3,
                'diagonal': False,
                'padding': 1,
                'stride': 1,
                'name': 'Laplacian_3x3_4conn'
            },
            {
                'size': 3,
                'diagonal': True,
                'padding': 1,
                'stride': 1,
                'name': 'Laplacian_3x3_8conn'
            },
            {
                'size': 5,
                'diagonal': False,
                'padding': 2,
                'stride': 1,
                'name': 'Laplacian_5x5'
            },
            {
                'size': 7,
                'diagonal': False,
                'padding': 3,
                'stride': 1,
                'name': 'Laplacian_7x7'
            }
        ]


class LaplacianLibrary:
    
    @staticmethod
    def laplacian_edge_detection_opencv(image, kernel_size=3, normalize=True):
        """
        Laplacian edge detection sử dụng cv2.Laplacian()
        
        Args:
            image (numpy.ndarray): Input grayscale image
            kernel_size (int): Kernel size (1, 3, 5, 7)
            normalize (bool): Normalize về [0, 255]
            
        Returns:
            numpy.ndarray: Laplacian result
        """
        image_float = image.astype(np.float32)
        
        # Compute Laplacian using OpenCV
        laplacian = cv2.Laplacian(image_float, cv2.CV_32F, ksize=kernel_size)
        
        if normalize:
            laplacian_normalized = LaplacianKernel.normalize_to_uint8(np.abs(laplacian))
            return laplacian_normalized
        else:
            return laplacian
    
    @staticmethod
    def log_edge_detection(image, kernel_size=5, sigma=1.0, normalize=True):
        """
        Laplacian of Gaussian (LoG) edge detection
        
        LoG = Laplacian ∘ Gaussian (blur trước rồi mới Laplacian)
        → Giảm noise trước khi tính second derivative
        
        Args:
            image (numpy.ndarray): Input grayscale image
            kernel_size (int): Gaussian kernel size
            sigma (float): Gaussian sigma
            normalize (bool): Normalize
            
        Returns:
            numpy.ndarray: LoG result
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Step 1: Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Step 2: Laplacian
        laplacian = cv2.Laplacian(blurred.astype(np.float32), cv2.CV_32F, ksize=3)
        
        if normalize:
            laplacian_normalized = LaplacianKernel.normalize_to_uint8(np.abs(laplacian))
            return laplacian_normalized
        else:
            return laplacian


class LaplacianComparison:
    
    @staticmethod
    def compare_manual_vs_library(image, kernel_size=3, diagonal=False):
        """
        So sánh kết quả giữa implementation tự code và OpenCV
        
        Args:
            image (numpy.ndarray): Input grayscale image
            kernel_size (int): Kernel size
            diagonal (bool): Diagonal neighbors (chỉ cho manual)
            
        Returns:
            dict: Results từ cả 2 methods
        """
        results = {}
        
        # 1. Manual implementation
        print("Đang xử lý với Manual implementation...")
        laplacian_manual = LaplacianKernel.laplacian_edge_detection(
            image,
            kernel_size=kernel_size,
            diagonal=diagonal
        )
        
        results['manual'] = {
            'laplacian': LaplacianKernel.normalize_to_uint8(np.abs(laplacian_manual)),
            'method': 'Manual (Tự code)'
        }
        
        # 2. OpenCV implementation
        print("Đang xử lý với OpenCV library...")
        laplacian_lib = LaplacianLibrary.laplacian_edge_detection_opencv(
            image,
            kernel_size=kernel_size,
            normalize=True
        )
        
        results['library'] = {
            'laplacian': laplacian_lib,
            'method': 'OpenCV (Thư viện)'
        }
        
        # 3. Compute differences
        diff_laplacian = np.abs(
            results['manual']['laplacian'].astype(np.float32) -
            results['library']['laplacian'].astype(np.float32)
        )
        
        results['difference'] = {
            'laplacian': diff_laplacian.astype(np.uint8),
            'mean_diff': np.mean(diff_laplacian),
            'max_diff': np.max(diff_laplacian),
            'method': 'Difference (Manual - OpenCV)'
        }
        
        print(f"\nSo sánh:")
        print(f"  Mean difference: {results['difference']['mean_diff']:.2f}")
        print(f"  Max difference: {results['difference']['max_diff']:.2f}")
        
        return results
