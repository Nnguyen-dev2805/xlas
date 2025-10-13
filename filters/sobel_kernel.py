import numpy as np
import cv2
from core.convolution import convolution_2d_manual

"""
    Sobel - tìm những vùng trong ảnh mà cường độ sáng thay đổi đột ngổ, tức là đạo hàm lớn nhất của ảnh
    Biên hất của Sobel là giảm nhiễu trước khi tính gradient sau đó tính gradient để phát hiện cạnh
    Đạo hàmảnh chính là nơi gradient(độ dốc) đạt cực đại
    Bảng c(gradient) đo mức thay đổi cường độ sáng giữa các pixel lân cận
    Nơi nào thay đổi mạnh -> khả năng là biên
    Nhưng nếu ảnh có nhiễu, nhiễu cũng tạo ra những thay đổi đột ngột giả -> dẫn đến biên giả
    Làm mờ Guassian nó giữ lại những thay đổi lớn và loại bỏ giao động nhỏ
    Công thức : Sobel = Gaussian ⊗ Derivative
    Trong đó:
    - Gaussian: Làm mượt ảnh
    - Derivative: Tính gradient
    - ⊗: là phép convolution (tích chập)
"""
class SobelKernel:

    # tạo Gaussian 1D để làm mượt 
    @staticmethod
    def create_gaussian_1d(size, sigma):
        # công thức: G(x) = (1 / √(2πσ²)) * exp(-x² / (2σ²))
        if size % 2 == 0:
            raise ValueError("Kernel phải là số lẻ")
        
        # xây dựng trục tọa độ đối xứng quanh 0
        center = size // 2
        x = np.arange(size) - center
        
        # áp dụng công thức Gaussian
        gaussian = np.exp(-(x**2) / (2 * sigma**2))
        
        # normalize để tổng = 1
        gaussian = gaussian / np.sum(gaussian)
        
        return gaussian.astype(np.float32)
    
    # tạo kernel 1D
    @staticmethod
    def create_derivative_1d(size, method='central'):
        if size % 2 == 0:
            raise ValueError("Kernel phải là số lẻ")
        
        derivative = np.zeros(size, dtype=np.float32)
        center = size // 2
        
        if method == 'central':
            # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / 2h
            if size == 3:
                derivative = np.array([-1, 0, 1], dtype=np.float32)
            elif size >= 5:
                # Lấy điểm cách trung tâm 1
                if center > 0:
                    derivative[center - 1] = -1 
                if center < size - 1:
                    derivative[center + 1] = 1
                
                # Điểm xa vẫn có giá trị thông tin về xu hướng thay đổi tổng thể
                # Nhưng càng xa thì đóng góp càng ít vào đọa hàm tại trung tâm -> giảm trọng số
                # Lấy điểm cách trung tâm 2
                if size >= 5 and center > 1 and center < size - 2:
                    derivative[center - 2] = -0.5
                    derivative[center + 2] = 0.5
                    
                # Lấy điểm cách trung tâm 3
                if size >= 7 and center > 2 and center < size - 3:
                    derivative[center - 3] = -0.25
                    derivative[center + 3] = 0.25
                    
        elif method == 'optimal':
            # custom kernel
            if size == 3:
                derivative = np.array([-1, 0, 1], dtype=np.float32)
            elif size == 5:
                derivative = np.array([-1, -2, 0, 2, 1], dtype=np.float32) / 4
            elif size == 7:
                derivative = np.array([-1, -4, -5, 0, 5, 4, 1], dtype=np.float32) / 10
            else:
                return SobelKernel.create_derivative_1d(size, 'central')
        
        return derivative
    
    # làm mượt theo hướng Y
    # tính gradient theo hướng X
    @staticmethod
    def create_sobel_x_kernel(size=3, sigma=1.0, method='central'):
        gaussian_y = SobelKernel.create_gaussian_1d(size, sigma)

        derivative_x = SobelKernel.create_derivative_1d(size, method)
        
        # Gaussian (column) × Derivative (row)
        sobel_x = np.outer(gaussian_y, derivative_x)
        
        return sobel_x.astype(np.float32)
    
    # làm mượt theo hướng X
    # tính gradient theo hướng Y
    @staticmethod
    def create_sobel_y_kernel(size=3, sigma=1.0, method='central'):
        derivative_y = SobelKernel.create_derivative_1d(size, method)
        
        gaussian_x = SobelKernel.create_gaussian_1d(size, sigma)
        
        # Derivative (column) × Gaussian (row)
        sobel_y = np.outer(derivative_y, gaussian_x)
        
        return sobel_y.astype(np.float32)
    
    # áp dụng convolution operation
    @staticmethod
    def apply_convolution(image, kernel, padding=0, stride=1):
        # ảnh xám
        image = image.astype(np.float32)
        kernel = kernel.astype(np.float32)

        output = convolution_2d_manual(image, kernel, padding=padding, stride=stride)

        return output
    
    # Tính gradient magnitude từ Gx và Gy
    # Công thức: G = √(Gx² + Gy²)
    @staticmethod
    def compute_gradient_magnitude(gx, gy):
        magnitude = np.sqrt(gx**2 + gy**2)
        return magnitude
    
    # Tính gradient direction (góc)
    # Công thức: θ = arctan(Gy / Gx)
    @staticmethod
    def compute_gradient_direction(gx, gy):
        direction = np.arctan2(gy, gx)
        return direction
    
    # Chuẩn hóa về [0, 255] và convert sang uint8
    @staticmethod
    def normalize_to_uint8(image):
        # image ở đây sẽ là mảng float sau khi tính gradient magnitude
        image_min = np.min(image) 
        image_max = np.max(image)
        
        if image_max - image_min == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # chuẩn hóa về [0, 255]
        normalized = (image - image_min) / (image_max - image_min) * 255
        
        return normalized.astype(np.uint8)
    
    # phát hiện biên bằng cách đo độ thay đổi cường độ sáng theo hai hướng X và Y
    @staticmethod
    def sobel_edge_detection(image, kernel_size=3, sigma=1.0, padding=None, stride=1, 
                            return_components=False):
        # thêm padding vào để giữ đúng kích thước input và output
        if padding is None:
            padding = kernel_size // 2
        
        # tạo Sobel kernels
        sobel_x = SobelKernel.create_sobel_x_kernel(kernel_size, sigma)
        sobel_y = SobelKernel.create_sobel_y_kernel(kernel_size, sigma)
        
        # áp dụng convolution
        gx = SobelKernel.apply_convolution(image, sobel_x, padding, stride)
        gy = SobelKernel.apply_convolution(image, sobel_y, padding, stride)
        
        # tính gradient magnitude
        magnitude = SobelKernel.compute_gradient_magnitude(gx, gy)
        
        if return_components:
            # Tính gradient direction
            direction = SobelKernel.compute_gradient_direction(gx, gy)
            return magnitude, gx, gy, direction
        else:
            return magnitude

class SobelProcessor:

    # hàm để xử lý nhiều cấu hình kernel khác nhau    
    @staticmethod
    def process_with_multiple_kernels(image, kernel_configs):
        results = {}
        
        for config in kernel_configs:
            size = config.get('size', 3)
            sigma = config.get('sigma', 1.0)
            padding = config.get('padding', size // 2)
            stride = config.get('stride', 1)
            name = config.get('name', f'Sobel_{size}x{size}')
            
            # thực hiện Sobel edge detection
            magnitude, gx, gy, direction = SobelKernel.sobel_edge_detection(
                image, 
                kernel_size=size, 
                sigma=sigma, 
                padding=padding, 
                stride=stride,
                return_components=True
            )
            
            # normalize về uint8 cho hiển thị
            magnitude_uint8 = SobelKernel.normalize_to_uint8(magnitude)
            gx_uint8 = SobelKernel.normalize_to_uint8(np.abs(gx))
            gy_uint8 = SobelKernel.normalize_to_uint8(np.abs(gy))
            
            results[name] = {
                'magnitude': magnitude,
                'magnitude_uint8': magnitude_uint8,
                'gx': gx,
                'gy': gy,
                'gx_uint8': gx_uint8,
                'gy_uint8': gy_uint8,
                'direction': direction,
                'config': config,
                'kernel_x': SobelKernel.create_sobel_x_kernel(size, sigma),
                'kernel_y': SobelKernel.create_sobel_y_kernel(size, sigma)
            }
        
        return results
    
    # hàm tạo các configs chuẩn theo yêu cầu
    @staticmethod
    def create_standard_configs():
        return [
            {
                'size': 3,
                'sigma': 0.8,
                'padding': 1,
                'stride': 1,
                'name': 'I1_Sobel_3x3'
            },
            {
                'size': 5,
                'sigma': 1.2,
                'padding': 2,
                'stride': 1,
                'name': 'I2_Sobel_5x5'
            },
            {
                'size': 7,
                'sigma': 1.5,
                'padding': 3,
                'stride': 2,
                'name': 'I3_Sobel_7x7'
            }
        ]
    
    # hàm in thông tin về kernel
    @staticmethod
    def print_kernel_info(kernel, name):
        print(f"\n{'='*60}")
        print(f"KERNEL: {name}")
        print(f"{'='*60}")
        print(f"Size: {kernel.shape}")
        print(f"Sum: {np.sum(kernel):.6f}")
        print(f"Min: {np.min(kernel):.6f}")
        print(f"Max: {np.max(kernel):.6f}")
        print(f"\nKernel values:")
        print(kernel)
        print(f"{'='*60}")

class SobelLibrary:

    @staticmethod
    def sobel_edge_detection_opencv(image, kernel_size=3, normalize=True):
        """
        Sobel edge detection sử dụng cv2.Sobel()
        
        Args:
            image (numpy.ndarray): Input grayscale image
            kernel_size (int): Kernel size (1, 3, 5, 7, ...)
            normalize (bool): Có normalize về [0, 255] không
            
        Returns:
            tuple: (magnitude, gx, gy)
        """
        image_float = image.astype(np.float32)
        
        # Compute Sobel gradients using OpenCV
        # cv2.Sobel(src, ddepth, dx, dy, ksize)
        # dx=1, dy=0 -> gradient theo X
        # dx=0, dy=1 -> gradient theo Y
        gx = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=kernel_size)
        gy = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=kernel_size)
        
        # Compute magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        if normalize:
            magnitude = SobelKernel.normalize_to_uint8(magnitude)
            gx = SobelKernel.normalize_to_uint8(np.abs(gx))
            gy = SobelKernel.normalize_to_uint8(np.abs(gy))
        
        return magnitude, gx, gy

class SobelComparison:

    @staticmethod
    def compare_manual_vs_library(image, kernel_size=3, sigma=1.0):
        """
        So sánh kết quả giữa implementation tự code và OpenCV
        
        Args:
            image (numpy.ndarray): Input grayscale image
            kernel_size (int): Kernel size
            sigma (float): Sigma cho Gaussian (chỉ dùng cho manual)
            
        Returns:
            dict: Results từ cả 2 methods
        """
        results = {}
        
        # 1. Manual implementation
        print("Đang xử lý với Manual implementation...")
        magnitude_manual, gx_manual, gy_manual, _ = SobelKernel.sobel_edge_detection(
            image, 
            kernel_size=kernel_size, 
            sigma=sigma,
            return_components=True
        )
        
        results['manual'] = {
            'magnitude': SobelKernel.normalize_to_uint8(magnitude_manual),
            'gx': SobelKernel.normalize_to_uint8(np.abs(gx_manual)),
            'gy': SobelKernel.normalize_to_uint8(np.abs(gy_manual)),
            'method': 'Manual (Tự code)'
        }
        
        # 2. OpenCV implementation
        print("Đang xử lý với OpenCV library...")
        magnitude_lib, gx_lib, gy_lib = SobelLibrary.sobel_edge_detection_opencv(
            image,
            kernel_size=kernel_size,
            normalize=True
        )
        
        results['library'] = {
            'magnitude': magnitude_lib,
            'gx': gx_lib,
            'gy': gy_lib,
            'method': 'OpenCV (Thư viện)'
        }
        
        # 3. Compute differences
        diff_magnitude = np.abs(
            results['manual']['magnitude'].astype(np.float32) - 
            results['library']['magnitude'].astype(np.float32)
        )
        
        results['difference'] = {
            'magnitude': diff_magnitude.astype(np.uint8),
            'mean_diff': np.mean(diff_magnitude),
            'max_diff': np.max(diff_magnitude),
            'method': 'Difference (Manual - OpenCV)'
        }
        
        print(f"\nSo sánh:")
        print(f"  Mean difference: {results['difference']['mean_diff']:.2f}")
        print(f"  Max difference: {results['difference']['max_diff']:.2f}")
        
        return results
