import numpy as np
import math
import cv2 as cv

class KernelGenerator:
    @staticmethod
    def gaussian_kernel(size, sigma=1.0):
        """
        GAUSSIAN KERNEL - Lọc làm mờ (Blur Filter)
        
        Lý thuyết:
        - Dựa trên phân phối Gaussian 2D
        - Công thức: G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
        - Tạo hiệu ứng làm mờ tự nhiên, giảm noise
        
        Ứng dụng:
        - Làm mờ ảnh (blur effect)
        - Giảm nhiễu (noise reduction)
        - Tiền xử lý cho edge detection
        - Tạo hiệu ứng depth of field
        
        Hoạt động:
        - Trọng số cao ở trung tâm, giảm dần ra ngoài
        - Tổng các trọng số = 1 (normalized)
        - Sigma càng lớn → blur càng mạnh
        
        Args:
            size (int): Kích thước kernel (3, 5, 7, ...)
            sigma (float): Độ lệch chuẩn, điều khiển độ mờ
            
        Returns:
            numpy.ndarray: Gaussian kernel
        """
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Tính toán Gaussian values
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        kernel = kernel / np.sum(kernel)
        return kernel
    
    @staticmethod
    def sobel_x_kernel(size=3, sigma=1.0):
        """
        Sobel_X Kernel - Phát hiện cạnh dọc (True Gaussian + Derivative)
        
        Lý thuyết:
        - Sobel = Gaussian smoothing ⊗ Derivative operator
        - Sx = Gy ⊗ Dx (outer product)
        - Gy: Gaussian 1D theo Y (smoothing)
        - Dx: Derivative 1D theo X (edge detection)
        
        Args:
            size (int): Kích thước kernel (3, 5, 7, ...)
            sigma (float): Sigma cho Gaussian smoothing
        
        Returns:
            numpy.ndarray: Sobel X kernel
        """
        # Tạo true Gaussian 1D theo Y
        def gaussian_1d(size, sigma):
            x = np.arange(size) - size // 2
            g = np.exp(-x**2 / (2 * sigma**2))
            return g / np.sum(g)
        
        # Tạo derivative operator 1D theo X
        def derivative_1d(size):
            d = np.zeros(size)
            center = size // 2
            
            if size == 3:
                # Standard Sobel derivative
                d = np.array([-1, 0, 1])
            else:
                # Central difference cho size lớn hơn
                if center > 0:
                    d[center-1] = -1
                if center < size-1:
                    d[center+1] = 1
                
                # Thêm trọng số cho các vị trí xa hơn nếu cần
                if size >= 5:
                    if center > 1:
                        d[center-2] = -0.5
                    if center < size-2:
                        d[center+2] = 0.5
            
            return d
        
        # Tạo Gaussian và Derivative vectors
        gy = gaussian_1d(size, sigma)
        dx = derivative_1d(size)
        
        # Outer product để tạo 2D kernel
        sobel_x = np.outer(gy, dx)
        
        return sobel_x.astype(np.float32)
    
    @staticmethod
    def sobel_y_kernel(size=3, sigma=1.0):
        """
        Sobel_Y kernel - Phát hiện cạnh ngang (True Gaussian + Derivative)
        
        Lý thuyết:
        - Sobel = Gaussian smoothing ⊗ Derivative operator
        - Sy = Gx ⊗ Dy (outer product)
        - Gx: Gaussian 1D theo X (smoothing)
        - Dy: Derivative 1D theo Y (edge detection)
        
        Args:
            size (int): Kích thước kernel (3, 5, 7, ...)
            sigma (float): Sigma cho Gaussian smoothing
        
        Returns:
            numpy.ndarray: Sobel Y kernel
        """
        # Tạo true Gaussian 1D theo X
        def gaussian_1d(size, sigma):
            x = np.arange(size) - size // 2
            g = np.exp(-x**2 / (2 * sigma**2))
            return g / np.sum(g)
        
        # Tạo derivative operator 1D theo Y
        def derivative_1d(size):
            d = np.zeros(size)
            center = size // 2
            
            if size == 3:
                # Standard Sobel derivative
                d = np.array([-1, 0, 1])
            else:
                # Central difference cho size lớn hơn
                if center > 0:
                    d[center-1] = -1
                if center < size-1:
                    d[center+1] = 1
                
                # Thêm trọng số cho các vị trí xa hơn nếu cần
                if size >= 5:
                    if center > 1:
                        d[center-2] = -0.5
                    if center < size-2:
                        d[center+2] = 0.5
            
            return d
        
        # Tạo Gaussian và Derivative vectors
        gx = gaussian_1d(size, sigma)
        dy = derivative_1d(size)
        
        # Outer product để tạo 2D kernel (transpose của Sobel X)
        sobel_y = np.outer(dy, gx)
        
        return sobel_y.astype(np.float32)
    
    @staticmethod
    def chuyen_doi_logarit(img, c=1):
        """
        Chuyển đổi logarit để cải thiện hiển thị
        
        Lý thuyết:
        - Công thức: s = c * log(1 + r)
        - c: hệ số khuếch đại
        - Làm tăng vùng sáng cho vùng tối
        - Nén mạnh vùng sáng mạnh
        - Giúp hiển thị chi tiết trong vùng tối mà không làm cháy vùng sáng
        
        Args:
            img: Ảnh input (numpy array)
            c: Hệ số khuếch đại (default=1)
            
        Returns:
            numpy.ndarray: Ảnh sau chuyển đổi logarit
        """
        # Đảm bảo img là float để tránh overflow
        img_float = img.astype(np.float64)
        
        # Áp dụng chuyển đổi logarit
        result = float(c) * cv.log(1.0 + img_float)
        
        # Normalize về range [0, 255] nếu cần
        if result.max() > 255:
            result = (result / result.max()) * 255
            
        return result.astype(np.uint8)
    
    @staticmethod
    def laplacian_kernel():
        """
        LAPLACIAN KERNEL - Phát hiện cạnh đa hướng
        
        Lý thuyết:
        - Second derivative operator
        - Tính đạo hàm bậc 2 theo mọi hướng
        - Phát hiện zero-crossings (edge locations)
        
        Ứng dụng:
        - Edge detection (tất cả hướng)
        - Image sharpening
        - Feature detection
        - Blob detection
        
        Hoạt động:
        - Tính tổng đạo hàm bậc 2 theo X và Y
        - Highlight các vùng thay đổi nhanh
        - Sensitive với noise
        
        Kernel cố định 3x3:
        [ 0 -1  0]
        [-1  4 -1]
        [ 0 -1  0]
        """
        return np.array([[ 0, -1,  0],
                        [-1,  4, -1],
                        [ 0, -1,  0]], dtype=np.float32)
    
    @staticmethod
    def laplacian_with_log_transform(image, c=1.0, apply_log=True):
        """
        Laplacian Edge Detection với Log Transform
        
        Lý thuyết:
        - Áp dụng Laplacian kernel để detect edges
        - Sử dụng log transform để cải thiện hiển thị
        - Giúp hiện chi tiết trong vùng tối mà không cháy vùng sáng
        
        Args:
            image: Ảnh input (grayscale)
            c: Hệ số khuếch đại cho log transform
            apply_log: Có áp dụng log transform hay không
            
        Returns:
            numpy.ndarray: Ảnh sau Laplacian + Log transform
        """
        from core.convolution import convolution_2d_manual
        
        # Lấy Laplacian kernel
        laplacian = KernelGenerator.laplacian_kernel()
        
        # Áp dụng convolution
        laplacian_result = convolution_2d_manual(image, laplacian, padding=1, stride=1)
        
        # Xử lý giá trị âm (Laplacian có thể tạo giá trị âm)
        # Cách 1: Lấy absolute value
        laplacian_abs = np.abs(laplacian_result)
        
        # Cách 2: Shift về range dương
        laplacian_shifted = laplacian_result - np.min(laplacian_result)
        
        # Chọn cách xử lý (có thể thay đổi)
        processed_result = laplacian_abs
        
        if apply_log:
            # Áp dụng log transform để cải thiện hiển thị
            enhanced_result = KernelGenerator.chuyen_doi_logarit(processed_result, c)
            return enhanced_result
        else:
            # Normalize về [0, 255]
            normalized = ((processed_result / processed_result.max()) * 255).astype(np.uint8)
            return normalized
    
    @staticmethod
    def laplacian_8_connected():
        """
        LAPLACIAN 8-CONNECTED - Laplacian với 8 kết nối
        
        Lý thuyết:
        - Mở rộng của Laplacian cơ bản
        - Xét 8 pixel lân cận (bao gồm đường chéo)
        - Sensitive hơn với edges ở mọi hướng
        
        Kernel cố định 3x3:
        [-1 -1 -1]
        [-1  8 -1]
        [-1 -1 -1]
        """
        return np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]], dtype=np.float32)
    
    @staticmethod
    def sharpen_kernel():
        """
        SHARPEN KERNEL - Làm sắc nét ảnh
        
        Lý thuyết:
        - Tăng cường contrast tại các edges
        - Kết hợp original image với Laplacian
        - Formula: Sharpened = Original + α * Laplacian
        
        Ứng dụng:
        - Tăng độ sắc nét ảnh
        - Enhance details
        - Unsharp masking
        - Photo enhancement
        
        Hoạt động:
        - Tăng cường sự khác biệt giữa pixel và neighbors
        - Làm nổi bật edges và details
        - Có thể tăng noise nếu dùng quá mức
        
        Kernel cố định 3x3:
        [ 0 -1  0]
        [-1  5 -1]
        [ 0 -1  0]
        """
        return np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]], dtype=np.float32)
    
    @staticmethod
    def emboss_kernel():
        """
        EMBOSS KERNEL - Tạo hiệu ứng nổi
        
        Lý thuyết:
        - Tạo hiệu ứng 3D bằng cách highlight edges
        - Simulate ánh sáng chiếu từ một góc
        - Tạo shadow và highlight effects
        
        Ứng dụng:
        - Artistic effects
        - Texture analysis
        - 3D appearance simulation
        - Stylistic image processing
        
        Kernel cố định 3x3:
        [-2 -1  0]
        [-1  1  1]
        [ 0  1  2]
        """
        return np.array([[-2, -1,  0],
                        [-1,  1,  1],
                        [ 0,  1,  2]], dtype=np.float32)
    
    @staticmethod
    def motion_blur_kernel(size, angle=0):
        """
        MOTION BLUR KERNEL - Mô phỏng chuyển động
        
        Lý thuyết:
        - Mô phỏng hiệu ứng chuyển động trong ảnh
        - Tạo blur theo một hướng cụ thể
        - Simulate camera shake hoặc object motion
        
        Ứng dụng:
        - Artistic motion effects
        - Simulate camera motion
        - Speed effect simulation
        - Dynamic image effects
        
        Args:
            size (int): Kích thước kernel
            angle (float): Góc chuyển động (degrees)
        """
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Create motion line
        for i in range(size):
            x = int(center + (i - center) * cos_angle)
            y = int(center + (i - center) * sin_angle)
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize
        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        
        return kernel
    
    @staticmethod
    def prewitt_x_kernel():
        """
        PREWITT X KERNEL - Edge detection (alternative to Sobel)
        
        Lý thuyết:
        - Gradient operator tương tự Sobel
        - Đơn giản hơn Sobel (không có Gaussian weighting)
        - Tính gradient theo hướng X
        
        Kernel cố định 3x3:
        [-1  0  1]
        [-1  0  1]
        [-1  0  1]
        """
        return np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float32)
    
    @staticmethod
    def prewitt_y_kernel():
        """
        PREWITT Y KERNEL - Edge detection theo hướng Y
        
        Kernel cố định 3x3:
        [-1 -1 -1]
        [ 0  0  0]
        [ 1  1  1]
        """
        return np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]], dtype=np.float32)
    
    @staticmethod
    def roberts_cross_x():
        """
        ROBERTS CROSS X - Simple edge detection
        
        Lý thuyết:
        - Đơn giản nhất trong các edge detectors
        - Sử dụng 2x2 kernel
        - Tính gradient theo đường chéo
        
        Kernel 2x2:
        [ 1  0]
        [ 0 -1]
        """
        return np.array([[ 1,  0],
                        [ 0, -1]], dtype=np.float32)
    
    @staticmethod
    def roberts_cross_y():
        """
        ROBERTS CROSS Y - Simple edge detection
        
        Kernel 2x2:
        [ 0  1]
        [-1  0]
        """
        return np.array([[ 0,  1],
                        [-1,  0]], dtype=np.float32)
    
    @staticmethod
    def high_pass_kernel():
        """
        HIGH PASS FILTER - Lọc tần số cao
        
        Lý thuyết:
        - Giữ lại thông tin tần số cao (edges, details)
        - Loại bỏ thông tin tần số thấp (smooth areas)
        - Complement của low-pass filter
        
        Ứng dụng:
        - Edge enhancement
        - Detail preservation
        - Noise analysis
        - Feature extraction
        
        Kernel cố định 3x3:
        [-1 -1 -1]
        [-1  8 -1]
        [-1 -1 -1]
        """
        return np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]], dtype=np.float32)
    
    @staticmethod
    def unsharp_mask_kernel(strength=1.0):
        """
        UNSHARP MASK - Tăng độ sắc nét thông minh
        
        Lý thuyết:
        - Technique từ photography truyền thống
        - Subtract blurred version từ original
        - Formula: Sharp = Original + strength * (Original - Blurred)
        
        Ứng dụng:
        - Professional image sharpening
        - Detail enhancement
        - Photo retouching
        - Print preparation
        
        Args:
            strength (float): Cường độ sharpening
        """
        center_value = 1 + 4 * strength
        edge_value = -strength
        
        return np.array([[0, edge_value, 0],
                        [edge_value, center_value, edge_value],
                        [0, edge_value, 0]], dtype=np.float32)


class KernelScaler:
    """
    Class để scale kernels từ 3x3 lên các kích thước lớn hơn
    """
    
    @staticmethod
    def scale_kernel_to_size(base_kernel, target_size):
        """
        Scale kernel từ kích thước nhỏ lên kích thước lớn hơn
        
        Args:
            base_kernel (numpy.ndarray): Kernel gốc (thường 3x3)
            target_size (int): Kích thước mục tiêu (5, 7, 9, ...)
            
        Returns:
            numpy.ndarray: Kernel đã được scale
        """
        if target_size == base_kernel.shape[0]:
            return base_kernel
        
        # Sử dụng numpy interpolation để scale up (thay thế scipy)
        from numpy import interp
        old_size = base_kernel.shape[0]
        
        # Tạo grid coordinates cho kernel mới
        old_coords = np.linspace(0, 1, old_size)
        new_coords = np.linspace(0, 1, target_size)
        
        # Interpolate từng row và column
        scaled_kernel = np.zeros((target_size, target_size))
        
        # Interpolate theo rows trước
        temp_kernel = np.zeros((target_size, old_size))
        for j in range(old_size):
            temp_kernel[:, j] = interp(new_coords, old_coords, base_kernel[:, j])
        
        # Interpolate theo columns
        for i in range(target_size):
            scaled_kernel[i, :] = interp(new_coords, old_coords, temp_kernel[i, :])
        
        # Normalize nếu cần
        if np.sum(np.abs(base_kernel)) > 0:
            original_sum = np.sum(base_kernel)
            if abs(original_sum - 1.0) < 0.01:  # Nếu kernel gốc đã normalized
                scaled_kernel = scaled_kernel / np.sum(scaled_kernel)
        
        return scaled_kernel
    
    @staticmethod
    def create_gaussian_family(sizes=[3, 5, 7], sigma=1.0):
        """
        Tạo family của Gaussian kernels với các kích thước khác nhau
        """
        kernels = {}
        for size in sizes:
            kernels[size] = KernelGenerator.gaussian_kernel(size, sigma)
        return kernels
    
    @staticmethod
    def create_edge_detection_family(sizes=[3, 5, 7]):
        """
        Tạo family của edge detection kernels
        
        Returns:
            dict: Dictionary chứa các loại edge kernels
        """
        base_kernels = {
            'sobel_x': KernelGenerator.sobel_x_kernel(),
            'sobel_y': KernelGenerator.sobel_y_kernel(),
            'laplacian': KernelGenerator.laplacian_kernel(),
            'prewitt_x': KernelGenerator.prewitt_x_kernel(),
            'prewitt_y': KernelGenerator.prewitt_y_kernel()
        }
        
        kernel_families = {}
        for kernel_name, base_kernel in base_kernels.items():
            kernel_families[kernel_name] = {}
            for size in sizes:
                if size == 3:
                    kernel_families[kernel_name][size] = base_kernel
                else:
                    kernel_families[kernel_name][size] = KernelScaler.scale_kernel_to_size(
                        base_kernel, size
                    )
        
        return kernel_families


def demonstrate_kernels():
    """
    Hàm demo để hiển thị các kernels và properties của chúng
    """
    print("=== KERNEL DEMONSTRATION ===\n")
    
    # Gaussian kernels
    print("1. GAUSSIAN KERNELS (Blur Effects)")
    for size in [3, 5, 7]:
        kernel = KernelGenerator.gaussian_kernel(size, sigma=1.0)
        print(f"Gaussian {size}x{size} (σ=1.0):")
        print(kernel)
        print(f"Sum: {np.sum(kernel):.6f}")
        print()
    
    # Edge detection kernels
    print("2. EDGE DETECTION KERNELS")
    edge_kernels = {
        'Sobel X': KernelGenerator.sobel_x_kernel(),
        'Sobel Y': KernelGenerator.sobel_y_kernel(),
        'Laplacian': KernelGenerator.laplacian_kernel(),
        'Prewitt X': KernelGenerator.prewitt_x_kernel()
    }
    
    for name, kernel in edge_kernels.items():
        print(f"{name}:")
        print(kernel)
        print()
    
    # Special effect kernels
    print("3. SPECIAL EFFECT KERNELS")
    special_kernels = {
        'Sharpen': KernelGenerator.sharpen_kernel(),
        'Emboss': KernelGenerator.emboss_kernel(),
        'High Pass': KernelGenerator.high_pass_kernel()
    }
    
    for name, kernel in special_kernels.items():
        print(f"{name}:")
        print(kernel)
        print()


if __name__ == "__main__":
    demonstrate_kernels()
