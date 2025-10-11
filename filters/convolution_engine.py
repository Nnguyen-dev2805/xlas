import numpy as np
from .kernel_types import KernelGenerator, KernelScaler


class ConvolutionEngine:
    """
    Engine thực hiện convolution với các loại kernel khác nhau
    """
    
    @staticmethod
    def apply_convolution(image, kernel, padding=0, stride=1):
        """
        Áp dụng convolution với kernel lên ảnh
        
        Args:
            image (numpy.ndarray): Ảnh input (grayscale)
            kernel (numpy.ndarray): Kernel để convolution
            padding (int): Số pixel padding
            stride (int): Bước nhảy khi convolution
            
        Returns:
            numpy.ndarray: Ảnh sau khi convolution
        """
        # Thêm padding nếu cần
        if padding > 0:
            image_padded = np.pad(image, padding, mode='constant', constant_values=0)
        else:
            image_padded = image
        
        # Kích thước sau convolution
        input_height, input_width = image_padded.shape
        kernel_height, kernel_width = kernel.shape
        
        output_height = (input_height - kernel_height) // stride + 1
        output_width = (input_width - kernel_width) // stride + 1
        
        # Khởi tạo output
        output = np.zeros((output_height, output_width))
        
        # Thực hiện convolution
        for i in range(0, output_height):
            for j in range(0, output_width):
                # Vị trí trong ảnh gốc
                start_i = i * stride
                start_j = j * stride
                end_i = start_i + kernel_height
                end_j = start_j + kernel_width
                
                # Lấy region of interest
                roi = image_padded[start_i:end_i, start_j:end_j]
                
                # Convolution (element-wise multiply và sum)
                output[i, j] = np.sum(roi * kernel)
        
        return output
    
    @staticmethod
    def apply_kernel_type(image, kernel_type, size=3, **kwargs):
        """
        Áp dụng một loại kernel cụ thể lên ảnh
        
        Args:
            image (numpy.ndarray): Ảnh input
            kernel_type (str): Loại kernel
            size (int): Kích thước kernel
            **kwargs: Các tham số bổ sung
            
        Returns:
            tuple: (output_image, kernel_used)
        """
        # Tạo kernel theo loại
        if kernel_type == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            kernel = KernelGenerator.gaussian_kernel(size, sigma)
            
            
        elif kernel_type == 'sobel_x':
            base_kernel = KernelGenerator.sobel_x_kernel()
            if size != 3:
                kernel = KernelScaler.scale_kernel_to_size(base_kernel, size)
            else:
                kernel = base_kernel
                
        elif kernel_type == 'sobel_y':
            base_kernel = KernelGenerator.sobel_y_kernel()
            if size != 3:
                kernel = KernelScaler.scale_kernel_to_size(base_kernel, size)
            else:
                kernel = base_kernel
                
        elif kernel_type == 'laplacian':
            base_kernel = KernelGenerator.laplacian_kernel()
            if size != 3:
                kernel = KernelScaler.scale_kernel_to_size(base_kernel, size)
            else:
                kernel = base_kernel
                
        elif kernel_type == 'sharpen':
            base_kernel = KernelGenerator.sharpen_kernel()
            if size != 3:
                kernel = KernelScaler.scale_kernel_to_size(base_kernel, size)
            else:
                kernel = base_kernel
                
        elif kernel_type == 'emboss':
            kernel = KernelGenerator.emboss_kernel()
            
        elif kernel_type == 'motion_blur':
            angle = kwargs.get('angle', 0)
            kernel = KernelGenerator.motion_blur_kernel(size, angle)
            
        elif kernel_type == 'high_pass':
            base_kernel = KernelGenerator.high_pass_kernel()
            if size != 3:
                kernel = KernelScaler.scale_kernel_to_size(base_kernel, size)
            else:
                kernel = base_kernel
                
        elif kernel_type == 'unsharp_mask':
            strength = kwargs.get('strength', 1.0)
            kernel = KernelGenerator.unsharp_mask_kernel(strength)
            
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Áp dụng convolution
        padding = kwargs.get('padding', 0)
        stride = kwargs.get('stride', 1)
        
        output = ConvolutionEngine.apply_convolution(image, kernel, padding, stride)
        
        return output, kernel


class MultiKernelProcessor:
    """
    Processor để áp dụng nhiều kernel cùng lúc và so sánh kết quả
    """
    
    @staticmethod
    def process_with_multiple_kernels(image, kernel_configs):
        """
        Xử lý ảnh với nhiều kernel khác nhau
        
        Args:
            image (numpy.ndarray): Ảnh input
            kernel_configs (list): List các config kernel
                Mỗi config là dict: {
                    'type': 'gaussian',
                    'size': 3,
                    'padding': 1,
                    'stride': 1,
                    'name': 'Gaussian 3x3',
                    **other_params
                }
                
        Returns:
            dict: Dictionary chứa kết quả cho mỗi kernel
        """
        results = {}
        
        for config in kernel_configs:
            kernel_type = config['type']
            size = config.get('size', 3)
            padding = config.get('padding', 0)
            stride = config.get('stride', 1)
            name = config.get('name', f"{kernel_type}_{size}x{size}")
            
            # Lấy các tham số bổ sung
            kwargs = {k: v for k, v in config.items() 
                     if k not in ['type', 'size', 'name']}
            
            try:
                output, kernel = ConvolutionEngine.apply_kernel_type(
                    image, kernel_type, size, **kwargs
                )
                
                results[name] = {
                    'output': output,
                    'kernel': kernel,
                    'config': config,
                    'success': True,
                    'error': None
                }
                
            except Exception as e:
                results[name] = {
                    'output': None,
                    'kernel': None,
                    'config': config,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    @staticmethod
    def create_standard_kernel_configs():
        """
        Tạo các config kernel chuẩn cho testing
        
        Returns:
            list: Danh sách các kernel configs
        """
        configs = []
        
        # Gaussian kernels với các kích thước khác nhau
        for size in [3, 5, 7]:
            padding = size // 2  # Để giữ nguyên kích thước
            configs.append({
                'type': 'gaussian',
                'size': size,
                'padding': padding,
                'stride': 1,
                'sigma': 1.0,
                'name': f'Gaussian {size}x{size} (σ=1.0)'
            })
        
        
        # Edge detection kernels
        edge_types = ['sobel_x', 'sobel_y', 'laplacian']
        for edge_type in edge_types:
            for size in [3, 5, 7]:
                padding = size // 2
                configs.append({
                    'type': edge_type,
                    'size': size,
                    'padding': padding,
                    'stride': 1,
                    'name': f'{edge_type.title()} {size}x{size}'
                })
        
        # Special effects
        special_configs = [
            {
                'type': 'sharpen',
                'size': 3,
                'padding': 1,
                'stride': 1,
                'name': 'Sharpen Filter'
            },
            {
                'type': 'emboss',
                'size': 3,
                'padding': 1,
                'stride': 1,
                'name': 'Emboss Effect'
            },
            {
                'type': 'motion_blur',
                'size': 7,
                'padding': 3,
                'stride': 1,
                'angle': 45,
                'name': 'Motion Blur (45°)'
            },
            {
                'type': 'high_pass',
                'size': 3,
                'padding': 1,
                'stride': 1,
                'name': 'High Pass Filter'
            }
        ]
        
        configs.extend(special_configs)
        
        return configs
    
    @staticmethod
    def create_assignment_configs():
        """
        Tạo configs theo yêu cầu bài tập:
        - Kernel 3x3, padding = 1 (I1)
        - Kernel 5x5, padding = 2 (I2)  
        - Kernel 7x7, padding = 3, stride = 2 (I3)
        
        Returns:
            list: Configs cho bài tập
        """
        return [
            {
                'type': 'gaussian',
                'size': 3,
                'padding': 1,
                'stride': 1,
                'sigma': 0.8,
                'name': 'I1: Gaussian 3x3 (pad=1)'
            },
            {
                'type': 'gaussian',
                'size': 5,
                'padding': 2,
                'stride': 1,
                'sigma': 1.2,
                'name': 'I2: Gaussian 5x5 (pad=2)'
            },
            {
                'type': 'gaussian',
                'size': 7,
                'padding': 3,
                'stride': 2,
                'sigma': 1.5,
                'name': 'I3: Gaussian 7x7 (pad=3, stride=2)'
            }
        ]


class KernelAnalyzer:
    """
    Analyzer để phân tích properties của kernels
    """
    
    @staticmethod
    def analyze_kernel_properties(kernel, name="Unknown"):
        """
        Phân tích các thuộc tính của kernel
        
        Args:
            kernel (numpy.ndarray): Kernel cần phân tích
            name (str): Tên kernel
            
        Returns:
            dict: Dictionary chứa các thuộc tính
        """
        properties = {
            'name': name,
            'size': kernel.shape,
            'sum': np.sum(kernel),
            'mean': np.mean(kernel),
            'std': np.std(kernel),
            'min': np.min(kernel),
            'max': np.max(kernel),
            'is_normalized': abs(np.sum(kernel) - 1.0) < 1e-6,
            'is_symmetric': np.allclose(kernel, kernel.T),
            'center_value': kernel[kernel.shape[0]//2, kernel.shape[1]//2],
            'energy': np.sum(kernel**2),
            'sparsity': np.count_nonzero(kernel) / kernel.size
        }
        
        # Phân loại kernel
        if abs(properties['sum'] - 1.0) < 1e-6:
            properties['type'] = 'Low-pass (Smoothing)'
        elif abs(properties['sum']) < 1e-6:
            properties['type'] = 'High-pass (Edge detection)'
        elif properties['sum'] > 1:
            properties['type'] = 'Amplifying'
        else:
            properties['type'] = 'Mixed'
        
        return properties
    
    @staticmethod
    def compare_kernels(kernels_dict):
        """
        So sánh nhiều kernels
        
        Args:
            kernels_dict (dict): Dictionary {name: kernel}
            
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        for name, kernel in kernels_dict.items():
            comparison[name] = KernelAnalyzer.analyze_kernel_properties(kernel, name)
        
        return comparison
    
    @staticmethod
    def visualize_kernel_effects(image, kernel_configs, max_kernels=6):
        """
        Tạo visualization cho effects của các kernels
        
        Args:
            image (numpy.ndarray): Ảnh test
            kernel_configs (list): Configs của kernels
            max_kernels (int): Số kernel tối đa để visualize
            
        Returns:
            dict: Results với images và analysis
        """
        # Giới hạn số kernels
        configs = kernel_configs[:max_kernels]
        
        # Process với các kernels
        results = MultiKernelProcessor.process_with_multiple_kernels(image, configs)
        
        # Thêm analysis cho mỗi kernel
        for name, result in results.items():
            if result['success']:
                kernel_analysis = KernelAnalyzer.analyze_kernel_properties(
                    result['kernel'], name
                )
                result['analysis'] = kernel_analysis
        
        return results


def demonstrate_convolution_engine():
    """
    Demo convolution engine với các loại kernel
    """
    print("=== CONVOLUTION ENGINE DEMONSTRATION ===\n")
    
    # Tạo ảnh test đơn giản
    test_image = np.random.randint(0, 256, (50, 50)).astype(np.float32)
    
    # Test với các kernel configs chuẩn
    configs = MultiKernelProcessor.create_assignment_configs()
    
    print("Processing with assignment kernels...")
    results = MultiKernelProcessor.process_with_multiple_kernels(test_image, configs)
    
    for name, result in results.items():
        if result['success']:
            print(f"\n{name}:")
            print(f"  Input size: {test_image.shape}")
            print(f"  Output size: {result['output'].shape}")
            print(f"  Kernel size: {result['kernel'].shape}")
            
            # Analyze kernel
            analysis = KernelAnalyzer.analyze_kernel_properties(result['kernel'], name)
            print(f"  Kernel sum: {analysis['sum']:.6f}")
            print(f"  Kernel type: {analysis['type']}")
        else:
            print(f"\n{name}: FAILED - {result['error']}")


if __name__ == "__main__":
    demonstrate_convolution_engine()
