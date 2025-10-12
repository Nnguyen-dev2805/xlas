import numpy as np
import cv2


class MeanKernel:
    """Mean Filter implementation"""
    
    @staticmethod
    def create_mean_kernel(kernel_size):
        """
        Tạo mean kernel (all 1s, normalized)
        
        Args:
            kernel_size: Size of kernel (3, 5, 7, ...)
            
        Returns:
            Mean kernel (kernel_size x kernel_size)
            
        Example:
            3x3 mean kernel:
            [[1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9],
             [1/9, 1/9, 1/9]]
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size phải là số lẻ")
        
        # All 1s, normalized
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel = kernel / (kernel_size * kernel_size)
        
        return kernel
    
    @staticmethod
    def mean_filter_manual(image, kernel_size=3):
        """
        Mean filter tự implement (slow, for learning)
        
        Args:
            image: Input grayscale image
            kernel_size: Size of averaging window
            
        Returns:
            Filtered image
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size phải là số lẻ")
        
        height, width = image.shape
        pad_size = kernel_size // 2
        
        # Padding với edge values
        padded = np.pad(image, pad_size, mode='edge')
        
        filtered = np.zeros((height, width), dtype=np.float32)
        
        # Sliding window
        for i in range(height):
            for j in range(width):
                # Extract window
                window = padded[i:i+kernel_size, j:j+kernel_size]
                
                # Calculate mean
                mean_val = np.mean(window)
                
                filtered[i, j] = mean_val
        
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    @staticmethod
    def mean_filter_convolve(image, kernel_size=3):
        """
        Mean filter using convolution (Faster)
        
        Args:
            image: Input grayscale image
            kernel_size: Size of averaging window
            
        Returns:
            Filtered image
        """
        # Create mean kernel
        kernel = MeanKernel.create_mean_kernel(kernel_size)
        
        # Convolve
        from core.convolution import convolution_2d_manual
        filtered = convolution_2d_manual(
            image, 
            kernel, 
            padding=kernel_size//2, 
            stride=1
        )
        
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    @staticmethod
    def mean_filter_opencv(image, kernel_size=3):
        """
        Mean filter using OpenCV (fastest)
        
        Args:
            image: Input grayscale image
            kernel_size: Size of averaging window
            
        Returns:
            Filtered image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # cv2.blur() = mean filter
        filtered = cv2.blur(image, (kernel_size, kernel_size))
        
        return filtered


class MeanProcessor:
    """Mean filter với nhiều cấu hình"""
    
    @staticmethod
    def process_with_multiple_kernels(image, kernel_configs):
        """
        Xử lý với nhiều kernel sizes
        
        Args:
            image: Input image
            kernel_configs: List of configs, each with 'size' and 'name'
            
        Returns:
            dict of results
        """
        results = {}
        
        for config in kernel_configs:
            size = config.get('size', 3)
            name = config.get('name', f'Mean_{size}x{size}')
            
            # Apply mean filter
            filtered = MeanKernel.mean_filter_convolve(image, kernel_size=size)
            
            results[name] = {
                'filtered': filtered,
                'config': config
            }
        
        return results
    
    @staticmethod
    def create_standard_configs():
        """
        Tạo configs chuẩn
        """
        return [
            {
                'size': 3,
                'name': 'Mean_3x3',
                'description': 'Light averaging'
            },
            {
                'size': 5,
                'name': 'Mean_5x5',
                'description': 'Medium averaging'
            },
            {
                'size': 7,
                'name': 'Mean_7x7',
                'description': 'Heavy averaging'
            }
        ]