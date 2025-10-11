"""
Các loại nhiễu hỗ trợ:
- Salt & Pepper noise (nhiễu muối tiêu)
- Gaussian noise (nhiễu Gaussian)
- Poisson noise (nhiễu Poisson)
- Speckle noise (nhiễu đốm)
- Uniform noise (nhiễu đều)
- Motion blur (mờ chuyển động)
- Gaussian blur (mờ Gaussian)
- Defocus blur (mờ ngoài tiêu cự)
"""

import numpy as np
import cv2

class NoiseGenerator:
    @staticmethod
    def salt_and_pepper(image, salt_prob=0.01, pepper_prob=0.01):
        """
        Thêm nhiễu muối tiêu (Salt & Pepper Noise)
        
        Lý thuyết:
        - Salt: Pixels trắng ngẫu nhiên (255)
        - Pepper: Pixels đen ngẫu nhiên (0)
        - Thường xảy ra do lỗi sensor hoặc transmission errors
        
        Args:
            image (numpy.ndarray): Input image
            salt_prob (float): Xác suất xuất hiện salt (0-1)
            pepper_prob (float): Xác suất xuất hiện pepper (0-1)
            
        Returns:
            numpy.ndarray: Image with salt & pepper noise
        """
        noisy_image = image.copy()
        
        # Salt noise (white pixels)
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255
        
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    @staticmethod
    def gaussian_noise(image, mean=0, std=25):
        """
        Thêm nhiễu Gaussian (Gaussian Noise)
        
        Lý thuyết:
        - Nhiễu có phân phối Gaussian/Normal
        - Thường xảy ra do electronic circuit noise
        - Công thức: noisy = image + N(mean, std²)
        
        Args:
            image (numpy.ndarray): Input image
            mean (float): Mean của Gaussian distribution
            std (float): Standard deviation (độ mạnh của nhiễu)
            
        Returns:
            numpy.ndarray: Image with Gaussian noise
        """
        gaussian = np.random.normal(mean, std, image.shape)
        noisy_image = image.astype(np.float32) + gaussian
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    @staticmethod
    def poisson_noise(image):
        """
        Thêm nhiễu Poisson (Poisson Noise / Shot Noise)
        
        Lý thuyết:
        - Nhiễu do bản chất lượng tử của ánh sáng
        - Phân phối Poisson
        - Thường xuất hiện trong low-light conditions
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Image with Poisson noise
        """
        # Normalize to [0, 1]
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        
        noisy_image = np.random.poisson(image.astype(np.float32) / 255.0 * vals) / float(vals) * 255
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    @staticmethod
    def speckle_noise(image, std=0.1):
        """
        Thêm nhiễu đốm (Speckle Noise)
        
        Lý thuyết:
        - Multiplicative noise
        - Công thức: noisy = image + image * N(0, std²)
        - Thường xuất hiện trong SAR, ultrasound imaging
        
        Args:
            image (numpy.ndarray): Input image
            std (float): Standard deviation của speckle
            
        Returns:
            numpy.ndarray: Image with speckle noise
        """
        gaussian = np.random.normal(0, std, image.shape)
        noisy_image = image.astype(np.float32) + image.astype(np.float32) * gaussian
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    @staticmethod
    def uniform_noise(image, low=-50, high=50):
        """
        Thêm nhiễu đều (Uniform Noise)
        
        Lý thuyết:
        - Nhiễu có phân phối đều (uniform distribution)
        - Mọi giá trị trong range có xác suất như nhau
        
        Args:
            image (numpy.ndarray): Input image
            low (float): Giá trị min của noise
            high (float): Giá trị max của noise
            
        Returns:
            numpy.ndarray: Image with uniform noise
        """
        uniform = np.random.uniform(low, high, image.shape)
        noisy_image = image.astype(np.float32) + uniform
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    @staticmethod
    def motion_blur(image, kernel_size=15, angle=45):
        """
        Tạo hiệu ứng mờ chuyển động (Motion Blur)
        
        Lý thuyết:
        - Mô phỏng camera hoặc object movement
        - Tạo blur theo một hướng cụ thể
        
        Args:
            image (numpy.ndarray): Input image
            kernel_size (int): Kích thước kernel (lẻ)
            angle (float): Góc chuyển động (degrees)
            
        Returns:
            numpy.ndarray: Motion blurred image
        """
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Draw line in the direction of motion
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * cos_angle)
            y = int(center + offset * sin_angle)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    @staticmethod
    def gaussian_blur(image, kernel_size=5, sigma=1.5):
        """
        Tạo hiệu ứng mờ Gaussian (Gaussian Blur)
        
        Lý thuyết:
        - Low-pass filter
        - Giảm high-frequency components (details, noise)
        - Mô phỏng out-of-focus hoặc atmospheric blur
        
        Args:
            image (numpy.ndarray): Input image
            kernel_size (int): Kích thước kernel (lẻ)
            sigma (float): Standard deviation
            
        Returns:
            numpy.ndarray: Gaussian blurred image
        """
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    @staticmethod
    def defocus_blur(image, radius=5):
        """
        Tạo hiệu ứng mờ ngoài tiêu cự (Defocus Blur)
        
        Lý thuyết:
        - Mô phỏng lens defocus
        - Circular/disk kernel
        - Uniform weights trong disk
        
        Args:
            image (numpy.ndarray): Input image
            radius (int): Bán kính của disk kernel
            
        Returns:
            numpy.ndarray: Defocused image
        """
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # Create circular kernel
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        kernel[mask] = 1
        
        # Normalize
        kernel = kernel / np.sum(kernel)
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    @staticmethod
    def mixed_noise(image, noise_types=['gaussian', 'salt_pepper'], intensities=None):
        """
        Kết hợp nhiều loại nhiễu
        
        Args:
            image (numpy.ndarray): Input image
            noise_types (list): List of noise types to apply
            intensities (dict): Intensities for each noise type
            
        Returns:
            numpy.ndarray: Image with mixed noise
        """
        if intensities is None:
            intensities = {}
        
        noisy_image = image.copy()
        
        for noise_type in noise_types:
            if noise_type == 'gaussian':
                std = intensities.get('gaussian_std', 15)
                noisy_image = NoiseGenerator.gaussian_noise(noisy_image, std=std)
            
            elif noise_type == 'salt_pepper':
                salt = intensities.get('salt_prob', 0.005)
                pepper = intensities.get('pepper_prob', 0.005)
                noisy_image = NoiseGenerator.salt_and_pepper(noisy_image, salt, pepper)
            
            elif noise_type == 'speckle':
                std = intensities.get('speckle_std', 0.1)
                noisy_image = NoiseGenerator.speckle_noise(noisy_image, std=std)
        
        return noisy_image


class NoiseTestSuite:
    """
    Test suite để tạo bộ ảnh test với nhiều loại nhiễu
    """
    
    @staticmethod
    def create_test_set(image, include_blurs=True):
        """
        Tạo bộ test đầy đủ với tất cả loại nhiễu
        
        Args:
            image (numpy.ndarray): Original image
            include_blurs (bool): Có tạo blur variations không
            
        Returns:
            dict: Dictionary chứa các ảnh với nhiễu khác nhau
        """
        test_set = {}
        
        # Original
        test_set['Original'] = image.copy()
        
        # Noise variations
        print("Generating noise variations...")
        
        # Salt & Pepper - các mức độ
        test_set['Salt&Pepper (Light)'] = NoiseGenerator.salt_and_pepper(
            image, salt_prob=0.005, pepper_prob=0.005
        )
        test_set['Salt&Pepper (Medium)'] = NoiseGenerator.salt_and_pepper(
            image, salt_prob=0.02, pepper_prob=0.02
        )
        test_set['Salt&Pepper (Heavy)'] = NoiseGenerator.salt_and_pepper(
            image, salt_prob=0.05, pepper_prob=0.05
        )
        
        # Gaussian noise - các mức độ
        test_set['Gaussian (Light)'] = NoiseGenerator.gaussian_noise(
            image, mean=0, std=10
        )
        test_set['Gaussian (Medium)'] = NoiseGenerator.gaussian_noise(
            image, mean=0, std=25
        )
        test_set['Gaussian (Heavy)'] = NoiseGenerator.gaussian_noise(
            image, mean=0, std=50
        )
        
        # Poisson noise
        test_set['Poisson'] = NoiseGenerator.poisson_noise(image)
        
        # Speckle noise
        test_set['Speckle (Light)'] = NoiseGenerator.speckle_noise(image, std=0.05)
        test_set['Speckle (Medium)'] = NoiseGenerator.speckle_noise(image, std=0.15)
        
        # Uniform noise
        test_set['Uniform'] = NoiseGenerator.uniform_noise(image, low=-30, high=30)
        
        if include_blurs:
            print("Generating blur variations...")
            
            # Motion blur - các góc
            test_set['Motion Blur (0°)'] = NoiseGenerator.motion_blur(
                image, kernel_size=15, angle=0
            )
            test_set['Motion Blur (45°)'] = NoiseGenerator.motion_blur(
                image, kernel_size=15, angle=45
            )
            test_set['Motion Blur (90°)'] = NoiseGenerator.motion_blur(
                image, kernel_size=15, angle=90
            )
            
            # Gaussian blur
            test_set['Gaussian Blur (Light)'] = NoiseGenerator.gaussian_blur(
                image, kernel_size=5, sigma=1.0
            )
            test_set['Gaussian Blur (Medium)'] = NoiseGenerator.gaussian_blur(
                image, kernel_size=9, sigma=2.0
            )
            test_set['Gaussian Blur (Heavy)'] = NoiseGenerator.gaussian_blur(
                image, kernel_size=15, sigma=3.0
            )
            
            # Defocus blur
            test_set['Defocus Blur (Light)'] = NoiseGenerator.defocus_blur(
                image, radius=3
            )
            test_set['Defocus Blur (Medium)'] = NoiseGenerator.defocus_blur(
                image, radius=5
            )
            test_set['Defocus Blur (Heavy)'] = NoiseGenerator.defocus_blur(
                image, radius=8
            )
        
        # Mixed noise
        test_set['Mixed (Gaussian+S&P)'] = NoiseGenerator.mixed_noise(
            image, 
            noise_types=['gaussian', 'salt_pepper'],
            intensities={'gaussian_std': 15, 'salt_prob': 0.01, 'pepper_prob': 0.01}
        )
        
        print(f"✓ Created {len(test_set)} test images")
        
        return test_set
    
    @staticmethod
    def get_noise_presets():
        """
        Trả về các preset configs cho các loại nhiễu
        
        Returns:
            dict: Preset configurations
        """
        return {
            'salt_pepper_light': {
                'type': 'salt_pepper',
                'salt_prob': 0.005,
                'pepper_prob': 0.005,
                'description': 'Nhiễu muối tiêu nhẹ'
            },
            'salt_pepper_medium': {
                'type': 'salt_pepper',
                'salt_prob': 0.02,
                'pepper_prob': 0.02,
                'description': 'Nhiễu muối tiêu trung bình'
            },
            'salt_pepper_heavy': {
                'type': 'salt_pepper',
                'salt_prob': 0.05,
                'pepper_prob': 0.05,
                'description': 'Nhiễu muối tiêu nặng'
            },
            'gaussian_light': {
                'type': 'gaussian',
                'mean': 0,
                'std': 10,
                'description': 'Nhiễu Gaussian nhẹ'
            },
            'gaussian_medium': {
                'type': 'gaussian',
                'mean': 0,
                'std': 25,
                'description': 'Nhiễu Gaussian trung bình'
            },
            'gaussian_heavy': {
                'type': 'gaussian',
                'mean': 0,
                'std': 50,
                'description': 'Nhiễu Gaussian nặng'
            },
            'motion_blur_horizontal': {
                'type': 'motion_blur',
                'kernel_size': 15,
                'angle': 0,
                'description': 'Mờ chuyển động ngang'
            },
            'motion_blur_diagonal': {
                'type': 'motion_blur',
                'kernel_size': 15,
                'angle': 45,
                'description': 'Mờ chuyển động chéo'
            },
            'gaussian_blur_medium': {
                'type': 'gaussian_blur',
                'kernel_size': 9,
                'sigma': 2.0,
                'description': 'Mờ Gaussian trung bình'
            }
        }
