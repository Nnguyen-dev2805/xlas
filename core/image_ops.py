import numpy as np
import cv2
from PIL import Image

# chuyển RGB sang Grayscale - tự code
def rgb_to_grayscale_manual(rgb_image):
    """
    Thuật toán:
    - Áp dụng công thức ITU-R BT.601
    - Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input must be RGB image with shape (H, W, 3)")
    
    height, width = rgb_image.shape[:2]
    gray_image = np.zeros((height, width), dtype=np.uint8)
    
    # Coefficients theo ITU-R BT.601
    r_coeff = 0.299
    g_coeff = 0.587
    b_coeff = 0.114
    
    for i in range(height):
        for j in range(width):
            r, g, b = rgb_image[i, j]
            # Công thức weighted average
            gray_value = r_coeff * r + g_coeff * g + b_coeff * b
            gray_image[i, j] = int(np.clip(gray_value, 0, 255))
    
    return gray_image

# chuyển rgb sang Grayscale - dùng thư viện
def rgb_to_grayscale_library(rgb_image):
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input must be RGB image with shape (H, W, 3)")
    
    # Sử dụng OpenCV
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return gray_image

# load ảnh từ file
def load_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Cannot load image from {image_path}: {e}")

# lưu ảnh ra file
def save_image(image_array, output_path):
    try:
        if len(image_array.shape) == 2:
            # Grayscale image
            image = Image.fromarray(image_array.astype(np.uint8), mode='L')
        else:
            # RGB image
            image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')
        
        image.save(output_path)
    except Exception as e:
        raise ValueError(f"Cannot save image to {output_path}: {e}")

# resize ảnh về kích thước target
def resize_image(image, target_size):
    if len(image.shape) == 2:
        # Grayscale
        return cv2.resize(image, target_size)
    else:
        # RGB
        return cv2.resize(image, target_size)

# tạo ảnh mẫu với các pattern khác nhau
def create_sample_images(count=10):
    images = []
    
    for i in range(count):
        size = 150
        image = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Tạo pattern khác nhau cho mỗi ảnh
        if i % 5 == 0:
            # Gradient
            for y in range(size):
                for x in range(size):
                    image[y, x] = [
                        (x * 255) // size,
                        (y * 255) // size,
                        ((x + y) * 255) // (2 * size)
                    ]
        
        elif i % 5 == 1:
            # Circles
            center = size // 2
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    if dist < size // 4:
                        image[y, x] = [255, 255, 255]
                    elif dist < size // 3:
                        image[y, x] = [128, 128, 128]
        
        elif i % 5 == 2:
            # Stripes
            stripe_width = size // 10
            for y in range(size):
                for x in range(size):
                    if (x // stripe_width) % 2 == 0:
                        image[y, x] = [255, 0, 0]
                    else:
                        image[y, x] = [0, 255, 0]
        
        elif i % 5 == 3:
            # Checkerboard
            block_size = size // 8
            for y in range(size):
                for x in range(size):
                    if ((x // block_size) + (y // block_size)) % 2 == 0:
                        image[y, x] = [255, 255, 255]
                    else:
                        image[y, x] = [0, 0, 0]
        
        else:
            # Random noise with structure
            np.random.seed(i)
            image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
            
            # Add some structure
            image[size//4:3*size//4, size//4:3*size//4] = [255, 255, 255]
        
        images.append(image)
    
    return images

# so sánh 2 ảnh
# phương pháp so sánh ('mse', 'psnr', 'ssim')
def compare_images(image1, image2, method='mse'):
    if image1.shape != image2.shape:
        raise ValueError("Images must have same shape")
    
    if method == 'mse':
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        return mse
    
    elif method == 'psnr':
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    else:
        raise ValueError(f"Unknown comparison method: {method}")

# lấy thống kê ảnh
def get_image_stats(image):
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': int(np.min(image)),
        'max': int(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'unique_values': len(np.unique(image))
    }
    
    return stats

# resize ảnh để match với target shape
def resize_to_match(image, target_shape):
    from PIL import Image as PILImage
    
    # Convert numpy array to PIL Image
    if len(image.shape) == 2:  # Grayscale
        pil_image = PILImage.fromarray(image, mode='L')
    else:  # RGB
        pil_image = PILImage.fromarray(image, mode='RGB')
    
    # Resize
    target_width, target_height = target_shape[1], target_shape[0]
    resized_pil = pil_image.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
    
    # Convert back to numpy array
    resized_array = np.array(resized_pil)
    
    return resized_array
