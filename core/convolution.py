import numpy as np
import cv2
from scipy import ndimage

# hàm giúp thêm padding cho ảnh
def add_padding(image, pad_size, pad_value=0):
    height, width = image.shape
    new_height = height + 2 * pad_size
    new_width = width + 2 * pad_size
    
    padded_image = np.full((new_height, new_width), pad_value, dtype=image.dtype)
    padded_image[pad_size:pad_size+height, pad_size:pad_size+width] = image
    
    return padded_image

# hàm tính tích chập
def convolution_2d_manual(image, kernel, padding=0, stride=1):
    if len(image.shape) != 2:
        raise ValueError("Ảnh phải là ảnh xám")
    
    if len(kernel.shape) != 2:
        raise ValueError("Kernel phải là ma trận 2D")
    
    if padding > 0:
        padded_image = add_padding(image, padding)
    else:
        padded_image = image.copy()
    
    input_height, input_width = padded_image.shape
    kernel_height, kernel_width = kernel.shape
    
    # tính kích thước output
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1
    
    output = np.zeros((output_height, output_width), dtype=np.float32)
    
    # convolution operation
    for i in range(0, output_height):
        for j in range(0, output_width):
            # Tính vị trí trong padded image
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kernel_height
            end_j = start_j + kernel_width
            
            # extract patch
            patch = padded_image[start_i:end_i, start_j:end_j]
            
            # convolution: element-wise multiply và sum
            conv_value = np.sum(patch * kernel)
            output[i, j] = conv_value
    
    # Không clip về [0, 255] để giữ giá trị âm (ví dụ Sobel)
    # Clip về [0, 255] và convert về uint8
    # output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

def convolution_2d_library(image, kernel, padding=0, stride=1):
    """
    Convolution 2D - DÙNG THƯ VIỆN
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be 2D image")
    
    # Sử dụng SciPy
    output = ndimage.convolve(image.astype(np.float32), kernel, mode='constant', cval=0.0)
    
    # Clip về [0, 255] và convert về uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output


def median_filter_manual(image, kernel_size):
    """
    Median Filter - TỰ CODE
    
    Thuật toán:
    1. Slide window kích thước kernel_size x kernel_size
    2. Tại mỗi vị trí: lấy median của các pixel trong window
    3. Median = giá trị ở giữa khi sort ascending
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be 2D image")
    
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    height, width = image.shape
    pad_size = kernel_size // 2
    
    # Thêm padding (mirror padding để tránh edge effect)
    padded_image = add_padding(image, pad_size)
    
    # Thêm padding để giúp kernel không bị tràn
    for i in range(pad_size):
        # Top và bottom
        padded_image[i, pad_size:-pad_size] = image[pad_size-1-i, :]
        padded_image[-(i+1), pad_size:-pad_size] = image[height-pad_size+i, :]
        
        # Left và right
        padded_image[pad_size:-pad_size, i] = image[:, pad_size-1-i]
        padded_image[pad_size:-pad_size, -(i+1)] = image[:, width-pad_size+i]
    
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    
    # Median filtering
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            start_i = i
            start_j = j
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size
            
            neighborhood = padded_image[start_i:end_i, start_j:end_j]
            
            # Flatten và sort để tìm median
            flat_neighborhood = neighborhood.flatten()
            flat_neighborhood.sort()
            
            # Median = element ở giữa
            median_idx = len(flat_neighborhood) // 2
            filtered_image[i, j] = flat_neighborhood[median_idx]
    
    return filtered_image


def median_filter_library(image, kernel_size):
    """
    Median Filter - DÙNG THƯ VIỆN
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be 2D image")
    
    # Sử dụng OpenCV
    filtered_image = cv2.medianBlur(image, kernel_size)
    
    return filtered_image


def min_filter_manual(image, kernel_size):
    """
    Min Filter (Erosion) - TỰ CODE
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be 2D image")
    
    height, width = image.shape
    pad_size = kernel_size // 2
    
    # Thêm padding với giá trị max để không ảnh hưởng min
    padded_image = add_padding(image, pad_size, pad_value=255)
    
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    
    # Min filtering
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            start_i = i
            start_j = j
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size
            
            neighborhood = padded_image[start_i:end_i, start_j:end_j]
            
            # Tìm minimum value
            min_value = np.min(neighborhood)
            filtered_image[i, j] = min_value
    
    return filtered_image


def max_filter_manual(image, kernel_size):
    """
    Max Filter (Dilation) - TỰ CODE
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be 2D image")
    
    height, width = image.shape
    pad_size = kernel_size // 2
    
    # Thêm padding với giá trị min để không ảnh hưởng max
    padded_image = add_padding(image, pad_size, pad_value=0)
    
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    
    # Max filtering
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            start_i = i
            start_j = j
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size
            
            neighborhood = padded_image[start_i:end_i, start_j:end_j]
            
            # Tìm maximum value
            max_value = np.max(neighborhood)
            filtered_image[i, j] = max_value
    
    return filtered_image


def threshold_comparison(image1, image2):
    """
    Thresholding comparison giữa 2 ảnh
    
    Logic: if image1(x,y) > image2(x,y) then output = 0 else output = image2(x,y)
    
    Args:
        image1: Ảnh thứ nhất (I4)
        image2: Ảnh thứ hai (I5)
        
    Returns:
        numpy.ndarray: Ảnh kết quả sau thresholding
    """
    # Đảm bảo 2 ảnh có cùng kích thước
    if image1.shape != image2.shape:
        # Resize image1 để match image2
        from core.image_ops import resize_to_match
        image1 = resize_to_match(image1, image2.shape)
    
    # Thực hiện thresholding comparison
    result = np.where(image1 > image2, 0, image2)
    
    return result.astype(np.uint8)
