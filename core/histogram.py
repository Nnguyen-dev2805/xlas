import numpy as np
import cv2

# tính histogram - tự code
def calculate_histogram_manual(image):
    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")
    
    histogram = np.zeros(256, dtype=int)
    
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            histogram[pixel_value] += 1
    
    return histogram

# tính histogram - dùng thư viện
def calculate_histogram_library(image):
    # if image.dtype != np.uint8:
    #     image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)).astype(np.uint8)

    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")
    
    # Sử dụng OpenCV
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram.flatten().astype(int)

# cân bằng histogram - tự code
def histogram_equalization_manual(image):
    """
    Thuật toán:
    1. Tính histogram gốc
    2. Tính p_in(r_k) = n_k / n
    3. Tính s_k = Σ(j=0 to k) p_in(r_j) (CDF)
    4. Scale s_k về [0, L-1]: new_intensity = round(s_k * (L-1))
    5. Áp dụng transformation
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")
    
    height, width = image.shape
    n = height * width  # Total pixels
    L = 256  # Number of intensity levels
    
    # Step 1: Tính histogram gốc
    hist = calculate_histogram_manual(image)
    
    # Step 2: Tính p_in(r_k) = n_k / n
    p_in = hist / n
    
    # Step 3: Tính s_k = CDF
    s = np.zeros(256)
    s[0] = p_in[0]
    for k in range(1, 256):
        s[k] = s[k-1] + p_in[k]
    
    # Step 4: Scale s_k về [0, L-1] và tạo LUT
    lut = np.zeros(256, dtype=np.uint8)
    for k in range(256):
        scaled_value = s[k] * (L - 1)
        lut[k] = int(np.clip(np.round(scaled_value), 0, L-1))
    
    # Step 5: Áp dụng transformation
    equalized_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            old_intensity = image[i, j]
            new_intensity = lut[old_intensity]
            equalized_image[i, j] = new_intensity
    
    # Tính histogram mới
    new_histogram = calculate_histogram_manual(equalized_image)
    
    return equalized_image, new_histogram

# cân bằng histogram - dùng thư viện
def histogram_equalization_library(image):
    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")
    
    # Sử dụng OpenCV
    equalized_image = cv2.equalizeHist(image)
    
    # Tính histogram mới
    new_histogram = calculate_histogram_library(equalized_image)
    
    return equalized_image, new_histogram

# thu hep histogram - tự code
def histogram_narrowing_manual(image, min_val=30, max_val=80):
    """
    Thuật toán Linear Mapping:
    new_value = (old_value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be grayscale image")
    
    height, width = image.shape
    current_min = np.min(image)
    current_max = np.max(image)
    
    narrowed_image = np.zeros_like(image)
    
    if current_max - current_min == 0:
        # Tất cả pixels có cùng giá trị
        narrowed_image.fill(min_val)
    else:
        for i in range(height):
            for j in range(width):
                old_value = image[i, j]
                
                # Công thức linear mapping
                new_value = ((old_value - current_min) / (current_max - current_min) * 
                           (max_val - min_val) + min_val)
                
                # Đảm bảo trong khoảng [min_val, max_val]
                new_value = max(min_val, min(max_val, int(new_value)))
                narrowed_image[i, j] = new_value
    
    # Tính histogram mới
    new_histogram = calculate_histogram_manual(narrowed_image)
    
    return narrowed_image, new_histogram

# phân tích histogram phục vụ mô tả ảnh
def analyze_histogram(histogram):
    total_pixels = np.sum(histogram)
    
    # Tính mean
    mean = np.sum(np.arange(256) * histogram) / total_pixels
    
    # Tính variance và std
    variance = np.sum(((np.arange(256) - mean) ** 2) * histogram) / total_pixels
    std = np.sqrt(variance)
    
    # Tính entropy
    prob = histogram / total_pixels
    prob = prob[prob > 0]  # Loại bỏ 0 để tránh log(0)
    entropy = -np.sum(prob * np.log2(prob))
    
    # Tìm range
    nonzero_indices = np.where(histogram > 0)[0]
    if len(nonzero_indices) > 0:
        min_intensity = np.min(nonzero_indices)
        max_intensity = np.max(nonzero_indices)
        intensity_range = max_intensity - min_intensity
    else:
        min_intensity = max_intensity = intensity_range = 0
    
    analysis = {
        'total_pixels': int(total_pixels),
        'mean': float(mean),
        'std': float(std),
        'variance': float(variance),
        'entropy': float(entropy),
        'min_intensity': int(min_intensity),
        'max_intensity': int(max_intensity),
        'intensity_range': int(intensity_range)
    }
    
    return analysis
