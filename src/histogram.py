"""
Histogram Module - Xử lý Histogram cho Bài 1
==========================================

Chức năng chính:
1. Tính histogram của ảnh grayscale (H1)
2. Histogram equalization - cân bằng histogram (H2)  
3. Thu hẹp histogram về khoảng [30, 80]

Thuật toán Histogram Equalization:
1. Tính histogram của ảnh gốc
2. Tính CDF (Cumulative Distribution Function)
3. Normalize CDF để map về range [0, 255]
4. Áp dụng transformation cho từng pixel

Author: Image Processing Team
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def calculate_histogram(image):
    """
    Tính histogram của ảnh grayscale
    
    Args:
        image: Ảnh grayscale (numpy array) shape (H, W)
        
    Returns:
        numpy array: Histogram với 256 bins (0-255)
    """
    # Kiểm tra input
    if len(image.shape) != 2:
        raise ValueError("Image phải là grayscale với shape (H, W)")
    
    # Khởi tạo histogram với 256 bins
    histogram = np.zeros(256, dtype=np.int32)
    
    # Đếm số lượng pixels cho mỗi intensity level
    flat_image = image.flatten()
    for pixel_value in flat_image:
        if 0 <= pixel_value <= 255:
            histogram[pixel_value] += 1
    
    return histogram


def plot_histogram_matplotlib(histogram, title="Histogram", color='blue'):
    """
    Vẽ histogram bằng matplotlib
    
    Args:
        histogram: Histogram data (array 256 phần tử)
        title: Tiêu đề của plot
        color: Màu của histogram bars
        
    Returns:
        matplotlib figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(256), histogram, color=color, alpha=0.7, width=1.0)
    ax.set_xlabel('Intensity Level (0-255)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xlim([0, 255])
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_histogram_plotly(histogram, title="Histogram", color='blue'):
    """
    Vẽ histogram bằng plotly (interactive)
    
    Args:
        histogram: Histogram data
        title: Tiêu đề
        color: Màu
        
    Returns:
        plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(256)),
        y=histogram,
        name=title,
        marker_color=color,
        opacity=0.7
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Intensity Level (0-255)",
        yaxis_title="Frequency",
        xaxis=dict(range=[0, 255]),
        height=400,
        showlegend=False
    )
    
    return fig


def histogram_equalization(image):
    """
    Cân bằng histogram của ảnh (Histogram Equalization)
    
    Thuật toán chi tiết:
    1. Tính histogram của ảnh gốc
    2. Tính CDF (Cumulative Distribution Function)
    3. Normalize CDF: cdf_norm = (cdf - cdf_min) / (total_pixels - cdf_min) * 255
    4. Tạo lookup table để map intensity cũ sang mới
    5. Áp dụng transformation cho toàn bộ ảnh
    
    Args:
        image: Ảnh grayscale (numpy array)
        
    Returns:
        tuple: (equalized_image, new_histogram, cdf, lookup_table)
    """
    # Bước 1: Tính histogram
    hist = calculate_histogram(image)
    
    # Bước 2: Tính CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()
    
    # Bước 3: Normalize CDF
    cdf_min = cdf[cdf > 0].min()  # Giá trị CDF nhỏ nhất khác 0
    total_pixels = image.shape[0] * image.shape[1]
    
    # Tạo lookup table để map intensity cũ sang mới
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if cdf[i] > 0:
            # Công thức chuẩn histogram equalization
            lut[i] = np.round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255)
        else:
            lut[i] = 0
    
    # Bước 4: Áp dụng transformation
    equalized_image = lut[image]
    
    # Tính histogram mới
    new_hist = calculate_histogram(equalized_image)
    
    return equalized_image, new_hist, cdf, lut


def narrow_histogram(image, min_val=30, max_val=80):
    """
    Thu hẹp histogram về khoảng [min_val, max_val]
    
    Thuật toán:
    1. Tìm min và max intensity trong ảnh hiện tại
    2. Áp dụng linear mapping từ [current_min, current_max] về [min_val, max_val]
    3. Công thức: new = (old - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    Args:
        image: Ảnh grayscale (numpy array)
        min_val: Giá trị intensity minimum mới (default=30)
        max_val: Giá trị intensity maximum mới (default=80)
        
    Returns:
        tuple: (narrowed_image, new_histogram)
    """
    # Tìm min và max hiện tại
    current_min = np.min(image)
    current_max = np.max(image)
    
    print(f"Current range: [{current_min}, {current_max}] -> New range: [{min_val}, {max_val}]")
    
    # Tránh chia cho 0
    if current_max - current_min == 0:
        # Nếu tất cả pixel có cùng giá trị, set về min_val
        narrowed_image = np.full_like(image, min_val, dtype=np.uint8)
    else:
        # Linear mapping
        narrowed_image = ((image.astype(np.float32) - current_min) / 
                         (current_max - current_min) * 
                         (max_val - min_val) + min_val)
        narrowed_image = np.clip(narrowed_image, min_val, max_val).astype(np.uint8)
    
    # Tính histogram mới
    new_hist = calculate_histogram(narrowed_image)
    
    return narrowed_image, new_hist


def process_task1(image):
    """
    Xử lý đầy đủ Bài 1 - Histogram Processing
    
    Args:
        image: Ảnh grayscale
        
    Returns:
        dict: Dictionary chứa tất cả kết quả
    """
    results = {}
    
    print("🔄 Đang xử lý Bài 1 - Histogram Processing...")
    
    # Ảnh gốc
    results['original_image'] = image
    print(f"✓ Ảnh gốc: {image.shape}")
    
    # H1: Histogram gốc
    results['h1'] = calculate_histogram(image)
    print("✓ Tính H1 - Histogram gốc")
    
    # H2: Histogram equalization
    h2_image, h2, cdf, lut = histogram_equalization(image)
    results['h2_image'] = h2_image
    results['h2'] = h2
    results['cdf'] = cdf
    results['lookup_table'] = lut
    print("✓ Tính H2 - Histogram Equalization")
    
    # Thu hẹp H2 về khoảng [30, 80]
    narrowed_image, narrowed_hist = narrow_histogram(h2_image, 30, 80)
    results['narrowed_image'] = narrowed_image
    results['narrowed_hist'] = narrowed_hist
    print("✓ Thu hẹp histogram về [30, 80]")
    
    print("✅ Hoàn thành Bài 1!")
    return results


def create_histogram_comparison_figure(original_hist, equalized_hist, narrowed_hist):
    """
    Tạo figure so sánh 3 histograms bằng matplotlib
    
    Args:
        original_hist: H1
        equalized_hist: H2  
        narrowed_hist: H3 (sau thu hẹp)
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # H1
    axes[0].bar(range(256), original_hist, color='blue', alpha=0.7, width=1.0)
    axes[0].set_title('H1 - Histogram Gốc', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Intensity Level')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim([0, 255])
    axes[0].grid(True, alpha=0.3)
    
    # H2
    axes[1].bar(range(256), equalized_hist, color='green', alpha=0.7, width=1.0)
    axes[1].set_title('H2 - Histogram sau Equalization', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Intensity Level')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([0, 255])
    axes[1].grid(True, alpha=0.3)
    
    # H3 (Narrowed)
    axes[2].bar(range(256), narrowed_hist, color='red', alpha=0.7, width=1.0)
    axes[2].set_title('H3 - Histogram Thu hẹp [30, 80]', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Intensity Level')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlim([0, 255])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_histogram_comparison_plotly(original_hist, equalized_hist, narrowed_hist):
    """
    Tạo interactive comparison với plotly
    
    Returns:
        plotly figure
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('H1 - Histogram Gốc', 'H2 - Histogram sau Equalization', 'H3 - Histogram Thu hẹp [30, 80]'),
        vertical_spacing=0.08
    )
    
    # H1
    fig.add_trace(go.Bar(
        x=list(range(256)),
        y=original_hist,
        name='H1',
        marker_color='blue',
        opacity=0.7
    ), row=1, col=1)
    
    # H2
    fig.add_trace(go.Bar(
        x=list(range(256)),
        y=equalized_hist,
        name='H2',
        marker_color='green',
        opacity=0.7
    ), row=2, col=1)
    
    # H3
    fig.add_trace(go.Bar(
        x=list(range(256)),
        y=narrowed_hist,
        name='H3',
        marker_color='red',
        opacity=0.7
    ), row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="So sánh Histograms - Bài 1"
    )
    
    # Update x-axis cho tất cả subplots
    for i in range(1, 4):
        fig.update_xaxes(title_text="Intensity Level", range=[0, 255], row=i, col=1)
        fig.update_yaxes(title_text="Frequency", row=i, col=1)
    
    return fig


def analyze_histogram_properties(hist):
    """
    Phân tích các tính chất của histogram
    
    Args:
        hist: Histogram array
        
    Returns:
        dict: Các thống kê
    """
    total_pixels = np.sum(hist)
    
    # Tính mean và std
    intensities = np.arange(256)
    mean_intensity = np.sum(intensities * hist) / total_pixels
    variance = np.sum(((intensities - mean_intensity) ** 2) * hist) / total_pixels
    std_intensity = np.sqrt(variance)
    
    # Tìm mode (intensity xuất hiện nhiều nhất)
    mode_intensity = np.argmax(hist)
    
    # Tính entropy
    prob = hist / total_pixels
    prob = prob[prob > 0]  # Loại bỏ 0 để tránh log(0)
    entropy = -np.sum(prob * np.log2(prob))
    
    return {
        'total_pixels': int(total_pixels),
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'mode_intensity': int(mode_intensity),
        'entropy': float(entropy),
        'min_intensity': int(np.min(np.where(hist > 0)[0])) if np.any(hist > 0) else 0,
        'max_intensity': int(np.max(np.where(hist > 0)[0])) if np.any(hist > 0) else 0
    }
