import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from core.image_ops import rgb_to_grayscale_manual
from core.convolution import convolution_2d_manual, median_filter_manual

from filters.gaussian_kernel import GaussianKernel
from filters.sobel_kernel import SobelKernel
from filters.laplacian_kernel import LaplacianKernel
from filters.median_kernel import MedianKernel
from filters.mean_kernel import MeanKernel
from filters.noise_generator import NoiseGenerator
from core.histogram import histogram_equalization_library

def safe_image_display(image):
    if image is None:
        return None
    
    img = np.array(image, copy=True)
    
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

def setup_bai3_styles():
    """CSS styles cho Bài 3"""
    st.markdown("""
    <style>
    .bai3-header {
        font-size: 2.5rem;
        color: #27ae60;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .bai3-step-header {
        font-size: 1.8rem;
        color: #16a085;
        margin: 1.5rem 0;
        padding: 0.8rem;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px;
        border-left: 5px solid #27ae60;
    }
    .bai3-info-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #81c784;
        margin: 1rem 0;
    }
    .bai3-result-box {
        background-color: #f1f8e9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #aed581;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def normalize_to_uint8(image):
    """Chuẩn hóa ảnh về uint8 [0, 255]"""
    if image.dtype == np.uint8:
        return image
    
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.uint8)
    
    normalized = (image - img_min) / (img_max - img_min) * 255
    return normalized.astype(np.uint8)

def apply_noise(image, noise_type, noise_params):
    """Áp dụng nhiễu lên ảnh"""
    if noise_type == "salt_pepper":
        return NoiseGenerator.salt_and_pepper(
            image,
            salt_prob=noise_params.get('salt_prob', 0.02),
            pepper_prob=noise_params.get('pepper_prob', 0.02)
        )
    elif noise_type == "gaussian":
        return NoiseGenerator.gaussian_noise(
            image,
            mean=noise_params.get('mean', 0),
            std=noise_params.get('std', 25)
        )
    elif noise_type == "uniform":
        return NoiseGenerator.uniform_noise(
            image,
            low=noise_params.get('low', -50),
            high=noise_params.get('high', 50)
        )
    elif noise_type == "speckle":
        return NoiseGenerator.speckle_noise(
            image,
            std=noise_params.get('std', 0.1)
        )
    elif noise_type == "poisson":
        return NoiseGenerator.poisson_noise(image)
    else:
        return image

def apply_preprocessing_step(image, filter_type, filter_params):
    """Áp dụng một bước tiền xử lý"""
    if filter_type == "none":
        return image
    
    elif filter_type == "median":
        kernel_size = filter_params.get('kernel_size', 3)
        return median_filter_manual(image, kernel_size=kernel_size)
    
    elif filter_type == "gaussian":
        kernel_size = filter_params.get('kernel_size', 3)
        sigma = filter_params.get('sigma', 1.0)
        kernel = GaussianKernel.create_gaussian_2d_separable(kernel_size, sigma)
        padding = kernel_size // 2
        result = convolution_2d_manual(image, kernel, padding=padding, stride=1)
        return normalize_to_uint8(result)
    
    elif filter_type == "mean":
        kernel_size = filter_params.get('kernel_size', 3)
        return MeanKernel.mean_filter_convolve(image, kernel_size=kernel_size)
    
    elif filter_type == "histogram_eq":
        # Histogram equalization
        equalized, _ = histogram_equalization_library(image)
        return equalized
    
    elif filter_type == "sharpen":
        from filters.sharpen_kernel import SharpenKernel
        kernel_size = filter_params.get('kernel_size', 3)
        alpha = filter_params.get('alpha', 1.0)
        
        if kernel_size == 3:
            kernel = SharpenKernel.create_sharpen_3x3(alpha=alpha, diagonal=False)
        elif kernel_size == 5:
            kernel = SharpenKernel.create_sharpen_5x5(alpha=alpha, diagonal=False)
        else:
            kernel = SharpenKernel.create_sharpen_7x7(alpha=alpha, diagonal=False)
        
        padding = kernel_size // 2
        result = convolution_2d_manual(image, kernel, padding=padding, stride=1)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
        # return result
        # return normalize_to_uint8(result)
    
    return image

def apply_edge_detection(image, detector_type, detector_params):
    """Áp dụng edge detection"""
    if detector_type == "sobel":
        sigma = detector_params.get('sigma', 1.0)
        kernel_size = detector_params.get('kernel_size', 3)
        
        # Tạo Sobel kernels
        # if kernel_size !=3:
        sobel_x = SobelKernel.create_sobel_x_kernel(kernel_size, sigma)
        sobel_y = SobelKernel.create_sobel_y_kernel(kernel_size, sigma)
        # else:
        #     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        #     sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        padding = kernel_size // 2
        
        # Tính gradient theo X và Y
        gx = convolution_2d_manual(image, sobel_x, padding=padding, stride=1)
        gy = convolution_2d_manual(image, sobel_y, padding=padding, stride=1)


        # Tính magnitude
        magnitude = SobelKernel.compute_gradient_magnitude(gx, gy)
        return normalize_to_uint8(magnitude)
    
    elif detector_type == "laplacian":
        kernel_size = detector_params.get('kernel_size', 3)
        diagonal = detector_params.get('diagonal', False)
        
        if kernel_size == 3:
            kernel = LaplacianKernel.create_laplacian_3x3(diagonal)
        elif kernel_size == 5:
            kernel = LaplacianKernel.create_laplacian_5x5(diagonal)
        else:
            kernel = LaplacianKernel.create_laplacian_7x7(diagonal)
        
        padding = kernel_size // 2
        result = convolution_2d_manual(image, kernel, padding=padding, stride=1)
        return normalize_to_uint8(np.abs(result))
    
    return image

def create_comparison_plot(images, titles):
    """Tạo comparison plot cho nhiều ảnh"""
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // cols
        col = idx % cols
        
        if len(img.shape) == 3:
            axes[row, col].imshow(img)
        else:
            axes[row, col].imshow(img, cmap='gray')
        
        axes[row, col].set_title(title, fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
    
    # Ẩn các subplot thừa
    for idx in range(len(images), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def create_report_figure_for_edge_detection(input_image, preprocessed_image, sobel_result, laplacian_result, 
                                            noise_type, preprocessing_steps):
    """
    Tạo tấm ảnh báo cáo đẹp cho Edge Detection Pipeline
    - Row 1: 4 ảnh (Input/Noisy, Preprocessed, Sobel, Laplacian)
    - Row 2: 4 histogram tương ứng
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # Pipeline description
    pipeline_desc = f"Noise: {noise_type}"
    if preprocessing_steps:
        steps_desc = " → ".join([f"{s['type']}" for s in preprocessing_steps if s['type'] != 'none'])
        if steps_desc:
            pipeline_desc += f" | Preprocessing: {steps_desc}"
    pipeline_desc += " | Edge Detection: Sobel + Laplacian"
    
    fig.suptitle(f'Edge Detection Pipeline Report\n{pipeline_desc}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Row 1: Ảnh
    images = [input_image, preprocessed_image, sobel_result, laplacian_result]
    input_title = f'Input ({noise_type})' if noise_type != 'none' else 'Input (No Noise)'
    titles = [input_title, 'After Preprocessing', 'Sobel Edge Detection', 'Laplacian Edge Detection']
    colors_border = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    for i, (img, title, color) in enumerate(zip(images, titles, colors_border)):
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[0, i].axis('off')
        
        # Thêm border màu
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        
        # Thêm thống kê nhỏ
        mean_val = np.mean(img)
        std_val = np.std(img)
        axes[0, i].text(0.5, -0.08, f'μ={mean_val:.1f}, σ={std_val:.1f}', 
                       transform=axes[0, i].transAxes, ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))
    
    # Row 2: Histogram
    hist_titles = ['H1: Input Histogram', 'H2: Preprocessed Histogram', 
                  'H3: Sobel Histogram', 'H4: Laplacian Histogram']
    
    for i, (img, title, color) in enumerate(zip(images, hist_titles, colors_border)):
        hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
        
        axes[1, i].plot(range(256), hist, color=color, linewidth=2, label=title)
        axes[1, i].fill_between(range(256), hist, alpha=0.3, color=color)
        axes[1, i].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[1, i].set_xlabel('Pixel Value (0-255)', fontsize=9)
        axes[1, i].set_ylabel('Frequency', fontsize=9)
        axes[1, i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[1, i].set_xlim([0, 255])
        
        # Highlight peak
        peak_idx = np.argmax(hist)
        peak_val = hist[peak_idx]
        axes[1, i].plot(peak_idx, peak_val, 'r*', markersize=10, 
                       label=f'Peak: {peak_idx}')
        axes[1, i].legend(fontsize=8, loc='upper right')
        
        # Style
        axes[1, i].spines['top'].set_visible(False)
        axes[1, i].spines['right'].set_visible(False)
        axes[1, i].spines['left'].set_color(color)
        axes[1, i].spines['bottom'].set_color(color)
        axes[1, i].spines['left'].set_linewidth(2)
        axes[1, i].spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    return fig

def create_histogram_plot(images, titles):
    """Tạo histogram plot cho nhiều ảnh"""
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig_hist = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', 
              '#34495e', '#e67e22', '#95a5a6', '#2ecc71', '#e84393']
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        # Tính histogram
        if len(img.shape) == 3:
            # RGB image - chỉ lấy grayscale
            gray = rgb_to_grayscale_manual(img)
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        else:
            hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        
        color = colors[idx % len(colors)]
        
        fig_hist.add_trace(
            go.Bar(
                x=list(range(256)),
                y=hist,
                name=title,
                marker_color=color,
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig_hist.update_xaxes(title_text="Pixel Value", row=row, col=col, showgrid=False)
        fig_hist.update_yaxes(title_text="Frequency", row=row, col=col, showgrid=True, gridcolor='#ecf0f1')
    
    fig_hist.update_layout(
        height=300 * rows,
        title_text="Histogram của các bước xử lý",
        title_x=0.5,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )
    
    return fig_hist

def main():
    """Hàm main cho Bài 3"""
    setup_bai3_styles()
    
    st.markdown('<div class="bai3-header">Bài 3: Pipeline Xử Lý Ảnh Linh Hoạt</div>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="bai3-info-box">', unsafe_allow_html=True)
    st.write("**Mô tả:**")
    st.write("Tạo pipeline xử lý ảnh linh hoạt với các bước:")
    st.write("")
    st.write("1. **Chọn ảnh gốc**")
    st.write("2. **Thêm nhiễu** (Salt & Pepper, Gaussian, Uniform, Speckle, Poisson, hoặc None)")
    st.write("3. **Tiền xử lý** (Median, Gaussian, Mean, Histogram EQ, Sharpen, hoặc None) - có thể chọn 0-5 bước theo thứ tự")
    st.write("4. **Edge Detection** (Sobel và Laplacian cùng lúc)")
    st.write("")
    st.success("**Tính năng linh hoạt:** Bạn có thể bỏ qua bất kỳ bước nào bằng cách chọn 'None' hoặc đặt số bước = 0")
    st.info("**Filters mới:** Mean Filter (tốt cho Uniform noise), Histogram Equalization (tăng contrast)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Cấu hình Pipeline")
    
    # Bước 1: Upload ảnh
    st.sidebar.markdown("### 1️⃣ Chọn ảnh gốc")
    uploaded_file = st.sidebar.file_uploader(
        "Upload ảnh", 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Chọn ảnh để xử lý"
    )
    
    if uploaded_file is None:
        st.info("👆 Vui lòng upload ảnh từ sidebar để bắt đầu!")
        return
    
    # Load và hiển thị ảnh gốc
    pil_image = Image.open(uploaded_file)
    rgb_array = np.array(pil_image)
    
    # Convert to grayscale
    if len(rgb_array.shape) == 3:
        gray_image = rgb_to_grayscale_manual(rgb_array)
    else:
        gray_image = rgb_array
    
    st.markdown('<div class="bai3-step-header">Ảnh gốc</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(safe_image_display(rgb_array), caption="RGB Original", use_container_width=True)
    with col2:
        st.image(safe_image_display(gray_image), caption="Grayscale", use_container_width=True)
    
    # Bước 2: Chọn và thêm nhiễu
    st.sidebar.markdown("### 2️⃣ Thêm nhiễu")
    noise_type = st.sidebar.selectbox(
        "Loại nhiễu:",
        ["none", "salt_pepper", "gaussian", "uniform", "speckle", "poisson"],
        format_func=lambda x: {
            "none": "Không có nhiễu",
            "salt_pepper": "Salt & Pepper",
            "gaussian": "Gaussian Noise",
            "uniform": "Uniform Noise",
            "speckle": "Speckle Noise",
            "poisson": "Poisson Noise"
        }[x]
    )
    
    noise_params = {}
    
    if noise_type == "salt_pepper":
        noise_params['salt_prob'] = st.sidebar.slider(
            "Salt probability", 0.0, 0.1, 0.02, 0.01,
            help="Xác suất xuất hiện pixel trắng"
        )
        noise_params['pepper_prob'] = st.sidebar.slider(
            "Pepper probability", 0.0, 0.1, 0.02, 0.01,
            help="Xác suất xuất hiện pixel đen"
        )
    elif noise_type == "gaussian":
        noise_params['mean'] = st.sidebar.slider(
            "Mean", -50, 50, 0, 5,
            help="Giá trị trung bình của nhiễu"
        )
        noise_params['std'] = st.sidebar.slider(
            "Std Dev", 1, 100, 25, 5,
            help="Độ lệch chuẩn - cường độ nhiễu"
        )
    elif noise_type == "uniform":
        noise_params['low'] = st.sidebar.slider(
            "Low value", -100, 0, -50, 10,
            help="Giá trị thấp nhất của nhiễu"
        )
        noise_params['high'] = st.sidebar.slider(
            "High value", 0, 100, 50, 10,
            help="Giá trị cao nhất của nhiễu"
        )
    elif noise_type == "speckle":
        noise_params['std'] = st.sidebar.slider(
            "Std Dev", 0.01, 0.5, 0.1, 0.01,
            help="Độ lệch chuẩn - cường độ nhiễu đốm"
        )
    elif noise_type == "poisson":
        st.sidebar.info("💡 Poisson noise không cần tham số - dựa vào bản chất lượng tử của ánh sáng")
    
    # Áp dụng nhiễu
    if noise_type != "none":
        noisy_image = apply_noise(gray_image, noise_type, noise_params)
        
        st.markdown('<div class="bai3-step-header">Ảnh sau khi thêm nhiễu</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(safe_image_display(gray_image), caption="Before Noise", use_container_width=True)
        with col2:
            st.image(safe_image_display(noisy_image), caption=f"After {noise_type.title()} Noise", 
                    use_container_width=True)
        
        current_image = noisy_image
    else:
        current_image = gray_image
        st.info("Không áp dụng nhiễu - sử dụng ảnh grayscale gốc")
    
    # Bước 3: Tiền xử lý (có thể nhiều bước)
    st.sidebar.markdown("### 3️⃣ Tiền xử lý")
    st.sidebar.markdown("**Chọn các bước tiền xử lý theo thứ tự:**")
    st.sidebar.info("💡 Bạn có thể chọn 0 bước để bỏ qua tiền xử lý, hoặc chọn 'None' để skip một bước cụ thể.")
    
    num_preprocessing_steps = st.sidebar.number_input(
        "Số bước tiền xử lý:", 
        min_value=0, 
        max_value=5, 
        value=2,
        help="Chọn số lượng bước tiền xử lý (0 = bỏ qua tiền xử lý)"
    )
    
    preprocessing_pipeline = []
    
    for i in range(num_preprocessing_steps):
        st.sidebar.markdown(f"**Bước {i+1}:**")
        
        filter_type = st.sidebar.selectbox(
            f"Loại filter:",
            ["none", "median", "gaussian", "mean", "histogram_eq", "sharpen"],
            key=f"filter_type_{i}",
            format_func=lambda x: {
                "none": "⛔ None (Skip bước này)",
                "median": "Median Filter",
                "gaussian": "Gaussian Filter",
                "mean": "Mean Filter",
                "histogram_eq": "Histogram Equalization",
                "sharpen": "Sharpen Filter"
            }[x]
        )
        
        filter_params = {}
        
        if filter_type == "none":
            st.sidebar.caption("Bỏ qua bước này")
        
        elif filter_type == "median":
            filter_params['kernel_size'] = st.sidebar.select_slider(
                f"Kernel size:",
                options=[3, 5, 7],
                value=3,
                key=f"median_size_{i}"
            )
        
        elif filter_type == "gaussian":
            filter_params['kernel_size'] = st.sidebar.select_slider(
                f"Kernel size:",
                options=[3, 5, 7],
                value=3,
                key=f"gaussian_size_{i}"
            )
            filter_params['sigma'] = st.sidebar.slider(
                f"Sigma:",
                0.1, 3.0, 1.0, 0.1,
                key=f"gaussian_sigma_{i}"
            )
        
        elif filter_type == "mean":
            filter_params['kernel_size'] = st.sidebar.select_slider(
                f"Kernel size:",
                options=[3, 5, 7],
                value=3,
                key=f"mean_size_{i}",
                help="Averaging window size"
            )
            st.sidebar.info("Tốt cho Uniform noise")
        
        elif filter_type == "histogram_eq":
            st.sidebar.caption("Cân bằng histogram")
            st.sidebar.info("Tăng contrast, phân phối đều pixel values")
        
        elif filter_type == "sharpen":
            filter_params['kernel_size'] = st.sidebar.select_slider(
                f"Kernel size:",
                options=[3, 5, 7],
                value=3,
                key=f"sharpen_size_{i}"
            )
            filter_params['alpha'] = st.sidebar.slider(
                f"Alpha (cường độ):",
                0.1, 3.0, 1.0, 0.1,
                key=f"sharpen_alpha_{i}",
                help="0.5=nhẹ, 1.0=trung bình, 2.0=mạnh"
            )
        
        preprocessing_pipeline.append({
            'type': filter_type,
            'params': filter_params,
            'name': f"Step {i+1}: {filter_type.title()}"
        })
    
    # Áp dụng preprocessing pipeline
    preprocessing_results = []
    preprocessing_images = [current_image]
    preprocessing_titles = ["Input Image"]
    
    if num_preprocessing_steps > 0:
        st.markdown('<div class="bai3-step-header">Tiền xử lý - Từng bước</div>', 
                    unsafe_allow_html=True)
        
        cols = st.columns(min(num_preprocessing_steps + 1, 4))
        
        # Hiển thị input image
        with cols[0]:
            st.image(safe_image_display(current_image), 
                    caption="Input", 
                    use_container_width=True)
        
        # Áp dụng từng bước
        for idx, step in enumerate(preprocessing_pipeline):
            # Chỉ áp dụng nếu không phải 'none'
            if step['type'] != 'none':
                current_image = apply_preprocessing_step(
                    current_image, 
                    step['type'], 
                    step['params']
                )
                preprocessing_images.append(current_image)
                preprocessing_titles.append(step['name'])
                
                # Hiển thị kết quả
                col_idx = (idx + 1) % 4
                if col_idx == 0 and idx < num_preprocessing_steps - 1:
                    cols = st.columns(min(num_preprocessing_steps - idx, 4))
                
                with cols[col_idx]:
                    caption = f"{step['type'].title()}"
                    if step['type'] == 'median':
                        caption += f" {step['params']['kernel_size']}x{step['params']['kernel_size']}"
                    elif step['type'] == 'gaussian':
                        caption += f" {step['params']['kernel_size']}x{step['params']['kernel_size']}, σ={step['params']['sigma']}"
                    elif step['type'] == 'mean':
                        caption += f" {step['params']['kernel_size']}x{step['params']['kernel_size']}"
                    elif step['type'] == 'histogram_eq':
                        caption = "Histogram Equalized"
                    elif step['type'] == 'sharpen':
                        caption += f" {step['params']['kernel_size']}x{step['params']['kernel_size']}, α={step['params']['alpha']}"
                    
                    st.image(safe_image_display(current_image), 
                            caption=caption, 
                            use_container_width=True)
            else:
                # Nếu chọn 'none', bỏ qua và không thêm vào results
                col_idx = (idx + 1) % 4
                if col_idx == 0 and idx < num_preprocessing_steps - 1:
                    cols = st.columns(min(num_preprocessing_steps - idx, 4))
                
                with cols[col_idx]:
                    st.info(f"⛔ Bước {idx+1}: Skipped")
    
    # Bước 4: Edge Detection - Cả Sobel và Laplacian
    st.sidebar.markdown("### 4️⃣ Edge Detection")
    st.sidebar.info("🔍 Hiển thị cả Sobel và Laplacian cùng lúc")
    
    # Tham số cho Sobel
    st.sidebar.markdown("**Sobel Parameters:**")
    sobel_params = {}
    sobel_params['kernel_size'] = st.sidebar.select_slider(
        "Sobel Kernel size:",
        options=[3, 5, 7],
        value=3,
        key="sobel_kernel_size"
    )
    sobel_params['sigma'] = st.sidebar.slider(
        "Sobel Sigma:",
        0.1, 3.0, 1.0, 0.1,
        key="sobel_sigma",
        help="Gaussian smoothing trong Sobel"
    )
    
    # Tham số cho Laplacian
    st.sidebar.markdown("**Laplacian Parameters:**")
    laplacian_params = {}
    laplacian_params['kernel_size'] = st.sidebar.select_slider(
        "Laplacian Kernel size:",
        options=[3, 5, 7],
        value=3,
        key="laplacian_kernel_size"
    )
    laplacian_params['diagonal'] = st.sidebar.checkbox(
        "Include diagonal",
        value=False,
        key="laplacian_diagonal",
        help="Bao gồm các hướng chéo"
    )
    
    # Áp dụng cả 2 edge detection
    sobel_result = apply_edge_detection(current_image, "sobel", sobel_params)
    laplacian_result = apply_edge_detection(current_image, "laplacian", laplacian_params)
    
    st.markdown('<div class="bai3-step-header">Edge Detection - So sánh Sobel và Laplacian</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(safe_image_display(current_image), 
                caption="Before Edge Detection", 
                use_container_width=True)
    
    with col2:
        sobel_caption = f"Sobel Result\n({sobel_params['kernel_size']}x{sobel_params['kernel_size']}, σ={sobel_params['sigma']})"
        st.image(safe_image_display(sobel_result), 
                caption=sobel_caption, 
                use_container_width=True)
    
    with col3:
        laplacian_caption = f"Laplacian Result\n({laplacian_params['kernel_size']}x{laplacian_params['kernel_size']})"
        if laplacian_params['diagonal']:
            laplacian_caption += "\n+ Diagonal"
        st.image(safe_image_display(laplacian_result), 
                caption=laplacian_caption, 
                use_container_width=True)
    
    # Tổng quan tất cả kết quả
    st.markdown('<div class="bai3-step-header">Tổng quan Pipeline - Tất cả bước xử lý</div>', 
                unsafe_allow_html=True)
    
    # Tạo danh sách tất cả các ảnh
    all_images = [rgb_array, gray_image]
    all_titles = ['RGB Original', 'Grayscale']
    
    if noise_type != "none":
        all_images.append(noisy_image)
        all_titles.append(f'Noisy ({noise_type})')
    
    all_images.extend(preprocessing_images[1:])  # Bỏ input image vì đã có
    all_titles.extend([f"Preproc Step {i+1}" for i in range(len(preprocessing_images[1:]))])
    
    # Thêm cả 2 kết quả edge detection
    all_images.append(sobel_result)
    all_titles.append('Sobel Edge')
    all_images.append(laplacian_result)
    all_titles.append('Laplacian Edge')
    
    # Hiển thị comparison plot
    comparison_fig = create_comparison_plot(all_images, all_titles)
    st.pyplot(comparison_fig)
    plt.close()
    
    # Hiển thị histogram
    st.markdown('<div class="bai3-step-header">Histogram - Phân phối pixel từng bước</div>', 
                unsafe_allow_html=True)
    
    # Chỉ lấy grayscale images cho histogram
    grayscale_images = [gray_image]
    grayscale_titles = ['Grayscale']
    
    if noise_type != "none":
        grayscale_images.append(noisy_image)
        grayscale_titles.append(f'Noisy')
    
    grayscale_images.extend(preprocessing_images[1:])
    grayscale_titles.extend([f"Preproc {i+1}" for i in range(len(preprocessing_images[1:]))])
    
    # Thêm cả 2 kết quả edge detection
    grayscale_images.append(sobel_result)
    grayscale_titles.append('Sobel')
    grayscale_images.append(laplacian_result)
    grayscale_titles.append('Laplacian')
    
    histogram_fig = create_histogram_plot(grayscale_images, grayscale_titles)
    st.plotly_chart(histogram_fig, use_container_width=True)
    
    # Thống kê chi tiết
    st.markdown('<div class="bai3-step-header">Thống kê chi tiết</div>', 
                unsafe_allow_html=True)
    
    stats_data = []
    for img, title in zip(grayscale_images, grayscale_titles):
        stats_data.append({
            'Stage': title,
            'Mean': f"{np.mean(img):.2f}",
            'Std Dev': f"{np.std(img):.2f}",
            'Min': int(np.min(img)),
            'Max': int(np.max(img)),
            'Median': f"{np.median(img):.2f}",
            'Shape': f"{img.shape[0]}x{img.shape[1]}"
        })
    
    df = pd.DataFrame(stats_data)
    st.dataframe(df, use_container_width=True)
    
    # Download kết quả
    st.markdown('<div class="bai3-step-header">Download kết quả</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if noise_type != "none":
            noisy_pil = Image.fromarray(noisy_image)
            buf = io.BytesIO()
            noisy_pil.save(buf, format='PNG')
            st.download_button(
                label="Download Noisy Image",
                data=buf.getvalue(),
                file_name=f"noisy_{noise_type}.png",
                mime="image/png"
            )
    
    with col2:
        if len(preprocessing_images) > 1:
            preprocessed_pil = Image.fromarray(preprocessing_images[-1])
            buf = io.BytesIO()
            preprocessed_pil.save(buf, format='PNG')
            st.download_button(
                label="Download Preprocessed",
                data=buf.getvalue(),
                file_name="preprocessed.png",
                mime="image/png"
            )
    
    with col3:
        # Download Sobel result
        sobel_pil = Image.fromarray(sobel_result)
        buf = io.BytesIO()
        sobel_pil.save(buf, format='PNG')
        st.download_button(
            label="Download Sobel",
            data=buf.getvalue(),
            file_name="edge_sobel.png",
            mime="image/png"
        )
    
    with col4:
        # Download Laplacian result
        laplacian_pil = Image.fromarray(laplacian_result)
        buf = io.BytesIO()
        laplacian_pil.save(buf, format='PNG')
        st.download_button(
            label="Download Laplacian",
            data=buf.getvalue(),
            file_name="edge_laplacian.png",
            mime="image/png"
        )
    
    # ===== PHẦN MỚI: Tấm ảnh báo cáo đẹp cho Edge Detection =====
    st.markdown("---")
    st.markdown('<div class="bai3-step-header">📊 Tấm Ảnh Báo Cáo (Report Figure)</div>', 
                unsafe_allow_html=True)
    st.info("💡 Tấm ảnh này chứa 4 ảnh + 4 histogram (Input, Preprocessed, Sobel, Laplacian), phù hợp để chèn vào báo cáo/bài viết khoa học")
    
    # Xác định input image cho report (noisy hoặc gray)
    report_input_image = noisy_image if noise_type != "none" else gray_image
    report_preprocessed_image = preprocessing_images[-1] if len(preprocessing_images) > 1 else gray_image
    
    # Tạo report figure
    report_fig = create_report_figure_for_edge_detection(
        report_input_image,
        report_preprocessed_image,
        sobel_result,
        laplacian_result,
        noise_type,
        preprocessing_pipeline
    )
    
    # Hiển thị figure
    st.pyplot(report_fig)
    
    # Nút download
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        # Lưu figure thành bytes
        buf = io.BytesIO()
        report_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Tạo filename dựa trên config
        filename_parts = ['edge_detection_report']
        if noise_type != "none":
            filename_parts.append(noise_type)
        if preprocessing_pipeline:
            preproc_names = "_".join([s['type'] for s in preprocessing_pipeline if s['type'] != 'none'])
            if preproc_names:
                filename_parts.append(preproc_names)
        filename = "_".join(filename_parts) + ".png"
        
        st.download_button(
            label="📥 Download Tấm Ảnh Báo Cáo (High Resolution PNG)",
            data=buf.getvalue(),
            file_name=filename,
            mime="image/png",
            type="primary",
            use_container_width=True,
            help="DPI 300, phù hợp cho báo cáo và bài viết khoa học"
        )
    
    plt.close(report_fig)

if __name__ == "__main__":
    main()
