import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from core.image_ops import (
    rgb_to_grayscale_manual
)

from core.convolution import (
    convolution_2d_manual,
    median_filter_manual,
    threshold_comparison
)

from filters.gaussian_kernel import GaussianKernel
from filters.sobel_kernel import SobelKernel
from filters.laplacian_kernel import LaplacianKernel
from filters.sharpen_kernel import SharpenKernel
from filters.median_kernel import MedianKernel
from filters.noise_generator import NoiseGenerator

def safe_image_display(image):
    if image is None:
        return None
    
    img = np.array(image, copy=True)
    
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

def setup_bai2_styles():
    """CSS styles cho Bài 2"""
    st.markdown("""
    <style>
    .bai2-header {
        font-size: 2.5rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .bai2-step-header {
        font-size: 1.8rem;
        color: #1f77b4;
        margin: 1.5rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, #e3f2fd, #ffffff);
        border-left: 5px solid #1f77b4;
    }
    .bai2-info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .bai2-result-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #32cd32;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_comparison_plot(images, titles):
    """Tạo plot so sánh nhiều ảnh"""
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n_images):
        img = images[i].copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        if len(img.shape) == 3:
            axes[i].imshow(img)
        else:
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Ẩn axes thừa
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_image_stats(image, name):
    """Phân tích thống kê ảnh"""
    stats = {
        'Name': name,
        'Shape': f"{image.shape}",
        'Min': int(np.min(image)),
        'Max': int(np.max(image)),
        'Mean': f"{np.mean(image):.2f}",
        'Std': f"{np.std(image):.2f}",
        'Unique values': len(np.unique(image))
    }
    return stats

def render_sidebar():
    """Hiển thị sidebar điều khiển cho Bài 2"""
    st.sidebar.markdown("## Điều khiển")
    
    # Upload ảnh
    uploaded_file = st.sidebar.file_uploader(
        "Upload ảnh màu (< 5MB)",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Chọn ảnh màu để xử lý",
        key="bai2_file_uploader"
    )
    
    # Thêm nhiễu (Noise)
    st.sidebar.markdown("### Thêm nhiễu")
    add_noise = st.sidebar.checkbox("Thêm nhiễu vào ảnh", value=False)
    
    noise_type = None
    noise_params = {}
    
    if add_noise:
        noise_type = st.sidebar.selectbox(
            "Loại nhiễu:",
            ["salt_pepper", "gaussian", "poisson", "speckle", "uniform"],
            format_func=lambda x: {
                "salt_pepper": "Salt & Pepper",
                "gaussian": "Gaussian",
                "poisson": "Poisson",
                "speckle": "Speckle",
                "uniform": "Uniform"
            }[x]
        )
        
        if noise_type == "salt_pepper":
            noise_params['salt_prob'] = st.sidebar.slider(
                "Salt probability", 0.0, 0.1, 0.01, 0.005
            )
            noise_params['pepper_prob'] = st.sidebar.slider(
                "Pepper probability", 0.0, 0.1, 0.01, 0.005
            )
        elif noise_type == "gaussian":
            noise_params['mean'] = st.sidebar.slider(
                "Mean", -50, 50, 0, 5
            )
            noise_params['std'] = st.sidebar.slider(
                "Standard deviation", 1, 100, 25, 5
            )
        elif noise_type == "speckle":
            noise_params['std'] = st.sidebar.slider(
                "Standard deviation", 0.01, 0.5, 0.1, 0.01
            )
        elif noise_type == "uniform":
            noise_params['low'] = st.sidebar.slider(
                "Low value", -100, 0, -50, 10
            )
            noise_params['high'] = st.sidebar.slider(
                "High value", 0, 100, 50, 10
            )
    
    # Chọn loại kernel
    st.sidebar.markdown("### Chọn Loại Kernel")
    kernel_type = st.sidebar.selectbox(
        "Loại Kernel:",
        ["gaussian", "sobel", "laplacian", "sharpen"],
        index=0,
        help="Chọn loại kernel để áp dụng"
    )
    
    # Hiển thị thông tin về kernel được chọn
    kernel_info = {
        "gaussian": "Gaussian Blur - Làm mờ tự nhiên, giảm nhiễu",
        "sobel": "Sobel Gradient Magnitude - Phát hiện cạnh theo mọi hướng",
        "laplacian": "Laplacian - Phát hiện cạnh tất cả hướng",
        "sharpen": "Sharpen - Tăng độ sắc nét"
    }
    
    st.sidebar.info(kernel_info[kernel_type])
    
    # Tham số cho kernel được chọn
    st.sidebar.markdown("### Tham Số Kernel")
    
    kernel_params = {}
    
    if kernel_type == "gaussian":
        st.sidebar.markdown("**Gaussian Kernel Parameters:**")
        
        # Sigma cho các kích thước khác nhau
        sigma_3x3 = st.sidebar.slider(
            "Sigma cho 3x3 (I1)", 
            0.1, 2.0, 0.8, 0.1,
            help="Độ lệch chuẩn - càng lớn càng mờ"
        )
        sigma_5x5 = st.sidebar.slider(
            "Sigma cho 5x5 (I2)", 
            0.1, 3.0, 1.2, 0.1,
            help="Độ lệch chuẩn - càng lớn càng mờ"
        )
        sigma_7x7 = st.sidebar.slider(
            "Sigma cho 7x7 (I3)", 
            0.1, 3.0, 1.5, 0.1,
            help="Độ lệch chuẩn - càng lớn càng mờ"
        )
        
        kernel_params = {
            'sigma_3x3': sigma_3x3,
            'sigma_5x5': sigma_5x5, 
            'sigma_7x7': sigma_7x7
        }
        
        # Hiển thị preview kernel values
        st.sidebar.markdown("**Preview Kernel 3x3:**")
        preview_kernel = GaussianKernel.create_gaussian_2d_separable(3, sigma_3x3)
        
        # Format kernel để hiển thị
        kernel_str = "```\n"
        for i in range(3):
            row = " ".join([f"{preview_kernel[i,j]:.3f}" for j in range(3)])
            kernel_str += f"[{row}]\n"
        kernel_str += "```"
        st.sidebar.markdown(kernel_str)
        st.sidebar.caption(f"Sum: {np.sum(preview_kernel):.6f}")
        
    elif kernel_type == "sobel":
        st.sidebar.markdown("**Sobel Kernel Parameters:**")
        
        # Sigma cho Gaussian smoothing trong Sobel
        sigma_sobel = st.sidebar.slider(
            "Sigma cho Gaussian smoothing", 
            0.3, 2.0, 1.0, 0.1,
            help="Độ mượt của Gaussian - ảnh hưởng đến độ nhạy edge detection"
        )
        
        kernel_params = {
            'sigma_sobel': sigma_sobel
        }
        
        # Hiển thị preview kernels
        st.sidebar.markdown("**Preview Sobel X Kernel 3x3:**")
        preview_kernel_x = SobelKernel.create_sobel_x_kernel(3, sigma_sobel)
        
        kernel_str = "```\n"
        for i in range(3):
            row = " ".join([f"{preview_kernel_x[i,j]:.3f}" for j in range(3)])
            kernel_str += f"[{row}]\n"
        kernel_str += "```"
        st.sidebar.markdown(kernel_str)
        
        st.sidebar.markdown("**Preview Sobel Y Kernel 3x3:**")
        preview_kernel_y = SobelKernel.create_sobel_y_kernel(3, sigma_sobel)
        
        kernel_str = "```\n"
        for i in range(3):
            row = " ".join([f"{preview_kernel_y[i,j]:.3f}" for j in range(3)])
            kernel_str += f"[{row}]\n"
        kernel_str += "```"
        st.sidebar.markdown(kernel_str)
        st.sidebar.info("Gradient Magnitude = √(Gx² + Gy²)")
        
    elif kernel_type == "laplacian":
        st.sidebar.markdown("**Laplacian Kernel Parameters:**")
        
        # Option để sử dụng log transform
        use_log_transform = st.sidebar.checkbox(
            "Sử dụng Log Transform", 
            value=True,
            help="Cải thiện hiển thị chi tiết trong vùng tối"
        )
        
        # Hệ số khuếch đại cho log transform
        if use_log_transform:
            log_c = st.sidebar.slider(
                "Hệ số khuếch đại (c)", 
                0.1, 100.0, 1.0, 0.1,
                help="c càng lớn càng tăng độ sáng vùng tối - có thể lên tới 100 cho hiệu ứng mạnh"
            )
        else:
            log_c = 1.0
        
        kernel_params = {
            'use_log_transform': use_log_transform,
            'log_c': log_c
        }
        
        # Hiển thị preview kernel values
        st.sidebar.markdown("**Preview Laplacian Kernel 3x3:**")
        preview_kernel = LaplacianKernel.create_laplacian_3x3(diagonal=False)
        
        # Format kernel để hiển thị
        kernel_str = "```\n"
        for i in range(3):
            row = " ".join([f"{preview_kernel[i,j]:.0f}" for j in range(3)])
            kernel_str += f"[{row}]\n"
        kernel_str += "```"
        st.sidebar.markdown(kernel_str)
        st.sidebar.caption(f"Sum: {np.sum(preview_kernel):.0f}")
        
        if use_log_transform:
            st.sidebar.info(f"Sẽ áp dụng Log Transform với c={log_c}")
        
    else:
        # Cho các kernel khác không có tham số
        kernel_params = {}
    
    # Tham số convolution
    st.sidebar.markdown("### Tham Số Convolution")
    
    # Padding và stride cho từng kích thước
    st.sidebar.markdown("**I1 (3x3):**")
    padding_3x3 = st.sidebar.number_input("Padding 3x3", 0, 5, 1, help="Padding = 1 giữ nguyên kích thước")
    stride_3x3 = st.sidebar.number_input("Stride 3x3", 1, 3, 1, help="Stride = 1 không downsampling")
    
    st.sidebar.markdown("**I2 (5x5):**") 
    padding_5x5 = st.sidebar.number_input("Padding 5x5", 0, 5, 2, help="Padding = 2 giữ nguyên kích thước")
    stride_5x5 = st.sidebar.number_input("Stride 5x5", 1, 3, 1, help="Stride = 1 không downsampling")
    
    st.sidebar.markdown("**I3 (7x7):**")
    padding_7x7 = st.sidebar.number_input("Padding 7x7", 0, 5, 3, help="Padding = 3 giữ nguyên kích thước")
    stride_7x7 = st.sidebar.number_input("Stride 7x7", 1, 3, 2, help="Stride = 2 downsampling 1/2")
    
    convolution_params = {
        'padding_3x3': padding_3x3, 'stride_3x3': stride_3x3,
        'padding_5x5': padding_5x5, 'stride_5x5': stride_5x5,
        'padding_7x7': padding_7x7, 'stride_7x7': stride_7x7
    }
    
    # Tham số Median Filter
    st.sidebar.markdown("### Tham Số Median Filter")
    
    median_size = st.sidebar.slider(
        "Median Filter size cho I3 → I4",
        min_value=3,
        max_value=9,
        value=3,
        step=2,
        help="Kích thước kernel cho median filter (phải là số lẻ)"
    )
    
    median2_size = st.sidebar.slider(
        "Median Filter size cho I1 → I5",
        min_value=3,
        max_value=9,
        value=3,
        step=2,
        help="Kích thước kernel cho median filter (phải là số lẻ)"
    )
    
    return uploaded_file, kernel_type, kernel_params, convolution_params, median_size, median2_size, noise_type, noise_params

def display_kernel_info(kernel, name):
    """Hiển thị thông tin về kernel"""
    st.write(f"**{name}:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"- Size: {kernel.shape}")
        st.write(f"- Sum: {np.sum(kernel):.6f}")
        st.write(f"- Max: {np.max(kernel):.6f}")
        st.write(f"- Min: {np.min(kernel):.6f}")
    
    with col2:
        # Hiển thị kernel dưới dạng heatmap
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(kernel, cmap='viridis')
        ax.set_title(f'{name} Kernel')
        
        # Thêm text annotations
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                ax.text(j, i, f'{kernel[i, j]:.3f}', 
                       ha='center', va='center', color='white', fontsize=8)
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close()

def apply_log_transform_to_result(result, c=1.0):
    """
    Áp dụng log transform cho kết quả convolution
    
    Args:
        result: Kết quả convolution (có thể có giá trị âm)
        c: Hệ số khuếch đại
        
    Returns:
        numpy.ndarray: Kết quả sau log transform
    """
    # Xử lý giá trị âm (Laplacian có thể tạo giá trị âm)
    # Lấy absolute value để đảm bảo giá trị dương
    result_abs = np.abs(result)
    
    # Áp dụng log transform: c * log(1 + r)
    enhanced_result = c * np.log1p(result_abs)
    
    return enhanced_result


def main():
    """Hàm main cho Bài 2"""
    # Thiết lập styles
    setup_bai2_styles()
    
    # Sidebar - sẽ chỉ hiển thị khi tab này được chọn
    uploaded_file, kernel_type, kernel_params, convolution_params, median_size, median2_size, noise_type, noise_params = render_sidebar()
    
    # Xử lý ảnh
    if uploaded_file is not None:
        try:
            # Kiểm tra kích thước file
            if uploaded_file.size > 5 * 1024 * 1024:
                st.error("File quá lớn! Vui lòng chọn ảnh < 5MB")
                return
            
            # Load ảnh
            image = Image.open(uploaded_file)
            
            # Resize nếu ảnh quá lớn
            if image.width > 1500 or image.height > 1500:
                st.info("Ảnh lớn, đang resize để xử lý nhanh hơn...")
                image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            rgb_array = np.array(image)
            
            # Thêm nhiễu nếu được chọn
            if noise_type:
                st.info(f"Đang thêm nhiễu {noise_type}...")
                gray_temp = rgb_to_grayscale_manual(rgb_array)
                
                if noise_type == "salt_pepper":
                    gray_temp = NoiseGenerator.salt_and_pepper(
                        gray_temp, 
                        salt_prob=noise_params.get('salt_prob', 0.01),
                        pepper_prob=noise_params.get('pepper_prob', 0.01)
                    )
                elif noise_type == "gaussian":
                    gray_temp = NoiseGenerator.gaussian_noise(
                        gray_temp,
                        mean=noise_params.get('mean', 0),
                        std=noise_params.get('std', 25)
                    )
                elif noise_type == "poisson":
                    gray_temp = NoiseGenerator.poisson_noise(gray_temp)
                elif noise_type == "speckle":
                    gray_temp = NoiseGenerator.speckle_noise(
                        gray_temp,
                        std=noise_params.get('std', 0.1)
                    )
                elif noise_type == "uniform":
                    gray_temp = NoiseGenerator.uniform_noise(
                        gray_temp,
                        low=noise_params.get('low', -50),
                        high=noise_params.get('high', 50)
                    )
                
                # Convert back to RGB for display
                rgb_array = np.stack([gray_temp, gray_temp, gray_temp], axis=-1)
                
        except Exception as e:
            st.error(f"Lỗi khi tải ảnh: {e}")
            return
    else:
        # Yêu cầu upload ảnh
        st.info("Vui lòng tải ảnh lên từ thanh bên trái để bắt đầu xử lý.")
        return
    
    # Hiển thị ảnh gốc
    st.markdown('<div class="bai2-step-header">Ảnh Input</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(safe_image_display(rgb_array), caption=f"Ảnh gốc - Kích thước: {rgb_array.shape}", use_container_width=True)
    
    with col2:
        st.markdown('<div class="bai2-info-box">', unsafe_allow_html=True)
        st.write("**Thông tin ảnh:**")
        st.write(f"- Kích thước: {rgb_array.shape}")
        st.write(f"- Channels: {rgb_array.shape[2]}")
        st.write(f"- Data type: {rgb_array.dtype}")
        st.write(f"- Size: {rgb_array.size:,} pixels")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bước 1: RGB to Grayscale
    st.markdown('<div class="bai2-step-header">Bước 1: Chuyển RGB sang Grayscale</div>', 
                unsafe_allow_html=True)
    
    with st.expander("Xem quá trình chuyển đổi", expanded=False):
        st.write("**Công thức ITU-R BT.601:**")
        st.latex(r"Gray = 0.299 \times R + 0.587 \times G + 0.114 \times B")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress (thực tế sẽ được update trong hàm)
        gray_image = rgb_to_grayscale_manual(rgb_array)
        progress_bar.progress(1.0)
        status_text.success("Chuyển đổi hoàn thành!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(safe_image_display(gray_image), caption="Ảnh Grayscale", use_container_width=True)
    
    with col2:
        gray_stats = analyze_image_stats(gray_image, "Grayscale")
        st.json(gray_stats)
    
    # Tạo kernels theo loại được chọn
    st.markdown(f'<div class="bai2-step-header">Tạo {kernel_type.title()} Kernels</div>', 
                unsafe_allow_html=True)
    
    # Hiển thị thông tin về kernel được chọn
    kernel_descriptions = {
        "gaussian": "**Gaussian Kernel** - Làm mờ tự nhiên dựa trên phân phối Gaussian. Hiệu quả trong việc giảm nhiễu và tạo hiệu ứng blur mượt mà.",
        "sobel": "**Sobel Gradient Magnitude** - Phát hiện cạnh theo mọi hướng bằng cách tính gradient magnitude = √(Gx² + Gy²). Kết hợp Sobel X và Sobel Y để có edge detection toàn diện.",
        "laplacian": "**Laplacian** - Phát hiện cạnh theo mọi hướng bằng second derivative operator. Sensitive với noise nhưng hiệu quả.",
        "sharpen": "**Sharpen Filter** - Tăng cường độ sắc nét bằng cách tăng contrast tại các edges. Làm nổi bật chi tiết."
    }
    
    st.info(f"**Kernel được chọn:** {kernel_descriptions[kernel_type]}")
    
    # Tạo kernels theo loại được chọn
    if kernel_type == "gaussian":
        kernel_3x3 = GaussianKernel.create_gaussian_2d_separable(3, kernel_params['sigma_3x3'])
        kernel_5x5 = GaussianKernel.create_gaussian_2d_separable(5, kernel_params['sigma_5x5'])
        kernel_7x7 = GaussianKernel.create_gaussian_2d_separable(7, kernel_params['sigma_7x7'])
        kernel_names = [
            f"Gaussian 3x3 (σ={kernel_params['sigma_3x3']})",
            f"Gaussian 5x5 (σ={kernel_params['sigma_5x5']})", 
            f"Gaussian 7x7 (σ={kernel_params['sigma_7x7']})"
        ]
        
    elif kernel_type == "sobel":
        sigma_sobel = kernel_params.get('sigma_sobel', 1.0)
        # Tạo Sobel X và Y kernels
        sobel_x_3x3 = SobelKernel.create_sobel_x_kernel(3, sigma_sobel)
        sobel_y_3x3 = SobelKernel.create_sobel_y_kernel(3, sigma_sobel)
        sobel_x_5x5 = SobelKernel.create_sobel_x_kernel(5, sigma_sobel)
        sobel_y_5x5 = SobelKernel.create_sobel_y_kernel(5, sigma_sobel)
        sobel_x_7x7 = SobelKernel.create_sobel_x_kernel(7, sigma_sobel)
        sobel_y_7x7 = SobelKernel.create_sobel_y_kernel(7, sigma_sobel)
        
        # Store both X and Y kernels for gradient magnitude calculation
        kernel_3x3 = {'x': sobel_x_3x3, 'y': sobel_y_3x3}
        kernel_5x5 = {'x': sobel_x_5x5, 'y': sobel_y_5x5}
        kernel_7x7 = {'x': sobel_x_7x7, 'y': sobel_y_7x7}
        
        kernel_names = [
            f"Sobel Magnitude 3x3 (σ={sigma_sobel})", 
            f"Sobel Magnitude 5x5 (σ={sigma_sobel})", 
            f"Sobel Magnitude 7x7 (σ={sigma_sobel})"
        ]
    elif kernel_type == "laplacian":
        kernel_3x3 = LaplacianKernel.create_laplacian_3x3(diagonal=False)
        kernel_5x5 = LaplacianKernel.create_laplacian_5x5(diagonal=False)
        kernel_7x7 = LaplacianKernel.create_laplacian_7x7(diagonal=False)
        
        # Thêm thông tin về log transform vào tên
        use_log = kernel_params.get('use_log_transform', False)
        log_c = kernel_params.get('log_c', 1.0)
        
        if use_log:
            kernel_names = [
                f"Laplacian 3x3 + Log(c={log_c})", 
                f"Laplacian 5x5 + Log(c={log_c})", 
                f"Laplacian 7x7 + Log(c={log_c})"
            ]
        else:
            kernel_names = ["Laplacian 3x3", "Laplacian 5x5", "Laplacian 7x7"]
    elif kernel_type == "sharpen":
        kernel_3x3 = SharpenKernel.create_sharpen_3x3(alpha=1.0, diagonal=False)
        kernel_5x5 = SharpenKernel.create_sharpen_5x5(alpha=1.0, diagonal=False)
        kernel_7x7 = SharpenKernel.create_sharpen_7x7(alpha=1.0, diagonal=False)
        kernel_names = ["Sharpen 3x3", "Sharpen 5x5", "Sharpen 7x7"]
    
    with st.expander("Xem chi tiết kernels", expanded=False):
        if kernel_type == "sobel":
            # Display both Sobel X and Y kernels
            st.markdown("**Sobel X Kernels:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                display_kernel_info(kernel_3x3['x'], "Sobel X 3x3")
            with col2:
                display_kernel_info(kernel_5x5['x'], "Sobel X 5x5")
            with col3:
                display_kernel_info(kernel_7x7['x'], "Sobel X 7x7")
            
            st.markdown("**Sobel Y Kernels:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                display_kernel_info(kernel_3x3['y'], "Sobel Y 3x3")
            with col2:
                display_kernel_info(kernel_5x5['y'], "Sobel Y 5x5")
            with col3:
                display_kernel_info(kernel_7x7['y'], "Sobel Y 7x7")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                display_kernel_info(kernel_3x3, kernel_names[0])
            with col2:
                display_kernel_info(kernel_5x5, kernel_names[1])
            with col3:
                display_kernel_info(kernel_7x7, kernel_names[2])
    
    # Bước 2-4: Convolutions
    st.markdown('<div class="bai2-step-header">Bước 2-4: Convolution Operations</div>', 
                unsafe_allow_html=True)
    
    # Progress tracking
    conv_progress = st.progress(0)
    conv_status = st.empty()
    
    # I1: Conv 3x3
    conv_status.text("Đang thực hiện Convolution 3x3...")
    if kernel_type == "sobel":
        # Compute gradient magnitude for Sobel
        I1_x = convolution_2d_manual(gray_image, kernel_3x3['x'], 
                                     padding=convolution_params['padding_3x3'], 
                                     stride=convolution_params['stride_3x3'])
        I1_y = convolution_2d_manual(gray_image, kernel_3x3['y'], 
                                     padding=convolution_params['padding_3x3'], 
                                     stride=convolution_params['stride_3x3'])
        I1 = SobelKernel.compute_gradient_magnitude(I1_x, I1_y)
    else:
        I1 = convolution_2d_manual(gray_image, kernel_3x3, 
                                  padding=convolution_params['padding_3x3'], 
                                  stride=convolution_params['stride_3x3'])
    
    # Áp dụng log transform cho Laplacian nếu được chọn
    if kernel_type == "laplacian" and kernel_params.get('use_log_transform', False):
        log_c = kernel_params.get('log_c', 1.0)
        I1 = apply_log_transform_to_result(I1, log_c)
    
    conv_progress.progress(0.33)
    
    # I2: Conv 5x5
    conv_status.text("Đang thực hiện Convolution 5x5...")
    if kernel_type == "sobel":
        # Compute gradient magnitude for Sobel
        I2_x = convolution_2d_manual(gray_image, kernel_5x5['x'], 
                                     padding=convolution_params['padding_5x5'], 
                                     stride=convolution_params['stride_5x5'])
        I2_y = convolution_2d_manual(gray_image, kernel_5x5['y'], 
                                     padding=convolution_params['padding_5x5'], 
                                     stride=convolution_params['stride_5x5'])
        I2 = SobelKernel.compute_gradient_magnitude(I2_x, I2_y)
    else:
        I2 = convolution_2d_manual(gray_image, kernel_5x5, 
                                  padding=convolution_params['padding_5x5'], 
                                  stride=convolution_params['stride_5x5'])
    
    # Áp dụng log transform cho Laplacian nếu được chọn
    if kernel_type == "laplacian" and kernel_params.get('use_log_transform', False):
        log_c = kernel_params.get('log_c', 1.0)
        I2 = apply_log_transform_to_result(I2, log_c)
    
    conv_progress.progress(0.66)
    
    # I3: Conv 7x7
    conv_status.text("Đang thực hiện Convolution 7x7...")
    if kernel_type == "sobel":
        # Compute gradient magnitude for Sobel
        I3_x = convolution_2d_manual(gray_image, kernel_7x7['x'], 
                                     padding=convolution_params['padding_7x7'], 
                                     stride=convolution_params['stride_7x7'])
        I3_y = convolution_2d_manual(gray_image, kernel_7x7['y'], 
                                     padding=convolution_params['padding_7x7'], 
                                     stride=convolution_params['stride_7x7'])
        I3 = SobelKernel.compute_gradient_magnitude(I3_x, I3_y)
    else:
        I3 = convolution_2d_manual(gray_image, kernel_7x7, 
                                  padding=convolution_params['padding_7x7'], 
                                  stride=convolution_params['stride_7x7'])
    
    # Áp dụng log transform cho Laplacian nếu được chọn
    if kernel_type == "laplacian" and kernel_params.get('use_log_transform', False):
        log_c = kernel_params.get('log_c', 1.0)
        I3 = apply_log_transform_to_result(I3, log_c)
    
    conv_progress.progress(1.0)
    conv_status.success("Tất cả convolutions hoàn thành!")
    
    # Hiển thị kết quả convolutions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(safe_image_display(I1), caption=f"I1: {kernel_names[0]} (pad={convolution_params['padding_3x3']}, stride={convolution_params['stride_3x3']}) - {I1.shape}", use_container_width=True)
        st.json(analyze_image_stats(I1, "I1"))
    
    with col2:
        st.image(safe_image_display(I2), caption=f"I2: {kernel_names[1]} (pad={convolution_params['padding_5x5']}, stride={convolution_params['stride_5x5']}) - {I2.shape}", use_container_width=True)
        st.json(analyze_image_stats(I2, "I2"))
    
    with col3:
        st.image(safe_image_display(I3), caption=f"I3: {kernel_names[2]} (pad={convolution_params['padding_7x7']}, stride={convolution_params['stride_7x7']}) - {I3.shape}", use_container_width=True)
        st.json(analyze_image_stats(I3, "I3"))
    
    # Bước 5: Median Filter
    st.markdown('<div class="bai2-step-header">Bước 5: Median Filter I3 → I4</div>', 
                unsafe_allow_html=True)
    
    with st.expander("Về Median Filter", expanded=False):
        st.write("""
        **Median Filter:**
        - Lọc nhiễu hiệu quả (đặc biệt salt-and-pepper noise)
        - Giữ được edges tốt hơn Gaussian filter
        - Thay mỗi pixel bằng median của neighborhood
        - Không làm mờ edges như linear filters
        """)
    
    median_progress = st.progress(0)
    median_status = st.empty()
    
    median_status.text(f"Đang thực hiện Median Filter {median_size}x{median_size}...")
    I4 = median_filter_manual(I3, kernel_size=median_size)
    median_progress.progress(1.0)
    median_status.success("Median Filter hoàn thành!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(safe_image_display(I3), caption=f"I3: Input cho Median Filter", use_container_width=True)
    with col2:
        st.image(safe_image_display(I4), caption=f"I4: Sau Median Filter {median_size}x{median_size}", use_container_width=True)
    
    # Bước 6: Median Filter I1
    st.markdown('<div class="bai2-step-header">Bước 6: Median Filter I1 → I5</div>', 
                unsafe_allow_html=True)
    
    with st.expander("Về Median Filter cho I1", expanded=False):
        st.write("""
        **Median Filter cho I1:**
        - Lọc trung bị (median filter) 
        - Loại bỏ nhiễu hiệu quả
        - Giữ được edges tốt
        - Thay mỗi pixel bằng median của neighborhood
        - Khác với min filter ở bước trước
        """)
    
    median2_progress = st.progress(0)
    median2_status = st.empty()
    
    median2_status.text(f"Đang thực hiện Median Filter {median2_size}x{median2_size} cho I1...")
    I5 = median_filter_manual(I1, kernel_size=median2_size)
    median2_progress.progress(1.0)
    median2_status.success("Median Filter I1 hoàn thành!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(safe_image_display(I1), caption=f"I1: Input cho Median Filter", use_container_width=True)
    with col2:
        st.image(safe_image_display(I5), caption=f"I5: Sau Median Filter {median2_size}x{median2_size}", use_container_width=True)
    
    # Bước 7: Thresholding
    st.markdown('<div class="bai2-step-header">Bước 7: Thresholding I4 vs I5 → I6</div>', 
                unsafe_allow_html=True)
    
    with st.expander("Về Thresholding Logic", expanded=False):
        st.write("""
        **Thresholding Rule:**
        ```python
        if I4(x,y) > I5(x,y):
            I6(x,y) = 0
        else:
            I6(x,y) = I5(x,y)
        ```
        
        **Ý nghĩa:**
        - So sánh kết quả 2 median filters
        - I4: Median filter của I3 (conv 7x7)
        - I5: Median filter của I1 (conv 3x3)
        - Nếu I4 > I5 → set pixel = 0 (đen)
        - Ngược lại → giữ giá trị từ I5
        """)
    
    # Kiểm tra kích thước trước khi threshold
    st.write(f"**Kích thước trước khi threshold:**")
    st.write(f"- I4: {I4.shape}")
    st.write(f"- I5: {I5.shape}")
    
    if I4.shape != I5.shape:
        st.warning("Hai ảnh có kích thước khác nhau. Sẽ resize để match...")
    
    threshold_status = st.empty()
    threshold_status.text("Đang thực hiện Thresholding...")
    I6 = threshold_comparison(I4, I5)
    threshold_status.success("Thresholding hoàn thành!")
    
    # Hiển thị kết quả cuối cùng
    st.markdown('<div class="bai2-step-header">Kết quả cuối cùng</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(safe_image_display(I4), caption=f"I4: Median Filter", use_container_width=True)
        st.json(analyze_image_stats(I4, "I4"))
    
    with col2:
        st.image(safe_image_display(I5), caption=f"I5: Median Filter I1", use_container_width=True)  
        st.json(analyze_image_stats(I5, "I5"))
    
    with col3:
        st.image(safe_image_display(I6), caption=f"I6: Final Result", use_container_width=True)
        st.json(analyze_image_stats(I6, "I6"))
    
    # Tổng quan tất cả kết quả
    st.markdown('<div class="bai2-step-header">Tổng quan tất cả kết quả</div>', 
                unsafe_allow_html=True)
    
    # Tạo comparison plot
    all_images = [rgb_array, gray_image, I1, I2, I3, I4, I5, I6]
    all_titles = ['Original RGB', 'Grayscale', 'I1: Conv 3x3', 'I2: Conv 5x5', 
                  'I3: Conv 7x7', 'I4: Median I3', 'I5: Median I1', 'I6: Final']
    
    comparison_fig = create_comparison_plot(all_images, all_titles)
    st.pyplot(comparison_fig)
    plt.close()
    
    # Hiển thị histogram cho từng kết quả
    st.markdown('<div class="bai2-step-header">Histogram của từng kết quả</div>', 
                unsafe_allow_html=True)
    
    # Tạo histogram plot cho các kết quả grayscale
    fig_hist = make_subplots(
        rows=3, cols=3,
        subplot_titles=('Grayscale', 'I1: Conv 3x3', 'I2: Conv 5x5',
                       'I3: Conv 7x7', 'I4: Median I3', 'I5: Median I1',
                       'I6: Final', '', ''),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Danh sách các ảnh cần vẽ histogram (chỉ grayscale images)
    hist_images = [gray_image, I1, I2, I3, I4, I5, I6]
    hist_titles = ['Grayscale', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6']
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    
    # Vẽ histogram cho từng ảnh
    for idx, (img, title, color) in enumerate(zip(hist_images, hist_titles, colors)):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        # Tính histogram
        hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
        
        # Thêm vào subplot
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
        
        # Cập nhật axes
        fig_hist.update_xaxes(title_text="Giá trị pixel", row=row, col=col, showgrid=False)
        fig_hist.update_yaxes(title_text="Tần suất", row=row, col=col, showgrid=True, gridcolor='#ecf0f1')
    
    # Cập nhật layout
    fig_hist.update_layout(
        height=900,
        title_text="Phân phối pixel của các kết quả",
        title_x=0.5,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Thêm thông tin thống kê histogram
    st.markdown('<div class="bai2-info-box">', unsafe_allow_html=True)
    st.write("**Nhận xét về histogram:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Grayscale:**")
        gray_mean = np.mean(gray_image)
        gray_std = np.std(gray_image)
        st.write(f"- Mean: {gray_mean:.2f}")
        st.write(f"- Std Dev: {gray_std:.2f}")
        st.write(f"- Range: [{np.min(gray_image)}, {np.max(gray_image)}]")
        
        st.write("\n**I1 (Conv 3x3):**")
        i1_mean = np.mean(I1)
        i1_std = np.std(I1)
        st.write(f"- Mean: {i1_mean:.2f}")
        st.write(f"- Std Dev: {i1_std:.2f}")
        st.write(f"- Range: [{np.min(I1)}, {np.max(I1)}]")
        
        st.write("\n**I2 (Conv 5x5):**")
        i2_mean = np.mean(I2)
        i2_std = np.std(I2)
        st.write(f"- Mean: {i2_mean:.2f}")
        st.write(f"- Std Dev: {i2_std:.2f}")
        st.write(f"- Range: [{np.min(I2)}, {np.max(I2)}]")
    
    with col2:
        st.write("**I3 (Conv 7x7):**")
        i3_mean = np.mean(I3)
        i3_std = np.std(I3)
        st.write(f"- Mean: {i3_mean:.2f}")
        st.write(f"- Std Dev: {i3_std:.2f}")
        st.write(f"- Range: [{np.min(I3)}, {np.max(I3)}]")
        
        st.write("\n**I4 (Median I3):**")
        i4_mean = np.mean(I4)
        i4_std = np.std(I4)
        st.write(f"- Mean: {i4_mean:.2f}")
        st.write(f"- Std Dev: {i4_std:.2f}")
        st.write(f"- Range: [{np.min(I4)}, {np.max(I4)}]")
        
        st.write("\n**I5 (Median I1):**")
        i5_mean = np.mean(I5)
        i5_std = np.std(I5)
        st.write(f"- Mean: {i5_mean:.2f}")
        st.write(f"- Std Dev: {i5_std:.2f}")
        st.write(f"- Range: [{np.min(I5)}, {np.max(I5)}]")
        
        st.write("\n**I6 (Final):**")
        i6_mean = np.mean(I6)
        i6_std = np.std(I6)
        st.write(f"- Mean: {i6_mean:.2f}")
        st.write(f"- Std Dev: {i6_std:.2f}")
        st.write(f"- Range: [{np.min(I6)}, {np.max(I6)}]")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bảng thống kê
    st.markdown('<div class="bai2-result-box">', unsafe_allow_html=True)
    st.write("**Bảng thống kê tổng hợp:**")
    
    stats_data = []
    for img, title in zip([gray_image, I1, I2, I3, I4, I5, I6], 
                         ['Grayscale', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6']):
        stats = analyze_image_stats(img, title)
        stats_data.append(stats)
    
    df = pd.DataFrame(stats_data)
    st.dataframe(df, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download kết quả
    st.markdown('<div class="bai2-step-header">Download kết quả</div>', 
                unsafe_allow_html=True)
    
    download_cols = st.columns(4)
    
    # Download I4
    with download_cols[0]:
        I4_pil = Image.fromarray(I4)
        buf = io.BytesIO()
        I4_pil.save(buf, format='PNG')
        st.download_button(
            "Download I4",
            buf.getvalue(),
            "I4_median_filter.png",
            "image/png"
        )
    
    # Download I5
    with download_cols[1]:
        I5_pil = Image.fromarray(I5)
        buf = io.BytesIO()
        I5_pil.save(buf, format='PNG')
        st.download_button(
            "Download I5",
            buf.getvalue(),
            "I5_min_filter.png",
            "image/png"
        )
    
    # Download I6
    with download_cols[2]:
        I6_pil = Image.fromarray(I6)
        buf = io.BytesIO()
        I6_pil.save(buf, format='PNG')
        st.download_button(
            "Download I6",
            buf.getvalue(),
            "I6_final_result.png",
            "image/png"
        )
    
    # Download comparison plot
    with download_cols[3]:
        buf = io.BytesIO()
        comparison_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        st.download_button(
            "Download All",
            buf.getvalue(),
            "bai2_all_results.png",
            "image/png"
        )

if __name__ == "__main__":
    main()
