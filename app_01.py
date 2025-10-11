import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.histogram import (
    calculate_histogram_manual, 
    calculate_histogram_library,
    histogram_equalization_manual,
    histogram_equalization_library,
    histogram_narrowing_manual,
    analyze_histogram
)
from core.image_ops import (
    rgb_to_grayscale_manual,
    rgb_to_grayscale_library
)

def safe_image_display(image):
    if image is None:
        return None
    
    img = np.array(image, copy=True)
    
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

# cấu hình web
def setup_bai1_styles():
    """CSS styles cho Bài 1"""
    st.markdown("""
    <style>
    .bai1-title {
        font-size: 2.2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .bai1-section-header {
        font-size: 1.5rem;
        color: #34495e;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, #ecf0f1, #ffffff);
        border-left: 4px solid #3498db;
        border-radius: 5px;
    }
    .bai1-info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .bai1-stats-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# tạo biểu đồ histogram
def create_histogram_plot(h1, h2, h3, min_val, max_val):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('H1 - Histogram gốc', 'H2 - Sau cân bằng', 
                       f'H3 - Thu hẹp [{min_val}-{max_val}]', 'So sánh'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Màu sắc phân biệt
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    # H1 - Histogram gốc
    fig.add_trace(
        go.Scatter(x=list(range(256)), y=h1, name='H1 - Gốc', 
                  line_color=colors[0], line_width=2),
        row=1, col=1
    )
    
    # H2 - Sau cân bằng
    fig.add_trace(
        go.Scatter(x=list(range(256)), y=h2, name='H2 - Cân bằng', 
                  line_color=colors[1], line_width=2),
        row=1, col=2
    )
    
    # H3 - Thu hẹp
    fig.add_trace(
        go.Scatter(x=list(range(256)), y=h3, name='H3 - Thu hẹp', 
                  line_color=colors[2], line_width=2),
        row=2, col=1
    )
    
    # So sánh tất cả
    fig.add_trace(
        go.Scatter(x=list(range(256)), y=h1, name='H1 - Gốc', 
                  line_color=colors[0], line_width=2),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(range(256)), y=h2, name='H2 - Cân bằng', 
                  line_color=colors[1], line_width=2),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(range(256)), y=h3, name='H3 - Thu hẹp', 
                  line_color=colors[2], line_width=2),
        row=2, col=2
    )
    
    # Layout với màu nền sáng
    fig.update_layout(
        height=700, 
        showlegend=True,
        title_text="Phân tích Histogram",
        title_x=0.5,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )
    
    # Cập nhật axes với màu grid nhẹ
    fig.update_xaxes(title_text="Mức xám", showgrid=True, gridcolor='#ecf0f1', gridwidth=1)
    fig.update_yaxes(title_text="Tần suất", showgrid=True, gridcolor='#ecf0f1', gridwidth=1)
    
    return fig

# phân tích thống kê histogram - sử dụng function từ core
def analyze_histogram_stats(hist, name):
    analysis = analyze_histogram(hist)
    
    return {
        'mean': analysis['mean'],
        'std': analysis['std'],
        'entropy': analysis['entropy'],
        'min': analysis['min_intensity'],
        'max': analysis['max_intensity'],
        'range': analysis['intensity_range']
    }

# hiển thị sidebar điều khiển
def render_sidebar():
    st.sidebar.markdown("## Điều khiển")
    
    images = []
    image_names = []
    
    # Upload nhiều ảnh
    uploaded_files = st.sidebar.file_uploader(
        "Tải ảnh lên (tối đa 10 ảnh, mỗi ảnh < 5MB)",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True,
        help="Chọn nhiều ảnh màu để xử lý histogram",
        key="bai1_file_uploader"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.sidebar.error("Tối đa 10 ảnh!")
        else:
            for uploaded_file in uploaded_files:
                try:
                    # Kiểm tra kích thước file (5MB = 5 * 1024 * 1024 bytes)
                    if uploaded_file.size > 5 * 1024 * 1024:
                        st.sidebar.error(f"{uploaded_file.name}: File quá lớn (> 5MB)")
                        continue
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Đọc và xử lý ảnh
                    image = Image.open(uploaded_file)
                    
                    # Kiểm tra kích thước ảnh
                    if image.width > 2000 or image.height > 2000:
                        st.sidebar.warning(f"{uploaded_file.name}: Ảnh lớn, đang resize...")
                        # Resize ảnh nếu quá lớn
                        image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Validate image array
                    img_array = np.array(image)
                    if img_array.shape[2] != 3:
                        st.sidebar.error(f"{uploaded_file.name}: Ảnh không đúng định dạng RGB")
                        continue
                    
                    images.append(img_array)
                    image_names.append(uploaded_file.name)
                    
                except Exception as e:
                    st.sidebar.error(f"Lỗi tải {uploaded_file.name}: {str(e)[:50]}...")
    
    # Chọn phương pháp
    method = st.sidebar.selectbox(
        "Phương pháp xử lý",
        ["Sử dụng thư viện OpenCV", "Tự lập trình"],
        help="Thư viện: nhanh, Tự lập trình: hiểu thuật toán"
    )
    
    # Tham số thu hẹp
    st.sidebar.markdown("### Tham số thu hẹp")
    min_val = st.sidebar.slider("Giá trị tối thiểu", 0, 100, 30)
    max_val = st.sidebar.slider("Giá trị tối đa", min_val+1, 255, 80)
    
    return images, image_names, method, min_val, max_val

# hiển thị tổng quan dataset
def render_dataset_overview(images, image_names):
    st.markdown('<div class="bai1-section-header">Tổng quan Dataset</div>', 
                unsafe_allow_html=True)
    
    # Thống kê tổng quan
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Số lượng ảnh", len(images))
    
    with col2:
        avg_size = np.mean([img.shape[0] * img.shape[1] for img in images])
        st.metric("Pixel trung bình", f"{avg_size:,.0f}")
    
    with col3:
        total_pixels = sum(img.shape[0] * img.shape[1] for img in images)
        st.metric("Tổng pixel", f"{total_pixels:,.0f}")
    
    with col4:
        memory_mb = sum(img.nbytes for img in images) / (1024 * 1024)
        st.metric("Bộ nhớ", f"{memory_mb:.1f} MB")
    
    # Gallery ảnh
    st.markdown("**Thư viện ảnh:**")
    cols_per_row = 5
    rows = (len(images) + cols_per_row - 1) // cols_per_row
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx
            if img_idx < len(images):
                with cols[col_idx]:
                    st.image(safe_image_display(images[img_idx]), caption=image_names[img_idx], use_container_width=True)

# xử lý một ảnh đơn lẻ - sử dụng core functions
def process_single_image(image, image_name, method, min_val, max_val):
    results = {'name': image_name}
    
    try:
        # Bước 1: Chuyển sang xám - sử dụng core functions
        if method == "Sử dụng thư viện OpenCV":
            gray_image = rgb_to_grayscale_library(image)
        else:
            gray_image = rgb_to_grayscale_manual(image)
        results['gray_image'] = gray_image
        
        # Bước 2: Tính histogram gốc - sử dụng core functions
        if method == "Sử dụng thư viện OpenCV":
            h1 = calculate_histogram_library(gray_image)
        else:
            h1 = calculate_histogram_manual(gray_image)
        results['h1'] = h1
        
        # Bước 3: Cân bằng histogram - sử dụng core functions
        if method == "Sử dụng thư viện OpenCV":
            equalized_image, h2 = histogram_equalization_library(gray_image)
        else:
            equalized_image, h2 = histogram_equalization_manual(gray_image)
        results['equalized_image'] = equalized_image
        results['h2'] = h2
        
        # Bước 4: Thu hẹp histogram - sử dụng core functions
        narrowed_image, h3 = histogram_narrowing_manual(equalized_image, min_val, max_val)
        results['narrowed_image'] = narrowed_image
        results['h3'] = h3
        
        results['success'] = True
        results['error'] = None
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results

# xử lý batch cho nhiều ảnh - vòng for xử lý tất cả ảnh
def render_batch_processing(images, image_names, method, min_val, max_val):
    st.markdown('<div class="bai1-section-header">Xử lý Batch</div>', 
                unsafe_allow_html=True)
    
    if st.button("Bắt đầu xử lý tất cả ảnh", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, (image, name) in enumerate(zip(images, image_names)):
            status_text.text(f"Đang xử lý: {name} ({i+1}/{len(images)})")
            
            result = process_single_image(image, name, method, min_val, max_val)
            results.append(result)
            
            progress_bar.progress((i + 1) / len(images))
        
        # Lưu kết quả vào session state
        st.session_state.batch_results = results
        st.session_state.processing_config = {
            'method': method,
            'min_val': min_val,
            'max_val': max_val
        }
        
        # Thống kê tổng kết
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        status_text.success(f"Hoàn thành! Thành công: {successful}/{len(results)} ảnh")
        progress_bar.empty()
        
        # Rerun để hiển thị kết quả
        st.rerun()

# hiển thị kết quả batch processing
def render_batch_results():
    if 'batch_results' not in st.session_state:
        return
    
    results = st.session_state.batch_results
    config = st.session_state.processing_config
    
    st.markdown('<div class="bai1-section-header">Kết quả Batch Processing</div>', 
                unsafe_allow_html=True)
    
    # Chọn ảnh để xem chi tiết
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        st.error("Không có ảnh nào được xử lý thành công!")
        return
    
    selected_idx = st.selectbox(
        "Chọn ảnh để xem chi tiết:",
        range(len(successful_results)),
        format_func=lambda x: successful_results[x]['name']
    )
    
    result = successful_results[selected_idx]
    
    # Hiển thị kết quả chi tiết cho ảnh được chọn
    render_single_result(result, config['min_val'], config['max_val'])

# hiển thị kết quả chi tiết cho một ảnh
def render_single_result(result, min_val, max_val):
    # Hiển thị 4 bước xử lý
    st.markdown("**Các bước xử lý:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image(safe_image_display(result['gray_image']), caption="1. Ảnh xám", use_container_width=True)
    
    with col2:
        st.image(safe_image_display(result['equalized_image']), caption="2. Cân bằng histogram", use_container_width=True)
    
    with col3:
        st.image(safe_image_display(result['narrowed_image']), caption=f"3. Thu hẹp [{min_val}-{max_val}]", use_container_width=True)
    
    with col4:
        # Hiển thị thống kê
        st.markdown("**Thống kê:**")
        stats_h1 = analyze_histogram_stats(result['h1'], "H1")
        stats_h2 = analyze_histogram_stats(result['h2'], "H2")
        stats_h3 = analyze_histogram_stats(result['h3'], "H3")
        
        st.write(f"H1 Mean: {stats_h1['mean']:.1f}")
        st.write(f"H2 Mean: {stats_h2['mean']:.1f}")
        st.write(f"H3 Mean: {stats_h3['mean']:.1f}")
    
    # Hiển thị histogram
    fig = create_histogram_plot(result['h1'], result['h2'], result['h3'], min_val, max_val)
    st.plotly_chart(fig, use_container_width=True)

# hàm chính
def main():
    # Thiết lập styles
    setup_bai1_styles()
    
    # Sidebar - sẽ chỉ hiển thị khi tab này được chọn
    images, image_names, method, min_val, max_val = render_sidebar()
    
    # Xử lý ảnh
    if len(images) > 0:
        # Hiển thị tổng quan dataset
        render_dataset_overview(images, image_names)
        
        # Xử lý batch
        render_batch_processing(images, image_names, method, min_val, max_val)
        
        # Hiển thị kết quả batch
        render_batch_results()
    
    else:
        # Thông báo upload ảnh
        st.info("Vui lòng tải ảnh lên từ thanh bên trái để bắt đầu xử lý.")

if __name__ == "__main__":
    main()
