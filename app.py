"""
Streamlit GUI cho Đồ án Xử lý Ảnh Số
==================================

Giao diện web đẹp mắt để demo các thuật toán xử lý ảnh:
- Bài 1: Histogram Processing
- Bài 2: Filtering Operations
- Batch processing cho 10 ảnh
- Export PDF report

Author: Image Processing Team
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import zipfile
import os
from datetime import datetime

# Import các modules tự tạo
from src.utils import load_image, rgb_to_grayscale, save_image, calculate_image_stats
from src.histogram import process_task1, create_histogram_comparison_plotly, analyze_histogram_properties
from src.filtering import process_task2, create_kernel_visualization, compare_filtering_methods, analyze_filter_effects
from src.pdf_generator import generate_pdf_report, create_sample_team_info


# Cấu hình trang
st.set_page_config(
    page_title="Xử lý Ảnh Số - Đồ án Cuối kỳ",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp giao diện
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def display_image_grid(images, titles, cols=3):
    """Hiển thị grid các ảnh"""
    rows = (len(images) + cols - 1) // cols
    
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(images):
                with columns[col]:
                    st.image(images[idx], caption=titles[idx], use_column_width=True)


def create_download_zip(results_dict, filename_prefix="processed_images"):
    """Tạo file ZIP chứa tất cả ảnh đã xử lý"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for key, image in results_dict.items():
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                if len(image.shape) == 2:  # Grayscale
                    pil_img = Image.fromarray(image, mode='L')
                else:  # RGB
                    pil_img = Image.fromarray(image, mode='RGB')
                
                # Save to bytes
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Add to ZIP
                zip_file.writestr(f"{filename_prefix}_{key}.png", img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer


def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Đồ án Xử lý Ảnh Số</h1>', unsafe_allow_html=True)
    
    # Thông tin nhóm
    st.markdown("""
    <div class="info-box">
        <h3>👥 Thông tin nhóm</h3>
        <ul>
            <li><strong>Môn học:</strong> Xử lý Ảnh Số</li>
            <li><strong>Đề tài:</strong> Histogram Processing & Image Filtering</li>
            <li><strong>Công nghệ:</strong> Python, OpenCV, Streamlit</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Điều khiển")
    
    # Upload ảnh
    uploaded_files = st.sidebar.file_uploader(
        "📤 Upload ảnh(s)",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True,
        help="Chọn 1 hoặc nhiều ảnh để xử lý"
    )
    
    # Chọn chế độ xử lý
    processing_mode = st.sidebar.selectbox(
        "🔧 Chế độ xử lý",
        ["Single Image Analysis", "Batch Processing (10 ảnh)", "Algorithm Comparison"]
    )
    
    # Chọn thuật toán
    algorithms = st.sidebar.multiselect(
        "🧮 Chọn thuật toán",
        ["Bài 1: Histogram Processing", "Bài 2: Filtering Operations"],
        default=["Bài 1: Histogram Processing", "Bài 2: Filtering Operations"]
    )
    
    if not uploaded_files:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Hướng dẫn sử dụng</h3>
            <ol>
                <li>Upload ảnh từ sidebar bên trái</li>
                <li>Chọn chế độ xử lý và thuật toán</li>
                <li>Xem kết quả và download</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo kernels
        st.markdown('<h2 class="sub-header">📊 Demo Convolution Kernels</h2>', unsafe_allow_html=True)
        kernels = create_kernel_visualization()
        
        cols = st.columns(3)
        for i, (name, kernel) in enumerate(kernels.items()):
            with cols[i % 3]:
                st.write(f"**{name}**")
                st.text(str(kernel))
        
        return
    
    # Xử lý ảnh
    if processing_mode == "Single Image Analysis":
        process_single_image(uploaded_files[0], algorithms)
    elif processing_mode == "Batch Processing (10 ảnh)":
        process_batch_images(uploaded_files, algorithms)
    else:
        process_algorithm_comparison(uploaded_files[0])


def process_single_image(uploaded_file, algorithms):
    """Xử lý một ảnh duy nhất"""
    st.markdown('<h2 class="sub-header">🔍 Phân tích ảnh đơn</h2>', unsafe_allow_html=True)
    
    try:
        # Load và convert ảnh
        rgb_image = load_image(uploaded_file)
        gray_image = rgb_to_grayscale(rgb_image)
        
        # Hiển thị ảnh gốc
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_image, caption="Ảnh gốc (RGB)", use_column_width=True)
        with col2:
            st.image(gray_image, caption="Ảnh Grayscale", use_column_width=True, cmap='gray')
        
        # Thống kê ảnh
        stats = calculate_image_stats(gray_image)
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Thông tin ảnh</h4>
            <ul>
                <li><strong>Kích thước:</strong> {stats['shape']}</li>
                <li><strong>Min/Max:</strong> {stats['min']}/{stats['max']}</li>
                <li><strong>Mean ± Std:</strong> {stats['mean']:.2f} ± {stats['std']:.2f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Xử lý Bài 1
        if "Bài 1: Histogram Processing" in algorithms:
            st.markdown('<h3 class="sub-header">📈 Bài 1: Histogram Processing</h3>', unsafe_allow_html=True)
            
            with st.spinner("Đang xử lý histogram..."):
                task1_results = process_task1(gray_image)
            
            # Hiển thị kết quả ảnh
            images = [
                task1_results['original_image'],
                task1_results['h2_image'],
                task1_results['narrowed_image']
            ]
            titles = ["Ảnh gốc", "Sau Equalization", "Thu hẹp [30,80]"]
            display_image_grid(images, titles)
            
            # Hiển thị histograms
            fig_hist = create_histogram_comparison_plotly(
                task1_results['h1'],
                task1_results['h2'],
                task1_results['narrowed_hist']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Phân tích histograms
            st.write("**📊 Phân tích Histograms:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                h1_analysis = analyze_histogram_properties(task1_results['h1'])
                st.json({"H1 (Gốc)": h1_analysis})
            
            with col2:
                h2_analysis = analyze_histogram_properties(task1_results['h2'])
                st.json({"H2 (Equalized)": h2_analysis})
            
            with col3:
                h3_analysis = analyze_histogram_properties(task1_results['narrowed_hist'])
                st.json({"H3 (Narrowed)": h3_analysis})
        
        # Xử lý Bài 2
        if "Bài 2: Filtering Operations" in algorithms:
            st.markdown('<h3 class="sub-header">🔧 Bài 2: Filtering Operations</h3>', unsafe_allow_html=True)
            
            with st.spinner("Đang xử lý filtering..."):
                task2_results = process_task2(gray_image)
            
            # Hiển thị kết quả ảnh
            images = [
                task2_results['original_image'],
                task2_results['i1'],
                task2_results['i2'],
                task2_results['i3'],
                task2_results['i4'],
                task2_results['i5'],
                task2_results['i6']
            ]
            titles = [
                "Ảnh gốc",
                "I1 (3x3, pad=1)",
                "I2 (5x5, pad=2)",
                "I3 (7x7, pad=3, stride=2)",
                "I4 (Median 3x3 trên I3)",
                "I5 (Min 5x5 trên I1)",
                "I6 (Threshold I4 vs I5)"
            ]
            display_image_grid(images, titles, cols=4)
            
            # Hiển thị kernels
            st.write("**🔧 Convolution Kernels:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Kernel 3x3:**")
                st.text(str(task2_results['kernel_3x3']))
            
            with col2:
                st.write("**Kernel 5x5:**")
                st.text(str(task2_results['kernel_5x5']))
            
            with col3:
                st.write("**Kernel 7x7:**")
                st.text(str(task2_results['kernel_7x7']))
            
            # Phân tích filter effects
            st.write("**📊 Phân tích hiệu ứng filters:**")
            
            effects = []
            for key in ['i1', 'i2', 'i3', 'i4', 'i5']:
                effect = analyze_filter_effects(
                    task2_results['original_image'],
                    task2_results[key],
                    key.upper()
                )
                effects.append(effect)
            
            # Tạo bảng so sánh
            import pandas as pd
            df_effects = pd.DataFrame([
                {
                    'Filter': effect['filter_name'],
                    'MSE': f"{effect['mse']:.2f}",
                    'PSNR': f"{effect['psnr']:.2f}",
                    'Correlation': f"{effect['correlation']:.4f}",
                    'Mean Change': f"{effect['filtered_stats']['mean'] - effect['original_stats']['mean']:.2f}"
                }
                for effect in effects
            ])
            st.dataframe(df_effects, use_container_width=True)
        
        # Download results
        st.markdown('<h3 class="sub-header">💾 Download kết quả</h3>', unsafe_allow_html=True)
        
        all_results = {}
        if "Bài 1: Histogram Processing" in algorithms:
            all_results.update({
                'original': gray_image,
                'h2_equalized': task1_results['h2_image'],
                'h3_narrowed': task1_results['narrowed_image']
            })
        
        if "Bài 2: Filtering Operations" in algorithms:
            all_results.update({
                'i1_conv3x3': task2_results['i1'],
                'i2_conv5x5': task2_results['i2'],
                'i3_conv7x7': task2_results['i3'],
                'i4_median': task2_results['i4'],
                'i5_min': task2_results['i5'],
                'i6_threshold': task2_results['i6']
            })
        
        if all_results:
            zip_buffer = create_download_zip(all_results, "single_image_results")
            st.download_button(
                label="📥 Download tất cả ảnh (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"image_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        
    except Exception as e:
        st.error(f"❌ Lỗi xử lý ảnh: {str(e)}")


def process_batch_images(uploaded_files, algorithms):
    """Xử lý batch nhiều ảnh"""
    st.markdown('<h2 class="sub-header">📦 Xử lý Batch (Tối đa 10 ảnh)</h2>', unsafe_allow_html=True)
    
    # Giới hạn 10 ảnh
    files_to_process = uploaded_files[:10]
    
    if len(uploaded_files) > 10:
        st.warning(f"⚠️ Chỉ xử lý 10 ảnh đầu tiên. Bạn đã upload {len(uploaded_files)} ảnh.")
    
    st.info(f"🔄 Đang xử lý {len(files_to_process)} ảnh...")
    
    all_batch_results = {}
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(files_to_process):
        try:
            # Load và convert ảnh
            rgb_image = load_image(uploaded_file)
            gray_image = rgb_to_grayscale(rgb_image)
            
            filename = uploaded_file.name.split('.')[0]
            
            # Xử lý theo thuật toán được chọn
            image_results = {'original_rgb': rgb_image, 'original_gray': gray_image}
            
            if "Bài 1: Histogram Processing" in algorithms:
                task1_results = process_task1(gray_image)
                image_results.update({
                    'h2_equalized': task1_results['h2_image'],
                    'h3_narrowed': task1_results['narrowed_image']
                })
            
            if "Bài 2: Filtering Operations" in algorithms:
                task2_results = process_task2(gray_image)
                image_results.update({
                    'i1_conv3x3': task2_results['i1'],
                    'i2_conv5x5': task2_results['i2'],
                    'i3_conv7x7': task2_results['i3'],
                    'i4_median': task2_results['i4'],
                    'i5_min': task2_results['i5'],
                    'i6_threshold': task2_results['i6']
                })
            
            all_batch_results[filename] = image_results
            
            # Update progress
            progress_bar.progress((i + 1) / len(files_to_process))
            
        except Exception as e:
            st.error(f"❌ Lỗi xử lý ảnh {uploaded_file.name}: {str(e)}")
    
    # Hiển thị kết quả
    st.markdown('<h3 class="sub-header">📊 Kết quả Batch Processing</h3>', unsafe_allow_html=True)
    
    # Tạo tabs cho từng ảnh
    if all_batch_results:
        tabs = st.tabs(list(all_batch_results.keys()))
        
        for tab, (filename, results) in zip(tabs, all_batch_results.items()):
            with tab:
                # Hiển thị ảnh gốc
                col1, col2 = st.columns(2)
                with col1:
                    st.image(results['original_rgb'], caption=f"{filename} - RGB", use_column_width=True)
                with col2:
                    st.image(results['original_gray'], caption=f"{filename} - Grayscale", use_column_width=True)
                
                # Hiển thị kết quả xử lý
                processed_images = []
                processed_titles = []
                
                for key, image in results.items():
                    if key not in ['original_rgb', 'original_gray']:
                        processed_images.append(image)
                        processed_titles.append(key)
                
                if processed_images:
                    display_image_grid(processed_images, processed_titles, cols=3)
        
        # Download batch results
        st.markdown('<h3 class="sub-header">💾 Download Batch Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tạo ZIP cho tất cả kết quả
            batch_zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(batch_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for filename, results in all_batch_results.items():
                    for result_type, image in results.items():
                        if isinstance(image, np.ndarray):
                            # Convert to PIL
                            if len(image.shape) == 2:
                                pil_img = Image.fromarray(image, mode='L')
                            else:
                                pil_img = Image.fromarray(image, mode='RGB')
                            
                            # Save to buffer
                            img_buffer = io.BytesIO()
                            pil_img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            # Add to ZIP
                            zip_file.writestr(f"{filename}_{result_type}.png", img_buffer.getvalue())
            
            batch_zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Images (ZIP)",
                data=batch_zip_buffer.getvalue(),
                file_name=f"batch_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        
        with col2:
            # Tạo PDF Report
            if st.button("📄 Tạo Báo cáo PDF", type="primary"):
                with st.spinner("Đang tạo báo cáo PDF..."):
                    try:
                        # Chuẩn bị dữ liệu cho PDF
                        pdf_data = {}
                        for filename, results in all_batch_results.items():
                            pdf_data[filename] = {}
                            
                            # Task 1 results
                            if "Bài 1: Histogram Processing" in algorithms:
                                gray_img = results['original_gray']
                                task1_results = process_task1(gray_img)
                                pdf_data[filename]['task1'] = task1_results
                            
                            # Task 2 results  
                            if "Bài 2: Filtering Operations" in algorithms:
                                gray_img = results['original_gray']
                                task2_results = process_task2(gray_img)
                                pdf_data[filename]['task2'] = task2_results
                        
                        # Tạo PDF
                        team_info = create_sample_team_info()
                        pdf_path = generate_pdf_report(pdf_data, team_info)
                        
                        # Download PDF
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_file.read(),
                                file_name=f"bao_cao_do_an_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        
                        st.success("✅ Báo cáo PDF đã được tạo thành công!")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi tạo PDF: {str(e)}")
        
        # Thống kê tổng quan
        st.markdown('<h3 class="sub-header">📈 Thống kê tổng quan</h3>', unsafe_allow_html=True)
        
        total_images = len(all_batch_results)
        total_processed = sum(len(results) - 2 for results in all_batch_results.values())  # -2 for original images
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng số ảnh gốc", total_images)
        with col2:
            st.metric("Tổng số ảnh đã xử lý", total_processed)
        with col3:
            st.metric("Thuật toán đã áp dụng", len(algorithms))


def process_algorithm_comparison(uploaded_file):
    """So sánh các thuật toán khác nhau"""
    st.markdown('<h2 class="sub-header">⚖️ So sánh thuật toán</h2>', unsafe_allow_html=True)
    
    try:
        rgb_image = load_image(uploaded_file)
        gray_image = rgb_to_grayscale(rgb_image)
        
        st.image(gray_image, caption="Ảnh gốc", use_column_width=True)
        
        # So sánh các phương pháp filtering
        st.markdown('<h3 class="sub-header">🔧 So sánh Filtering Methods</h3>', unsafe_allow_html=True)
        
        with st.spinner("Đang so sánh các phương pháp filtering..."):
            comparison_results = compare_filtering_methods(gray_image)
        
        # Hiển thị kết quả so sánh
        filter_names = list(comparison_results.keys())
        filter_images = list(comparison_results.values())
        
        display_image_grid(filter_images, filter_names, cols=4)
        
        # Phân tích định lượng
        st.markdown('<h3 class="sub-header">📊 Phân tích định lượng</h3>', unsafe_allow_html=True)
        
        analysis_results = []
        for name, filtered_img in comparison_results.items():
            if name != 'original':
                analysis = analyze_filter_effects(gray_image, filtered_img, name)
                analysis_results.append(analysis)
        
        # Tạo DataFrame để hiển thị
        import pandas as pd
        df_analysis = pd.DataFrame([
            {
                'Filter': result['filter_name'],
                'MSE': f"{result['mse']:.2f}",
                'PSNR': f"{result['psnr']:.2f}",
                'Correlation': f"{result['correlation']:.4f}",
                'Mean': f"{result['filtered_stats']['mean']:.2f}",
                'Std': f"{result['filtered_stats']['std']:.2f}"
            }
            for result in analysis_results
        ])
        
        st.dataframe(df_analysis, use_container_width=True)
        
        # Biểu đồ so sánh
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[r['filter_name'] for r in analysis_results],
            y=[r['psnr'] for r in analysis_results],
            mode='markers+lines',
            name='PSNR',
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="So sánh PSNR của các phương pháp filtering",
            xaxis_title="Phương pháp",
            yaxis_title="PSNR (dB)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Lỗi so sánh thuật toán: {str(e)}")


if __name__ == "__main__":
    main()
