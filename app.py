"""
Main Application - Xử lý Ảnh Số
===============================

Giao diện chính tích hợp cả Bài 1 và Bài 2 với tab system
- Tab 1: Histogram Processing (Bài 1)
- Tab 2: Image Filtering & Convolution (Bài 2)

Tác giả: Nhóm xử lý ảnh số
"""

import streamlit as st
import sys
import os

# Thêm đường dẫn để import các modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các modules cho từng bài
from app_01 import main as bai1_main
from app_02 import main as bai2_main

def setup_main_page():
    """Cấu hình trang chính"""
    st.set_page_config(
        page_title="Xử lý Ảnh Số - XLAS",
        # page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Khởi tạo session state
    if 'session_initialized' not in st.session_state:
        st.session_state.session_initialized = True
        # Clear any old cached data
        if 'test_images' in st.session_state:
            del st.session_state.test_images
        if 'test_image_names' in st.session_state:
            del st.session_state.test_image_names
        if 'batch_results' in st.session_state:
            del st.session_state.batch_results
    
    # CSS chung cho toàn bộ app
    st.markdown("""
    <style>
    .main-app-title {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3498db, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .tab-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .info-banner {
        background: linear-gradient(90deg, #e3f2fd, #f3e5f5);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Custom radio button styling to look like tabs */
    .stRadio > div {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stRadio > div > label {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 250px;
        text-align: center;
    }
    
    .stRadio > div > label:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background-color: #ffffff;
        border-color: #3498db;
        color: #2c3e50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def render_main_header():
    """Hiển thị header chính của app"""
    st.markdown('<h1 class="main-app-title">Xử lý Ảnh Số - XLAS</h1>', 
                unsafe_allow_html=True)
    
    # Nút reset session nếu có lỗi
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Reset Session (nếu có lỗi)", help="Nhấn nếu gặp lỗi upload hoặc hiển thị"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Đã reset session! Trang sẽ tự động reload.")
            st.rerun()
    
    st.markdown("""
    <div class="info-banner">
    <h3>Chào mừng đến với hệ thống xử lý ảnh số!</h3>
    <p>Chọn tab bên dưới để thực hiện các bài tập:</p>
    <ul>
    <li><strong>Bài 1:</strong> Histogram Processing - Cân bằng và thu hẹp histogram</li>
    <li><strong>Bài 2:</strong> Image Filtering & Convolution - Lọc ảnh và phép tích chập</li>
    </ul>
    <p><small>💡 <strong>Lưu ý:</strong> Nếu gặp lỗi upload ảnh, hãy nhấn nút "Reset Session" ở trên.</small></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Hàm main chính"""
    # Cấu hình trang
    setup_main_page()
    
    # Header chính
    render_main_header()
    
    # Tạo tab selector ở main content thay vì sidebar
    st.markdown("### Chọn bài tập:")
    selected_tab = st.radio(
        "Chọn bài tập để thực hiện:",
        ["Bài 1: Histogram Processing", "Bài 2: Image Filtering & Convolution"],
        horizontal=True,
        key="tab_selector"
    )
    
    st.markdown("---")
    
    if selected_tab == "Bài 1: Histogram Processing":
        st.session_state.current_tab = 'bai1'
        
        st.markdown('<div class="tab-header">Bài 1: Xử lý Histogram</div>', 
                    unsafe_allow_html=True)
        
        # Gọi main function của bài 1
        try:
            bai1_main()
        except Exception as e:
            st.error(f"Lỗi trong Bài 1: {e}")
            st.info("Đảm bảo file app_01.py tồn tại và có function main()")
    
    else:  # Bài 2
        st.session_state.current_tab = 'bai2'
        
        st.markdown('<div class="tab-header">Bài 2: Image Filtering & Convolution</div>', 
                    unsafe_allow_html=True)
        
        # Gọi main function của bài 2
        try:
            bai2_main()
        except Exception as e:
            st.error(f"Lỗi trong Bài 2: {e}")
            st.info("Đảm bảo file app_02.py tồn tại và có function main()")

if __name__ == "__main__":
    main()
