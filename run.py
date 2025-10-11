"""
Run Main Application
====================

Script để chạy ứng dụng chính với tab system
Tích hợp cả Bài 1 và Bài 2 trong một giao diện

Usage:
    streamlit run run_main.py

Tác giả: Nhóm xử lý ảnh số
"""

import subprocess
import sys
import os

def main():
    """Chạy main application"""
    # Đảm bảo đang ở đúng thư mục
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Chạy streamlit với main_app.py
    main_app_path = os.path.join(script_dir, "app.py")
    
    if not os.path.exists(main_app_path):
        print("Không tìm thấy app.py!")
        print("Đảm bảo file app.py tồn tại trong thư mục hiện tại.")
        return
    
    print("Đang khởi động ứng dụng xử lý ảnh số...")
    print("Bài 1: Histogram Processing")
    print("Bài 2: Image Filtering & Convolution")
    print("Mở trình duyệt để xem giao diện...")
    
    # Chạy streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            main_app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nĐã dừng ứng dụng!")
    except Exception as e:
        print(f"Lỗi khi chạy ứng dụng: {e}")

if __name__ == "__main__":
    main()
