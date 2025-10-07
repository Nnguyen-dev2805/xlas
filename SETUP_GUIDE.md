# 🚀 Hướng dẫn Setup và Chạy Project

## 📋 Yêu cầu hệ thống

### 💻 Phần mềm cần thiết
- **Python 3.8+** (khuyến nghị Python 3.9 hoặc 3.10)
- **pip** (Python package manager)
- **Git** (tùy chọn, để clone project)

### 🖥️ Hệ điều hành hỗ trợ
- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Ubuntu 18.04+
- ✅ Linux distributions khác

### 💾 Dung lượng
- **Disk space:** ~500MB (bao gồm dependencies)
- **RAM:** Tối thiểu 4GB (khuyến nghị 8GB+)

---

## 🔧 Cách 1: Setup Tự động (Khuyến nghị)

### Windows
```bash
# 1. Mở Command Prompt hoặc PowerShell
# 2. Navigate đến thư mục project
cd path/to/XLAS

# 3. Chạy script tự động
run.bat
```

### macOS/Linux
```bash
# 1. Mở Terminal
# 2. Navigate đến thư mục project
cd path/to/XLAS

# 3. Cấp quyền thực thi
chmod +x run.sh

# 4. Chạy script
./run.sh
```

---

## ⚙️ Cách 2: Setup Thủ công

### Bước 1: Kiểm tra Python
```bash
# Kiểm tra version Python
python --version
# hoặc
python3 --version

# Kết quả mong đợi: Python 3.8.x hoặc cao hơn
```

### Bước 2: Tạo Virtual Environment (Khuyến nghị)
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Sau khi kích hoạt, prompt sẽ hiện (venv)
```

### Bước 3: Cài đặt Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Cài đặt packages
pip install -r requirements.txt

# Kiểm tra cài đặt thành công
pip list
```

### Bước 4: Chạy ứng dụng
```bash
# Chạy Streamlit
streamlit run app.py

# Hoặc với Python 3
python -m streamlit run app.py
```

### Bước 5: Mở trình duyệt
- Tự động mở: `http://localhost:8501`
- Thủ công: Copy URL từ terminal

---

## 🐛 Troubleshooting

### ❌ Lỗi Python không tìm thấy

**Windows:**
```bash
# Cài đặt Python từ Microsoft Store
# Hoặc download từ python.org
# Đảm bảo check "Add Python to PATH"
```

**macOS:**
```bash
# Cài đặt qua Homebrew
brew install python3

# Hoặc download từ python.org
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# CentOS/RHEL
sudo yum install python3 python3-pip
```

### ❌ Lỗi pip không tìm thấy
```bash
# Windows
python -m ensurepip --upgrade

# macOS/Linux
python3 -m ensurepip --upgrade

# Hoặc cài đặt thủ công
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### ❌ Lỗi cài đặt package

**Permission denied:**
```bash
# Thêm --user flag
pip install --user -r requirements.txt

# Hoặc dùng sudo (Linux/macOS)
sudo pip3 install -r requirements.txt
```

**Network error:**
```bash
# Dùng mirror khác
pip install -r requirements.txt -i https://pypi.org/simple/

# Hoặc upgrade certificates
pip install --upgrade certifi
```

### ❌ Lỗi import module
```bash
# Kiểm tra virtual environment đã activate chưa
# Kiểm tra Python path
python -c "import sys; print(sys.path)"

# Reinstall packages
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### ❌ Streamlit không chạy
```bash
# Kiểm tra port 8501 có bị chiếm không
netstat -an | grep 8501

# Chạy trên port khác
streamlit run app.py --server.port 8502

# Clear cache
streamlit cache clear
```

### ❌ Lỗi memory khi xử lý ảnh lớn
- Resize ảnh nhỏ hơn (< 1000x1000 pixels)
- Đóng các ứng dụng khác
- Tăng RAM hoặc dùng máy mạnh hơn

---

## 🔍 Kiểm tra Installation

### Test cơ bản
```python
# Tạo file test.py
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

print("✅ Tất cả packages đã được cài đặt thành công!")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Streamlit version: {st.__version__}")
```

```bash
# Chạy test
python test.py
```

### Test Streamlit
```bash
# Chạy hello world
streamlit hello

# Nếu mở được browser → Setup thành công
```

---

## 📁 Cấu trúc thư mục sau setup

```
XLAS/
├── 📁 src/                    ✅ Source code
├── 📁 docs/                   ✅ Documentation  
├── 📁 data/
│   ├── 📁 input/             📤 Đặt ảnh test vào đây
│   └── 📁 output/            📥 Kết quả sẽ lưu ở đây
├── 📁 venv/                   🐍 Virtual environment (nếu tạo)
├── 📄 app.py                  🚀 Main application
├── 📄 requirements.txt        📦 Dependencies
├── 📄 run.bat                 🏃 Windows script
├── 📄 run.sh                  🏃 Unix script
└── 📄 README.md               📖 Documentation
```

---

## 🎯 Bước tiếp theo sau setup

### 1. Chuẩn bị ảnh test
- Đặt 5-10 ảnh vào `data/input/`
- Format: PNG, JPG, JPEG, BMP
- Kích thước khuyến nghị: 200x200 đến 1000x1000 pixels

### 2. Chạy ứng dụng
```bash
streamlit run app.py
```

### 3. Test các chức năng
- ✅ Upload ảnh
- ✅ Chọn Single Image Analysis
- ✅ Chọn cả 2 thuật toán
- ✅ Xem kết quả
- ✅ Download ZIP

### 4. Test Batch Processing
- ✅ Upload 5-10 ảnh
- ✅ Chọn Batch Processing
- ✅ Xem kết quả theo tabs
- ✅ Download batch results

---

## 🆘 Hỗ trợ thêm

### 📞 Khi gặp vấn đề
1. **Đọc error message** cẩn thận
2. **Google error message** cụ thể
3. **Kiểm tra version** Python và packages
4. **Thử virtual environment** mới
5. **Restart** terminal/computer

### 📚 Tài liệu tham khảo
- [Python Installation Guide](https://docs.python.org/3/using/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NumPy Installation](https://numpy.org/install/)
- [OpenCV Installation](https://opencv.org/releases/)

### 🔗 Links hữu ích
- **Python Download:** https://python.org/downloads/
- **Streamlit Cloud:** https://streamlit.io/cloud
- **Stack Overflow:** https://stackoverflow.com/questions/tagged/streamlit

---

## ✅ Checklist hoàn thành

- [ ] Python 3.8+ đã cài đặt
- [ ] pip hoạt động bình thường
- [ ] Virtual environment đã tạo (khuyến nghị)
- [ ] Dependencies đã cài đặt thành công
- [ ] Streamlit chạy được
- [ ] Browser mở được localhost:8501
- [ ] Upload ảnh thành công
- [ ] Các thuật toán chạy không lỗi
- [ ] Download kết quả thành công

**🎉 Chúc mừng! Bạn đã setup thành công project!**
