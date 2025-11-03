<p align="center">
# XLAS - Ứng Dụng Xử Lý Ảnh Số

  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.0+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-4.0+-green.svg" alt="OpenCV">
</p>

Ứng dụng web tương tác cho xử lý ảnh số, được xây dựng với Streamlit và các thư viện xử lý ảnh mạnh mẽ như OpenCV, NumPy và SciPy.

## Mục Lục

- [Tác Giả](#tác-giả)
- [Tổng Quan](#tổng-quan)
- [Tính Năng](#tính-năng)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Công Nghệ](#công-nghệ)
- [Ghi Chú](#ghi-chú)
- [Đóng Góp](#đóng-góp)
- [Liên Hệ](#liên-hệ)

## Tác Giả

Dự án được phát triển bởi nhóm sinh viên:

- **Trương Nhật Nguyên** - 23110273
- **Nguyễn Hoàng Hà** - 23110207
- **Nghiêm Quang Huy** - 23110222

## Tổng Quan

XLAS (Xử Lý Ảnh Số) là một ứng dụng web tương tác giúp người dùng thực hiện các kỹ thuật xử lý ảnh số phổ biến một cách trực quan và dễ dàng. Ứng dụng được thiết kế với giao diện thân thiện, phù hợp cho cả mục đích học tập và nghiên cứu.

## Tính Năng

### Bài 1: Histogram Processing
- **Phân tích Histogram**: Trực quan hóa phân bố độ sáng của ảnh
- **Histogram Equalization**: Cân bằng histogram để cải thiện độ tương phản
- **Histogram Matching**: Khớp histogram giữa hai ảnh
- **Thống kê ảnh**: Phân tích chi tiết các thông số thống kê

### Bài 2: Image Filtering & Convolution
- **Bộ lọc làm mịn**:
  - Mean Filter (Trung bình)
  - Median Filter (Trung vị)
  - Gaussian Filter (Gauss)
  
- **Bộ lọc phát hiện cạnh**:
  - Sobel Filter (Sobel)
  - Laplacian Filter (Laplace)
  
- **Bộ lọc làm sắc nét**:
  - Sharpen Filter
  
- **Tạo nhiễu**:
  - Gaussian Noise
  - Salt & Pepper Noise
  - Speckle Noise

### Bài 3: Pipeline Xử Lý Ảnh Linh Hoạt
- **Xây dựng pipeline tùy chỉnh**: Kết hợp nhiều bộ lọc trong một quy trình
- **Xử lý batch**: Áp dụng pipeline cho nhiều ảnh cùng lúc
- **So sánh kết quả**: Trực quan hóa kết quả trước và sau xử lý
- **Xuất kết quả**: Lưu ảnh đã xử lý

## Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8 trở lên
- pip hoặc conda

### Bước 1: Clone Repository

```bash
git clone https://github.com/Nnguyen-dev2805/xlas.git
cd xlas
```

### Bước 2: Tạo Virtual Environment

#### Sử dụng venv (pip)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

#### Sử dụng conda
```bash
conda create -n xlas python=3.8
conda activate xlas
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

## Sử Dụng

### Chạy Ứng Dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: `http://localhost:8501`

### Chạy từng Module Riêng Lẻ

```bash
# Bài 1: Histogram Processing
streamlit run app_01.py

# Bài 2: Image Filtering & Convolution
streamlit run app_02.py

# Bài 3: Pipeline Xử Lý Ảnh
streamlit run app_03.py
```

### Hướng Dẫn Sử Dụng Chi Tiết

1. **Tải ảnh lên**: Sử dụng nút "Browse files" để tải ảnh từ máy tính
2. **Chọn bài tập**: Chọn bài tập muốn thực hiện từ menu
3. **Điều chỉnh tham số**: Thay đổi các tham số của bộ lọc theo nhu cầu
4. **Xem kết quả**: Kết quả sẽ được hiển thị ngay lập tức
5. **Tải xuống**: Lưu ảnh đã xử lý về máy

## Cấu Trúc Dự Án

```
xlas/
│
├── app.py                  # Ứng dụng chính (điều phối 3 bài)
├── app_01.py              # Bài 1: Histogram Processing
├── app_02.py              # Bài 2: Image Filtering & Convolution
├── app_03.py              # Bài 3: Pipeline Xử Lý Ảnh
├── run.py                 # Script khởi chạy nhanh
│
├── core/                  # Module xử lý ảnh cốt lõi
│   ├── __init__.py
│   ├── convolution.py     # Thuật toán convolution
│   ├── histogram.py       # Xử lý histogram
│   └── image_ops.py       # Các phép toán ảnh cơ bản
│
├── filters/               # Bộ lọc ảnh
│   ├── gaussian_kernel.py # Bộ lọc Gaussian
│   ├── laplacian_kernel.py# Bộ lọc Laplacian
│   ├── mean_kernel.py     # Bộ lọc Mean
│   ├── median_kernel.py   # Bộ lọc Median
│   ├── sharpen_kernel.py  # Bộ lọc Sharpen
│   ├── sobel_kernel.py    # Bộ lọc Sobel
│   └── noise_generator.py # Tạo nhiễu
│
├── data/                  # Thư mục lưu trữ ảnh mẫu
├── requirements.txt       # Danh sách dependencies
├── .gitignore            # Git ignore file
└── README.md             # File này
```

## Công Nghệ

### Thư Viện Chính

- **[Streamlit](https://streamlit.io/)**: Framework web app tương tác
- **[OpenCV](https://opencv.org/)**: Xử lý ảnh và computer vision
- **[NumPy](https://numpy.org/)**: Tính toán số học và mảng
- **[Matplotlib](https://matplotlib.org/)**: Trực quan hóa dữ liệu
- **[Pillow](https://python-pillow.org/)**: Xử lý ảnh Python
- **[SciPy](https://scipy.org/)**: Tính toán khoa học
- **[Pandas](https://pandas.pydata.org/)**: Phân tích dữ liệu
- **[Plotly](https://plotly.com/)**: Biểu đồ tương tác
- **[Seaborn](https://seaborn.pydata.org/)**: Trực quan hóa thống kê
- **[Albumentations](https://albumentations.ai/)**: Augmentation ảnh

### Các Thuật Toán Được Sử Dụng

- Convolution 2D
- Histogram Equalization
- Histogram Matching
- Gaussian Blur
- Median Filtering
- Sobel Edge Detection
- Laplacian Edge Detection
- Image Sharpening
- Noise Generation (Gaussian, Salt & Pepper, Speckle)

## Ghi Chú

### Lưu Ý Khi Sử Dụng

- Ảnh đầu vào nên có định dạng: JPG, PNG, JPEG, BMP
- Kích thước ảnh quá lớn có thể làm chậm quá trình xử lý
- Một số bộ lọc có thể tốn nhiều tài nguyên với kernel size lớn
- Khuyến nghị sử dụng ảnh có kích thước <= 2000x2000 pixels

### Xử Lý Lỗi Thường Gặp

**Lỗi: Module not found**
```bash
pip install -r requirements.txt
```

**Lỗi: Port đã được sử dụng**
```bash
streamlit run app.py --server.port 8502
```

**Lỗi: Memory error với ảnh lớn**
- Giảm kích thước ảnh trước khi xử lý
- Giảm kernel size của bộ lọc

## Đóng Góp

Mọi đóng góp đều được hoan nghênh! Nếu bạn muốn đóng góp:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

### Hướng Dẫn Phát Triển

- Tuân thủ PEP 8 style guide
- Viết docstring cho functions và classes
- Thêm unit tests cho code mới
- Cập nhật README nếu thêm tính năng mới

## Liên Hệ

- **Email**: tnhatnguyen.dev2805@gmail.com
- **GitHub**: [@Nnguyen-dev2805](https://github.com/Nnguyen-dev2805)

---

**Lưu ý**: Dự án này được phát triển cho mục đích học tập và nghiên cứu.
