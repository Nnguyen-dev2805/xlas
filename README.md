# 🖼️ Đồ án Xử lý Ảnh Số - Digital Image Processing Project

## 📋 Tổng quan

Đây là project đồ án cuối kỳ môn **Xử lý Ảnh Số**, implement các thuật toán cơ bản về histogram processing và image filtering với giao diện web đẹp mắt sử dụng Streamlit.

### 🎯 Yêu cầu đồ án

**Bài 1: Histogram Processing**
- Chuyển ảnh màu sang grayscale
- Tính và vẽ histogram gốc (H1)
- Histogram equalization (H2)
- Thu hẹp histogram về khoảng [30, 80]

**Bài 2: Image Filtering**
- Convolution với kernel 3×3, padding=1 (I1)
- Convolution với kernel 5×5, padding=2 (I2)  
- Convolution với kernel 7×7, padding=3, stride=2 (I3)
- Median filter 3×3 trên I3 (I4)
- Min filter 5×5 trên I1 (I5)
- Thresholding I4 vs I5 (I6)

**Bài 3: Batch Processing**
- Áp dụng cho 10 ảnh
- Tạo báo cáo PDF

---

## 🏗️ Cấu trúc Project

```
XLAS/
├── 📁 src/                          # Source code chính
│   ├── 📄 __init__.py              # Package initialization
│   ├── 📄 utils.py                 # Hàm tiện ích (load, save, convert ảnh)
│   ├── 📄 histogram.py             # Bài 1: Histogram processing
│   └── 📄 filtering.py             # Bài 2: Image filtering
│
├── 📁 docs/                         # Documentation chi tiết
│   ├── 📄 utils_documentation.md
│   ├── 📄 histogram_documentation.md
│   └── 📄 filtering_documentation.md
│
├── 📁 data/                         # Thư mục chứa ảnh test (tự tạo)
│   ├── 📁 input/                   # Ảnh đầu vào
│   └── 📁 output/                  # Kết quả xử lý
│
├── 📄 app.py                       # Streamlit GUI chính
├── 📄 requirements.txt             # Dependencies
└── 📄 README.md                    # File này
```

---

## 🚀 Cách chạy Project

### 1. Cài đặt Dependencies

```bash
# Clone hoặc download project về máy
cd XLAS

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chạy ứng dụng Streamlit

```bash
# Chạy từ thư mục gốc của project
streamlit run app.py
```

### 3. Mở trình duyệt

Ứng dụng sẽ tự động mở tại: `http://localhost:8501`

---

## 🎮 Hướng dẫn sử dụng GUI

### 📤 Upload ảnh
1. Sử dụng sidebar bên trái
2. Click "Browse files" để chọn ảnh
3. Hỗ trợ: PNG, JPG, JPEG, BMP
4. Có thể upload nhiều ảnh cùng lúc

### 🔧 Chọn chế độ xử lý

#### **Single Image Analysis**
- Phân tích chi tiết 1 ảnh
- Hiển thị từng bước xử lý
- Thống kê và metrics đầy đủ

#### **Batch Processing (10 ảnh)**
- Xử lý hàng loạt tối đa 10 ảnh
- Kết quả theo tabs
- Download ZIP tất cả kết quả

#### **Algorithm Comparison**
- So sánh các phương pháp filtering
- Phân tích định lượng (MSE, PSNR)
- Biểu đồ interactive

### 🧮 Chọn thuật toán
- ✅ **Bài 1: Histogram Processing**
- ✅ **Bài 2: Filtering Operations**
- Có thể chọn 1 hoặc cả 2

### 📊 Xem kết quả
- **Ảnh:** Grid layout đẹp mắt
- **Histograms:** Interactive plots với Plotly
- **Thống kê:** JSON format dễ đọc
- **Kernels:** Hiển thị ma trận kernel

### 💾 Download kết quả
- **Single file:** Download từng ảnh
- **ZIP package:** Tất cả kết quả trong 1 file
- **Tên file:** Có timestamp tự động

---

## 🔧 Chi tiết Implementation

### 📈 Bài 1: Histogram Processing

#### **Histogram Calculation**
```python
def calculate_histogram(image):
    histogram = np.zeros(256)
    for pixel_value in image.flatten():
        histogram[pixel_value] += 1
    return histogram
```

#### **Histogram Equalization**
```python
# 1. Tính CDF
cdf = histogram.cumsum()

# 2. Normalize CDF  
lut[i] = (cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255

# 3. Apply transformation
equalized_image = lut[image]
```

#### **Histogram Narrowing**
```python
# Linear mapping về [30, 80]
new_value = (old - old_min) / (old_max - old_min) * 50 + 30
```

### 🔧 Bài 2: Image Filtering

#### **Convolution 2D (từ scratch)**
```python
def convolution_2d(image, kernel, padding=0, stride=1):
    # Add padding
    padded = pad_image(image, padding)
    
    # Calculate output size
    out_h = (img_h - kernel_h) // stride + 1
    out_w = (img_w - kernel_w) // stride + 1
    
    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            roi = padded[i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            output[i,j] = np.sum(roi * kernel)
```

#### **Median Filter**
```python
def apply_median_filter(image, kernel_size=3):
    return median_filter(image, size=kernel_size)
```

#### **Min Filter**  
```python
def apply_min_filter(image, kernel_size=5):
    return minimum_filter(image, size=kernel_size)
```

#### **Thresholding**
```python
def threshold_operation(image1, image2):
    return np.where(image1 > image2, 0, image2)
```

---

## 📊 Features nổi bật

### 🎨 Giao diện đẹp mắt
- **Modern UI:** Streamlit với custom CSS
- **Responsive:** Tự động điều chỉnh theo màn hình
- **Interactive:** Zoom, pan trên biểu đồ
- **Professional:** Color scheme và typography

### ⚡ Performance tối ưu
- **Vectorized operations:** Sử dụng NumPy hiệu quả
- **Progress bars:** Theo dõi tiến trình xử lý
- **Memory efficient:** Xử lý ảnh lớn không bị crash
- **Caching:** Streamlit cache để tăng tốc

### 🔍 Phân tích chi tiết
- **Quantitative metrics:** MSE, PSNR, Correlation
- **Statistical analysis:** Mean, std, entropy
- **Visual comparison:** Side-by-side plots
- **Interactive exploration:** Hover information

### 📱 User Experience
- **Drag & drop:** Upload ảnh dễ dàng
- **Real-time preview:** Xem kết quả ngay lập tức
- **Error handling:** Thông báo lỗi rõ ràng
- **Help tooltips:** Hướng dẫn sử dụng

---

## 🧪 Testing & Validation

### ✅ Test Cases
1. **Ảnh grayscale:** Kiểm tra xử lý ảnh xám
2. **Ảnh màu:** Convert RGB → Grayscale
3. **Ảnh nhỏ:** < 100×100 pixels
4. **Ảnh lớn:** > 2000×2000 pixels
5. **Edge cases:** Ảnh toàn đen, toàn trắng

### 🔬 Validation Methods
- **OpenCV comparison:** So sánh với cv2.filter2D()
- **Mathematical verification:** Kiểm tra công thức
- **Visual inspection:** Đánh giá chất lượng ảnh
- **Performance benchmarks:** Đo thời gian xử lý

---

## 📚 Dependencies

```txt
numpy==1.24.3          # Numerical computing
opencv-python==4.8.1.78 # Computer vision
matplotlib==3.8.0       # Plotting (backup)
Pillow==10.1.0          # Image I/O
streamlit==1.28.0       # Web GUI
reportlab==4.0.7        # PDF generation
scipy==1.11.3           # Scientific computing
plotly==5.17.0          # Interactive plots
```

---

## 🎓 Kiến thức áp dụng

### 📖 Lý thuyết
- **Digital Image Processing:** Gonzalez & Woods
- **Computer Vision:** Szeliski
- **Linear Algebra:** Matrix operations
- **Statistics:** Histogram analysis

### 💻 Kỹ thuật lập trình
- **NumPy:** Vectorized operations
- **Object-oriented design:** Modular code
- **Error handling:** Robust implementation
- **Documentation:** Comprehensive docs

### 🎨 UI/UX Design
- **Streamlit:** Modern web apps
- **CSS customization:** Beautiful styling
- **Information architecture:** Logical flow
- **User feedback:** Progress indicators

---

## 🚨 Troubleshooting

### ❌ Lỗi thường gặp

#### **Import Error**
```bash
ModuleNotFoundError: No module named 'streamlit'
```
**Giải pháp:** `pip install -r requirements.txt`

#### **Memory Error**
```bash
MemoryError: Unable to allocate array
```
**Giải pháp:** Resize ảnh nhỏ hơn hoặc tăng RAM

#### **File Not Found**
```bash
FileNotFoundError: [Errno 2] No such file or directory
```
**Giải pháp:** Kiểm tra đường dẫn file và thư mục

### 🔧 Performance Issues

#### **Chậm khi xử lý ảnh lớn**
- Resize ảnh về kích thước nhỏ hơn
- Sử dụng stride lớn hơn cho convolution
- Giảm số lượng ảnh trong batch processing

#### **GUI không responsive**
- Đóng các tabs không cần thiết
- Refresh browser
- Restart Streamlit server

---

## 🤝 Đóng góp và Phát triển

### 🔮 Tính năng có thể mở rộng
- [ ] **More filters:** Gaussian, Laplacian, Sobel
- [ ] **Color processing:** HSV, LAB color spaces  
- [ ] **Advanced algorithms:** Bilateral filter, Non-local means
- [ ] **Machine learning:** CNN-based enhancement
- [ ] **Video processing:** Frame-by-frame analysis
- [ ] **Cloud deployment:** Heroku, AWS, GCP

### 🎯 Cải thiện hiệu suất
- [ ] **GPU acceleration:** CUDA, OpenCL
- [ ] **Parallel processing:** Multiprocessing
- [ ] **Optimized algorithms:** FFT convolution
- [ ] **Memory mapping:** Large file handling

---

## 📞 Liên hệ và Hỗ trợ

### 👥 Team Information
- **Môn học:** Xử lý Ảnh Số
- **Học kỳ:** [Điền thông tin]
- **Giảng viên:** [Điền tên thầy/cô]

### 🆘 Hỗ trợ kỹ thuật
- **Issues:** Tạo issue trên GitHub
- **Documentation:** Đọc files trong thư mục `docs/`
- **Email:** [Điền email liên hệ]

---

## 📜 License

Đây là project học tập, sử dụng cho mục đích giáo dục. Tham khảo và sử dụng code với trích dẫn nguồn phù hợp.

---

## 🎉 Kết luận

Project này demonstrate việc implementation từ scratch các thuật toán cơ bản trong xử lý ảnh số, kết hợp với giao diện web hiện đại để tạo ra một công cụ học tập và demo hiệu quả.

**Chúc bạn thành công với đồ án! 🚀**
