# Utils Module Documentation

## 📋 Tổng quan

Module `utils.py` chứa các hàm tiện ích cơ bản cho việc xử lý ảnh, bao gồm:
- Load và save ảnh
- Chuyển đổi RGB sang Grayscale
- Padding và resize ảnh
- Tạo convolution kernels
- Normalize ảnh

## 🔧 Các hàm chính

### 1. `load_image(image_path)`

**Chức năng:** Load ảnh từ file path hoặc uploaded file (Streamlit)

**Input:**
- `image_path`: Đường dẫn file hoặc file object từ Streamlit

**Output:**
- `numpy array`: Ảnh RGB với shape (H, W, 3)

**Cách hoạt động:**
1. Mở ảnh bằng PIL.Image
2. Convert về numpy array
3. Đảm bảo format RGB (xử lý RGBA, Grayscale)

**Ví dụ sử dụng:**
```python
from src.utils import load_image

# Load từ file path
image = load_image("path/to/image.jpg")

# Load từ Streamlit uploaded file
uploaded_file = st.file_uploader("Upload image")
image = load_image(uploaded_file)
```

---

### 2. `rgb_to_grayscale(image)`

**Chức năng:** Chuyển ảnh RGB sang Grayscale theo chuẩn ITU-R BT.601

**Input:**
- `image`: Ảnh RGB với shape (H, W, 3)

**Output:**
- `numpy array`: Ảnh grayscale với shape (H, W)

**Công thức:**
```
Gray = 0.299 × R + 0.587 × G + 0.114 × B
```

**Lý do sử dụng công thức này:**
- Mắt người nhạy cảm nhất với màu xanh lá (Green) - hệ số 0.587
- Ít nhạy cảm với màu xanh dương (Blue) - hệ số 0.114
- Màu đỏ (Red) có độ nhạy trung bình - hệ số 0.299

**Ví dụ sử dụng:**
```python
rgb_image = load_image("color_image.jpg")
gray_image = rgb_to_grayscale(rgb_image)
```

---

### 3. `create_kernel(size, kernel_type)`

**Chức năng:** Tạo các loại convolution kernel

**Input:**
- `size`: Kích thước kernel (phải là số lẻ)
- `kernel_type`: Loại kernel ('average', 'gaussian', 'sharpen', 'edge')

**Output:**
- `numpy array`: Kernel matrix

**Các loại kernel:**

#### Average Kernel
- **Mục đích:** Làm mờ ảnh (blur)
- **Công thức:** Tất cả phần tử = 1/(size×size)
- **Ví dụ 3x3:**
```
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]
```

#### Gaussian Kernel
- **Mục đích:** Làm mờ tự nhiên hơn average
- **Công thức:** Phân phối Gaussian 2D
- **Đặc điểm:** Trọng số cao ở trung tâm, giảm dần ra ngoài

#### Sharpen Kernel
- **Mục đích:** Làm sắc nét ảnh
- **Ví dụ 3x3:**
```
[ 0  -1   0]
[-1   5  -1]
[ 0  -1   0]
```

#### Edge Detection Kernel
- **Mục đích:** Phát hiện cạnh
- **Ví dụ 3x3:**
```
[-1  -1  -1]
[-1   8  -1]
[-1  -1  -1]
```

---

### 4. `pad_image(image, padding, pad_value=0)`

**Chức năng:** Thêm padding vào ảnh

**Input:**
- `image`: Ảnh input
- `padding`: Số pixel padding
- `pad_value`: Giá trị fill (mặc định=0)

**Output:**
- `numpy array`: Ảnh đã được pad

**Tại sao cần padding:**
- Giữ nguyên kích thước ảnh sau convolution
- Xử lý pixels ở biên ảnh
- Công thức: `output_size = input_size + 2×padding - kernel_size + 1`

---

### 5. `resize_to_match(image1, image2)`

**Chức năng:** Resize image1 để match kích thước image2

**Input:**
- `image1`: Ảnh cần resize
- `image2`: Ảnh reference

**Output:**
- `numpy array`: image1 đã được resize

**Cách hoạt động:**
1. So sánh kích thước hai ảnh
2. Nếu image1 nhỏ hơn → padding
3. Nếu image1 lớn hơn → cropping

---

## 📊 Hàm phân tích

### `calculate_image_stats(image)`

**Chức năng:** Tính các thống kê cơ bản của ảnh

**Output:**
```python
{
    'shape': (height, width),
    'min': giá_trị_min,
    'max': giá_trị_max, 
    'mean': giá_trị_trung_bình,
    'std': độ_lệch_chuẩn,
    'dtype': kiểu_dữ_liệu
}
```

---

## 💡 Tips sử dụng

### 1. Xử lý lỗi
```python
try:
    image = load_image(path)
    gray = rgb_to_grayscale(image)
except Exception as e:
    print(f"Lỗi: {e}")
```

### 2. Kiểm tra kích thước
```python
print(f"Ảnh gốc: {image.shape}")
padded = pad_image(image, padding=2)
print(f"Sau padding: {padded.shape}")
```

### 3. Tạo kernel tùy chỉnh
```python
# Kernel làm mờ
blur_kernel = create_kernel(5, 'gaussian')

# Kernel làm sắc nét
sharp_kernel = create_kernel(3, 'sharpen')
```

---

## ⚠️ Lưu ý quan trọng

1. **Kernel size phải là số lẻ** (3, 5, 7, ...)
2. **Ảnh input phải là RGB** cho hàm `rgb_to_grayscale`
3. **Padding giúp giữ nguyên kích thước** sau convolution
4. **Normalize ảnh** nếu giá trị vượt quá [0, 255]

---

## 🔗 Liên kết với modules khác

- **histogram.py**: Sử dụng `rgb_to_grayscale()` để convert ảnh
- **filtering.py**: Sử dụng `create_kernel()` và `pad_image()` cho convolution
- **app.py**: Sử dụng `load_image()` để xử lý upload từ Streamlit
