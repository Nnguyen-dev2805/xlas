# Histogram Module Documentation

## 📋 Tổng quan

Module `histogram.py` implement các thuật toán xử lý histogram cho **Bài 1** của đồ án:
1. **H1**: Tính histogram của ảnh gốc
2. **H2**: Histogram equalization (cân bằng histogram)
3. **H3**: Thu hẹp histogram về khoảng [30, 80]

## 🎯 Mục tiêu Bài 1

> Cho 1 ảnh màu I kích thước n×m, chuyển đổi ảnh I thành ảnh xám:
> - Vẽ Histogram của I (H1)
> - Histogram cân bằng của I (H2)  
> - Hiệu chỉnh thu hẹp H2 trong khoảng (30,80)

## 🔧 Các hàm chính

### 1. `calculate_histogram(image)`

**Chức năng:** Tính histogram của ảnh grayscale

**Input:**
- `image`: Ảnh grayscale với shape (H, W)

**Output:**
- `numpy array`: Histogram với 256 bins (0-255)

**Thuật toán:**
```python
# Khởi tạo histogram với 256 bins
histogram = np.zeros(256)

# Đếm số lượng pixels cho mỗi intensity level
for pixel_value in image.flatten():
    histogram[pixel_value] += 1
```

**Ví dụ sử dụng:**
```python
from src.histogram import calculate_histogram

gray_image = rgb_to_grayscale(color_image)
hist = calculate_histogram(gray_image)
print(f"Histogram shape: {hist.shape}")  # (256,)
```

---

### 2. `histogram_equalization(image)` ⭐

**Chức năng:** Cân bằng histogram để cải thiện contrast

**Input:**
- `image`: Ảnh grayscale

**Output:**
- `tuple`: (equalized_image, new_histogram, cdf, lookup_table)

**Thuật toán chi tiết:**

#### Bước 1: Tính Histogram
```python
hist = calculate_histogram(image)
```

#### Bước 2: Tính CDF (Cumulative Distribution Function)
```python
cdf = hist.cumsum()
```

#### Bước 3: Normalize CDF
```python
cdf_min = cdf[cdf > 0].min()
total_pixels = image.shape[0] * image.shape[1]

# Công thức chuẩn histogram equalization
for i in range(256):
    lut[i] = (cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255
```

#### Bước 4: Áp dụng Transformation
```python
equalized_image = lut[image]
```

**Lý thuyết:**
- **Mục đích:** Phân phối lại intensity để histogram gần đều nhất
- **Kết quả:** Tăng contrast, chi tiết rõ nét hơn
- **Ứng dụng:** Cải thiện ảnh tối, ảnh có contrast thấp

**Ví dụ sử dụng:**
```python
equalized_img, new_hist, cdf, lut = histogram_equalization(gray_image)
```

---

### 3. `narrow_histogram(image, min_val=30, max_val=80)`

**Chức năng:** Thu hẹp histogram về khoảng [min_val, max_val]

**Input:**
- `image`: Ảnh grayscale
- `min_val`: Giá trị minimum mới (default=30)
- `max_val`: Giá trị maximum mới (default=80)

**Output:**
- `tuple`: (narrowed_image, new_histogram)

**Thuật toán:**

#### Linear Mapping
```python
current_min = np.min(image)
current_max = np.max(image)

# Công thức linear mapping
new_value = (old_value - current_min) / (current_max - current_min) * (max_val - min_val) + min_val
```

**Ví dụ:**
- Ảnh gốc có range [50, 200]
- Thu hẹp về [30, 80]
- Pixel có giá trị 125 → `(125-50)/(200-50) * (80-30) + 30 = 55`

---

### 4. `process_task1(image)` 🎯

**Chức năng:** Xử lý đầy đủ Bài 1

**Input:**
- `image`: Ảnh grayscale

**Output:**
- `dict`: Dictionary chứa tất cả kết quả
```python
{
    'original_image': ảnh_gốc,
    'h1': histogram_gốc,
    'h2_image': ảnh_sau_equalization,
    'h2': histogram_sau_equalization,
    'narrowed_image': ảnh_sau_thu_hẹp,
    'narrowed_hist': histogram_sau_thu_hẹp,
    'cdf': cumulative_distribution_function,
    'lookup_table': bảng_lookup_cho_equalization
}
```

**Workflow:**
```python
# Bước 1: Tính H1
h1 = calculate_histogram(image)

# Bước 2: Histogram Equalization → H2
h2_image, h2, cdf, lut = histogram_equalization(image)

# Bước 3: Thu hẹp H2 → H3
narrowed_image, h3 = narrow_histogram(h2_image, 30, 80)
```

---

## 📊 Hàm Visualization

### 1. `plot_histogram_plotly(histogram, title, color)`

**Chức năng:** Tạo interactive histogram với Plotly

**Features:**
- Interactive zoom, pan
- Hover information
- Professional styling
- Export options

### 2. `create_histogram_comparison_plotly(h1, h2, h3)`

**Chức năng:** So sánh 3 histograms trong 1 figure

**Layout:**
- 3 subplots theo chiều dọc
- H1: Màu xanh dương
- H2: Màu xanh lá  
- H3: Màu đỏ

---

## 📈 Hàm Phân tích

### `analyze_histogram_properties(hist)`

**Chức năng:** Phân tích các tính chất của histogram

**Output:**
```python
{
    'total_pixels': tổng_số_pixel,
    'mean_intensity': cường_độ_trung_bình,
    'std_intensity': độ_lệch_chuẩn,
    'mode_intensity': cường_độ_xuất_hiện_nhiều_nhất,
    'entropy': entropy_của_histogram,
    'min_intensity': cường_độ_min,
    'max_intensity': cường_độ_max
}
```

**Công thức Entropy:**
```python
entropy = -Σ(p_i × log2(p_i))
```
- `p_i`: Xác suất của intensity level i
- Entropy cao → Ảnh có nhiều chi tiết
- Entropy thấp → Ảnh đơn giản, ít chi tiết

---

## 🎓 Lý thuyết nền tảng

### Histogram là gì?
- **Định nghĩa:** Biểu đồ thống kê phân phối cường độ sáng trong ảnh
- **Trục X:** Intensity levels (0-255)
- **Trục Y:** Số lượng pixels có intensity đó

### Tại sao cần Histogram Equalization?
1. **Cải thiện contrast:** Ảnh tối → sáng hơn
2. **Tăng chi tiết:** Làm nổi bật features ẩn
3. **Chuẩn hóa:** Đưa ảnh về phân phối chuẩn

### Ứng dụng thực tế:
- **Y học:** Cải thiện ảnh X-ray, CT scan
- **Vệ tinh:** Xử lý ảnh từ không gian
- **Photography:** Auto-enhance trong camera
- **Security:** Cải thiện ảnh từ camera giám sát

---

## 💡 Tips sử dụng

### 1. Kiểm tra histogram trước khi xử lý
```python
hist = calculate_histogram(image)
plt.plot(hist)
plt.title("Histogram gốc")
plt.show()
```

### 2. So sánh trước và sau equalization
```python
# Trước
original_stats = analyze_histogram_properties(h1)
print(f"Entropy gốc: {original_stats['entropy']:.2f}")

# Sau
equalized_stats = analyze_histogram_properties(h2)
print(f"Entropy sau equalization: {equalized_stats['entropy']:.2f}")
```

### 3. Tùy chỉnh khoảng thu hẹp
```python
# Thu hẹp về khoảng khác
narrowed_img, _ = narrow_histogram(image, min_val=50, max_val=150)
```

---

## ⚠️ Lưu ý quan trọng

### 1. Histogram Equalization không phải lúc nào cũng tốt
- **Tốt:** Ảnh có contrast thấp, tối
- **Không tốt:** Ảnh đã có contrast tốt → có thể làm mất tự nhiên

### 2. Thu hẹp histogram
- Giảm dynamic range
- Có thể mất thông tin
- Phù hợp khi cần giới hạn intensity range

### 3. Xử lý ảnh màu
- Không nên apply trực tiếp lên RGB
- Convert sang HSV, chỉ equalize channel V
- Hoặc convert sang LAB, equalize channel L

---

## 🔗 Integration với GUI

Trong Streamlit app:
```python
# Xử lý
results = process_task1(gray_image)

# Hiển thị histograms
fig = create_histogram_comparison_plotly(
    results['h1'], 
    results['h2'], 
    results['narrowed_hist']
)
st.plotly_chart(fig)

# Hiển thị ảnh
st.image(results['h2_image'], caption="Sau Equalization")
```

---

## 📚 Tài liệu tham khảo

1. **Gonzalez & Woods** - Digital Image Processing
2. **OpenCV Documentation** - Histogram Equalization
3. **Wikipedia** - Histogram Equalization Algorithm
