# Filtering Module Documentation

## 📋 Tổng quan

Module `filtering.py` implement các thuật toán filtering cho **Bài 2** của đồ án:
1. **I1**: Convolution với kernel 3×3, padding=1
2. **I2**: Convolution với kernel 5×5, padding=2
3. **I3**: Convolution với kernel 7×7, padding=3, stride=2
4. **I4**: Median filter 3×3 trên I3
5. **I5**: Min filter 5×5 trên I1
6. **I6**: Thresholding I4 vs I5

## 🎯 Mục tiêu Bài 2

> Cho 1 ảnh màu I kích thước n×m, chuyển đổi ảnh I thành ảnh xám:
> - Dùng phép tích chập với các kernel khác nhau
> - Áp dụng median và min filtering
> - Thực hiện thresholding operation

## 🔧 Hàm Core: Convolution 2D

### `convolution_2d(image, kernel, padding=0, stride=1)` ⭐

**Chức năng:** Thực hiện phép tích chập 2D từ scratch

**Input:**
- `image`: Ảnh grayscale (H, W)
- `kernel`: Convolution kernel (K, K)
- `padding`: Số pixel padding
- `stride`: Bước nhảy (default=1)

**Output:**
- `numpy array`: Ảnh sau convolution

**Thuật toán chi tiết:**

#### Bước 1: Thêm Padding
```python
if padding > 0:
    padded_image = pad_image(image, padding)
```

#### Bước 2: Tính kích thước Output
```python
out_h = (img_h - kernel_h) // stride + 1
out_w = (img_w - kernel_w) // stride + 1
```

#### Bước 3: Thực hiện Convolution
```python
for i in range(0, out_h):
    for j in range(0, out_w):
        # Tính vị trí trong ảnh gốc
        start_i = i * stride
        start_j = j * stride
        
        # Lấy region of interest
        roi = padded_image[start_i:start_i+kernel_h, start_j:start_j+kernel_w]
        
        # Tính tích chập (element-wise multiply và sum)
        conv_sum = np.sum(roi * kernel)
        output[i, j] = conv_sum
```

**Lý thuyết Convolution:**
- **Mục đích:** Áp dụng filter lên từng vùng nhỏ của ảnh
- **Cách hoạt động:** Nhân từng phần tử rồi cộng lại
- **Ứng dụng:** Blur, sharpen, edge detection, feature extraction

---

## 🎯 Các hàm xử lý chính

### 1. I1, I2, I3 - Convolution Operations

#### I1: Kernel 3×3, Padding=1
```python
kernel_3x3 = create_kernel(3, 'average')
i1 = convolution_2d(image, kernel_3x3, padding=1, stride=1)
```
- **Kích thước output:** Giống input (nhờ padding=1)
- **Hiệu ứng:** Làm mờ nhẹ, giảm noise

#### I2: Kernel 5×5, Padding=2  
```python
kernel_5x5 = create_kernel(5, 'average')
i2 = convolution_2d(image, kernel_5x5, padding=2, stride=1)
```
- **Kích thước output:** Giống input (nhờ padding=2)
- **Hiệu ứng:** Làm mờ mạnh hơn I1

#### I3: Kernel 7×7, Padding=3, Stride=2
```python
kernel_7x7 = create_kernel(7, 'average')
i3 = convolution_2d(image, kernel_7x7, padding=3, stride=2)
```
- **Kích thước output:** Giảm một nửa (do stride=2)
- **Hiệu ứng:** Làm mờ mạnh + downsampling

**Công thức tính kích thước:**
```
output_size = (input_size + 2×padding - kernel_size) / stride + 1
```

---

### 2. `apply_median_filter(image, kernel_size=3)`

**Chức năng:** Áp dụng median filter để loại bỏ noise

**Input:**
- `image`: Ảnh input
- `kernel_size`: Kích thước kernel (default=3)

**Output:**
- `numpy array`: Ảnh đã được lọc

**Thuật toán:**
1. Với mỗi pixel, lấy vùng lân cận kích thước kernel_size×kernel_size
2. Sắp xếp tất cả giá trị trong vùng đó
3. Lấy giá trị median (ở giữa) làm giá trị mới

**Ưu điểm:**
- **Loại bỏ salt-and-pepper noise** hiệu quả
- **Bảo toàn cạnh** tốt hơn average filter
- **Không làm mờ** chi tiết quan trọng

**Ví dụ:**
```
Vùng 3×3:     Sắp xếp:      Median:
[10 255 12]   [10 11 12     →  20
 11  20  13]   13 20 25
 25  30  40]   30 40 255]
```

---

### 3. `apply_min_filter(image, kernel_size=5)`

**Chức năng:** Áp dụng min filter (erosion-like operation)

**Input:**
- `image`: Ảnh input  
- `kernel_size`: Kích thước kernel (default=5)

**Output:**
- `numpy array`: Ảnh đã được lọc

**Thuật toán:**
1. Với mỗi pixel, lấy vùng lân cận kích thước kernel_size×kernel_size
2. Tìm giá trị minimum trong vùng đó
3. Gán giá trị minimum làm giá trị mới

**Hiệu ứng:**
- **Làm tối ảnh** (erosion effect)
- **Thu nhỏ vùng sáng**
- **Mở rộng vùng tối**
- **Loại bỏ bright noise**

---

### 4. `threshold_operation(image1, image2)`

**Chức năng:** Thực hiện thresholding theo yêu cầu đề bài

**Input:**
- `image1`: Ảnh thứ nhất (I4)
- `image2`: Ảnh thứ hai (I5)

**Output:**
- `numpy array`: Ảnh sau thresholding

**Thuật toán:**
```python
# Nếu I4(x,y) > I5(x,y) thì I6(x,y) = 0
# Ngược lại I6(x,y) = I5(x,y)
result = np.where(image1 > image2, 0, image2)
```

**Xử lý kích thước khác nhau:**
```python
if image1.shape != image2.shape:
    image1 = resize_to_match(image1, image2)
```

---

## 🎯 Hàm tổng hợp: `process_task2(image)`

**Chức năng:** Xử lý đầy đủ Bài 2

**Workflow:**
```python
def process_task2(image):
    # I1: Conv 3×3, pad=1
    i1 = convolution_2d(image, kernel_3x3, padding=1, stride=1)
    
    # I2: Conv 5×5, pad=2  
    i2 = convolution_2d(image, kernel_5x5, padding=2, stride=1)
    
    # I3: Conv 7×7, pad=3, stride=2
    i3 = convolution_2d(image, kernel_7x7, padding=3, stride=2)
    
    # I4: Median filter 3×3 trên I3
    i4 = apply_median_filter(i3, kernel_size=3)
    
    # I5: Min filter 5×5 trên I1
    i5 = apply_min_filter(i1, kernel_size=5)
    
    # I6: Thresholding I4 vs I5
    i6 = threshold_operation(i4, i5)
    
    return results_dict
```

**Output:**
```python
{
    'original_image': ảnh_gốc,
    'i1': kết_quả_I1,
    'i2': kết_quả_I2, 
    'i3': kết_quả_I3,
    'i4': kết_quả_I4,
    'i5': kết_quả_I5,
    'i6': kết_quả_I6,
    'kernel_3x3': kernel_3×3,
    'kernel_5x5': kernel_5×5,
    'kernel_7x7': kernel_7×7
}
```

---

## 📊 Hàm phân tích và so sánh

### 1. `analyze_filter_effects(original, filtered, filter_name)`

**Chức năng:** Phân tích hiệu ứng của filter

**Metrics tính toán:**

#### MSE (Mean Squared Error)
```python
mse = np.mean((original - filtered) ** 2)
```

#### PSNR (Peak Signal-to-Noise Ratio)
```python
psnr = 20 * log10(255 / sqrt(mse))
```
- **PSNR cao:** Ảnh ít bị thay đổi
- **PSNR thấp:** Ảnh bị thay đổi nhiều

#### Correlation
```python
correlation = np.corrcoef(original.flatten(), filtered.flatten())[0,1]
```
- **Correlation = 1:** Hoàn toàn giống nhau
- **Correlation = 0:** Không có mối liên hệ

---

### 2. `compare_filtering_methods(image)`

**Chức năng:** So sánh các phương pháp filtering khác nhau

**Các phương pháp được so sánh:**
- Average filters (3×3, 5×5)
- Gaussian filters (3×3, 5×5)  
- Sharpen filter
- Edge detection filter
- Median filters (3×3, 5×5)
- Min filters (3×3, 5×5)

---

### 3. `custom_convolution_with_opencv_comparison(image, kernel)`

**Chức năng:** Validate implementation bằng cách so sánh với OpenCV

**So sánh:**
- Custom implementation vs `cv2.filter2D()`
- Tính độ khác biệt maximum và mean
- Threshold để xác định tính tương đồng

---

## 🎓 Lý thuyết nền tảng

### Convolution trong Computer Vision

#### Tại sao dùng Convolution?
1. **Local feature detection:** Phát hiện patterns cục bộ
2. **Translation invariant:** Không phụ thuộc vị trí
3. **Parameter sharing:** Dùng chung kernel cho toàn ảnh
4. **Hierarchical learning:** Từ low-level đến high-level features

#### Padding strategies:
- **Valid:** Không padding → output nhỏ hơn input
- **Same:** Padding để output = input  
- **Full:** Padding maximum → output lớn hơn input

#### Stride effects:
- **Stride = 1:** Giữ nguyên resolution
- **Stride > 1:** Downsampling, giảm kích thước

---

### Morphological Operations

#### Erosion (Min Filter)
- **Mục đích:** Thu nhỏ vùng sáng
- **Ứng dụng:** Loại bỏ noise nhỏ, tách các object dính nhau

#### Dilation (Max Filter)  
- **Mục đích:** Mở rộng vùng sáng
- **Ứng dụng:** Lấp đầy lỗ hổng, nối các object gần nhau

#### Opening = Erosion + Dilation
#### Closing = Dilation + Erosion

---

### Median Filtering

#### Ưu điểm:
- **Edge-preserving:** Không làm mờ cạnh
- **Noise removal:** Loại bỏ impulse noise hiệu quả
- **Non-linear:** Không phải convolution tuyến tính

#### Nhược điểm:
- **Computational cost:** Chậm hơn linear filters
- **Detail loss:** Có thể mất chi tiết nhỏ
- **Kernel size sensitive:** Kích thước kernel ảnh hưởng nhiều

---

## 💡 Tips sử dụng

### 1. Chọn kernel size phù hợp
```python
# Noise nhỏ → kernel nhỏ
median_3x3 = apply_median_filter(image, 3)

# Noise lớn → kernel lớn  
median_7x7 = apply_median_filter(image, 7)
```

### 2. Kiểm tra kích thước output
```python
print(f"Input: {image.shape}")
result = convolution_2d(image, kernel, padding=1, stride=2)
print(f"Output: {result.shape}")
```

### 3. Validate với OpenCV
```python
comparison = custom_convolution_with_opencv_comparison(image, kernel)
print(f"Max difference: {comparison['max_difference']}")
print(f"Are similar: {comparison['are_similar']}")
```

---

## ⚠️ Lưu ý quan trọng

### 1. Memory và Performance
- **Large kernels:** Chậm, tốn memory
- **Stride > 1:** Nhanh hơn nhưng mất thông tin
- **Separable kernels:** Tách thành 2 convolutions 1D

### 2. Boundary effects
- **Zero padding:** Tạo artifacts ở biên
- **Reflect padding:** Tự nhiên hơn
- **Wrap padding:** Cho ảnh periodic

### 3. Data types
- **Float32:** Tính toán chính xác
- **Uint8:** Tiết kiệm memory nhưng có thể overflow
- **Clipping:** Luôn clip về [0, 255] cuối cùng

---

## 🔗 Integration với GUI

Trong Streamlit:
```python
# Xử lý
results = process_task2(gray_image)

# Hiển thị grid ảnh
images = [results[f'i{i}'] for i in range(1, 7)]
titles = [f'I{i}' for i in range(1, 7)]
display_image_grid(images, titles)

# Hiển thị kernels
st.text("Kernel 3×3:")
st.text(str(results['kernel_3x3']))
```

---

## 📚 Tài liệu tham khảo

1. **Gonzalez & Woods** - Digital Image Processing, Chapter 3
2. **OpenCV Documentation** - Image Filtering
3. **CS231n Stanford** - Convolutional Neural Networks
4. **Scipy Documentation** - ndimage filters
