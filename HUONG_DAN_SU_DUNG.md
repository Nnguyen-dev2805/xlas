# 📖 Hướng dẫn Sử dụng Chi tiết

## 🚀 Bắt đầu nhanh (Quick Start)

### 1. Chạy ứng dụng
```bash
# Windows
run.bat

# macOS/Linux  
./run.sh

# Hoặc thủ công
streamlit run app.py
```

### 2. Mở trình duyệt
- URL: `http://localhost:8501`
- Ứng dụng sẽ tự động mở

---

## 🎮 Hướng dẫn từng bước

### Bước 1: Upload ảnh 📤

1. **Tìm sidebar bên trái** với tiêu đề "🎛️ Điều khiển"
2. **Click "Browse files"** trong phần "📤 Upload ảnh(s)"
3. **Chọn ảnh** từ máy tính:
   - Format hỗ trợ: PNG, JPG, JPEG, BMP
   - Kích thước khuyến nghị: 200x200 đến 1000x1000 pixels
   - Có thể chọn nhiều ảnh cùng lúc (tối đa 10 ảnh)

### Bước 2: Chọn chế độ xử lý 🔧

#### **Single Image Analysis** (Phân tích 1 ảnh)
- **Mục đích:** Xem chi tiết từng bước xử lý
- **Phù hợp:** Demo, học tập, debug
- **Kết quả:** Hiển thị đầy đủ ảnh, histograms, metrics

#### **Batch Processing (10 ảnh)** (Xử lý hàng loạt)
- **Mục đích:** Xử lý nhiều ảnh cùng lúc
- **Phù hợp:** Nộp bài, báo cáo
- **Kết quả:** Tabs cho từng ảnh + PDF report

#### **Algorithm Comparison** (So sánh thuật toán)
- **Mục đích:** So sánh các phương pháp filtering
- **Phù hợp:** Nghiên cứu, phân tích
- **Kết quả:** Bảng so sánh PSNR, correlation

### Bước 3: Chọn thuật toán 🧮

- ✅ **Bài 1: Histogram Processing**
  - H1: Histogram gốc
  - H2: Histogram Equalization
  - H3: Thu hẹp về [30, 80]

- ✅ **Bài 2: Filtering Operations**
  - I1: Convolution 3×3, padding=1
  - I2: Convolution 5×5, padding=2
  - I3: Convolution 7×7, padding=3, stride=2
  - I4: Median filter 3×3 trên I3
  - I5: Min filter 5×5 trên I1
  - I6: Thresholding I4 vs I5

### Bước 4: Xem kết quả 📊

#### **Ảnh được hiển thị:**
- Grid layout đẹp mắt
- Caption mô tả rõ ràng
- Zoom được khi click

#### **Histograms:**
- Interactive plots với Plotly
- Hover để xem chi tiết
- Zoom, pan, export

#### **Thống kê:**
- JSON format dễ đọc
- MSE, PSNR, Correlation
- Mean, std, entropy

### Bước 5: Download kết quả 💾

#### **Single Image:**
- **ZIP file:** Tất cả ảnh đã xử lý
- **Tên file:** Có timestamp tự động

#### **Batch Processing:**
- **Images ZIP:** Tất cả ảnh từ tất cả files
- **PDF Report:** Báo cáo chuyên nghiệp đầy đủ
- **Format:** `filename_resulttype.png`

---

## 📋 Chi tiết từng chức năng

### 🔍 Single Image Analysis

#### **Thông tin ảnh gốc:**
```
📊 Thông tin ảnh
• Kích thước: (height, width, channels)
• Min/Max: giá_trị_min/giá_trị_max  
• Mean ± Std: trung_bình ± độ_lệch_chuẩn
```

#### **Bài 1 - Histogram Processing:**

**Ảnh hiển thị:**
- Ảnh gốc (grayscale)
- Sau Histogram Equalization
- Sau thu hẹp [30, 80]

**Histograms interactive:**
- H1: Màu xanh dương
- H2: Màu xanh lá
- H3: Màu đỏ

**Phân tích histograms:**
```json
{
  "total_pixels": số_pixel_tổng,
  "mean_intensity": cường_độ_trung_bình,
  "std_intensity": độ_lệch_chuẩn,
  "mode_intensity": cường_độ_xuất_hiện_nhiều_nhất,
  "entropy": entropy_histogram,
  "min_intensity": cường_độ_min,
  "max_intensity": cường_độ_max
}
```

#### **Bài 2 - Filtering Operations:**

**Ảnh hiển thị (grid 4 cột):**
- Ảnh gốc
- I1, I2, I3 (convolutions)
- I4, I5, I6 (median, min, threshold)

**Kernels hiển thị:**
- Ma trận 3×3, 5×5, 7×7
- Giá trị số thực hiện

**Bảng phân tích hiệu ứng:**
| Filter | MSE | PSNR | Correlation | Mean Change |
|--------|-----|------|-------------|-------------|
| I1     | ... | ...  | ...         | ...         |

### 📦 Batch Processing

#### **Upload nhiều ảnh:**
- Chọn 1-10 ảnh cùng lúc
- Progress bar hiển thị tiến trình
- Xử lý song song để tăng tốc

#### **Kết quả theo tabs:**
- Mỗi ảnh 1 tab riêng
- Hiển thị RGB gốc + Grayscale
- Grid kết quả xử lý

#### **Download options:**
1. **📥 Download Images (ZIP)**
   - Tất cả ảnh đã xử lý
   - Format: `filename_resulttype.png`
   - Nén ZIP để tiết kiệm dung lượng

2. **📄 Tạo Báo cáo PDF**
   - Click button "Tạo Báo cáo PDF"
   - Chờ processing (có progress)
   - Download PDF report hoàn chỉnh

#### **Thống kê tổng quan:**
- Tổng số ảnh gốc
- Tổng số ảnh đã xử lý  
- Thuật toán đã áp dụng

### ⚖️ Algorithm Comparison

#### **So sánh filtering methods:**
- Original, Average 3×3, Average 5×5
- Gaussian 3×3, Gaussian 5×5
- Sharpen, Edge detection
- Median 3×3, Median 5×5
- Min 3×3, Min 5×5

#### **Phân tích định lượng:**
- Bảng so sánh đầy đủ
- Biểu đồ PSNR interactive
- Ranking theo chất lượng

---

## 📄 Báo cáo PDF

### **Cấu trúc báo cáo:**

#### **1. Trang bìa**
- Tên đồ án và môn học
- Thông tin nhóm (tên, lớp, giảng viên)
- Danh sách thành viên (MSSV, tỉ lệ đóng góp)
- Ngày tháng

#### **2. Cơ sở lý thuyết**
- Histogram và Histogram Equalization
- Convolution và các loại filters
- Công thức toán học chi tiết

#### **3. Kết quả xử lý (từng ảnh)**
- Ảnh gốc, kết quả Bài 1, Bài 2
- Histograms H1, H2, H3
- Grid I1-I6 với caption

#### **4. Kết luận**
- Tóm tắt kết quả đạt được
- Kỹ năng học được
- Ứng dụng thực tế

### **Tùy chỉnh thông tin nhóm:**

Sửa file `src/pdf_generator.py`, hàm `create_sample_team_info()`:

```python
return {
    'team_name': 'Tên nhóm của bạn',
    'class': 'Lớp của bạn', 
    'instructor': 'Tên giảng viên',
    'semester': 'HK - Năm học',
    'members': [
        {
            'name': 'Họ tên thành viên 1',
            'student_id': 'MSSV1',
            'contribution': '33.33'
        },
        # Thêm thành viên khác...
    ]
}
```

---

## 🎯 Tips sử dụng hiệu quả

### 💡 Chọn ảnh test tốt:
- **Ảnh có contrast thấp:** Để thấy rõ hiệu quả histogram equalization
- **Ảnh có noise:** Để test median filter
- **Ảnh có chi tiết:** Để thấy sự khác biệt các filters
- **Kích thước đa dạng:** Test khả năng xử lý

### 🚀 Tăng tốc xử lý:
- **Resize ảnh nhỏ hơn** nếu quá chậm
- **Đóng tabs không cần thiết** trong browser
- **Chọn ít thuật toán hơn** nếu chỉ cần test
- **Dùng ảnh grayscale** thay vì RGB

### 📊 Phân tích kết quả:
- **So sánh PSNR:** Cao = ít thay đổi, thấp = nhiều thay đổi
- **Xem entropy:** Cao = nhiều chi tiết, thấp = đơn giản
- **Check correlation:** Gần 1 = giữ nguyên structure
- **Visual inspection:** Mắt thường vẫn quan trọng nhất

### 🎨 Presentation tips:
- **Screenshot kết quả** để đưa vào slide
- **Export histograms** từ Plotly (click camera icon)
- **Sử dụng PDF report** làm tài liệu tham khảo
- **Demo trực tiếp** trong lớp bằng Streamlit

---

## ⚠️ Xử lý lỗi thường gặp

### 🔧 Lỗi kỹ thuật:

#### **"Module not found"**
```bash
pip install -r requirements.txt
```

#### **"Port 8501 already in use"**
```bash
streamlit run app.py --server.port 8502
```

#### **Memory error với ảnh lớn**
- Resize ảnh < 1000×1000 pixels
- Đóng các ứng dụng khác
- Restart browser

#### **PDF không tạo được**
- Kiểm tra quyền ghi file
- Đảm bảo thư mục `data/output/` tồn tại
- Restart ứng dụng

### 🎯 Lỗi sử dụng:

#### **Không thấy kết quả**
- Đảm bảo đã chọn thuật toán
- Check console browser (F12) xem lỗi
- Refresh page và thử lại

#### **Ảnh hiển thị sai**
- Kiểm tra format ảnh (PNG, JPG)
- Đảm bảo ảnh không bị corrupt
- Thử ảnh khác

#### **Download không hoạt động**
- Disable popup blocker
- Check download folder
- Thử browser khác

---

## 🏆 Checklist hoàn thành đồ án

### ✅ Yêu cầu cơ bản:
- [ ] Bài 1: H1, H2, H3 hoạt động đúng
- [ ] Bài 2: I1-I6 hoạt động đúng  
- [ ] Xử lý được 10 ảnh
- [ ] Tạo được báo cáo PDF
- [ ] Giao diện đẹp và rõ ràng

### ✅ Yêu cầu nâng cao:
- [ ] Code có comments đầy đủ
- [ ] Documentation chi tiết
- [ ] Error handling tốt
- [ ] Performance tối ưu
- [ ] UI/UX chuyên nghiệp

### ✅ Chuẩn bị vấn đáp:
- [ ] Hiểu thuật toán Histogram Equalization
- [ ] Giải thích được Convolution
- [ ] Biết ứng dụng từng filter
- [ ] Demo được trực tiếp
- [ ] Trả lời được câu hỏi mở rộng

---

## 🎓 Mở rộng và phát triển

### 🔮 Tính năng có thể thêm:
- **More filters:** Gaussian blur, Sobel edge detection
- **Color processing:** HSV, LAB color spaces
- **Advanced algorithms:** Bilateral filter, CLAHE
- **Real-time processing:** Webcam input
- **Cloud deployment:** Heroku, Streamlit Cloud

### 📚 Học thêm:
- **Computer Vision:** OpenCV advanced
- **Deep Learning:** CNN for image processing  
- **Image Enhancement:** HDR, super-resolution
- **Medical Imaging:** DICOM processing
- **Satellite Imagery:** Remote sensing

---

**🎉 Chúc bạn thành công với đồ án! Hãy tận dụng tối đa công cụ này để học tập và nghiên cứu!**
