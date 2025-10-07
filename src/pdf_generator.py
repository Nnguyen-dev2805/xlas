"""
PDF Generator Module - Tạo báo cáo PDF cho Bài 3
===============================================

Chức năng:
- Tạo báo cáo PDF tự động từ kết quả xử lý
- Bao gồm ảnh gốc, kết quả, histograms
- Format chuyên nghiệp với thông tin nhóm
- Hỗ trợ batch processing 10 ảnh

Author: Image Processing Team
"""

from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime


class PDFReportGenerator:
    """Class để tạo báo cáo PDF chuyên nghiệp"""
    
    def __init__(self, output_path="report.pdf"):
        """
        Khởi tạo PDF generator
        
        Args:
            output_path: Đường dẫn file PDF output
        """
        self.output_path = output_path
        self.doc = SimpleDocTemplate(output_path, pagesize=A4)
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkgreen
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
    
    def add_cover_page(self, team_info):
        """
        Thêm trang bìa
        
        Args:
            team_info: Dictionary chứa thông tin nhóm
        """
        # Tiêu đề chính
        title = Paragraph("BÁO CÁO ĐỒ ÁN CUỐI KỲ", self.title_style)
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Môn học
        subject = Paragraph("MÔN: XỬ LÝ ẢNH SỐ", self.heading_style)
        self.story.append(subject)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Đề tài
        topic = Paragraph("ĐỀ TÀI: HISTOGRAM PROCESSING & IMAGE FILTERING", self.heading_style)
        self.story.append(topic)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Thông tin nhóm
        team_data = [
            ['Thông tin', 'Chi tiết'],
            ['Tên nhóm', team_info.get('team_name', 'Nhóm [Số]')],
            ['Lớp', team_info.get('class', '[Tên lớp]')],
            ['Giảng viên', team_info.get('instructor', '[Tên giảng viên]')],
            ['Học kỳ', team_info.get('semester', '[Học kỳ - Năm học]')]
        ]
        
        team_table = Table(team_data, colWidths=[2*inch, 3*inch])
        team_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.story.append(team_table)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Thành viên nhóm
        if 'members' in team_info:
            members_title = Paragraph("THÀNH VIÊN NHÓM", self.heading_style)
            self.story.append(members_title)
            
            members_data = [['STT', 'Họ và Tên', 'MSSV', 'Tỉ lệ đóng góp (%)']]
            for i, member in enumerate(team_info['members'], 1):
                members_data.append([
                    str(i),
                    member.get('name', f'Thành viên {i}'),
                    member.get('student_id', '[MSSV]'),
                    member.get('contribution', '33.33')
                ])
            
            members_table = Table(members_data, colWidths=[0.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
            members_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(members_table)
        
        # Ngày tháng
        date_str = datetime.now().strftime("%d/%m/%Y")
        date_para = Paragraph(f"<para align=center>Ngày: {date_str}</para>", self.normal_style)
        self.story.append(Spacer(1, 1*inch))
        self.story.append(date_para)
        
        # Page break
        self.story.append(PageBreak())
    
    def add_theory_section(self):
        """Thêm phần lý thuyết"""
        # Tiêu đề
        theory_title = Paragraph("I. CƠ SỞ LÝ THUYẾT", self.title_style)
        self.story.append(theory_title)
        
        # Bài 1: Histogram Processing
        bai1_title = Paragraph("1. Histogram Processing", self.heading_style)
        self.story.append(bai1_title)
        
        bai1_content = """
        <b>Histogram</b> là biểu đồ thống kê phân phối cường độ sáng trong ảnh số. 
        Trục hoành biểu diễn các mức cường độ (0-255), trục tung biểu diễn số lượng pixel có cường độ tương ứng.
        
        <b>Histogram Equalization</b> là kỹ thuật cải thiện contrast bằng cách phân phối lại các mức cường độ 
        sao cho histogram trở nên đều hơn. Công thức:
        
        s = T(r) = (L-1) × CDF(r)
        
        Trong đó: CDF là Cumulative Distribution Function, L=256 (số mức xám).
        
        <b>Thu hẹp Histogram</b> là quá trình ánh xạ tuyến tính để giới hạn dải cường độ về một khoảng nhỏ hơn.
        """
        
        bai1_para = Paragraph(bai1_content, self.normal_style)
        self.story.append(bai1_para)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Bài 2: Image Filtering
        bai2_title = Paragraph("2. Image Filtering", self.heading_style)
        self.story.append(bai2_title)
        
        bai2_content = """
        <b>Convolution</b> là phép tích chập giữa ảnh và kernel (ma trận lọc). 
        Công thức: g(x,y) = Σ Σ f(x+i, y+j) × h(i,j)
        
        <b>Padding</b>: Thêm pixel ở biên để giữ nguyên kích thước ảnh sau convolution.
        
        <b>Stride</b>: Bước nhảy khi di chuyển kernel, stride > 1 sẽ giảm kích thước output.
        
        <b>Median Filter</b>: Thay thế pixel bằng giá trị trung vị trong vùng lân cận, 
        hiệu quả loại bỏ salt-and-pepper noise mà vẫn bảo toàn cạnh.
        
        <b>Min Filter</b>: Thay thế pixel bằng giá trị minimum trong vùng lân cận, 
        có tác dụng tương tự phép erosion trong morphology.
        """
        
        bai2_para = Paragraph(bai2_content, self.normal_style)
        self.story.append(bai2_para)
        
        self.story.append(PageBreak())
    
    def numpy_to_pil_image(self, np_array):
        """
        Convert numpy array thành PIL Image
        
        Args:
            np_array: Numpy array
            
        Returns:
            PIL Image
        """
        if len(np_array.shape) == 2:  # Grayscale
            return Image.fromarray(np_array.astype(np.uint8), mode='L')
        else:  # RGB
            return Image.fromarray(np_array.astype(np.uint8), mode='RGB')
    
    def create_image_buffer(self, image, format='PNG'):
        """
        Tạo buffer cho ảnh để embed vào PDF
        
        Args:
            image: PIL Image hoặc numpy array
            format: Format ảnh (PNG, JPEG)
            
        Returns:
            BytesIO buffer
        """
        if isinstance(image, np.ndarray):
            image = self.numpy_to_pil_image(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer
    
    def create_histogram_plot(self, histogram, title="Histogram", color='blue'):
        """
        Tạo plot histogram và return buffer
        
        Args:
            histogram: Histogram data
            title: Tiêu đề
            color: Màu
            
        Returns:
            BytesIO buffer
        """
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(256), histogram, color=color, alpha=0.7, width=1.0)
        ax.set_xlabel('Intensity Level')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_xlim([0, 255])
        ax.grid(True, alpha=0.3)
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer
    
    def add_image_results(self, image_name, task1_results, task2_results):
        """
        Thêm kết quả xử lý cho 1 ảnh
        
        Args:
            image_name: Tên ảnh
            task1_results: Kết quả Bài 1
            task2_results: Kết quả Bài 2
        """
        # Tiêu đề ảnh
        img_title = Paragraph(f"KẾT QUẢ XỬ LÝ: {image_name.upper()}", self.heading_style)
        self.story.append(img_title)
        
        # Bài 1 Results
        if task1_results:
            bai1_subtitle = Paragraph("Bài 1: Histogram Processing", self.heading_style)
            self.story.append(bai1_subtitle)
            
            # Tạo table cho ảnh Bài 1
            images_row1 = []
            
            # Ảnh gốc
            orig_buffer = self.create_image_buffer(task1_results['original_image'])
            orig_img = RLImage(orig_buffer, width=1.5*inch, height=1.5*inch)
            
            # Ảnh equalized
            eq_buffer = self.create_image_buffer(task1_results['h2_image'])
            eq_img = RLImage(eq_buffer, width=1.5*inch, height=1.5*inch)
            
            # Ảnh narrowed
            narrow_buffer = self.create_image_buffer(task1_results['narrowed_image'])
            narrow_img = RLImage(narrow_buffer, width=1.5*inch, height=1.5*inch)
            
            # Table ảnh
            img_data = [
                ['Ảnh gốc', 'Sau Equalization', 'Thu hẹp [30,80]'],
                [orig_img, eq_img, narrow_img]
            ]
            
            img_table = Table(img_data, colWidths=[2*inch, 2*inch, 2*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(img_table)
            self.story.append(Spacer(1, 0.2*inch))
            
            # Histograms
            hist_title = Paragraph("Histograms:", self.normal_style)
            self.story.append(hist_title)
            
            # Tạo histogram plots
            h1_buffer = self.create_histogram_plot(task1_results['h1'], 'H1 - Gốc', 'blue')
            h2_buffer = self.create_histogram_plot(task1_results['h2'], 'H2 - Equalized', 'green')
            h3_buffer = self.create_histogram_plot(task1_results['narrowed_hist'], 'H3 - Narrowed', 'red')
            
            h1_img = RLImage(h1_buffer, width=2*inch, height=1*inch)
            h2_img = RLImage(h2_buffer, width=2*inch, height=1*inch)
            h3_img = RLImage(h3_buffer, width=2*inch, height=1*inch)
            
            hist_data = [[h1_img, h2_img, h3_img]]
            hist_table = Table(hist_data, colWidths=[2*inch, 2*inch, 2*inch])
            hist_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            
            self.story.append(hist_table)
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Bài 2 Results
        if task2_results:
            bai2_subtitle = Paragraph("Bài 2: Image Filtering", self.heading_style)
            self.story.append(bai2_subtitle)
            
            # I1, I2, I3
            i1_buffer = self.create_image_buffer(task2_results['i1'])
            i2_buffer = self.create_image_buffer(task2_results['i2'])
            i3_buffer = self.create_image_buffer(task2_results['i3'])
            
            i1_img = RLImage(i1_buffer, width=1.3*inch, height=1.3*inch)
            i2_img = RLImage(i2_buffer, width=1.3*inch, height=1.3*inch)
            i3_img = RLImage(i3_buffer, width=1.3*inch, height=1.3*inch)
            
            # I4, I5, I6
            i4_buffer = self.create_image_buffer(task2_results['i4'])
            i5_buffer = self.create_image_buffer(task2_results['i5'])
            i6_buffer = self.create_image_buffer(task2_results['i6'])
            
            i4_img = RLImage(i4_buffer, width=1.3*inch, height=1.3*inch)
            i5_img = RLImage(i5_buffer, width=1.3*inch, height=1.3*inch)
            i6_img = RLImage(i6_buffer, width=1.3*inch, height=1.3*inch)
            
            # Table kết quả filtering
            filt_data = [
                ['I1 (Conv 3x3)', 'I2 (Conv 5x5)', 'I3 (Conv 7x7)'],
                [i1_img, i2_img, i3_img],
                ['I4 (Median)', 'I5 (Min)', 'I6 (Threshold)'],
                [i4_img, i5_img, i6_img]
            ]
            
            filt_table = Table(filt_data, colWidths=[2*inch, 2*inch, 2*inch])
            filt_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(filt_table)
        
        self.story.append(PageBreak())
    
    def add_conclusion(self):
        """Thêm phần kết luận"""
        conclusion_title = Paragraph("III. KẾT LUẬN", self.title_style)
        self.story.append(conclusion_title)
        
        conclusion_content = """
        Qua đồ án này, nhóm đã thành công implement và áp dụng các thuật toán cơ bản trong xử lý ảnh số:
        
        <b>1. Histogram Processing:</b>
        - Hiểu được ý nghĩa và cách tính histogram của ảnh
        - Áp dụng thành công thuật toán Histogram Equalization để cải thiện contrast
        - Thực hiện thu hẹp histogram về khoảng giá trị mong muốn
        
        <b>2. Image Filtering:</b>
        - Implement convolution 2D từ scratch với các tham số padding và stride
        - Áp dụng median filter để loại bỏ noise hiệu quả
        - Sử dụng min filter cho morphological operations
        - Thực hiện thresholding để tạo ảnh nhị phân
        
        <b>3. Kỹ năng lập trình:</b>
        - Sử dụng thành thạo NumPy cho xử lý ma trận
        - Tạo giao diện web đẹp mắt với Streamlit
        - Tự động hóa tạo báo cáo PDF
        - Viết code có cấu trúc, dễ bảo trì
        
        <b>4. Ứng dụng thực tế:</b>
        - Hiểu được ứng dụng của từng thuật toán trong thực tế
        - Biết cách đánh giá chất lượng ảnh qua các metrics
        - Có thể mở rộng cho các bài toán phức tạp hơn
        
        Đồ án đã hoàn thành đầy đủ các yêu cầu và có thể được sử dụng như một công cụ học tập hiệu quả.
        """
        
        conclusion_para = Paragraph(conclusion_content, self.normal_style)
        self.story.append(conclusion_para)
    
    def generate_report(self, team_info, batch_results):
        """
        Tạo báo cáo PDF hoàn chỉnh
        
        Args:
            team_info: Thông tin nhóm
            batch_results: Dictionary chứa kết quả xử lý batch
        """
        print("📄 Đang tạo báo cáo PDF...")
        
        # Trang bìa
        self.add_cover_page(team_info)
        
        # Lý thuyết
        self.add_theory_section()
        
        # Kết quả từng ảnh
        results_title = Paragraph("II. KẾT QUẢ XỬ LÝ", self.title_style)
        self.story.append(results_title)
        
        for i, (image_name, results) in enumerate(batch_results.items(), 1):
            print(f"📄 Đang xử lý ảnh {i}/{len(batch_results)}: {image_name}")
            
            # Extract task results
            task1_results = None
            task2_results = None
            
            if 'task1' in results:
                task1_results = results['task1']
            if 'task2' in results:
                task2_results = results['task2']
            
            self.add_image_results(image_name, task1_results, task2_results)
        
        # Kết luận
        self.add_conclusion()
        
        # Build PDF
        print("📄 Đang build PDF...")
        self.doc.build(self.story)
        print(f"✅ Báo cáo PDF đã được tạo: {self.output_path}")


def create_sample_team_info():
    """Tạo thông tin nhóm mẫu"""
    return {
        'team_name': 'Nhóm Image Processing',
        'class': 'Xử lý ảnh số - K65',
        'instructor': 'TS. Nguyễn Văn A',
        'semester': 'HK1 - 2024-2025',
        'members': [
            {
                'name': 'Nguyễn Văn A',
                'student_id': '20210001',
                'contribution': '33.33'
            },
            {
                'name': 'Trần Thị B',
                'student_id': '20210002', 
                'contribution': '33.33'
            },
            {
                'name': 'Lê Văn C',
                'student_id': '20210003',
                'contribution': '33.34'
            }
        ]
    }


def generate_pdf_report(batch_results, team_info=None, output_path="data/output/bao_cao_do_an.pdf"):
    """
    Hàm tiện ích để tạo báo cáo PDF
    
    Args:
        batch_results: Kết quả xử lý batch
        team_info: Thông tin nhóm (nếu None sẽ dùng mẫu)
        output_path: Đường dẫn file PDF
    """
    if team_info is None:
        team_info = create_sample_team_info()
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Tạo PDF
    generator = PDFReportGenerator(output_path)
    generator.generate_report(team_info, batch_results)
    
    return output_path
