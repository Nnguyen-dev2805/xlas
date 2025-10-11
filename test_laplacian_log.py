"""
Test Script: Laplacian với Log Transform
=======================================

Demo hiệu quả của log transform trong việc cải thiện
hiển thị kết quả Laplacian edge detection

Author: Image Processing Team
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from filters.kernel_types import KernelGenerator
from core.convolution import convolution_2d_manual
from core.image_ops import rgb_to_grayscale_manual

def create_test_image():
    """Tạo ảnh test với vùng tối và sáng"""
    size = 200
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Tạo gradient từ tối đến sáng
    for i in range(size):
        for j in range(size):
            # Gradient ngang
            intensity = int(j * 255 / (size - 1))
            
            # Thêm một số shapes để tạo edges
            center_x, center_y = size//2, size//2
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            
            # Tạo circle
            if 50 < dist < 70:
                intensity = 255
            elif 30 < dist < 50:
                intensity = 50
            
            # Tạo rectangle
            if 80 < i < 120 and 80 < j < 120:
                intensity = 200
            
            image[i, j] = [intensity, intensity, intensity]
    
    return image

def test_laplacian_log_transform():
    """Test Laplacian với và không có log transform"""
    print("LAPLACIAN LOG TRANSFORM TEST")
    print("=" * 50)
    
    # Tạo ảnh test
    rgb_image = create_test_image()
    gray_image = rgb_to_grayscale_manual(rgb_image)
    
    print(f"Ảnh test: {gray_image.shape}")
    print(f"Range: [{np.min(gray_image)}, {np.max(gray_image)}]")
    
    # Lấy Laplacian kernel
    laplacian_kernel = KernelGenerator.laplacian_kernel()
    print(f"\nLaplacian kernel:")
    print(laplacian_kernel)
    print(f"Sum: {np.sum(laplacian_kernel)}")
    
    # Áp dụng Laplacian convolution
    laplacian_result = convolution_2d_manual(gray_image, laplacian_kernel, padding=1, stride=1)
    
    print(f"\nLaplacian result:")
    print(f"Shape: {laplacian_result.shape}")
    print(f"Range: [{np.min(laplacian_result):.2f}, {np.max(laplacian_result):.2f}]")
    print(f"Mean: {np.mean(laplacian_result):.2f}")
    print(f"Std: {np.std(laplacian_result):.2f}")
    
    # Xử lý giá trị âm
    laplacian_abs = np.abs(laplacian_result)
    print(f"\nSau absolute value:")
    print(f"Range: [{np.min(laplacian_abs):.2f}, {np.max(laplacian_abs):.2f}]")
    
    # Normalize thông thường
    normalized = ((laplacian_abs / laplacian_abs.max()) * 255).astype(np.uint8)
    
    # Test với các giá trị c khác nhau (bao gồm giá trị cao)
    c_values = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    print(f"\nTEST LOG TRANSFORM VỚI CÁC GIÁ TRỊ C:")
    print("-" * 40)
    
    results = {}
    results['original'] = gray_image
    results['laplacian_normal'] = normalized
    
    for c in c_values:
        log_result = KernelGenerator.chuyen_doi_logarit(laplacian_abs, c)
        results[f'log_c_{c}'] = log_result
        
        print(f"c = {c}:")
        print(f"  Range: [{np.min(log_result)}, {np.max(log_result)}]")
        print(f"  Mean: {np.mean(log_result):.2f}")
        print(f"  Std: {np.std(log_result):.2f}")
        
        # Đếm số pixel trong các range khác nhau
        dark_pixels = np.sum(log_result < 85)
        mid_pixels = np.sum((log_result >= 85) & (log_result < 170))
        bright_pixels = np.sum(log_result >= 170)
        
        print(f"  Phân bố: Tối={dark_pixels}, Vừa={mid_pixels}, Sáng={bright_pixels}")
    
    # Visualization
    create_comparison_plot(results)
    
    return results

def create_comparison_plot(results):
    """Tạo plot so sánh các kết quả"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    titles = [
        'Original Image',
        'Laplacian (Normal)',
        'Log Transform (c=0.5)',
        'Log Transform (c=1.0)',
        'Log Transform (c=2.0)',
        'Log Transform (c=5.0)',
        'Log Transform (c=10.0)',
        'Log Transform (c=50.0)',
        'Log Transform (c=100.0)'
    ]
    
    images = [
        results['original'],
        results['laplacian_normal'],
        results['log_c_0.5'],
        results['log_c_1.0'],
        results['log_c_2.0'],
        results['log_c_5.0'],
        results['log_c_10.0'],
        results['log_c_50.0'],
        results['log_c_100.0']
    ]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
        
        # Thêm thông tin thống kê
        axes[i].text(0.02, 0.98, f'Range: [{np.min(img)}, {np.max(img)}]', 
                    transform=axes[i].transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('d:/CODE/AI/XLAS/laplacian_log_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Đã lưu comparison plot: laplacian_log_comparison.png")

def analyze_edge_enhancement():
    """Phân tích hiệu quả enhancement của log transform"""
    print("\n" + "="*50)
    print("PHÂN TÍCH EDGE ENHANCEMENT")
    print("="*50)
    
    # Tạo ảnh với edges rõ ràng
    test_img = np.zeros((100, 100), dtype=np.uint8)
    
    # Tạo step edge (từ 50 lên 200)
    test_img[:, :50] = 50
    test_img[:, 50:] = 200
    
    # Áp dụng Laplacian
    laplacian_kernel = KernelGenerator.laplacian_kernel()
    laplacian_result = convolution_2d_manual(test_img, laplacian_kernel, padding=1, stride=1)
    laplacian_abs = np.abs(laplacian_result)
    
    # So sánh normal vs log transform
    normal = ((laplacian_abs / laplacian_abs.max()) * 255).astype(np.uint8)
    log_c1 = KernelGenerator.chuyen_doi_logarit(laplacian_abs, 1.0)
    log_c2 = KernelGenerator.chuyen_doi_logarit(laplacian_abs, 2.0)
    
    print("STEP EDGE TEST:")
    print(f"Original edge strength: {laplacian_abs.max():.2f}")
    print(f"Normal normalization max: {normal.max()}")
    print(f"Log transform (c=1.0) max: {log_c1.max()}")
    print(f"Log transform (c=2.0) max: {log_c2.max()}")
    
    # Tính contrast ratio
    def calculate_contrast(img):
        return (np.max(img) - np.min(img)) / (np.max(img) + np.min(img))
    
    print(f"\nCONTRAST RATIO:")
    print(f"Normal: {calculate_contrast(normal):.3f}")
    print(f"Log (c=1.0): {calculate_contrast(log_c1):.3f}")
    print(f"Log (c=2.0): {calculate_contrast(log_c2):.3f}")
    
    # Test với giá trị c cao
    log_c10 = KernelGenerator.chuyen_doi_logarit(laplacian_abs, 10.0)
    log_c100 = KernelGenerator.chuyen_doi_logarit(laplacian_abs, 100.0)
    
    print(f"Log (c=10.0): {calculate_contrast(log_c10):.3f}")
    print(f"Log (c=100.0): {calculate_contrast(log_c100):.3f}")
    
    print(f"\nHIỆU ỨNG CỦA GIÁ TRỊ C CAO:")
    print(f"c=10.0  - Max: {log_c10.max()}, Mean: {np.mean(log_c10):.1f}")
    print(f"c=100.0 - Max: {log_c100.max()}, Mean: {np.mean(log_c100):.1f}")


def test_extreme_c_values():
    """Test với các giá trị c cực cao"""
    print("\n" + "="*50)
    print("TEST GIÁ TRỊ C CỰC CAO")
    print("="*50)
    
    # Tạo ảnh test đơn giản
    test_img = np.zeros((50, 50), dtype=np.uint8)
    test_img[20:30, 20:30] = 100  # Square nhỏ
    
    # Áp dụng Laplacian
    laplacian_kernel = KernelGenerator.laplacian_kernel()
    laplacian_result = convolution_2d_manual(test_img, laplacian_kernel, padding=1, stride=1)
    laplacian_abs = np.abs(laplacian_result)
    
    extreme_c_values = [1, 10, 50, 100]
    
    print("HIỆU ỨNG CỦA CÁC GIÁ TRỊ C:")
    print(f"{'c':<5} {'Max':<5} {'Mean':<8} {'Std':<8} {'Unique':<8}")
    print("-" * 40)
    
    for c in extreme_c_values:
        result = KernelGenerator.chuyen_doi_logarit(laplacian_abs, c)
        print(f"{c:<5} {result.max():<5} {np.mean(result):<8.1f} {np.std(result):<8.1f} {len(np.unique(result)):<8}")
    
    print(f"\n💡 QUAN SÁT:")
    print(f"- c=1: Cân bằng tự nhiên")
    print(f"- c=10: Tăng cường mạnh, chi tiết rõ")
    print(f"- c=50: Hiệu ứng dramatic, có thể over-enhance")
    print(f"- c=100: Cực mạnh, có thể tạo artifacts")

def main():
    """Main function"""
    print("🔬 TESTING LAPLACIAN WITH LOG TRANSFORM")
    print("=" * 60)
    
    # Test chính
    results = test_laplacian_log_transform()
    
    # Phân tích edge enhancement
    analyze_edge_enhancement()
    
    # Test giá trị c cực cao
    test_extreme_c_values()
    
    print("\n" + "="*60)
    print("KẾT LUẬN")
    print("="*60)
    print("✅ Log Transform cho Laplacian:")
    print("  - Cải thiện hiển thị vùng tối")
    print("  - Tăng contrast cho edges yếu")
    print("  - Không làm cháy vùng sáng")
    print("  - c càng lớn càng tăng độ sáng vùng tối")
    print("\n🎯 Khuyến nghị:")
    print("  - c = 1.0: Cân bằng tốt, tự nhiên")
    print("  - c = 2.0-5.0: Tăng cường mạnh cho ảnh tối")
    print("  - c = 10.0-20.0: Hiệu ứng dramatic, chi tiết cực rõ")
    print("  - c = 50.0-100.0: Cực mạnh, artistic effect")
    print("  - c < 1.0: Giảm enhancement, giữ tự nhiên")
    print("\n⚠️  Lưu ý:")
    print("  - c > 20: Có thể tạo artifacts")
    print("  - c > 50: Chỉ dùng cho hiệu ứng đặc biệt")
    print("  - c = 100: Maximum dramatic effect")

if __name__ == "__main__":
    main()
