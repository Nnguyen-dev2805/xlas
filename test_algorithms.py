"""
Test Script cho các thuật toán xử lý ảnh
=====================================

Script này để test các module một cách độc lập,
không cần GUI Streamlit.

Usage: python test_algorithms.py
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import modules tự tạo
from src.utils import rgb_to_grayscale, create_kernel, calculate_image_stats
from src.histogram import process_task1, analyze_histogram_properties
from src.filtering import process_task2, analyze_filter_effects


def create_test_image():
    """Tạo ảnh test đơn giản"""
    # Tạo ảnh gradient
    height, width = 200, 200
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            image[i, j] = [i % 256, j % 256, (i + j) % 256]
    
    return image


def test_utils():
    """Test module utils.py"""
    print("🔧 Testing Utils Module...")
    
    # Tạo ảnh test
    rgb_image = create_test_image()
    print(f"✓ Created test image: {rgb_image.shape}")
    
    # Test RGB to Grayscale
    gray_image = rgb_to_grayscale(rgb_image)
    print(f"✓ RGB to Grayscale: {gray_image.shape}")
    
    # Test create kernel
    kernel_3x3 = create_kernel(3, 'average')
    kernel_5x5 = create_kernel(5, 'gaussian')
    print(f"✓ Created kernels: 3x3 {kernel_3x3.shape}, 5x5 {kernel_5x5.shape}")
    
    # Test image stats
    stats = calculate_image_stats(gray_image)
    print(f"✓ Image stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    return rgb_image, gray_image


def test_histogram(gray_image):
    """Test module histogram.py"""
    print("\n📈 Testing Histogram Module...")
    
    try:
        # Xử lý Bài 1
        results = process_task1(gray_image)
        print(f"✓ Task 1 completed successfully")
        
        # Phân tích histograms
        h1_analysis = analyze_histogram_properties(results['h1'])
        h2_analysis = analyze_histogram_properties(results['h2'])
        
        print(f"✓ H1 entropy: {h1_analysis['entropy']:.2f}")
        print(f"✓ H2 entropy: {h2_analysis['entropy']:.2f}")
        print(f"✓ Narrowed range: [{np.min(results['narrowed_image'])}, {np.max(results['narrowed_image'])}]")
        
        return results
        
    except Exception as e:
        print(f"❌ Histogram test failed: {e}")
        return None


def test_filtering(gray_image):
    """Test module filtering.py"""
    print("\n🔧 Testing Filtering Module...")
    
    try:
        # Xử lý Bài 2
        results = process_task2(gray_image)
        print(f"✓ Task 2 completed successfully")
        
        # Phân tích filter effects
        original = results['original_image']
        
        for key in ['i1', 'i2', 'i3', 'i4', 'i5']:
            if key in results:
                analysis = analyze_filter_effects(original, results[key], key.upper())
                print(f"✓ {key.upper()}: PSNR={analysis['psnr']:.2f}dB, Correlation={analysis['correlation']:.4f}")
        
        print(f"✓ Final I6 shape: {results['i6'].shape}")
        
        return results
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")
        return None


def visualize_results(hist_results, filt_results):
    """Tạo visualization cho kết quả"""
    print("\n📊 Creating visualizations...")
    
    try:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Row 1: Histogram results
        if hist_results:
            axes[0, 0].imshow(hist_results['original_image'], cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(hist_results['h2_image'], cmap='gray')
            axes[0, 1].set_title('H2 Equalized')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(hist_results['narrowed_image'], cmap='gray')
            axes[0, 2].set_title('H3 Narrowed')
            axes[0, 2].axis('off')
            
            axes[0, 3].plot(hist_results['h1'], color='blue', alpha=0.7)
            axes[0, 3].plot(hist_results['h2'], color='green', alpha=0.7)
            axes[0, 3].set_title('Histograms')
            axes[0, 3].legend(['H1', 'H2'])
        
        # Row 2: Filtering results I1-I3
        if filt_results:
            for i, key in enumerate(['i1', 'i2', 'i3']):
                if i < 3 and key in filt_results:
                    axes[1, i].imshow(filt_results[key], cmap='gray')
                    axes[1, i].set_title(f'{key.upper()}')
                    axes[1, i].axis('off')
            
            # Kernel visualization
            if 'kernel_3x3' in filt_results:
                im = axes[1, 3].imshow(filt_results['kernel_3x3'], cmap='viridis')
                axes[1, 3].set_title('Kernel 3x3')
                plt.colorbar(im, ax=axes[1, 3])
        
        # Row 3: Filtering results I4-I6
        if filt_results:
            for i, key in enumerate(['i4', 'i5', 'i6']):
                if i < 3 and key in filt_results:
                    axes[2, i].imshow(filt_results[key], cmap='gray')
                    axes[2, i].set_title(f'{key.upper()}')
                    axes[2, i].axis('off')
            
            # Stats
            axes[2, 3].text(0.1, 0.8, f"Original shape: {filt_results['original_image'].shape}", transform=axes[2, 3].transAxes)
            axes[2, 3].text(0.1, 0.6, f"I3 shape: {filt_results['i3'].shape}", transform=axes[2, 3].transAxes)
            axes[2, 3].text(0.1, 0.4, f"I6 shape: {filt_results['i6'].shape}", transform=axes[2, 3].transAxes)
            axes[2, 3].set_title('Info')
            axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        # Lưu kết quả
        os.makedirs('data/output', exist_ok=True)
        plt.savefig('data/output/test_results.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to data/output/test_results.png")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


def performance_test(gray_image):
    """Test performance với ảnh kích thước khác nhau"""
    print("\n⚡ Performance Testing...")
    
    import time
    
    sizes = [(100, 100), (200, 200), (400, 400)]
    
    for size in sizes:
        # Resize ảnh test
        pil_img = Image.fromarray(gray_image)
        resized = pil_img.resize(size)
        test_img = np.array(resized)
        
        # Test Histogram
        start_time = time.time()
        hist_results = process_task1(test_img)
        hist_time = time.time() - start_time
        
        # Test Filtering  
        start_time = time.time()
        filt_results = process_task2(test_img)
        filt_time = time.time() - start_time
        
        print(f"✓ Size {size}: Histogram={hist_time:.3f}s, Filtering={filt_time:.3f}s")


def main():
    """Main test function"""
    print("🚀 Starting Algorithm Tests...")
    print("=" * 50)
    
    # Test 1: Utils
    rgb_image, gray_image = test_utils()
    
    # Test 2: Histogram
    hist_results = test_histogram(gray_image)
    
    # Test 3: Filtering
    filt_results = test_filtering(gray_image)
    
    # Test 4: Visualization
    if hist_results and filt_results:
        visualize_results(hist_results, filt_results)
    
    # Test 5: Performance
    performance_test(gray_image)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\nNext steps:")
    print("1. Check data/output/test_results.png")
    print("2. Run 'streamlit run app.py' for GUI")
    print("3. Upload real images for testing")


if __name__ == "__main__":
    main()
