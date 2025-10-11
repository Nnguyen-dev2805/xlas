"""
EXAMPLE: Sử dụng Sobel API
===========================

File này minh họa cách sử dụng Sobel module trong code Python.
"""

import numpy as np
from PIL import Image
from filters.sobel_kernel import SobelKernel, SobelProcessor
from core.image_ops import rgb_to_grayscale_manual


def example_1_basic_usage():
    """
    Example 1: Sử dụng cơ bản - edge detection với một ảnh
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Sobel Edge Detection")
    print("="*70)
    
    # 1. Load ảnh
    print("\n[1] Loading image...")
    # Thay đổi path này thành ảnh của bạn
    image_path = "path/to/your/image.jpg"
    
    # Hoặc tạo ảnh test
    print("    Creating test image...")
    test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    test_image[50:150, 50:150] = [255, 255, 255]  # White square
    
    # 2. Chuyển sang grayscale
    print("[2] Converting to grayscale...")
    gray_image = rgb_to_grayscale_manual(test_image)
    
    # 3. Áp dụng Sobel
    print("[3] Applying Sobel edge detection...")
    magnitude = SobelKernel.sobel_edge_detection(
        gray_image, 
        kernel_size=3, 
        sigma=1.0
    )
    
    print(f"    Output shape: {magnitude.shape}")
    print(f"    Range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    
    # 4. Lưu kết quả
    print("[4] Saving result...")
    magnitude_uint8 = SobelKernel.normalize_to_uint8(magnitude)
    Image.fromarray(magnitude_uint8).save('example_output_basic.png')
    print("    ✓ Saved as: example_output_basic.png")


def example_2_multiple_kernels():
    """
    Example 2: Xử lý với nhiều kernel sizes (theo yêu cầu bài tập)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Kernel Sizes (Assignment Requirements)")
    print("="*70)
    
    # 1. Load và convert
    print("\n[1] Preparing image...")
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Add some patterns
    test_image[60:100, 60:100] = [200, 200, 200]
    test_image[150:200, 150:200] = [100, 100, 100]
    
    gray_image = rgb_to_grayscale_manual(test_image)
    
    # 2. Tạo configs theo yêu cầu
    print("[2] Creating standard configs...")
    configs = SobelProcessor.create_standard_configs()
    
    for config in configs:
        print(f"    • {config['name']}: {config['size']}x{config['size']}, "
              f"padding={config['padding']}, stride={config['stride']}")
    
    # 3. Process
    print("[3] Processing with all configs...")
    results = SobelProcessor.process_with_multiple_kernels(gray_image, configs)
    
    # 4. Hiển thị và lưu kết quả
    print("[4] Results:")
    for name, result in results.items():
        print(f"\n    {name}:")
        print(f"      Output shape: {result['magnitude'].shape}")
        print(f"      Range: [{result['magnitude'].min():.2f}, "
              f"{result['magnitude'].max():.2f}]")
        
        # Lưu
        Image.fromarray(result['magnitude_uint8']).save(f'example_{name}.png')
        print(f"      ✓ Saved as: example_{name}.png")


def example_3_custom_config():
    """
    Example 3: Custom configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Configuration")
    print("="*70)
    
    # Tạo ảnh test
    print("\n[1] Creating test image...")
    gray_image = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    
    # Add edges
    gray_image[100:200, 100:200] = 255
    gray_image[140:160, :] = 0
    
    # Custom config
    print("[2] Creating custom config...")
    custom_configs = [
        {
            'size': 5,
            'sigma': 0.8,      # Sigma nhỏ → edges sắc nét
            'padding': 2,
            'stride': 1,
            'name': 'Sharp_5x5'
        },
        {
            'size': 7,
            'sigma': 2.5,      # Sigma lớn → edges mượt
            'padding': 3,
            'stride': 1,
            'name': 'Smooth_7x7'
        }
    ]
    
    # Process
    print("[3] Processing...")
    results = SobelProcessor.process_with_multiple_kernels(gray_image, custom_configs)
    
    # Kết quả
    print("[4] Results:")
    for name, result in results.items():
        print(f"    {name}: shape={result['magnitude'].shape}")
        Image.fromarray(result['magnitude_uint8']).save(f'example_{name}.png')


def example_4_access_components():
    """
    Example 4: Truy cập Gradient X, Y components và direction
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Access Gradient Components")
    print("="*70)
    
    # Tạo ảnh
    print("\n[1] Creating test image...")
    gray_image = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    
    # Sobel với return components
    print("[2] Applying Sobel with components...")
    magnitude, gx, gy, direction = SobelKernel.sobel_edge_detection(
        gray_image,
        kernel_size=3,
        sigma=1.0,
        return_components=True
    )
    
    print("\n[3] Results:")
    print(f"    Magnitude shape: {magnitude.shape}")
    print(f"    Gx shape: {gx.shape}")
    print(f"    Gy shape: {gy.shape}")
    print(f"    Direction shape: {direction.shape}")
    print(f"    Direction range: [{direction.min():.2f}, {direction.max():.2f}] radians")
    
    # Lưu components
    print("\n[4] Saving components...")
    Image.fromarray(SobelKernel.normalize_to_uint8(magnitude)).save('example_magnitude.png')
    Image.fromarray(SobelKernel.normalize_to_uint8(np.abs(gx))).save('example_gx.png')
    Image.fromarray(SobelKernel.normalize_to_uint8(np.abs(gy))).save('example_gy.png')
    print("    ✓ Saved magnitude, gx, gy")


def example_5_kernel_inspection():
    """
    Example 5: Kiểm tra kernel values
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Inspect Kernel Values")
    print("="*70)
    
    # Tạo kernels với sizes khác nhau
    print("\n[1] Creating Sobel kernels...")
    
    sizes = [3, 5, 7]
    for size in sizes:
        print(f"\n    Sobel X {size}x{size} (sigma=1.0):")
        kernel = SobelKernel.create_sobel_x_kernel(size, sigma=1.0)
        print(kernel)
        print(f"    Sum: {np.sum(kernel):.6f}")
        print(f"    Range: [{kernel.min():.6f}, {kernel.max():.6f}]")


def main():
    """
    Chạy tất cả examples
    """
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "SOBEL USAGE EXAMPLES" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Uncomment example bạn muốn chạy:
        
        example_1_basic_usage()
        # example_2_multiple_kernels()
        # example_3_custom_config()
        # example_4_access_components()
        # example_5_kernel_inspection()
        
        print("\n" + "="*70)
        print("✓ All examples completed!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
