import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from core.image_ops import rgb_to_grayscale_manual
from filters.sobel_kernel import SobelKernel, SobelProcessor, SobelLibrary, SobelComparison
from filters.noise_generator import NoiseGenerator, NoiseTestSuite


def setup_page():
    st.set_page_config(
        page_title="Sobel Edge Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(90deg, #e3f2fd, #ffffff);
        border-left: 5px solid #1f77b4;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #32cd32;
        margin: 1rem 0;
    }
    .kernel-display {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown("## Cấu Hình Sobel")
    
    # Upload ảnh
    st.sidebar.markdown("### Upload Ảnh")
    uploaded_file = st.sidebar.file_uploader(
        "Chọn ảnh màu",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload ảnh < 5MB"
    )
    
    use_sample = st.sidebar.checkbox("Dùng ảnh mẫu", value=(uploaded_file is None))
    
    st.sidebar.markdown("---")
    
    # Noise/Blur options
    st.sidebar.markdown("### Thêm Nhiễu/Mờ (Optional)")
    
    apply_noise = st.sidebar.checkbox("Áp dụng nhiễu/mờ", value=False, 
                                      help="Test Sobel với ảnh bị nhiễu")
    
    noise_config = None
    if apply_noise:
        noise_type = st.sidebar.selectbox(
            "Loại nhiễu:",
            [
                "Không có",
                "Salt & Pepper (Muối Tiêu)",
                "Gaussian Noise",
                "Poisson Noise",
                "Speckle Noise",
                "Motion Blur",
                "Gaussian Blur",
                "Defocus Blur",
                "Mixed (Gaussian + S&P)"
            ],
            help="Chọn loại nhiễu để thêm vào ảnh"
        )
        
        noise_config = {'type': noise_type}
        
        # Parameters cho từng loại
        if noise_type == "Salt & Pepper (Muối Tiêu)":
            intensity = st.sidebar.select_slider(
                "Độ mạnh:",
                options=["Light", "Medium", "Heavy"],
                value="Medium"
            )
            if intensity == "Light":
                noise_config['salt_prob'] = 0.005
                noise_config['pepper_prob'] = 0.005
            elif intensity == "Medium":
                noise_config['salt_prob'] = 0.02
                noise_config['pepper_prob'] = 0.02
            else:
                noise_config['salt_prob'] = 0.05
                noise_config['pepper_prob'] = 0.05
        
        elif noise_type == "Gaussian Noise":
            std = st.sidebar.slider("Standard Deviation:", 5, 80, 25, 5)
            noise_config['std'] = std
        
        elif noise_type == "Speckle Noise":
            std = st.sidebar.slider("Speckle Std:", 0.05, 0.5, 0.15, 0.05)
            noise_config['std'] = std
        
        elif noise_type == "Motion Blur":
            kernel_size = st.sidebar.slider("Kernel Size:", 5, 25, 15, 2)
            angle = st.sidebar.slider("Angle (degrees):", 0, 180, 45, 15)
            noise_config['kernel_size'] = kernel_size
            noise_config['angle'] = angle
        
        elif noise_type == "Gaussian Blur":
            kernel_size = st.sidebar.slider("Kernel Size:", 3, 21, 9, 2)
            sigma = st.sidebar.slider("Sigma:", 0.5, 5.0, 2.0, 0.5)
            noise_config['kernel_size'] = kernel_size
            noise_config['sigma'] = sigma
        
        elif noise_type == "Defocus Blur":
            radius = st.sidebar.slider("Radius:", 2, 15, 5, 1)
            noise_config['radius'] = radius
    
    st.sidebar.markdown("---")
    
    # Implementation method
    st.sidebar.markdown("### Implementation")
    implementation = st.sidebar.radio(
        "Chọn implementation:",
        ["Manual (Tự code)", "OpenCV (Thư viện)", "So Sánh (Manual vs OpenCV)"],
        help="Manual: Tự code | OpenCV: Dùng thư viện | So Sánh: Xem sự khác biệt"
    )
    
    st.sidebar.markdown("---")
    
    # Mode selection
    st.sidebar.markdown("### 🎯 Chế Độ")
    mode = st.sidebar.radio(
        "Chọn chế độ:",
        ["Single Config", "Multi Config (I1, I2, I3)", "Compare Sigma"],
        help="Single: Test một config | Multi: Theo yêu cầu bài tập | Compare: So sánh sigma"
    )
    
    st.sidebar.markdown("---")
    
    if mode == "Single Config":
        # Single config parameters
        st.sidebar.markdown("### ⚙️ Tham Số Kernel")
        
        kernel_size = st.sidebar.select_slider(
            "Kernel Size",
            options=[3, 5, 7, 9, 11],
            value=3,
            help="Kích thước kernel (lẻ)"
        )
        
        sigma = st.sidebar.slider(
            "Sigma (Gaussian)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Sigma càng lớn → edges càng mượt"
        )
        
        st.sidebar.markdown("### 🔧 Tham Số Convolution")
        
        padding = st.sidebar.number_input(
            "Padding",
            min_value=0,
            max_value=10,
            value=kernel_size // 2,
            help="Padding để giữ nguyên kích thước output"
        )
        
        stride = st.sidebar.selectbox(
            "Stride",
            options=[1, 2, 3],
            index=0,
            help="Stride = 2 → downsampling 50%"
        )
        
        config = {
            'size': kernel_size,
            'sigma': sigma,
            'padding': padding,
            'stride': stride,
            'name': f'Sobel_{kernel_size}x{kernel_size}'
        }
        
        return uploaded_file, use_sample, mode, [config], noise_config, implementation
    
    elif mode == "Multi Config (I1, I2, I3)":
        # Standard configs theo bài tập
        st.sidebar.markdown("### Configs Chuẩn")
        st.sidebar.info("""
        **I1:** 3×3, pad=1, stride=1
        **I2:** 5×5, pad=2, stride=1
        **I3:** 7×7, pad=3, stride=2
        """)
        
        # Allow customize sigma
        st.sidebar.markdown("### Tùy Chỉnh Sigma")
        
        sigma_3x3 = st.sidebar.slider("Sigma cho 3×3", 0.1, 2.0, 0.8, 0.1)
        sigma_5x5 = st.sidebar.slider("Sigma cho 5×5", 0.1, 2.0, 1.2, 0.1)
        sigma_7x7 = st.sidebar.slider("Sigma cho 7×7", 0.1, 3.0, 1.5, 0.1)
        
        configs = [
            {'size': 3, 'sigma': sigma_3x3, 'padding': 1, 'stride': 1, 'name': 'I1_Sobel_3x3'},
            {'size': 5, 'sigma': sigma_5x5, 'padding': 2, 'stride': 1, 'name': 'I2_Sobel_5x5'},
            {'size': 7, 'sigma': sigma_7x7, 'padding': 3, 'stride': 2, 'name': 'I3_Sobel_7x7'}
        ]
        
        return uploaded_file, use_sample, mode, configs, noise_config, implementation
    
    else:  # Compare Sigma
        st.sidebar.markdown("### Compare Sigma Values")
        
        kernel_size = st.sidebar.select_slider(
            "Kernel Size",
            options=[3, 5, 7],
            value=5
        )
        
        sigma_low = st.sidebar.slider("Sigma Low", 0.1, 2.0, 0.5, 0.1)
        sigma_mid = st.sidebar.slider("Sigma Mid", 0.1, 3.0, 1.0, 0.1)
        sigma_high = st.sidebar.slider("Sigma High", 0.1, 3.0, 2.0, 0.1)
        
        configs = [
            {'size': kernel_size, 'sigma': sigma_low, 'padding': kernel_size//2, 'stride': 1, 
             'name': f'Low_σ={sigma_low}'},
            {'size': kernel_size, 'sigma': sigma_mid, 'padding': kernel_size//2, 'stride': 1, 
             'name': f'Mid_σ={sigma_mid}'},
            {'size': kernel_size, 'sigma': sigma_high, 'padding': kernel_size//2, 'stride': 1, 
             'name': f'High_σ={sigma_high}'}
        ]
        
        return uploaded_file, use_sample, mode, configs, noise_config, implementation


def create_sample_image():
    """Tạo ảnh mẫu có edges rõ ràng"""
    size = 400
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(size):
        image[i, :] = [int(128 + 127 * np.sin(i/50)), 150, 170]
    
    # White rectangle
    image[80:180, 80:180] = [255, 255, 255]
    
    # Dark circle
    center = (size//2 + 50, size//2 + 50)
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if dist < 60:
                image[i, j] = [50, 50, 50]
    
    # Horizontal and vertical lines
    image[280:290, :] = [200, 100, 100]
    image[:, 280:290] = [100, 200, 100]
    
    # Diagonal line
    for i in range(size):
        if 0 <= i < size and 0 <= i < size:
            image[i, i] = [100, 100, 200]
    
    return image


def display_kernel_info(kernel, name):
    """Hiển thị thông tin kernel"""
    st.markdown(f"**{name}**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Thống kê:**")
        st.write(f"- Size: {kernel.shape}")
        st.write(f"- Sum: {np.sum(kernel):.6f}")
        st.write(f"- Min: {np.min(kernel):.6f}")
        st.write(f"- Max: {np.max(kernel):.6f}")
    
    with col2:
        # Kernel heatmap
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(kernel, cmap='RdBu_r', vmin=-kernel.max(), vmax=kernel.max())
        ax.set_title(f'{name}')
        
        # Add values
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                text = ax.text(j, i, f'{kernel[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close()


def main():
    """Main function"""
    setup_page()
    
    # Header
    st.markdown('<h1 class="main-title">🔍 Sobel Edge Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Về Sobel Edge Detection</h3>
    <p>Sobel operator phát hiện cạnh bằng cách tính <strong>gradient</strong> của ảnh.</p>
    <ul>
    <li><strong>Sobel X:</strong> Phát hiện cạnh dọc (gradient theo X)</li>
    <li><strong>Sobel Y:</strong> Phát hiện cạnh ngang (gradient theo Y)</li>
    <li><strong>Magnitude:</strong> G = √(Gx² + Gy²)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    uploaded_file, use_sample, mode, configs, noise_config, implementation = render_sidebar()
    
    # Load image
    st.markdown('<div class="section-header">Ảnh Input</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None and not use_sample:
        try:
            image = Image.open(uploaded_file)
            
            # Resize nếu quá lớn
            if image.width > 1000 or image.height > 1000:
                st.info("Ảnh lớn, đang resize...")
                image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            rgb_array = np.array(image)
            st.success(f"✓ Đã load ảnh: {rgb_array.shape}")
        except Exception as e:
            st.error(f"Lỗi khi load ảnh: {e}")
            rgb_array = create_sample_image()
    else:
        rgb_array = create_sample_image()
        st.info("Đang sử dụng ảnh mẫu. Upload ảnh của bạn ở sidebar!")
    
    # Display original
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(rgb_array, caption=f"Ảnh gốc - {rgb_array.shape}", use_container_width=True)
    with col2:
        st.markdown("**Thông tin ảnh:**")
        st.json({
            'Shape': f"{rgb_array.shape}",
            'Size': f"{rgb_array.size:,} pixels",
            'Type': str(rgb_array.dtype)
        })
    
    # Convert to grayscale
    st.markdown('<div class="section-header">Chuyển Sang Grayscale</div>', unsafe_allow_html=True)
    
    with st.spinner("Đang chuyển đổi..."):
        gray_image = rgb_to_grayscale_manual(rgb_array)
    
    # Apply noise/blur if selected
    if noise_config and noise_config['type'] != "Không có":
        st.markdown('<div class="section-header">Thêm Nhiễu/Mờ</div>', unsafe_allow_html=True)
        
        noise_type = noise_config['type']
        
        with st.spinner(f"Đang áp dụng {noise_type}..."):
            if noise_type == "Salt & Pepper (Muối Tiêu)":
                gray_image_noisy = NoiseGenerator.salt_and_pepper(
                    gray_image, 
                    salt_prob=noise_config['salt_prob'],
                    pepper_prob=noise_config['pepper_prob']
                )
            elif noise_type == "Gaussian Noise":
                gray_image_noisy = NoiseGenerator.gaussian_noise(
                    gray_image,
                    std=noise_config['std']
                )
            elif noise_type == "Poisson Noise":
                gray_image_noisy = NoiseGenerator.poisson_noise(gray_image)
            elif noise_type == "Speckle Noise":
                gray_image_noisy = NoiseGenerator.speckle_noise(
                    gray_image,
                    std=noise_config['std']
                )
            elif noise_type == "Motion Blur":
                gray_image_noisy = NoiseGenerator.motion_blur(
                    gray_image,
                    kernel_size=noise_config['kernel_size'],
                    angle=noise_config['angle']
                )
            elif noise_type == "Gaussian Blur":
                gray_image_noisy = NoiseGenerator.gaussian_blur(
                    gray_image,
                    kernel_size=noise_config['kernel_size'],
                    sigma=noise_config['sigma']
                )
            elif noise_type == "Defocus Blur":
                gray_image_noisy = NoiseGenerator.defocus_blur(
                    gray_image,
                    radius=noise_config['radius']
                )
            elif noise_type == "Mixed (Gaussian + S&P)":
                gray_image_noisy = NoiseGenerator.mixed_noise(
                    gray_image,
                    noise_types=['gaussian', 'salt_pepper'],
                    intensities={'gaussian_std': 15, 'salt_prob': 0.01, 'pepper_prob': 0.01}
                )
            else:
                gray_image_noisy = gray_image
        
        st.success(f"✓ Đã áp dụng {noise_type}")
        
        # Show comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_image, caption="Grayscale gốc", use_container_width=True, clamp=True)
        with col2:
            st.image(gray_image_noisy, caption=f"Với {noise_type}", use_container_width=True, clamp=True)
        
        # Use noisy image for edge detection
        gray_image = gray_image_noisy
    else:
        st.image(gray_image, caption="Ảnh Grayscale", use_container_width=True, clamp=True)
    
    # Apply Sobel
    st.markdown('<div class="section-header">🔧 Edge Detection</div>', unsafe_allow_html=True)
    
    st.info(f"**Implementation:** {implementation} | **Chế độ:** {mode} | **Số configs:** {len(configs)}")
    
    # Process based on implementation choice
    if implementation == "Manual (Tự code)":
        with st.spinner("Đang xử lý với Sobel Manual..."):
            results = SobelProcessor.process_with_multiple_kernels(gray_image, configs)
        st.success(f"✓ Hoàn thành! Đã xử lý {len(results)} configs với Manual implementation")
    
    elif implementation == "OpenCV (Thư viện)":
        with st.spinner("Đang xử lý với OpenCV..."):
            results = {}
            for config in configs:
                size = config.get('size', 3)
                name = config.get('name', f'Sobel_{size}x{size}')
                
                magnitude, gx, gy = SobelLibrary.sobel_edge_detection_opencv(
                    gray_image, kernel_size=size, normalize=True
                )
                
                results[name] = {
                    'magnitude': magnitude,
                    'magnitude_uint8': magnitude,
                    'gx': gx,
                    'gy': gy,
                    'gx_uint8': gx,
                    'gy_uint8': gy,
                    'direction': None,
                    'config': config,
                    'kernel_x': None,
                    'kernel_y': None
                }
        st.success(f"✓ Hoàn thành! Đã xử lý {len(results)} configs với OpenCV")
    
    elif implementation == "So Sánh (Manual vs OpenCV)":
        with st.spinner("Đang so sánh Manual vs OpenCV..."):
            # Chỉ dùng config đầu tiên để so sánh
            config = configs[0]
            size = config.get('size', 3)
            sigma = config.get('sigma', 1.0)
            
            comparison_results = SobelComparison.compare_manual_vs_library(
                gray_image, kernel_size=size, sigma=sigma
            )
            
            # Convert to results format
            results = {}
            results['Manual'] = {
                'magnitude_uint8': comparison_results['manual']['magnitude'],
                'gx_uint8': comparison_results['manual']['gx'],
                'gy_uint8': comparison_results['manual']['gy'],
                'magnitude': comparison_results['manual']['magnitude'].astype(np.float32),
                'config': config
            }
            results['OpenCV'] = {
                'magnitude_uint8': comparison_results['library']['magnitude'],
                'gx_uint8': comparison_results['library']['gx'],
                'gy_uint8': comparison_results['library']['gy'],
                'magnitude': comparison_results['library']['magnitude'].astype(np.float32),
                'config': config
            }
            results['Difference'] = {
                'magnitude_uint8': comparison_results['difference']['magnitude'],
                'magnitude': comparison_results['difference']['magnitude'].astype(np.float32),
                'config': config
            }
        
        st.success(f"✓ Hoàn thành! Mean diff: {comparison_results['difference']['mean_diff']:.2f}, Max diff: {comparison_results['difference']['max_diff']:.2f}")
    
    # Display results
    if len(results) == 1:
        # Single result - detailed view
        st.markdown('<div class="section-header">Kết Quả</div>', unsafe_allow_html=True)
        
        name, result = list(results.items())[0]
        config = result['config']
        
        # Tabs cho các views khác nhau
        tab1, tab2, tab3, tab4 = st.tabs(["Magnitude", "Gradient X", "Gradient Y", "Kernels"])
        
        with tab1:
            st.markdown("### Gradient Magnitude (Kết quả chính)")
            st.image(result['magnitude_uint8'], caption=f"{name} - {result['magnitude'].shape}", 
                    use_container_width=True, clamp=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min", f"{result['magnitude'].min():.2f}")
            with col2:
                st.metric("Max", f"{result['magnitude'].max():.2f}")
            with col3:
                st.metric("Mean", f"{result['magnitude'].mean():.2f}")
            
            # Histogram
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(result['magnitude'].flatten(), bins=50, color='steelblue', alpha=0.7)
            ax.set_xlabel('Gradient Magnitude')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Gradient Magnitude')
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            st.markdown("### Gradient X (Cạnh dọc)")
            st.image(result['gx_uint8'], caption="Sobel X Result", 
                    use_container_width=True, clamp=True)
        
        with tab3:
            st.markdown("### Gradient Y (Cạnh ngang)")
            st.image(result['gy_uint8'], caption="Sobel Y Result", 
                    use_container_width=True, clamp=True)
        
        with tab4:
            st.markdown("### Sobel Kernels Used")
            col1, col2 = st.columns(2)
            with col1:
                display_kernel_info(result['kernel_x'], f"Sobel X {config['size']}×{config['size']}")
            with col2:
                display_kernel_info(result['kernel_y'], f"Sobel Y {config['size']}×{config['size']}")
        
        # Download
        st.markdown("### 💾 Download")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            buf = io.BytesIO()
            Image.fromarray(result['magnitude_uint8']).save(buf, format='PNG')
            st.download_button("📥 Magnitude", buf.getvalue(), f"{name}_magnitude.png", "image/png")
        
        with col2:
            buf = io.BytesIO()
            Image.fromarray(result['gx_uint8']).save(buf, format='PNG')
            st.download_button("📥 Gradient X", buf.getvalue(), f"{name}_gx.png", "image/png")
        
        with col3:
            buf = io.BytesIO()
            Image.fromarray(result['gy_uint8']).save(buf, format='PNG')
            st.download_button("📥 Gradient Y", buf.getvalue(), f"{name}_gy.png", "image/png")
    
    else:
        # Multiple results - comparison view
        st.markdown('<div class="section-header">📊 So Sánh Kết Quả</div>', unsafe_allow_html=True)
        
        # Display in grid
        n_results = len(results)
        cols_per_row = min(3, n_results)
        
        # Magnitude comparison
        st.markdown("### Gradient Magnitude")
        cols = st.columns(cols_per_row)
        for idx, (name, result) in enumerate(results.items()):
            with cols[idx % cols_per_row]:
                st.image(result['magnitude_uint8'], caption=f"{name}\n{result['magnitude'].shape}", 
                        use_container_width=True, clamp=True)
                st.caption(f"Range: [{result['magnitude'].min():.1f}, {result['magnitude'].max():.1f}]")
        
        # Gradient X comparison
        with st.expander("👁️ Xem Gradient X", expanded=False):
            cols = st.columns(cols_per_row)
            for idx, (name, result) in enumerate(results.items()):
                with cols[idx % cols_per_row]:
                    st.image(result['gx_uint8'], caption=f"{name} - Gx", 
                            use_container_width=True, clamp=True)
        
        # Gradient Y comparison
        with st.expander("👁️ Xem Gradient Y", expanded=False):
            cols = st.columns(cols_per_row)
            for idx, (name, result) in enumerate(results.items()):
                with cols[idx % cols_per_row]:
                    st.image(result['gy_uint8'], caption=f"{name} - Gy", 
                            use_container_width=True, clamp=True)
        
        # Statistics table
        st.markdown("### 📈 Bảng Thống Kê")
        stats_data = []
        for name, result in results.items():
            stats_data.append({
                'Config': name,
                'Output Shape': f"{result['magnitude'].shape}",
                'Min': f"{result['magnitude'].min():.2f}",
                'Max': f"{result['magnitude'].max():.2f}",
                'Mean': f"{result['magnitude'].mean():.2f}",
                'Std': f"{result['magnitude'].std():.2f}"
            })
        
        import pandas as pd
        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True)
        
        # Download all
        st.markdown("### Download Tất Cả")
        cols = st.columns(len(results))
        for idx, (name, result) in enumerate(results.items()):
            with cols[idx]:
                buf = io.BytesIO()
                Image.fromarray(result['magnitude_uint8']).save(buf, format='PNG')
                st.download_button(f"📥 {name}", buf.getvalue(), f"{name}.png", "image/png")
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <h4>💡 Tips:</h4>
    <ul>
    <li><strong>Sigma nhỏ (0.5-1.0):</strong> Edges sắc nét, chi tiết cao</li>
    <li><strong>Sigma lớn (1.5-3.0):</strong> Edges mượt, giảm nhiễu</li>
    <li><strong>Stride=2:</strong> Downsampling 50%, xử lý nhanh hơn</li>
    <li><strong>Padding=kernel_size//2:</strong> Giữ nguyên kích thước output</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
