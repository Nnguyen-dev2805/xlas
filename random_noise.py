import streamlit as st
import cv2
import numpy as np
import albumentations as A
from io import BytesIO
from PIL import Image

# --- App title ---
st.set_page_config(page_title="Noise Generator", page_icon="🖼️")
st.title("🖼️ Image Noise Generator App")
st.write("Upload an image, apply noise, and download the result.")

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # Display original
    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # --- Noise options ---
    noise_options = {
        "Gaussian Noise": A.GaussNoise(var_limit=(10, 100), mean=0, p=1.0),
        "Salt & Pepper (ISO Noise)": A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
        "Poisson Noise (Multiplicative)": A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, p=1.0),
        "Speckle Noise": A.ISONoise(color_shift=(0.05, 0.2), intensity=(0.3, 0.5), p=1.0),
        "Motion Blur": A.MotionBlur(blur_limit=15, p=1.0),
        "Defocus Blur": A.Defocus(radius=(2, 6), alias_blur=(0.1, 0.5), p=1.0),
        "JPEG Compression": A.ImageCompression(quality_lower=10, quality_upper=30, p=1.0),
        "Pixel Dropout": A.PixelDropout(dropout_prob=0.1, per_channel=True, p=1.0),
        "Downscale (Low Resolution)": A.Downscale(scale_min=0.3, scale_max=0.7, p=1.0),
        "Random Brightness/Contrast": A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)
    }

    # --- Select noise type ---
    noise_name = st.selectbox("🧩 Chọn loại nhiễu bạn muốn thêm:", list(noise_options.keys()))
    transform = noise_options[noise_name]

    # --- Apply noise ---
    if st.button("✨ Tạo ảnh nhiễu"):
        augmented = transform(image=image)
        noisy_image = augmented["image"]

        # Hiển thị ảnh nhiễu
        st.image(noisy_image, caption=f"Ảnh sau khi thêm {noise_name}", use_column_width=True)

        # --- Download button ---
        result = Image.fromarray(noisy_image)
        buf = BytesIO()
        result.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Tải ảnh nhiễu v   ề",
            data=byte_im,
            file_name=f"noisy_{noise_name.replace(' ', '_')}.jpg",
            mime="image/jpeg"
        )

else:
    st.info("👆 Hãy upload một ảnh để bắt đầu!")

# --- Footer ---
st.markdown("---")
st.markdown("🧠 *Tạo bởi ChatGPT – Noise Generator Demo*")
