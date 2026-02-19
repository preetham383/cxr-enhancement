import streamlit as st
import cv2
import numpy as np

from features import extract_features
from model import load_model

# EXPLICIT imports (Pylance-safe)
from enhance import (
    histogram_equalization,
    apply_clahe,
    gamma_correction,
    bilateral_filter,
    clahe_gamma,
    enhance_image,
    normalize_image,
    compute_psnr,
    compute_ssim
)

st.set_page_config(page_title="CXR Enhancement App", layout="wide")
st.title("Chest X-ray Enhancement – Grid Comparison with Metrics")

uploaded = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded:
    img = cv2.imdecode(
        np.frombuffer(uploaded.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )
    img = cv2.resize(img, (512, 512))

    st.subheader("Grid Comparison (All Methods)")

    # Generate enhanced images
    he_img = normalize_image(histogram_equalization(img))
    clahe_img = normalize_image(apply_clahe(img))
    gamma_img = normalize_image(gamma_correction(img))
    bilateral_img = normalize_image(bilateral_filter(img))
    clahe_gamma_img = normalize_image(clahe_gamma(img))

    # Our Method
    model = load_model()
    features = extract_features(img).reshape(1, -1)
    params = model.predict(features)[0]
    our_img = normalize_image(enhance_image(img, params))

    # Show parameters
    st.subheader("Predicted Parameters (Our Method)")
    names = [
        "CLAHE Clip Limit",
        "Gamma Value",
        "Laplacian μ1 (High freq)",
        "Laplacian μ2 (Mid freq)",
        "Laplacian μ3 (Low freq)"
    ]

    for n, v in zip(names, params):
        st.write(f"**{n}:** {v:.3f}")

    # Grid
    cols = st.columns(3)
    cols2 = st.columns(3)

    def show(col, image, title):
        psnr_val = compute_psnr(img, image)
        ssim_val = compute_ssim(img, image)

        with col:
            st.image(image, caption=title, width=260)
            st.markdown(f"**PSNR:** {psnr_val:.2f} dB  \n**SSIM:** {ssim_val:.4f}")

    show(cols[0], img, "Original")
    show(cols[1], he_img, "Histogram Equalization")
    show(cols[2], clahe_img, "CLAHE")

    show(cols2[0], gamma_img, "Gamma (0.8)")
    show(cols2[1], bilateral_img, "Bilateral Filter")
    show(cols2[2], our_img, "CLAHE + Gamma + Laplacian (Our Method)")

    st.download_button(
        "Download Our Method Output",
        cv2.imencode(".png", our_img)[1].tobytes(),
        "enhanced_our_method.png",
        "image/png"
    )
