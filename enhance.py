import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

# ---------- BASIC METHODS ----------
def histogram_equalization(img):
    return cv2.equalizeHist(img)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

# ---------- CLAHE ----------
def apply_clahe(img, clip_limit=2.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img)

# ---------- GAMMA ----------
def gamma_correction(img, gamma=0.8):
    img_norm = img / 255.0
    img_gamma = np.power(img_norm, gamma)
    return np.uint8(img_gamma * 255)

# ---------- HYBRID ----------
def clahe_gamma(img):
    return gamma_correction(apply_clahe(img))

# ---------- LAPLACIAN PYRAMID ----------
def laplacian_enhance(img, mu):
    img = img.astype(np.float32)
    gaussian = img.copy()
    laplacian_pyr = []

    for _ in range(3):
        down = cv2.pyrDown(gaussian)
        up = cv2.pyrUp(down, dstsize=gaussian.shape[::-1])
        laplacian_pyr.append(gaussian - up)
        gaussian = down

    enhanced = gaussian
    for i in reversed(range(3)):
        enhanced = cv2.pyrUp(enhanced, dstsize=laplacian_pyr[i].shape[::-1])
        enhanced += mu[i] * laplacian_pyr[i]

    return np.clip(enhanced, 0, 255).astype(np.uint8)

# ---------- NORMALIZATION ----------
def normalize_image(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# ---------- OUR FULL METHOD ----------
def enhance_image(img, params):
    clip, gamma, mu1, mu2, mu3 = params
    img1 = apply_clahe(img, clip)
    img2 = gamma_correction(img1, gamma)
    img3 = laplacian_enhance(img2, [mu1, mu2, mu3])
    return img3

# ---------- METRICS ----------
def compute_psnr(original, enhanced):
    mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(original, enhanced):
    score, _ = ssim(
        original,
        enhanced,
        full=True,
        data_range=enhanced.max() - enhanced.min()
    )
    return score
