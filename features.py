import cv2
import numpy as np

def extract_features(img):
    img_uint8 = img.astype(np.uint8)
    img_float = img_uint8.astype(np.float32)

    mean_intensity = np.mean(img_float)
    std_intensity = np.std(img_float)

    laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
    edge_strength = laplacian.var()

    hist = cv2.calcHist([img_uint8], [0], None, [8], [0, 256]).flatten()
    if hist.sum() != 0:
        hist = hist / hist.sum()

    features = np.hstack([
        mean_intensity,
        std_intensity,
        edge_strength,
        hist
    ])

    return features
