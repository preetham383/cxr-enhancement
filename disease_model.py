# ---------------- IMPORTS ----------------
import numpy as np
import cv2
import joblib

from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops

# ---------------- CONSTANTS ----------------
MODEL_PATH = "disease_model.pkl"
CLASS_NAMES = ["Normal", "Pneumonia"]

# ---------------- FEATURE EXTRACTION ----------------
def extract_disease_features(img):
    # Ensure grayscale uint8
    img = img.astype(np.uint8)
    img = cv2.resize(img, (256, 256))

    features = []

    # ---------- BASIC INTENSITY ----------
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    features.extend([mean_intensity, std_intensity])

    # ---------- EDGE DENSITY ----------
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)

    # ---------- CONTRAST RANGE ----------
    p2, p98 = np.percentile(img, (2, 98))
    contrast_range = p98 - p2
    features.append(contrast_range)

    # ---------- LUNG REGION VARIANCE (CENTRAL ROI) ----------
    h, w = img.shape
    lung_roi = img[int(0.25*h):int(0.75*h), int(0.25*w):int(0.75*w)]
    lung_variance = np.var(lung_roi)
    features.append(lung_variance)

    # ---------- GLCM TEXTURE FEATURES ----------
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features.append(graycoprops(glcm, "contrast")[0, 0])
    features.append(graycoprops(glcm, "energy")[0, 0])
    features.append(graycoprops(glcm, "homogeneity")[0, 0])
    features.append(graycoprops(glcm, "correlation")[0, 0])

    return np.array(features)

# ---------------- TRAIN MODEL ----------------
def train_disease_model():
    """
    NOTE:
    This is a DEMO / PROTOTYPE training routine.
    Replace with real dataset training for clinical use.
    """

    X = []
    y = []

    # ---- Dummy NORMAL samples ----
    for _ in range(40):
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        X.append(extract_disease_features(img))
        y.append(0)

    # ---- Dummy PNEUMONIA samples ----
    for _ in range(40):
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        X.append(extract_disease_features(img))
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    print("Disease model trained and saved.")

# ---------------- LOAD MODEL ----------------
def load_disease_model():
    return joblib.load(MODEL_PATH)

# ---------------- PREDICT ----------------
def predict_disease(img, model):
    feats = extract_disease_features(img).reshape(1, -1)
    probs = model.predict_proba(feats)[0]

    idx = np.argmax(probs)
    return CLASS_NAMES[idx], float(probs[idx])
