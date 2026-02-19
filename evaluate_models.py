import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import os

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 16
DATA_DIR = "dataset_split/test"

MODELS = {
    "Baseline (Original)": "cnn_original.h5",
    "Enhanced": "cnn_enhanced.keras"
}

# ================= LOAD TEST DATA =================
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

y_true = test_data.classes

# ================= EVALUATION =================
plt.figure(figsize=(10, 4))

for i, (name, model_path) in enumerate(MODELS.items()):
    print(f"\nEvaluating: {name}")

    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(test_data).ravel()
    y_pred = (y_prob >= 0.6).astype(int)   # threshold = 0.6

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Pneumonia"]
    )

    plt.figure()
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.show()

    # ---------- ROC Curve ----------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

# ---------- ROC Final Plot ----------
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
