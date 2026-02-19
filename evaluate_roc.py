import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

MODEL_PATH = "cnn_pneumonia_model.h5"
DATA_DIR = "dataset_split/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_PATH)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

y_true = test_data.classes
y_scores = model.predict(test_data).ravel()

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Pneumonia Detection")
plt.legend(loc="lower right")
plt.show()

# Find threshold closest to 0.6
idx = np.argmin(np.abs(thresholds - 0.6))
print(f"At threshold 0.6 → TPR={tpr[idx]:.2f}, FPR={fpr[idx]:.2f}")
