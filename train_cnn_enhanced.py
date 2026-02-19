import tensorflow as tf
import numpy as np
import cv2
import os

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
DATA_DIR = "dataset_split"

# ================= IMAGE ENHANCEMENT =================
def enhance_image(img):
    # img comes as float32 from generator, convert to uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (now valid)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    # Convert back to 3-channel
    cl = cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)

    # Normalize to [0,1] for CNN
    cl = cl.astype(np.float32) / 255.0

    return cl

# ================= CUSTOM GENERATOR =================
def enhanced_generator(directory):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = datagen.flow_from_directory(
        directory,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    while True:
        x, y = next(generator)
        x_enhanced = np.array([enhance_image(img) for img in x])
        yield x_enhanced, y

train_data = enhanced_generator(os.path.join(DATA_DIR, "train"))
val_data   = enhanced_generator(os.path.join(DATA_DIR, "val"))
test_data  = enhanced_generator(os.path.join(DATA_DIR, "test"))

# ================= MODEL =================
base_model = tf.keras.applications.DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# ================= TRAIN =================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=100,
    validation_steps=50
)

# ================= EVALUATE =================
results = model.evaluate(test_data, steps=50)
print("Enhanced Test Results:", results)

# ================= SAVE =================
model.save("cnn_enhanced.keras")
print("Enhanced model saved as cnn_enhanced.keras")
