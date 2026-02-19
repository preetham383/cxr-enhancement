import tensorflow as tf
import os

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
DATA_DIR = "dataset_split"

# ================= DATA GENERATORS (NO ENHANCEMENT) =================
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_gen   = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

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
    epochs=EPOCHS
)

# ================= EVALUATE =================
results = model.evaluate(test_data)
print("Baseline Test Results:", results)

# ================= SAVE =================
model.save("cnn_original.keras")
print("Baseline model saved as cnn_original.keras")
