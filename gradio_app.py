import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# ================= LOAD CNN MODEL =================
MODEL_PATH = "cnn_pneumonia_model.h5"  # or .keras
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

# ================= RELAXED INPUT VALIDATION =================
def is_valid_chest_xray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(gray)
    if mean_intensity < 15 or mean_intensity > 240:
        return False

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.003 or edge_density > 0.5:
        return False

    h, w = gray.shape
    ratio = w / h
    if ratio < 0.5 or ratio > 2.0:
        return False

    return True

# ================= IMAGE ENHANCEMENT =================
def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

def gamma_correction(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** invGamma * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)

def proposed_enhancement(img):
    img = apply_clahe(img)
    img = gamma_correction(img, gamma=1.1)
    return img

# ================= CNN PREPROCESS =================
def preprocess_for_cnn(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================= CONFIDENCE BANDS =================
def interpret_probability(prob):
    if prob < 0.4:
        return "Low", "Low risk – likely normal."
    elif prob < 0.6:
        return "Moderate", "Uncertain – borderline patterns detected."
    else:
        return "High", "High risk – consult a medical professional."

# ================= GRAD-CAM (NO keras.backend) =================
def generate_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # REPLACEMENT FOR K.mean
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

# ================= MAIN PIPELINE =================
def process_image(image):
    if image is None:
        return [None]*6 + ["", ""]

    if not is_valid_chest_xray(image):
        return (
            image, None, None, None, None, None,
            "❌ Invalid Input",
            "The uploaded image is not a valid chest X-ray.\n"
            "Please upload a chest X-ray image."
        )

    original = image
    he = histogram_equalization(image)
    clahe = apply_clahe(image)
    gamma = gamma_correction(image)
    proposed = proposed_enhancement(image)

    cnn_input = preprocess_for_cnn(proposed)
    prob = model.predict(cnn_input)[0][0]

    band, advice = interpret_probability(prob)
    label = "🦠 Pneumonia Detected" if prob >= 0.6 else "✅ Normal Chest X-ray"

    confidence = (
        f"P(Pneumonia): {prob*100:.2f}%\n"
        f"P(Normal): {(1-prob)*100:.2f}%\n"
        f"Confidence Band: {band}\n"
        f"Advice: {advice}"
    )

    heatmap = generate_gradcam(
        model,
        cnn_input,
        last_conv_layer_name="conv5_block16_concat"
    )
    gradcam_img = overlay_gradcam(proposed, heatmap)

    return original, he, clahe, gamma, proposed, gradcam_img, label, confidence

# ================= GRADIO UI =================
with gr.Blocks(title="Chest X-ray Enhancement & Disease Prediction") as demo:
    gr.Markdown("## 🩺 Chest X-ray Enhancement & Disease Prediction System")
    gr.Markdown(
        "**Pipeline:** Validation → Enhancement → CNN → Grad-CAM\n\n"
        "**Disclaimer:** Academic & decision-support use only."
    )

    image_input = gr.Image(type="numpy", label="Upload Chest X-ray")

    with gr.Row():
        out1 = gr.Image(label="Original")
        out2 = gr.Image(label="Histogram Equalization")
        out3 = gr.Image(label="CLAHE")
        out4 = gr.Image(label="Gamma Correction")

    with gr.Row():
        out5 = gr.Image(label="Proposed Enhancement")
        out6 = gr.Image(label="Grad-CAM Explanation")

    label_out = gr.Textbox(label="Disease Prediction")
    conf_out = gr.Textbox(label="Confidence & Advice")

    btn = gr.Button("Run Enhancement & Prediction")

    btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[out1, out2, out3, out4, out5, out6, label_out, conf_out]
    )
iface=gr.Interface(
    fn=process_image,
    inputs=image_input,
    outputs=[out1, out2, out3, out4, out5, out6, label_out, conf_out],
    title="Chest X-ray Enhancement & Disease Prediction System",
    description="Upload a chest X-ray image to enhance and predict disease."
)
iface.launch(share=True)

