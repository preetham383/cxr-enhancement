import cv2
import matplotlib.pyplot as plt
from features import extract_features
from enhance import enhance_image
from model import load_model

img = cv2.imread("sample_cxr.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))

features = extract_features(img).reshape(1, -1)
model = load_model()
params = model.predict(features)[0]

enhanced = enhance_image(img, params)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(enhanced, cmap='gray')
plt.title("Enhanced (Automatic)")
plt.axis('off')

plt.show()
