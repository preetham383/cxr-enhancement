import os
import pydicom
import cv2
import pandas as pd
from tqdm import tqdm

CSV_PATH = r"C:\Users\preet\OneDrive\Desktop\cxr_enhancement\drive-download-20240112T131344Z-002\stage_2_train_labels.csv"
DICOM_DIR = r"C:\Users\preet\OneDrive\Desktop\cxr_enhancement\drive-download-20240112T131344Z-002\stage_2_train_images"
OUT_DIR = r"C:\Users\preet\OneDrive\Desktop\cxr_enhancement\dataset"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for _, row in tqdm(df.iterrows(), total=len(df)):
    pid = row["patientId"]
    label = row["Target"]

    dicom_path = os.path.join(DICOM_DIR, pid + ".dcm")
    if not os.path.exists(dicom_path):
        continue

    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype("uint8")

    cls = "Pneumonia" if label == 1 else "Normal"
    out_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, pid + ".png"), img)
