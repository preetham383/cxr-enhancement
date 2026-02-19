import os
import random
import shutil

DATASET_DIR = "dataset"
BALANCED_DIR = "dataset_balanced"

NORMAL_DIR = os.path.join(DATASET_DIR, "Normal")
PNEUMONIA_DIR = os.path.join(DATASET_DIR, "Pneumonia")

OUT_NORMAL = os.path.join(BALANCED_DIR, "Normal")
OUT_PNEUMONIA = os.path.join(BALANCED_DIR, "Pneumonia")

os.makedirs(OUT_NORMAL, exist_ok=True)
os.makedirs(OUT_PNEUMONIA, exist_ok=True)

normal_imgs = os.listdir(NORMAL_DIR)
pneumonia_imgs = os.listdir(PNEUMONIA_DIR)

min_count = min(len(normal_imgs), len(pneumonia_imgs))

print(f"Using {min_count} images per class")

normal_sample = random.sample(normal_imgs, min_count)
pneumonia_sample = random.sample(pneumonia_imgs, min_count)

for img in normal_sample:
    shutil.copy(os.path.join(NORMAL_DIR, img), OUT_NORMAL)

for img in pneumonia_sample:
    shutil.copy(os.path.join(PNEUMONIA_DIR, img), OUT_PNEUMONIA)

print("Dataset balancing completed.")
