import os
import random
import shutil

SOURCE_DIR = "dataset_balanced"
DEST_DIR = "dataset_split"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

classes = ["Normal", "Pneumonia"]

for split in SPLITS:
    for cls in classes:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

for cls in classes:
    imgs = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(imgs)

    total = len(imgs)
    train_end = int(SPLITS["train"] * total)
    val_end = train_end + int(SPLITS["val"] * total)

    splits = {
        "train": imgs[:train_end],
        "val": imgs[train_end:val_end],
        "test": imgs[val_end:]
    }

    for split, files in splits.items():
        for img in files:
            src = os.path.join(SOURCE_DIR, cls, img)
            dst = os.path.join(DEST_DIR, split, cls, img)
            shutil.copy(src, dst)

print("Dataset split completed.")
