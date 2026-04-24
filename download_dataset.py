"""
Organizes the already-downloaded EuroSAT dataset into data/raw/images and data/raw/masks.
Expects the dataset at: data/EuroSAT/2750/<ClassName>/
"""
import os
import cv2
import numpy as np

EUROSAT_DIR = os.path.join('data', 'EuroSAT', '2750')
IMG_OUT = "data/raw/images"
MASK_OUT = "data/raw/masks"

CLASS_MAP = {
    "Industrial":            "high",
    "Residential":           "high",
    "Highway":               "medium",
    "AnnualCrop":            "medium",
    "PermanentCrop":         "medium",
    "Forest":                "low",
    "HerbaceousVegetation":  "low",
    "Pasture":               "low",
    "River":                 "low",
    "SeaLake":               "low",
}

LIMIT_PER_CLASS = 30  # 300 images total max

def organize():
    os.makedirs(IMG_OUT, exist_ok=True)
    os.makedirs(MASK_OUT, exist_ok=True)

    if not os.path.isdir(EUROSAT_DIR):
        print(f"ERROR: Could not find {EUROSAT_DIR}")
        print("Make sure data/EuroSAT/2750 exists with class subfolders.")
        return

    count = 0
    for cls, growth in CLASS_MAP.items():
        cls_dir = os.path.join(EUROSAT_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  Skipping missing class: {cls}")
            continue

        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg', '.png', '.tif', '.jpeg'))][:LIMIT_PER_CLASS]

        for fname in files:
            src = os.path.join(cls_dir, fname)
            dst_name = f"{cls}_{fname.replace('.tif', '.jpg')}"
            dst_img  = os.path.join(IMG_OUT, dst_name)
            dst_mask = os.path.join(MASK_OUT, dst_name)

            img = cv2.imread(src)
            if img is None:
                continue
            img = cv2.resize(img, (400, 400))
            cv2.imwrite(dst_img, img)

            mask = np.zeros((400, 400), dtype=np.uint8)
            if growth == "high":
                mask[:] = 1
            elif growth == "medium":
                mask[100:300, 100:300] = 1
            # low -> all zeros

            cv2.imwrite(dst_mask, mask)
            count += 1
            print(f"  [{count}] {cls}/{fname}", end='\r')

    print(f"\nDone! Organized {count} images into {IMG_OUT} and {MASK_OUT}.")
    print("Now run:  python -m ml.train_model")

if __name__ == "__main__":
    organize()
