import cv2
import numpy as np
import os

from src.data_loader import load_all_images
from src.feature_extraction import compute_density
from src.model import classify_growth


def preprocess_image(img, size=(64, 64)):
    img_resized = cv2.resize(img, size)
    return img_resized.flatten()


def build_dataset(data_path, limit=50):
    images, masks = load_all_images(data_path, limit=limit)

    X = []
    y = []

    for i in range(len(images)):
        img = images[i]
        mask = masks[i]

        # 🔹 Generate label using your existing pipeline
        b_density, r_density = compute_density(mask)
        label = classify_growth(b_density, r_density)

        # 🔹 Convert label to number
        if label == "Low":
            y_val = 0
        elif label == "Medium":
            y_val = 1
        else:
            y_val = 2  # High

        # 🔹 Convert image to feature vector
        features = preprocess_image(img)

        X.append(features)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)

    print("Dataset built:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y