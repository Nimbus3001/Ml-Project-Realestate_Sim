import cv2
import numpy as np
import joblib


# 🔹 Load trained model
model = joblib.load("ml/model.pkl")


def preprocess_image(img_path, size=(64, 64)):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Image not found or invalid path")

    img_resized = cv2.resize(img, size)
    return img_resized.flatten()


def predict_growth(img_path):
    features = preprocess_image(img_path)

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    # 🔹 Convert numeric label back to text
    if prediction == 0:
        return "Low"
    elif prediction == 1:
        return "Medium"
    else:
        return "High"


# 🔹 TEST WITH IMAGE
if __name__ == "__main__":
    test_image = "test1.jpg"  # 👈 change this

    result = predict_growth(test_image)

    print("\nPrediction:", result)