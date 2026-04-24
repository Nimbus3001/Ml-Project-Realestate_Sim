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
    # Predict proba returns an array for the classes it was trained on
    proba_array = model.predict_proba(features)[0]
    classes = getattr(model, 'classes_', [0, 1, 2])

    # Initialize all to 0
    probs = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    
    # Map the probabilities to the correct class labels
    for cls, prob in zip(classes, proba_array):
        if cls == 0:
            probs["Low"] = float(prob)
        elif cls == 1:
            probs["Medium"] = float(prob)
        elif cls == 2:
            probs["High"] = float(prob)

    # 🔹 Convert numeric label back to text
    if prediction == 0:
        label = "Low"
    elif prediction == 1:
        label = "Medium"
    else:
        label = "High"

    return {
        "label": label,
        "probabilities": probs
    }


# 🔹 TEST WITH IMAGE
if __name__ == "__main__":
    test_images = ["test.jpg", "test1.jpg", "test2.jpg"]
    
    for test_image in test_images:
        try:
            result = predict_growth(test_image)
            print(f"Prediction for {test_image}: {result}")
        except Exception as e:
            print(f"Error processing {test_image}: {e}")