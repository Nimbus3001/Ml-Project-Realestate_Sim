import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.dataset_builder import build_dataset

DATA_PATH = "data/raw"


# 🔹 Build dataset
X, y = build_dataset(DATA_PATH, limit=100)


# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 🔹 Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)


# 🔹 Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 🔹 Save model
joblib.dump(model, "ml/model.pkl")

print("\nModel saved as ml/model.pkl")