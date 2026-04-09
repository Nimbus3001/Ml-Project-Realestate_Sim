from ml.dataset_builder import build_dataset

DATA_PATH = "data/raw"

X, y = build_dataset(DATA_PATH, limit=20)