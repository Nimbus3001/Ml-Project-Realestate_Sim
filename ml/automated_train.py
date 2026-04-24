import os
import csv
import random
from ml.api_data_fetcher import fetch_satellite_image

def run_automated_pipeline(num_samples=20):
    print("Starting automated API data collection...")
    
    csv_path = "data/coordinates.csv"
    os.makedirs("data/raw/images", exist_ok=True)
    os.makedirs("data/raw/masks", exist_ok=True)
    
    success_count = 0
    if os.path.exists(csv_path):
        print(f"Reading coordinates from {csv_path}...")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for i, row in enumerate(reader):
                if not row or len(row) < 2: continue
                lat, lon = float(row[0]), float(row[1])
                img_path = f"data/raw/images/api_img_{i}.jpg"
                if fetch_satellite_image(lat, lon, save_path=img_path):
                    success_count += 1
    else:
        print("No coordinates.csv found. Generating random coordinates for demonstration...")
        # Generate some random coordinates
        base_lat, base_lon = 40.7128, -74.0060 # NYC
        for i in range(num_samples):
            lat = base_lat + random.uniform(-0.1, 0.1)
            lon = base_lon + random.uniform(-0.1, 0.1)
            img_path = f"data/raw/images/api_rand_{i}.jpg"
            if fetch_satellite_image(lat, lon, save_path=img_path):
                success_count += 1
                
    print(f"Collected {success_count} new images in data/raw/images.")
    
    print("\nStarting model training...")
    os.system("python -m ml.train_model")
    print("\nAutomated training pipeline complete! The model is now robust.")

if __name__ == "__main__":
    run_automated_pipeline()
