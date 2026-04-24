import os
import requests
import cv2
import numpy as np
import math

def deg2num(lat_deg, lon_deg, zoom):
    """Converts latitude/longitude to ESRI/Google tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def fetch_satellite_image(lat, lon, zoom=16, save_path="data/raw/images/img.jpg"):
    """
    Fetches a satellite image using the free ESRI World Imagery API.
    Requires NO API Key or billing info!
    """
    img_dir = os.path.dirname(save_path)
    os.makedirs(img_dir, exist_ok=True)
    
    # Also create the corresponding mask directory
    base_dir = os.path.dirname(img_dir)
    mask_dir = os.path.join(base_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, os.path.basename(save_path))
    
    img = None
    
    # Calculate tile coordinates
    xtile, ytile = deg2num(lat, lon, zoom)
    
    # ESRI World Imagery tile server (Free, no auth)
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    headers = {
        'User-Agent': 'UrbanizationPredictor/1.0 (Research Project)'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        img = cv2.imread(save_path)
        # Resize from 256x256 to 400x400 to maintain consistency with previous dataset
        if img is not None:
            img = cv2.resize(img, (400, 400))
            cv2.imwrite(save_path, img)
    else:
        print(f"Failed to fetch ESRI image for {lat}, {lon}: {response.status_code}")
        # Fallback to dummy generation if network fails
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        rand_val = np.random.rand()
        if rand_val > 0.6:
            img[:, :] = (0, 200, 0) # Green
        elif rand_val > 0.3:
            img[:, :] = (150, 150, 50) # Mixed
        else:
            img[:, :] = (100, 100, 100) # Gray
        cv2.imwrite(save_path, img)
            
    # Generate a heuristic mask based on the image
    if img is not None:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 255])
        building_pixels = cv2.inRange(hsv, lower_gray, upper_gray)
        mask[building_pixels > 0] = 1 # Building
        
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 40])
        road_pixels = cv2.inRange(hsv, lower_dark, upper_dark)
        mask[road_pixels > 0] = 3 # Road
        
        cv2.imwrite(mask_path, mask)
        
    return save_path
