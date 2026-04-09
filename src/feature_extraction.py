import numpy as np

def compute_density(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # 🔍 ADD THIS
    unique_vals, counts = np.unique(mask, return_counts=True)


    total_pixels = mask.shape[0] * mask.shape[1]

    BUILDING_CLASS = 1
    ROAD_CLASS = 3

    building_pixels = np.sum(mask == BUILDING_CLASS)
    road_pixels = np.sum(mask == ROAD_CLASS)

    building_density = building_pixels / total_pixels
    road_density = road_pixels / total_pixels

    return building_density, road_density