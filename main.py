import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_all_images
from src.feature_extraction import compute_density
from src.model import classify_growth

DATA_PATH = "data/raw"

images, masks = load_all_images(DATA_PATH, limit=20)

results = {"Low": 0, "Medium": 0, "High": 0}

for i in range(len(images)):
    img = images[i]
    mask = masks[i]

    b_density, r_density = compute_density(mask)
    label = classify_growth(b_density, r_density)

    print(f"\nImage {i + 1}")
    print("Building Density:", b_density)
    print("Road Density:", r_density)
    print("Growth:", label)

    # 🎨 Convert mask to single channel
    if len(mask.shape) == 3:
        mask_single = mask[:, :, 0]
    else:
        mask_single = mask

    # 🎨 Create colored mask
    colored_mask = np.zeros((mask_single.shape[0], mask_single.shape[1], 3), dtype=np.uint8)

    colored_mask[mask_single == 1] = [255, 0, 0]   # Buildings → Red
    colored_mask[mask_single == 3] = [0, 255, 0]   # Roads → Green

    # 🖼️ SHOW IMAGE + MASK
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"Original (Image {i + 1})")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title(f"Mask | Growth: {label}")
    plt.imshow(colored_mask)

    plt.tight_layout()
    plt.show()

    results[label] += 1

print("\n===== FINAL SUMMARY =====")
for key in results:
    print(f"{key}: {results[key]} images")

# 📊 Plot results
labels = list(results.keys())
values = list(results.values())

plt.bar(labels, values)
plt.title("Growth Distribution")
plt.xlabel("Growth Category")
plt.ylabel("Number of Images")

plt.show()