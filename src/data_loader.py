import cv2
import os

def load_all_images(data_path, limit=20):
    image_path = os.path.join(data_path, "images")
    mask_path = os.path.join(data_path, "masks")

    image_files = os.listdir(image_path)

    images = []
    masks = []

    for img_file in image_files[:limit]:
        img = cv2.imread(os.path.join(image_path, img_file))
        mask = cv2.imread(os.path.join(mask_path, img_file))

        if img is not None and mask is not None:
            images.append(img)
            masks.append(mask)

    return images, masks