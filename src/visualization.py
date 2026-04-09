import matplotlib.pyplot as plt

def show_results(image, mask, label):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title(f"Growth: {label}")
    plt.imshow(mask)

    plt.show()