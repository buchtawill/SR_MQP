import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def nerf_rgb_to_rgb565(image: Image) -> Image:
    """
    Degrade a PIL image by reducing the amount of information in each color channel 
    to match an RGB565 image. 
    
    Args:
        image (Image): A PIL image to degrade.

    Returns:
        Image: A PIL image with the same dimensions as the input image, but with the
    
    """
    # Convert to numpy array
    rgb_array = np.array(image, dtype=np.uint8)

    # zero out the lower bits
    r = rgb_array[:, :, 0] & 0b11111000  # 5 bits for red
    g = rgb_array[:, :, 1] & 0b11111100  # 6 bits for green
    b = rgb_array[:, :, 2] & 0b11111000  # 5 bits for blue

    # Stack and create RGB image
    rgb_array = np.stack((r, g, b), axis=-1).astype(np.uint8)
    return Image.fromarray(rgb_array)
    
def display_images(original_image :Image, rgb565_image: Image):
    """
    Display the original image and the RGB565 image side by side.
    
    Args:
        original_image (Image): The original image.
        rgb565_image (Image): The RGB 565 nerfed image
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # RGB565 image
    axes[1].imshow(rgb565_image)
    axes[1].set_title("RGB565 Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the original image
    image_path = "C:\\Users\\bucht\\OneDrive\\Pictures\\desktop_slideshow\\IMG_5599.jpg"
    image_path = "C:\\Users\\bucht\\OneDrive\\Pictures\\desktop_slideshow\\bartlett.jpg"
    image_path = "./16777216colors.png"
    original_image = Image.open(image_path).convert("RGB")

    # Convert to RGB565
    rgb565_equivalent = nerf_rgb_to_rgb565(original_image)

    # Display the images
    display_images(original_image, rgb565_equivalent)