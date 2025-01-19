import imageio
import numpy as np
import matplotlib.pyplot as plt

# Framebuffer details
width = 1920
height = 2160

# RGB565 has 16 bits per pixel (2 bytes per pixel)
depth = 2  

# Decode RGB565 to RGB888
def rgb565_to_rgb888(pixel):
    r = ((pixel >> 11) & 0x1F) * 255 // 31  # Extract red and scale to 0-255
    g = ((pixel >> 5) & 0x3F) * 255 // 63   # Extract green and scale to 0-255
    b = (pixel & 0x1F) * 255 // 31          # Extract blue and scale to 0-255
    return r, g, b

if __name__ == '__main__':
    # Read the raw data
    with open('screen_cap.raw', 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint16)

    # Ensure the data size matches the expected framebuffer size
    if raw_data.size != width * height:
        print(raw_data.size)
        raise ValueError("The size of the raw data does not match the expected resolution.")

    # Reshape the data into the 2D image format
    image_2d = raw_data.reshape((height, width))


    # Vectorized conversion of RGB565 to RGB888
    rgb_image = np.zeros((height // 2, width, 3), dtype=np.uint8)
    for y in range(height // 2):
        for x in range(width):
            rgb_image[y, x] = rgb565_to_rgb888(image_2d[y, x])

    print(rgb_image[0,22])
    print(rgb_image[0,23])

    # Display the image
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()
    
    # Save the image as a PNG file
    output_file = './screenshot.png'
    imageio.imwrite(output_file, rgb_image)
    print(f"Image saved as {output_file}")
