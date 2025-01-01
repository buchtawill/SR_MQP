import numpy as np

import matplotlib.pyplot as plt

INPUT_VIDEO_HEIGHT = 576
INPUT_VIDEO_WIDTH  = 720
UPSCALE_FACTOR     = 1

TILE_SIZE = 28

def tile_coord_to_pixel_coord(tile_x, tile_y):
    pixel_x = tile_x * TILE_SIZE
    pixel_y = tile_y * TILE_SIZE
    
    if(pixel_x >= INPUT_VIDEO_WIDTH - TILE_SIZE):
        pixel_x = INPUT_VIDEO_WIDTH - TILE_SIZE
        
    if(pixel_y >= INPUT_VIDEO_HEIGHT - TILE_SIZE):
        pixel_y = INPUT_VIDEO_HEIGHT - TILE_SIZE
        
    return pixel_x, pixel_y

if __name__ == "__main__":
    
    # Create a 720x576 pixel image with random colors
    image = np.ones((576, 720, 3))
    
    # Set the outermost pixels to black
    image[0, :, :]  = 0  # Top row
    image[-1, :, :] = 0  # Bottom row
    image[:, 0, :]  = 0  # Left column
    image[:, -1, :] = 0  # Right column
    
    num_vertical_tiles = INPUT_VIDEO_HEIGHT // TILE_SIZE
    num_horizontal_tiles = INPUT_VIDEO_WIDTH // TILE_SIZE
    
    if(INPUT_VIDEO_HEIGHT % TILE_SIZE != 0):
        num_vertical_tiles += 1
    
    if(INPUT_VIDEO_WIDTH % TILE_SIZE != 0):
        num_horizontal_tiles += 1
    
    for tile_x in range(num_horizontal_tiles):
        for tile_y in range(num_vertical_tiles):
            pix_x, pix_y = tile_coord_to_pixel_coord(tile_x, tile_y)
            image[pix_y, pix_x, :] = 0
            image[pix_y, pix_x, 0] = 1

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.show()