import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

TILE_WIDTH = 28
TILE_HEIGHT = 28

# [x,y]
TILE_LOCATION=[3,1]

def rgb_to_yuv(r, g, b):
    """Convert RGB to YUV using BT.601 coefficients."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.500 * b + 128
    v = 0.500 * r - 0.419 * g - 0.081 * b + 128
    return int(y), int(u), int(v)

def rgb_to_yuyv(image):
    """Convert an RGB image (H, W, 3) to YUYV 4:2:2 format."""
    h, w, _ = image.shape
    assert w % 2 == 0, "Width must be even for YUYV 4:2:2"

    # Correct output shape: (H, W//2, 2)
    yuyv = np.zeros((h, w // 2, 2), dtype=np.uint8)

    for i in range(h):
        for j in range(0, w, 2):  # Process 2 pixels at a time
            r0, g0, b0 = image[i, j]
            r1, g1, b1 = image[i, j+1]

            y0, u0, v0 = rgb_to_yuv(r0, g0, b0)
            y1, u1, v1 = rgb_to_yuv(r1, g1, b1)

            # Average U and V over 2 pixels
            u = (u0 + u1) // 2
            v = (v0 + v1) // 2

            # Store interleaved as: Y0 U, Y1 V
            yuyv[i, j // 2, 0] = y0  # Y0
            yuyv[i, j // 2, 1] = u   # U (shared)
            yuyv[i, j // 2, 0] = y1  # Y1 (overwrite issue)
            yuyv[i, j // 2, 1] = v   # V (overwrite issue)

    return yuyv  # Shape: (H, W//2, 2)

if __name__ == '__main__':
    
    path = './images/frame2984.png'
    
    image = Image.open(path)
    image = image.resize((720, 576))  # Resize the image to the desired shape
    image_data = np.array(image)
    
    tile_name = input("Enter a memorable name for this image: ")
    
    image_tile = image_data[TILE_HEIGHT*TILE_LOCATION[0]:TILE_HEIGHT*(TILE_LOCATION[0]+1), 
                            TILE_WIDTH*TILE_LOCATION[1]:TILE_WIDTH*(TILE_LOCATION[1]+1)]
    
    # Convert the tile to YUYV format
    yuyv_tile = rgb_to_yuyv(image_tile)
    
    # Open a new file, image_data.hpp, and write image data to an array therein
    
    with open(f'image_data_{tile_name}.hpp', 'w') as f:
        f.write(f"#ifndef IMAGE_DATA_{tile_name}_HPP\n")
        f.write(f"#define IMAGE_DATA_{tile_name}_HPP\n\n")
        f.write('#define IMAGE_WIDTH 720\n')
        f.write('#define IMAGE_HEIGHT 576\n')
        f.write('const unsigned char image_data[] = {\n')
        for row in image_data:
            for pixel in row:
                f.write(f'\t{pixel[0]}, {pixel[1]}, {pixel[2]}, \n')
        f.write('};\n')
        f.write("\n#endif\n")
        
    # Open a new file, save it as a 28x28 tile
    with open(f'image_tile_{tile_name}.hpp', 'w') as f:
        f.write(f"#ifndef IMAGE_TILE_{tile_name}_HPP\n")
        f.write(f"#define IMAGE_TILE_{tile_name}_HPP\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f'const uint8_t {tile_name}_tile_low_res_rgb[] = {{\n')
        for row in image_tile:
            for pixel in row:
                f.write(f'\t{pixel[0]}, {pixel[1]}, {pixel[2]}, \n')
        f.write('};\n')
        
        f.write('\n')
        f.write(f'const uint8_t {tile_name}_tile_low_res_yuyv[] = {{\n')
        for row in yuyv_tile:
            for pixel in row:
                f.write(f'\t{pixel[0]}, {pixel[1]}, \n')
                
        f.write('};\n')
        
        # Upscale the image using bilinear interpolation        
        upscaled_tile = zoom(image_tile, (2, 2, 1), order=1)  # order=1 for bilinear interpolation
        
        f.write(f'const uint8_t {tile_name}_tile_interpolated_rgb[] = {{\n')
        for row in upscaled_tile:
            for pixel in row:
                f.write(f'\t{pixel[0]}, {pixel[1]}, {pixel[2]},\n')
        f.write('};\n')
        
        f.write("\n#endif\n")
    
    # display the image tile
    img = Image.fromarray(image_tile)
    img.show()
        