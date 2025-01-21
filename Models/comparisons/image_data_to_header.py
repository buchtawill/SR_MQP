import numpy as np
from PIL import Image
from scipy.ndimage import zoom

TILE_WIDTH = 28
TILE_HEIGHT = 28

# [x,y]
TILE_LOCATION=[3,1]

if __name__ == '__main__':
    
    path = './images/frame2984.png'
    
    image = Image.open(path)
    image = image.resize((720, 576))  # Resize the image to the desired shape
    image_data = np.array(image)
    
    tile_name = input("Enter a memorable name for this image: ")
    
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
        f.write(f'const uint8_t {tile_name}_tile_low_res[] = {{\n')
        for row in image_data[TILE_HEIGHT*TILE_LOCATION[0]:TILE_HEIGHT*(TILE_LOCATION[0]+1)]:
            for pixel in row[TILE_WIDTH*TILE_LOCATION[1]:TILE_WIDTH*(TILE_LOCATION[1]+1)]:
                f.write(f'\t{pixel[0]}, {pixel[1]}, {pixel[2]}, \n')
        f.write('};\n')
        
        
        # Upscale the image using bilinear interpolation
        tile_data = image_data[TILE_HEIGHT*TILE_LOCATION[0]:TILE_HEIGHT*(TILE_LOCATION[0]+1), 
                                TILE_WIDTH*TILE_LOCATION[1]:TILE_WIDTH*(TILE_LOCATION[1]+1)]
        
        upscaled_tile = zoom(tile_data, (2, 2, 1), order=1)  # order=1 for bilinear interpolation
        
        f.write(f'const uint8_t {tile_name}_tile_interpolated[] = {{\n')
        for row in upscaled_tile:
            for pixel in row:
                f.write(f'\t{pixel[0]}, {pixel[1]}, {pixel[2]},\n')
        f.write('};\n')
        
        f.write("\n#endif\n")
    
    # display the image tile
    # img = Image.fromarray(tile_data)
    # img.show()
        