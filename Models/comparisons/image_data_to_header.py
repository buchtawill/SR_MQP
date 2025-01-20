from PIL import Image
import numpy as np

TILE_WIDTH = 28
TILE_HEIGHT = 28

# [x,y]
TILE_LOCATION=[3,1]

if __name__ == '__main__':
    
    path = './images/frame2984.png'
    
    image = Image.open(path)
    image = image.resize((720, 576))  # Resize the image to the desired shape
    image_data = np.array(image)
    
    # Open a new file, image_data.hpp, and write image data to an array therein
    
    with open('image_data.hpp', 'w') as f:
        f.write("#ifndef IMAGE_DATA_HPP\n")
        f.write("#define IMAGE_DATA_HPP\n\n")
        f.write('#define IMAGE_WIDTH 720\n')
        f.write('#define IMAGE_HEIGHT 576\n')
        f.write('const unsigned char image_data[] = {\n')
        for row in image_data:
            for pixel in row:
                f.write(f'\t{pixel[0]}, {pixel[1]}, {pixel[2]}, \n')
        f.write('};\n')
        
        f.write("\n#endif\n")
        
    # Open a new file, save it as a 28x28 tile
    with open('image_tile.hpp', 'w') as f:
        f.write("#ifndef IMAGE_TILE_HPP\n")
        f.write("#define IMAGE_TILE_HPP\n\n")
        f.write('const unsigned char image_tile[] = {\n')
        for row in image_data[TILE_HEIGHT*TILE_LOCATION[0]:TILE_HEIGHT*(TILE_LOCATION[0]+1)]:
            for pixel in row[TILE_WIDTH*TILE_LOCATION[1]:TILE_WIDTH*(TILE_LOCATION[1]+1)]:
                f.write(f'\t{pixel[0]}, {pixel[1]}, {pixel[2]}, \n')
        f.write('};\n')
        
        f.write("\n#endif\n")
        
    # display the image tile
    
    image_tile = image_data[TILE_HEIGHT*TILE_LOCATION[0]:TILE_HEIGHT*(TILE_LOCATION[0]+1), 
                            TILE_WIDTH*TILE_LOCATION[1]:TILE_WIDTH*(TILE_LOCATION[1]+1)]
    img = Image.fromarray(image_tile)
    img.show()
        