import numpy as np
import matplotlib.pyplot as plt

def yuyv_to_rgb888(yuyv, width, height):
    yuyv = np.frombuffer(yuyv, dtype=np.uint8)  # Keep as a 1D array
    rgb888 = np.zeros((height * width * 3), dtype=np.uint8)

    for i in range(0, width * height, 2):
        y0 = yuyv[i * 2]
        u  = yuyv[i * 2 + 1] - 128
        y1 = yuyv[i * 2 + 2]
        v  = yuyv[i * 2 + 3] - 128

        r0 = np.clip(y0 + 1.403 * v, 0, 255).astype(np.uint8)
        g0 = np.clip(y0 - 0.344 * u - 0.714 * v, 0, 255).astype(np.uint8)
        b0 = np.clip(y0 + 1.770 * u, 0, 255).astype(np.uint8)

        r1 = np.clip(y1 + 1.403 * v, 0, 255).astype(np.uint8)
        g1 = np.clip(y1 - 0.344 * u - 0.714 * v, 0, 255).astype(np.uint8)
        b1 = np.clip(y1 + 1.770 * u, 0, 255).astype(np.uint8)

        rgb888[i * 3]     = r0
        rgb888[i * 3 + 1] = g0
        rgb888[i * 3 + 2] = b0
        rgb888[i * 3 + 3] = r1
        rgb888[i * 3 + 4] = g1
        rgb888[i * 3 + 5] = b1

    return rgb888.reshape((height, width, 3))


def load_raw_image(filename, shape):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def write_header(image_tile, yuyv_tile):
    # Open a new file, save it as a 28x28 tile
    tile_name = 'conversion'
    with open(f'image_tile_{tile_name}.hpp', 'w') as f:
        f.write(f"#ifndef IMAGE_TILE_{tile_name}_HPP\n")
        f.write(f"#define IMAGE_TILE_{tile_name}_HPP\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f'const uint8_t {tile_name}_tile_rgb[] = {{\n')
        for row in image_tile:
            for pixel in row:
                f.write(f'\t{pixel[0]:>3}, {pixel[1]:>3}, {pixel[2]:>3},')
                f.write(f' // {(pixel[0] / 256.0) : .6f} {(pixel[1] / 256.0) : .6f} {(pixel[2] / 256.0) : .6f} \n')
                
        f.write('};\n')
        
        f.write('\n')
        f.write(f'const uint8_t {tile_name}_tile_yuyv[] = {{\n')
        for row in yuyv_tile:
            f.write('\t')
            for i in range(len(row)):
                f.write(f' {row[i]}, ')
                
            f.write('\n')
                
        f.write('};\n')
        

def main():
    width, height = 720, 576  # Adjust dimensions as needed
    
    # Load the YUYV file
    yuyv_filename = 'images/input_yuyv.raw'
    yuyv_data = load_raw_image(yuyv_filename, (height, width * 2))
    # rgb_from_yuyv = yuyv_to_rgb888(yuyv_data, width, height)
    
    # Load the RGB888 file
    rgb_filename = 'images/rgb888.raw'
    rgb_data = load_raw_image(rgb_filename, (height, width, 3))
    
    x, y = 80, 320
    tile_size = 28
    tile_rgb888 = rgb_data[y:y+tile_size, x:x+tile_size, :]

    # Crop the correct region from YUYV
    tile_yuyv = yuyv_data[y:y+tile_size, x*2:(x+tile_size)*2]

    # Convert the cropped YUYV tile to RGB
    yuyv_tile_rgb = yuyv_to_rgb888(tile_yuyv.flatten(), tile_size, tile_size)
    
    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(yuyv_tile_rgb)
    # axes[0].imshow(rgb_from_yuyv)
    axes[0].set_title('Converted RGB from YUYV')
    axes[0].axis('off')
    
    axes[1].imshow(tile_rgb888)
    # axes[1].imshow(rgb_data)
    axes[1].set_title('Original RGB888')
    axes[1].axis('off')
    
    plt.show()
    
    write_header(yuyv_tile_rgb, tile_yuyv)
    

if __name__ == "__main__":
    main()