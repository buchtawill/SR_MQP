import numpy as np
from PIL import Image


WIDTH           = 720
HEIGHT          = 576
BYTES_PER_PIXEL = 3

# Set to either '565' or '888' depending on the format of the raw image
FORMAT = '888'

if __name__ == '__main__':
    
    input_file = './images/input888.raw'
    
    import matplotlib.pyplot as plt

    if FORMAT == '888':
        image = np.fromfile(input_file, dtype=np.uint8)
        image = image.reshape((HEIGHT, WIDTH, BYTES_PER_PIXEL))
    elif FORMAT == '565':
        image = np.fromfile(input_file, dtype=np.uint16)
        image = image.reshape((HEIGHT, WIDTH))
        r = (image >> 11) & 0x1F
        g = (image >> 5) & 0x3F
        b = image & 0x1F
        r = (r << 3).astype(np.uint8)
        g = (g << 2).astype(np.uint8)
        b = (b << 3).astype(np.uint8)
        image = np.stack((r, g, b), axis=-1)

    plt.imshow(image)
    plt.show()
    
    # Save the image
    img = Image.fromarray(image)
    img.save('images/raw_wiiframe.png')