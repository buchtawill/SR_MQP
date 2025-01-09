import cv2
import numpy as np

import matplotlib.pyplot as plt

def read_and_process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image to 720x576
    resized_image = cv2.resize(image, (720, 576))
    
    # Convert the image to a numpy array
    image_array = np.array(resized_image)
    
    return image_array

def show_image(image_array):
    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Display the image using matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide the axis
    plt.show()

if __name__ == '__main__':
    image_path = "./IMG_5599.jpg"
    image_array = read_and_process_image(image_path)
    # show_image(image_array)
    
    print(image_array.shape)
    print(image_array[0, 0, :])
    
    # Reshape the image array to be linear (1 dimension)
    linear_image_array = image_array.reshape(-1)
    
    # Print the shape of the linear image array
    print(linear_image_array.shape)
    
    # Print the first 10 elements of the linear image array
    print(linear_image_array[:10])
    
    # Save the linear image array as "lanterns.bin"
    with open("lanterns.bin", "wb") as file:
        file.write(linear_image_array.tobytes())