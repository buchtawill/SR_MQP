import numpy as np
from scipy.ndimage import zoom
import torch.nn.functional as F
from queue import Queue

WIDTH = 28
HEIGHT = 28
CHANNELS = 3
SCALE = 2

WIDTH_OUT = 56
HEIGHT_OUT = 56

SLICE_WIDTH = 7
SLICE_HEIGHT = 7
BUFFER = 1
SLICE_BUFFERED = SLICE_WIDTH + BUFFER*2
NUM_SLICES_WIDTH = int(WIDTH / SLICE_WIDTH)
NUM_SLICES_HEIGHT = int(HEIGHT / SLICE_HEIGHT)

#image is padded slice, width and height is slice size unpadded
def bilinear_interpolation(image, width, height, channels, scale):
    #dimensions of output image unpadded
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    #outputs unpadded image
    output_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    #based on total image, not subsection
    width_ratio = (WIDTH - 1) / (WIDTH_OUT - 1) if new_width > 1 else 0.0
    height_ratio = (HEIGHT - 1) / (HEIGHT_OUT - 1) if new_height > 1 else 0.0

    smallest_x = 100
    smallest_y = 100

    biggest_x = 0
    biggest_y = 0

    for y_out in range(new_height):
        for x_out in range(new_width):

            x_in = x_out * width_ratio
            y_in = y_out * height_ratio
            
            #shouldn't be affected by doing slice instead of whole image
            #finds location of point
            x0 = int(np.floor(x_in))
            y0 = int(np.floor(y_in))
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)

            #used for testing which pixels are referrenced, not in calculations
            smallest_x = min(x0, x1, smallest_x)
            smallest_y = min(y0, y1, smallest_y)

            biggest_x = max(x0, x1, biggest_x)
            biggest_y = max(y0, y1, biggest_y)
            
            dx = x_in - x0
            dy = y_in - y0
            
            w00 = (1 - dx) * (1 - dy)
            w10 = dx * (1 - dy)
            w01 = (1 - dx) * dy
            w11 = dx * dy
            
            for c in range(channels):
                R_val = (
                    w00 * image[y0, x0, c] +
                    w10 * image[y0, x1, c] +
                    w01 * image[y1, x0, c] +
                    w11 * image[y1, x1, c]
                )
                output_image[y_out, x_out, c] = np.clip(round(R_val), 0, 255)
                
    return output_image

def upscale_image(image_tile):
    upscaled_tile = zoom(image_tile, (2, 2, 1), order=1)  # order=1 for bilinear interpolation
    return upscaled_tile

def extract_and_upscale(image, x, y, scale=2, method='scipy'):
    """Extracts a 7x7 section, adds reflection padding, and upscales."""
    assert image.shape == (28, 28, 3), "Input image must be 28x28 with 3 channels"

    # Extract 7x7 section
    section = image[y:y+SLICE_HEIGHT, x:x+SLICE_WIDTH, :]

    # Apply reflection padding (1 pixel on each side)
    #padded_section = np.pad(section, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    #padded_section = np.pad(section, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    # print(padded_section.shape)
    # print(padded_section)
    # print("\n")

    # Create a 9x9x3 array filled with zeros (padding)
    padded_section = np.full((SLICE_BUFFERED, SLICE_BUFFERED, CHANNELS), 0, dtype=section.dtype)

    # Place the 7x7 array in the center
    padded_section[1:SLICE_BUFFERED-1, 1:SLICE_BUFFERED-1, :] = section

    #top left corner
    if(y > 0 and x > 0):
        padded_section[0, 0, :] = image[y-1, x-1, :]

    # Top-right corner
    if(y > 0 and x < WIDTH - 1):
        padded_section[0, -1, :] = image[y-1, x+SLICE_WIDTH, :]

    # Bottom-left corner
    if(y < HEIGHT - 1 and x > 0):
        padded_section[-1, 0, :] = image[y+SLICE_HEIGHT, x-1, :]

    # Bottom-right corner
    if(y < HEIGHT - 1 and x < WIDTH - 1):
        padded_section[-1, -1, :] = image[y+SLICE_HEIGHT, x+SLICE_WIDTH, :]

    #top row
    if(y > 0):
        padded_section[0, 1:-1, :] = image[y-1, x:x+SLICE_WIDTH, :]
    
    #bottom row
    if(y < HEIGHT-1):
        padded_section[-1, 1:-1, :] = image[y+SLICE_HEIGHT, x:x+SLICE_WIDTH, :]

    #left column
    if(x > 0):
        padded_section[1:-1, 0, :] = image[y:y+SLICE_HEIGHT, x-1, :]
    
    #right column:
    if(x < WIDTH-1):
        padded_section[1:-1, -1, :] = image[y:y+SLICE_HEIGHT, x+SLICE_WIDTH, :]

    #print(padded_section)
    #print("End of Slice\n\n")

    # Upscale using the selected method
    if method == 'scipy':
        upscaled_section = zoom(padded_section, (scale, scale, 1), order=1)
    elif method == 'bilin':
        #print(padded_section)
        upscaled_section = bilin_interpolation(padded_section, x, y)
        #upscaled_section = bilin_interpolation(section, 7, 7, 3, 2)
        #upscaled_section = bilin_interpolation(image, x, y)
        #print(upscaled_section)
        #print("\n\n")
        #print(upscaled_section[2:-2, 2:-2, :])
    else:
        raise ValueError("Invalid method. Choose either 'scipy' or 'bilin.")

    # Clip values and convert to uint8
    upscaled_section = np.clip(upscaled_section, 0, 255).astype(np.uint8)

    # Remove the 2-pixel border to get the final 14x14 result
    #return upscaled_section[2:-2, 2:-2, :]
    return upscaled_section

def compare_upscaled_to_ideal(ideal, upscaled_slice, start_row, start_col, threshold=5):
    """
    Compares a section of the upscaled image to the ideal image and counts discrepancies.

    Args:
        ideal (numpy array): The ideal (target) image.
        upscaled_slice (numpy array): The upscaled slice of the image to compare.
        start_row (int): The starting row index of the region to compare in the ideal image.
        start_col (int): The starting column index of the region to compare in the ideal image.
        threshold (int): The threshold for the allowed error between ideal and upscaled values.

    Returns:
        int: The number of discrepancies found.
    """
    num_wrong = 0

    # Iterate over the upscaled slice region (scaled size of SLICE_HEIGHT and SLICE_WIDTH)
    for i in range(SLICE_HEIGHT * SCALE):
        for j in range(SLICE_WIDTH * SCALE):
            for c in range(CHANNELS):  # Iterate through RGB channels
                ideal_value = ideal[start_row + i, start_col + j, c]
                upscaled_value = upscaled_slice[i, j, c]

                # If the difference between ideal and upscaled value is greater than the threshold
                if abs(ideal_value - upscaled_value) > threshold:
                    num_wrong += 1
                    print(f"Error at row {start_row + i}, col {start_col + j}, channel {c}: Ideal: {ideal_value}, Calculated: {upscaled_value}")
                # Uncomment the next line if you want to print correct matches
                # else:
                #     print(f"Correct at row {start_row + i}, col {start_col + j}, channel {c}: Ideal: {ideal_value}, Calculated: {upscaled_value}")

    print("Number of errors in slice:", num_wrong)
    return num_wrong

#image is padded slice, width and height is slice size unpadded
def bilin_interpolation3(image, x_start, y_start, print_result):

    image_for_calcs = np.zeros((28, 28, 3), dtype=np.uint8)

    if y_start == 0:
        for row in range(y_start+8): #loops through rows 0 to 7
            image_for_calcs[row, :, :] = image[row, :, :]

    if y_start == 7 or y_start == 14:
        for row in range(y_start-1, y_start+7): #loops through 6 to 14 or 13 to 21
            image_for_calcs[row, :, :] = image[row, :, :]

    if y_start == 21:
        for row in range(y_start-1, y_start+6):
            image_for_calcs[row, :, :]

    image_for_calcs = np.copy(image)

    #dimensions of output image unpadded
    new_width = int(SLICE_WIDTH * SCALE)
    new_height = int(SLICE_HEIGHT * SCALE)
    
    #outputs unpadded image
    output_image = np.zeros((new_height, new_width, CHANNELS), dtype=np.uint8)
    
    #based on total image, not subsection
    width_ratio = (WIDTH - 1) / (WIDTH_OUT - 1) if new_width > 1 else 0.0
    height_ratio = (HEIGHT - 1) / (HEIGHT_OUT - 1) if new_height > 1 else 0.0

    #location of pixel being calculated in output_image
    x_location = 0
    y_location = 0 

    smallest_x = 100
    smallest_y = 100

    biggest_x = 0
    biggest_y = 0

    big_x_pos = 0
    big_y_pos = 0

    for y_out in range(y_start*SCALE, y_start*SCALE+new_height):
        for x_out in range(x_start*SCALE, x_start*SCALE+new_height):

            x_in = x_out * width_ratio
            y_in = y_out * height_ratio
            
            #shouldn't be affected by doing slice instead of whole image
            #finds location of point
            x0 = int(np.floor(x_in))
            y0 = int(np.floor(y_in))
            x1 = min(x0 + 1, WIDTH - 1)
            y1 = min(y0 + 1, HEIGHT - 1)

            #used for testing which pixels are referrenced, not in calculations
            smallest_x = min(x0, x1, smallest_x)
            smallest_y = min(y0, y1, smallest_y)

            biggest_x = max(x0, x1, biggest_x)
            biggest_y = max(y0, y1, biggest_y)
            
            dx = x_in - x0
            dy = y_in - y0
            
            w00 = (1 - dx) * (1 - dy)
            w10 = dx * (1 - dy)
            w01 = (1 - dx) * dy
            w11 = dx * dy
            
            for c in range(CHANNELS):
                top_left = image_for_calcs[y0, x0, c]
                top_right = image_for_calcs[y0, x1, c]
                bottom_left = image_for_calcs[y1, x0, c]
                bottom_right = image_for_calcs[y1, x1, c]

                R_val = (
                    w00 * image_for_calcs[y0, x0, c] +
                    w10 * image_for_calcs[y0, x1, c] +
                    w01 * image_for_calcs[y1, x0, c] +
                    w11 * image_for_calcs[y1, x1, c]
                )
                output_image[y_location, x_location, c] = np.clip(round(R_val), 0, 255)

            x_location += 1

        x_location = 0
        y_location += 1
                
    if print_result: print(f"Block at {x_start}, {y_start} uses rows {smallest_y} to {biggest_y} and columns {smallest_x} to {biggest_x}")
    return output_image#, smallest_x, smallest_y, biggest_x, biggest_y


#image is padded slice, width and height is slice size unpadded
def bilin_interpolation2(image_section, x_start, y_start, print_result):

    #BUILD IMAGE SECTION

    image_for_calcs = np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
    temp_row = 0

    if y_start == 0:
        for row in range(y_start+SLICE_HEIGHT+1): #loops through rows 0 to 7
            image_for_calcs[row, :, :] = image_section[temp_row, :, :]
            temp_row += 1

    elif y_start == HEIGHT - SLICE_HEIGHT:
        for row in range(y_start-1, y_start+7):
            image_for_calcs[row, :, :] = image_section[temp_row, :, :]
            temp_row += 1

    elif y_start % SLICE_HEIGHT == 0:
        for row in range(y_start-1, y_start+8): #loops through 6 to 14 or 13 to 21
            image_for_calcs[row, :, :] = image_section[temp_row, :, :]
            temp_row += 1


    #dimensions of output image unpadded
    new_width = int(SLICE_WIDTH * SCALE)
    new_height = int(SLICE_HEIGHT * SCALE)
    
    #outputs unpadded image
    output_image = np.zeros((new_height, new_width, CHANNELS), dtype=np.uint8)
    
    #based on total image, not subsection
    width_ratio = (WIDTH - 1) / (WIDTH_OUT - 1) if new_width > 1 else 0.0
    height_ratio = (HEIGHT - 1) / (HEIGHT_OUT - 1) if new_height > 1 else 0.0

    #location of pixel being calculated in output_image
    x_location = 0
    y_location = 0 

    smallest_x = 100
    smallest_y = 100

    biggest_x = 0
    biggest_y = 0


    for y_out in range(y_start*SCALE, y_start*SCALE+new_height):
        for x_out in range(x_start*SCALE, x_start*SCALE+new_height):

            x_in = x_out * width_ratio
            y_in = y_out * height_ratio
            
            #shouldn't be affected by doing slice instead of whole image
            #finds location of point
            x0 = int(np.floor(x_in))
            y0 = int(np.floor(y_in))
            x1 = min(x0 + 1, WIDTH - 1)
            y1 = min(y0 + 1, HEIGHT - 1)

            #used for testing which pixels are referrenced, not in calculations
            smallest_x = min(x0, x1, smallest_x)
            smallest_y = min(y0, y1, smallest_y)

            biggest_x = max(x0, x1, biggest_x)
            biggest_y = max(y0, y1, biggest_y)
            
            dx = x_in - x0
            dy = y_in - y0
            
            w00 = (1 - dx) * (1 - dy)
            w10 = dx * (1 - dy)
            w01 = (1 - dx) * dy
            w11 = dx * dy
            
            for c in range(CHANNELS):

                R_val = (
                    w00 * image_for_calcs[y0, x0, c] +
                    w10 * image_for_calcs[y0, x1, c] +
                    w01 * image_for_calcs[y1, x0, c] +
                    w11 * image_for_calcs[y1, x1, c]
                )
                output_image[y_location, x_location, c] = np.clip(round(R_val), 0, 255)

            x_location += 1

        x_location = 0
        y_location += 1
                
    if print_result: print(f"Block at {x_start}, {y_start} uses rows {smallest_y} to {biggest_y} and columns {smallest_x} to {biggest_x}")
    return output_image#, smallest_x, smallest_y, biggest_x, biggest_y

if __name__ == '__main__':
    
    image_tile = np.load('image_coin_tile.npy')
    image_tile = image_tile.astype(np.float32)

    image = image_tile.reshape(HEIGHT, WIDTH, CHANNELS)

    ideal = upscale_image(image_tile)
    ideal = np.round(ideal)  

    slices = [np.zeros((SLICE_HEIGHT*SCALE, SLICE_WIDTH*SCALE, CHANNELS), dtype=np.uint8) for _ in range(NUM_SLICES_HEIGHT*NUM_SLICES_WIDTH)]
    slice_idx = 0

    for row in range(0, HEIGHT, SLICE_HEIGHT):
        for col in range(0, WIDTH, SLICE_WIDTH):
            slices[slice_idx] = bilin_interpolation3(image, col, row, False)
            slice_idx += 1


    #image section to hold values from FIFOs that are passed into bilinear calculation
    image_section = np.zeros((SLICE_HEIGHT + BUFFER*2, WIDTH, CHANNELS), dtype=np.uint8)

    #first set of fifos and overlap fifos
    fifos_first = [Queue() for _ in range(SLICE_HEIGHT + BUFFER*2)]
    fifo_overlap_first = [Queue() for _ in range(BUFFER*2)]

    #second set of fifos and overlap fifos
    fifos_second = [Queue() for _ in range(SLICE_HEIGHT + BUFFER*2)]
    fifo_overlap_second = [Queue() for _ in range(BUFFER*2)]

    pull_from_top = True

    fifo_output_slices = [np.zeros((SLICE_HEIGHT*SCALE, SLICE_WIDTH*SCALE, CHANNELS), dtype=np.uint8) for _ in range(NUM_SLICES_HEIGHT*NUM_SLICES_WIDTH)]
    slice_idx = 0

    start_row = 0

    # Loop through all rows in the input image
    for row_idx in range(HEIGHT):

        #tracks which fifo is being written into next
        if row_idx < SLICE_HEIGHT + BUFFER:
            fifo_idx = row_idx + 1 
        else:
            fifo_idx = (row_idx - (SLICE_HEIGHT + BUFFER)) % SLICE_HEIGHT + BUFFER*2

        # Fill FIFOs with pixel values
        for col_idx in range(WIDTH):
            for ch in image[row_idx, col_idx]:
                if pull_from_top:
                    fifos_first[fifo_idx].put(ch)
                if not pull_from_top:
                    fifos_second[fifo_idx].put(ch)


        if row_idx >= SLICE_HEIGHT and (row_idx % SLICE_HEIGHT == 0 or row_idx == HEIGHT - 1):

            num_rows = SLICE_HEIGHT + (BUFFER if row_idx in {SLICE_HEIGHT, HEIGHT - 1} else BUFFER * 2)
            
            #on first fill, fill image rows 0-7 from first set of fifos
            if row_idx == SLICE_HEIGHT:

                for col_idx in range(WIDTH):
                    for ch in range(CHANNELS):
                        for i in range(num_rows): #num_rows = 8 for rows 0 through SLICE_HEIGHT
                            temp_value = fifos_first[i+1].get()
                            image_section[i, col_idx, ch] = temp_value

                            if i == SLICE_HEIGHT - 1:
                                fifo_overlap_first[0].put(temp_value)
                            if i == SLICE_HEIGHT:
                                fifo_overlap_first[1].put(temp_value)


            #otherwise if filling rows in the middle, switch between which FIFO is being used
            elif row_idx % SLICE_HEIGHT == 0:
                for col_idx in range(WIDTH):
                    for ch in range(CHANNELS):
                        for i in range(num_rows): #6 through 14

                            #if currently pulling from first fifos (odd numbered slice rows)
                            if pull_from_top:
                                temp_value = fifos_first[i].get()
                                image_section[i, col_idx, ch] = temp_value

                                if i == SLICE_HEIGHT:
                                    fifo_overlap_first[0].put(temp_value)
                                if i == SLICE_HEIGHT + 1:
                                    fifo_overlap_first[1].put(temp_value)
                            
                            #if currently pulling from second fifos (even numbered slice rows)
                            if not pull_from_top:
                                temp_value = fifos_second[i].get()
                                image_section[i, col_idx, ch] = temp_value

                                if i == SLICE_HEIGHT:
                                    fifo_overlap_second[0].put(temp_value)
                                if i == SLICE_HEIGHT + 1:
                                    fifo_overlap_second[1].put(temp_value)
                
            
            #on last row fill;
            elif row_idx == HEIGHT - 1:
                for col_idx in range(WIDTH):
                    for ch in range(CHANNELS):
                        for i in range(num_rows): #20 through 27

                            #if currently pulling from first fifos (odd numbered slice rows)
                            if pull_from_top:
                                temp_value = fifos_first[i].get()
                                image_section[i, col_idx, ch] = temp_value

                                if i == SLICE_HEIGHT - 1:
                                    fifo_overlap_first[0].put(temp_value)
                                if i == SLICE_HEIGHT:
                                    fifo_overlap_first[1].put(temp_value)
                            
                            #if currently pulling from second fifos (even numbered slice rows)
                            if not pull_from_top:
                                temp_value = fifos_second[i].get()
                                image_section[i, col_idx, ch] = temp_value

                                if i == SLICE_HEIGHT - 1:
                                    fifo_overlap_second[0].put(temp_value)
                                if i == SLICE_HEIGHT:
                                    fifo_overlap_second[1].put(temp_value)

   
            #FIFO usage doesn't change this step yet
            for col in range(0, WIDTH, SLICE_WIDTH):
                fifo_output_slices[slice_idx] = bilin_interpolation2(image_section, col, start_row, False)
                slice_idx += 1

            start_row += SLICE_HEIGHT
            
        #     #print("\n\n")

            # Verify slices
            for temp_slice_idx in range(slice_idx - NUM_SLICES_WIDTH, slice_idx):
                print(f"Slices {temp_slice_idx} are_equal: {np.array_equal(slices[temp_slice_idx], fifo_output_slices[temp_slice_idx])}")

            
            if row_idx % SLICE_HEIGHT == 0:
                for col in range(WIDTH):
                    for ch in range(CHANNELS):

                        if pull_from_top:
                            fifos_second[0].put(fifo_overlap_first[0].get())
                            fifos_second[1].put(fifo_overlap_first[1].get())
                        if not pull_from_top:
                            fifos_first[0].put(fifo_overlap_second[0].get())
                            fifos_first[1].put(fifo_overlap_second[1].get())

                #switch which FIFOs are written to/pulled from next
                pull_from_top = not pull_from_top


    row_one = np.hstack((fifo_output_slices[0], fifo_output_slices[1], fifo_output_slices[2], fifo_output_slices[3]))
    row_two = np.hstack((fifo_output_slices[4], fifo_output_slices[5], fifo_output_slices[6], fifo_output_slices[7]))
    row_three = np.hstack((fifo_output_slices[8], fifo_output_slices[9], fifo_output_slices[10], fifo_output_slices[11]))
    row_four = np.hstack((fifo_output_slices[12], fifo_output_slices[13], fifo_output_slices[14], fifo_output_slices[15]))

    # Vertically stack the rows to form the final 4x4 grid
    combined_array = np.vstack((row_one, row_two, row_three, row_four))  

    #upscaled_image = upscale_image_pytorch(image_tile)
    #upscaled_image = bilinear_interpolation(image_tile, 28, 28, 3, 2)

    num_wrong = 0
    num_right = 0

    for i in range(56):
        for j in range(56):
            # Check each of the three channels (RGB)
            for c in range(3):  # Channels 0 (R), 1 (G), 2 (B)
                if abs(ideal[i, j, c] - combined_array[i, j, c]) > 5:
                    num_wrong += 1
                    #print(f"Error at row {i}, col {j}, channel {c}: Ideal: {ideal[i, j, c]}, Calculated: {combined_array[i, j, c]}")
                #else:
                    #print(f"Matches at row {i}, col {j}, channel {c}: Ideal: {ideal[i, j, c]}, Calculated: {combined_array[i, j, c]}")
         
    #print("Num correct:     ", 56*56*3 - num_wrong)
    print("Number of errors:", num_wrong)
