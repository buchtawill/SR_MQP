import numpy as np
import torch
import torch.nn.functional as F

def bilinear_interpolate(npy_array, new_height, new_width):
    # Convert numpy array to PyTorch tensor
    tensor = torch.tensor(npy_array, dtype=torch.float32)

    # Ensure it's in the format (1, 1, H, W) for interpolation
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform bilinear interpolation
    interpolated_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Remove batch and channel dimensions to get back 2D array
    interpolated_array = interpolated_tensor.squeeze().numpy()

    return interpolated_array


if __name__ == '__main__':
    
    image_tile = np.load('../comparisons/images/image_coin_tile.npy')
    image_tile = image_tile.astype(np.float32) / 256.
    
    image_tile = np.transpose(image_tile, (2, 0, 1))
    image_tile = np.pad(image_tile, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
    # print(image_tile.shape) # (3, 32, 32)
    # print(image_tile[0,0])  # Red channel, first row
    
    channel = image_tile[0]
    
    # print("\nComputed via FIFOs and partial sums:")
    # channel[row, col]
    # fifod_conv = fifo_psum_conv(channel, WEIGHTS[0], print_slider=False, add_bias_and_prelu=False)
    # print(fifod_conv[0])
    # exit()
    first_fmap = emulate_pytorch(image_tile, bias_prelu=True)
    print(first_fmap[0])
    exit()
    
    print("Ideal convolution results after bias and prelu:")
    ideal = np.zeros((28, 28))
    for row in range(28):
        for col in range(28):
            ideal[row, col] = get_real_conv_result(col, row, channel)
    # print(ideal)
    
    num_wrong = 0
    for i in range(28):
        for j in range(28):
            if(not float_compare(ideal[i,j], fifod_conv[i,j])):
                num_wrong+=1
                print(f"Error at row {i}, col {j}: Ideal: {ideal[i,j]:9.6f}, FIFO: {fifod_conv[i,j]:9.6f}")
    print("Num correct:     ", 28*28 - num_wrong)
    print("Number of errors:", num_wrong)
    
    # print(channel[2, 0:5]) # first five columns of the third row