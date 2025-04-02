import torch
import numpy as np
from matplotlib import pyplot as plt

from FSRCNN import *

def prelu(value:float, weight:float):
    if(value >= 0.0):
        return value
    
    else:
        return weight * value

def text_to_featuremaps(path:str, num_feature_maps)->np.ndarray:
    """
    Reads a text file containing feature maps and returns them as a numpy array.
    Args:
        path (str): Path to the text file containing the feature maps.
        num_feature_maps (int): Number of feature maps to read from the file.
    Returns:
        np.ndarray: Feature maps as a numpy array of shape (44, 28, 28)
    """
    feature_maps = []
    
    with open(path, "r") as file:
        lines = file.readlines()
    
    # Find the start of the feature maps
    start_idx = None
    for i, line in enumerate(lines):
        if "INFO [conv2d] Feature map 0:" in line:
            start_idx = i + 1  # Start from the next line
            break

    if start_idx is None:
        raise ValueError("Feature map start not found in the file.")
    
    # Read 44 feature maps, each consisting of 28x28 lines
    feature_map_size = 28 * 28

    for i in range(num_feature_maps):
        # Skip the first two lines (header and empty line)
        start = start_idx + i * (feature_map_size + 2)
        end = start + feature_map_size
        feature_map = np.array([float(x.strip()) for x in lines[start:end]]).reshape(28, 28)
        feature_maps.append(feature_map)

    return np.array(feature_maps)  # Shape: (num_feature_maps, 28, 28)

def low_level_extraction(input_streams, weight_matrix:np.ndarray, biases:np.ndarray, prelus:np.ndarray):
    
    in_ch  = 3
    out_ch = 44
    # Input: a list of 3 FIFOs (lists), each FIFO has 28*28 values (range 0-1)
    # Output: a list of 44 FIFOs (lists), each FIFO has 28*28 values
    output_streams = []
    for i in range(out_ch):
        output_streams.append([])
    
    # input_buffer = []
    # for i in range(in_ch):
    #     input_buffer.append([])
    
    # Each processing element will be 44 x 5
    # psumx[channel, idx]
    slider = np.zeros((in_ch, 5))
    
    psum1 = []
    psum2 = []
    psum3 = []
    psum4 = []
    
    for i in range(out_ch):
        psum1.append([])
        psum2.append([])
        psum3.append([])
        psum4.append([])
    
    for row in range(32):
        # 1. Prepare the slider
        for ch in range(in_ch):
            for idx in range(4):
                if((row < 2) or (row >= 30) or (idx < 2)):
                    slider[ch][idx] = 0.0
                else:
                    slider[ch][idx] = input_streams[ch].pop(0)
        
        # 2. Go across the row cols 4-31
        for col in range(4, 32): 
            # Read the next value into the slider
            for ch in range(in_ch):
                if((row < 2) or (row >= 30) or (col >= 30)):
                    slider[ch][4] = 0.0
                else:
                    slider[ch][4] = input_streams[ch].pop(0)
            
            for map in range(out_ch):
                
                row1_psum, row2_psum, row3_psum, row4_psum = 0.0, 0.0, 0.0, 0.0
                mac0, mac1, mac2, mac3, mac4 = 0.0, 0.0, 0.0, 0.0, 0.0
                
                # MAC across all input channels to compute the partial sum
                for ch in range(in_ch):
                    if(row < 28):
                        # mac and save it for later
                        mac0 += np.dot(weight_matrix[map][ch][0], slider[ch])
                        
                    if(row >=1 and row < 29):
                        mac1 += np.dot(weight_matrix[map][ch][1], slider[ch])

                    if(row >=2 and row < 30):
                        mac2 += np.dot(weight_matrix[map][ch][2], slider[ch])
                    
                    if(row >=3 and row < 31):
                        mac3 += np.dot(weight_matrix[map][ch][3], slider[ch])
                    
                    if(row >= 4):
                        mac4 += np.dot(weight_matrix[map][ch][4], slider[ch])
                
                if(row < 28):
                    psum1[map].append(mac0)
                    
                if(row >=1 and row < 29):
                    row1_psum = psum1[map].pop(0)
                    psum2[map].append(row1_psum + mac1)
                    
                if(row >=2 and row < 30):
                    row2_psum = psum2[map].pop(0)
                    psum3[map].append(row2_psum + mac2)
                    
                if(row >=3 and row < 31):
                    row3_psum = psum3[map].pop(0)
                    psum4[map].append(row3_psum + mac3)
                    
                if(row >= 4):
                    row4_psum = psum4[map].pop(0)
                    final_sum = row4_psum + mac4
                    final_sum += biases[map]
                    final_value = prelu(final_sum, prelus[map])
                    output_streams[map].append(final_value)
                
            for ch in range(in_ch):
                slider[ch][0] = slider[ch][1]
                slider[ch][1] = slider[ch][2]
                slider[ch][2] = slider[ch][3]
                slider[ch][3] = slider[ch][4]
                    
    return output_streams


def fifo_psum_conv(channel_arr, weight_mat, add_bias_and_prelu=True, print_slider=False):
    
    psums = [[], [], [], []]
    outputs = np.zeros((28,28))
    for row in range(32):
        for col in range(28):
            slider = channel_arr[row, col:col+5]
            if(print_slider):
                print(slider)
            if(row < 28):
                psums[0].append(np.sum(slider * weight_mat[0]))
            if(row < 29 and row > 0):
                psums[1].append(np.sum(slider * weight_mat[1]) + psums[0].pop(0))
            if(row < 30 and row > 1):
                psums[2].append(np.sum(slider * weight_mat[2]) + psums[1].pop(0))
            if(row < 31 and row > 2):
                psums[3].append(np.sum(slider * weight_mat[3]) + psums[2].pop(0))
                
            if(row > 3):
                pre_activation = np.sum(slider * weight_mat[4]) + psums[3].pop(0)
                if(add_bias_and_prelu):
                    pre_activation += BIAS_1
                    output = np.maximum(0, pre_activation) + PRELU_1 * np.minimum(0, pre_activation)
                else:
                    output = pre_activation    
                outputs[row - 4, col] = output
        if(print_slider):
            print("\n")
    
    ############################## Do the first 4 rows ##############################
    # print("First row slider")
    '''for i in range(28):
        slider = channel_arr[0, i:i+5]
        # print(slider)
        psums[0].append(np.sum(slider * WEIGHTS[0,0]))
        
    # print("\nSecond row slider")
    for i in range(28):
        slider = channel_arr[1, i:i+5]
        # print(slider)
        psums[0].append(np.sum(slider * WEIGHTS[0,0]))
        psums[1].append(np.sum(slider * WEIGHTS[0,1]) + psums[0].pop(0))
        
    # print("\nThird row slider")
    for i in range(28):
        slider = channel_arr[2, i:i+5]
        # print(slider)
        psums[0].append(np.sum(slider * WEIGHTS[0,0]))
        psums[1].append(np.sum(slider * WEIGHTS[0,1]) + psums[0].pop(0))
        psums[2].append(np.sum(slider * WEIGHTS[0,2]) + psums[1].pop(0))
        
    # print("\nFourth row slider")
    for i in range(28):
        slider = channel_arr[3, i:i+5]
        # print(slider)
        psums[0].append(np.sum(slider * WEIGHTS[0,0]))
        psums[1].append(np.sum(slider * WEIGHTS[0,1]) + psums[0].pop(0))
        psums[2].append(np.sum(slider * WEIGHTS[0,2]) + psums[1].pop(0))
        psums[3].append(np.sum(slider * WEIGHTS[0,3]) + psums[2].pop(0))
        
    # # print("\nFifth row slider / start printing results")
    ############################## Pipeline is ready ##############################
    for row in range(4, 32):
        for i in range(28):
            slider = channel_arr[row, i:i+5]
            # print(slider)
            if(row < 28):
                psums[0].append(np.sum(slider * WEIGHTS[0,0]))
            if(row < 29):
                psums[1].append(np.sum(slider * WEIGHTS[0,1]) + psums[0].pop(0))
            if(row < 30):
                psums[2].append(np.sum(slider * WEIGHTS[0,2]) + psums[1].pop(0))
            if(row < 31):
                psums[3].append(np.sum(slider * WEIGHTS[0,3]) + psums[2].pop(0))
            pre_activation = np.sum(slider * WEIGHTS[0,4]) + psums[3].pop(0) + BIAS_1
            output = np.maximum(0, pre_activation) + PRELU_1 * np.minimum(0, pre_activation)
            outputs[row-4, i] = output'''
        
    # print(len(psums[0]))
    # print(len(psums[1]))
    # print(len(psums[2]))
    # print(len(psums[3]))       
    
    return outputs

def float_compare(a, b, epsilon=0.000001):
    return abs(a - b) < epsilon

def emulate_pytorch(tile, bias_prelu=True):
    fifod_conv0 = fifo_psum_conv(tile[0], WEIGHTS[0], False)
    fifod_conv1 = fifo_psum_conv(tile[1], WEIGHTS[1], False)
    fifod_conv2 = fifo_psum_conv(tile[2], WEIGHTS[2], False)
    result = fifod_conv0 + fifod_conv1 + fifod_conv2
    
    if(bias_prelu):
        result += BIAS_1
        result = np.maximum(0, result) + PRELU_1 * np.minimum(0, result)
    return result

def manual_2d_convolution(input_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform 2D convolution manually using two for loops and np.dot().

    Args:
        input_array (np.ndarray): Input feature map of shape (H, W).
        kernel (np.ndarray): Filter kernel of shape (kH, kW).

    Returns:
        np.ndarray: Convolved feature map.
    """
    H, W = input_array.shape  # Input height and width
    kH, kW = kernel.shape     # Kernel height and width

    # Compute output shape
    output_H = H - kH + 1
    output_W = W - kW + 1
    output = np.zeros((output_H, output_W))  # Initialize output feature map

    # Perform convolution using two loops
    for i in range(output_H):  # Slide vertically
        for j in range(output_W):  # Slide horizontally
            region = input_array[i:i+kH, j:j+kW]  # Extract region matching kernel size
            output[i, j] = np.dot(region.flatten(), kernel.flatten())  # Compute dot product

    return output

def transposed_convolution_9x9(input_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Performs transposed convolution using standard convolution by padding and upsampling.

    Args:
        input_array (np.ndarray): Input feature map of shape (28, 28).
        kernel (np.ndarray): 9x9 filter.

    Returns:
        np.ndarray: Output feature map of shape (56, 56).
    """

    # Step 1: Insert zeros between pixels (upsampling by 2x)
    # upsampled = np.zeros((56, 56))  # (28 * 2, 28 * 2) -> Now correct
    # upsampled[1::2, 1::2] = input_array  # Place input at odd indices


    # # Step 2: Pad to match 56x56 output after convolution
    # padded_input = np.pad(upsampled, pad_width=(
    #     (4, 4),  # Padding for height
    #     (4, 4)   # Padding for width
    # ), mode='constant', constant_values=0)

    output = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            # Each input pixel contributes to a 9x9 patch in the output
            output[i*2 : i*2 + 9, j*2 : j*2 + 9] += input_array[i, j] * kernel
    
    return output

def compare_transposed(image_tile_2828):
    weight_tensor = torch.tensor(WEIGHTS_DECONV).unsqueeze(0).unsqueeze(0)

    # Define transposed convolution layer
    deconv = torch.nn.ConvTranspose2d(
        in_channels=1, 
        out_channels=1, 
        kernel_size=(9, 9), 
        stride=2,  # Adjust stride if necessary
        padding=4,  # Adjust padding if necessary to get 56x56 output
        output_padding=1,
        bias=False  # No bias to match manual convolution
    )

    # Set weights manually
    with torch.no_grad():
        deconv.weight.copy_(weight_tensor)
    
    input_ch = image_tile_2828[0,:,:]
    input_ch_tensor = torch.tensor(input_ch).unsqueeze(0)
    pytorch_deconv = deconv(input_ch_tensor)
    pytorch_deconv = pytorch_deconv.squeeze(0).detach().numpy()

    # input ch is shape (28x28)
    # manual_deconv = transposed_convolution_9x9(input_ch, WEIGHTS_DECONV)
    
    diff = np.abs(pytorch_deconv - manual_deconv)

    return diff

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN(upscale_factor=2, color_space='rgb').to(device)
    model.load_state_dict(torch.load('./saved_weights/example_vitis_hls_weights_44.pth', weights_only=True))
    
    low_res_coin = torch.from_numpy(np.load('../comparisons/images/image_coin_tile.npy'))
    low_res_coin = low_res_coin.float()
    
    # Change shape from (28, 28, 3) → (1, 3, 28, 28) for pytorch
    low_res_coin = low_res_coin.permute(2, 0, 1).unsqueeze(0) / 256.
    image_tile_2828 = low_res_coin.squeeze(0).detach().cpu().numpy()
    
    conv_weights = np.load('./saved_weights/extraction_conv_44w.npy')   # 44, 3, 5, 5
    conv_bias    = np.load('./saved_weights/extraction_conv_44b.npy')   # 44
    prelu_weight = np.load('./saved_weights/extraction_conv_44pre.npy') # 44
    
    inference = model.feature_extraction(low_res_coin.to(device))
    inference = model.shrink(inference)
    inference = model.map(inference)
    inference = model.expand(inference)
    inference = inference.squeeze(0).cpu().detach().numpy()
    
    input_channels = [[], [], []]
    
    for ch in range(3):
        for row in range(28):
            for col in range(28):
                input_channels[ch].append(image_tile_2828[ch][row][col])
             
    # To check that the input is stored correctly:     
    # for ch in range(3):
    #     print(f"Channel {ch}: ", end='')
    #     print(input_channels[ch][:5])

    resulting_maps = low_level_extraction(input_channels, conv_weights, conv_bias, prelu_weight)
    
    # print(len(resulting_maps[0]))
    # for i in range(28):
    #     result = resulting_maps[0].pop(0)
    #     print(result - inference[0,0,i])
        
    last_conv = text_to_featuremaps('../../HLS/build/conv2d_proj/solution1/csim/report/conv2d_top_csim.log', 44)
    errors = np.abs(inference - last_conv).flatten()
    
    pct_err = np.abs((last_conv - inference) / inference).flatten()
    pct_err[pct_err > 20] = np.nan # divide by 0
    
    avg = np.mean(errors)
    worst = np.max(errors)
    print(f"Average error: {avg:9.6f} ({np.mean(pct_err):2.2f}%)")
    print(f"Worst error:   {worst:9.6f} ({np.max(pct_err):2.2f}%)")
    
    print(f"Average error * 256: {avg*256:9.6f}")
    print(f"Worst error * 256:   {worst*256:9.6f}")
    
    # First histogram (percent error)
    # plt.hist(errors, bins=50, alpha=0.5, label="Error", edgecolor='black', color='cyan')
    plt.hist(errors*256, bins=50, alpha=0.5, label="Error", edgecolor='black', color='cyan')
    plt.xlabel("Error (Pixel scale)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Error")
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()

    
    
    