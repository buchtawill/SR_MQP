import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt

from FSRCNN import *

def prelu(value:float, weight:float):
    if(value >= 0.0):
        return value
    
    else:
        return weight * value

def text_to_featuremaps(path:str, num_feature_maps, feature_map_size=784)->np.ndarray:
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
    
    for i in range(num_feature_maps):
        # Skip the first two lines (header and empty line)
        start = start_idx + i * (feature_map_size + 2)
        end = start + feature_map_size
        
        # Totally horrible hard-coded value that needs to be changed
        if(feature_map_size == 784):
            feature_map = np.array([float(x.strip()) for x in lines[start:end]]).reshape(28, 28)
        else:
            feature_map = np.array([float(x.strip()) for x in lines[start:end]]).reshape(56, 56)
            
        feature_maps.append(feature_map)

    return np.array(feature_maps)  # Shape: (num_feature_maps, 28, 28)

def low_level_extraction(input_streams, weight_matrix:np.ndarray, biases:np.ndarray, prelus:np.ndarray)->list:
    
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

def get_next_val_tconv(row, col, input_stream):
    # Padding follows this logic:
    # Return 0 under the following conditions:
    # if row is 0, 1, 2, or 3
    # if row is 5, 7, 9, ...,  57
    # if row is >= 59
    # if col is 0, 1, 2, or 3
    # if col is 5, 7, 9, ..., 57
    # if col is >= 59
    # Otherwise, return actual value
    if((row <= 3) or ((row % 2) == 1) or (row >= 59)):
        return 0.0
    
    if((col <= 3) or ((col % 2) == 1) or (col >= 59)):
        return 0.0
    
    else:
        return input_stream.pop(0)

def low_level_deconv(input_streams, weight_matrix:np.ndarray, biases:np.ndarray)->list:
    
    in_ch  = 44
    out_ch = 3
    kernel_size = 9
    in_size = 64
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
    slider = np.zeros((in_ch, kernel_size))
    
    psum1 = []
    psum2 = []
    psum3 = []
    psum4 = []
    psum5 = []
    psum6 = []
    psum7 = []
    psum8 = []
    
    for i in range(out_ch):
        psum1.append([])
        psum2.append([])
        psum3.append([])
        psum4.append([])
        psum5.append([])
        psum6.append([])
        psum7.append([])
        psum8.append([])
        
    # input is a 28x28 tile - unpadded
    
    for row in range(64):
        # 1. Prepare the slider
        for ch in range(in_ch):
            for idx in range(kernel_size-1):
                slider[ch][idx] = get_next_val_tconv(row, idx, input_stream=input_streams[ch])
        
        # 2. Go across the row cols 4-31
        for col in range(8, in_size): 
            # Read the next value into the slider
            for ch in range(in_ch):
                slider[ch][8] = get_next_val_tconv(row, col, input_stream=input_streams[ch])
            
            
            for map in range(out_ch):
                
                row1_psum, row2_psum, row3_psum, row4_psum = 0.0, 0.0, 0.0, 0.0
                row5_psum, row6_psum, row7_psum, row8_psum = 0.0, 0.0, 0.0, 0.0
                mac0, mac1, mac2, mac3, mac4 = 0.0, 0.0, 0.0, 0.0, 0.0
                mac5, mac6, mac7, mac8       = 0.0, 0.0, 0.0, 0.0
                
                last_row_kernel = in_size - kernel_size
                # MAC across all input channels to compute the partial sum
                for ch in range(in_ch):
                    if(row <= last_row_kernel):
                        # mac and save it for later
                        mac0 += np.dot(weight_matrix[map][ch][0], slider[ch])
                        
                    if(row >= 1 and row <= last_row_kernel + 1):
                        mac1 += np.dot(weight_matrix[map][ch][1], slider[ch])
                    if(row >= 2 and row <= last_row_kernel + 2):
                        mac2 += np.dot(weight_matrix[map][ch][2], slider[ch])
                    if(row >= 3 and row <= last_row_kernel + 3):
                        mac3 += np.dot(weight_matrix[map][ch][3], slider[ch])
                    
                    if(row >= 4 and row <= last_row_kernel + 4):
                        mac4 += np.dot(weight_matrix[map][ch][4], slider[ch])
                    
                    if(row >= 5 and row <= last_row_kernel + 5):
                        mac5 += np.dot(weight_matrix[map][ch][5], slider[ch])
                    
                    if(row >= 6 and row <= last_row_kernel + 6):
                        mac6 += np.dot(weight_matrix[map][ch][6], slider[ch])
                    
                    if(row >= 7 and row <= last_row_kernel + 7):
                        mac7 += np.dot(weight_matrix[map][ch][7], slider[ch])
                    
                    if(row >= 8):
                        mac8 += np.dot(weight_matrix[map][ch][8], slider[ch])
                
                if(row <= last_row_kernel):
                    psum1[map].append(mac0)
                if(row >= 1 and row <= last_row_kernel + 1):
                    row1_psum = psum1[map].pop(0)
                    psum2[map].append(row1_psum + mac1)
                if(row >= 2 and row <= last_row_kernel + 2):
                    row2_psum = psum2[map].pop(0)
                    psum3[map].append(row2_psum + mac2)
                    
                if(row >= 3 and row <= last_row_kernel + 3):
                    row3_psum = psum3[map].pop(0)
                    psum4[map].append(row3_psum + mac3)
                  
                if(row >= 4 and row <= last_row_kernel + 4):
                    row4_psum = psum4[map].pop(0)
                    psum5[map].append(row4_psum + mac4)
                
                if(row >= 5 and row <= last_row_kernel + 5):
                    row5_psum = psum5[map].pop(0)
                    psum6[map].append(row5_psum + mac5)
                
                if(row >= 6 and row <= last_row_kernel + 6):
                    row6_psum = psum6[map].pop(0)
                    psum7[map].append(row6_psum + mac6)
                
                if(row >= 7 and row <= last_row_kernel + 7):
                    row7_psum = psum7[map].pop(0)
                    psum8[map].append(row7_psum + mac7)    
                    
                if(row >= 8):
                    row8_psum = psum8[map].pop(0)
                    final_sum = row8_psum + mac8
                    final_sum += biases[map]
                    # final_value = prelu(final_sum, prelus[map])
                    final_value = final_sum # no prelu for deconv
                    output_streams[map].append(final_value)
                
            for ch in range(in_ch):
                slider[ch][0] = slider[ch][1]
                slider[ch][1] = slider[ch][2]
                slider[ch][2] = slider[ch][3]
                slider[ch][3] = slider[ch][4]
                slider[ch][4] = slider[ch][5]
                slider[ch][5] = slider[ch][6]
                slider[ch][6] = slider[ch][7]
                slider[ch][7] = slider[ch][8]
                    
    return output_streams

def float_compare(a, b, epsilon=0.000001)->bool:
    """
    Return true if a == b with specified tolerance
    """
    return abs(a - b) < epsilon

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

def compare_hls_pytorch(pytorch_inference, num_per_map=784, nmaps=44):
    last_conv = text_to_featuremaps('../../HLS/build/conv2d_proj/solution1/csim/report/conv2d_top_csim.log', nmaps, num_per_map)
    
    errors = np.abs(pytorch_inference - last_conv).flatten()
    
    pct_err = np.abs((last_conv - pytorch_inference) / pytorch_inference).flatten()
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

def compare_tconv_conv_fmod(input:torch.tensor=None, fsrcnn:FSRCNN=None):
    # Get transposed convolution results. Operate on single channel input, single channel output of 28x28 input, 56x56 output
    upscale_factor = 2
    in_ch = 44
    out_ch = 3
    
    tconv_pyt = nn.ConvTranspose2d(
        in_channels=in_ch, out_channels=out_ch, 
        kernel_size=(9, 9),
        stride=(upscale_factor, upscale_factor),
        padding=(4, 4),
        output_padding=(upscale_factor-1, upscale_factor-1)
    )
    init.normal_(tconv_pyt.weight, mean=0.0, std=0.1)
    if(tconv_pyt.bias is not None):
        init.uniform_(tconv_pyt.bias, a=-0.1, b=0.1)
    
    # Override if given model
    if(fsrcnn is not None):
        tconv_pyt = fsrcnn.deconv
    
    tconv_weight_tensor = tconv_pyt.weight.detach().clone()
    tconv_bias_tensor = tconv_pyt.bias.detach().clone()
    
    # print(tconv_weight_tensor.shape) # [in_ch, out_ch, H, W]. Opposite from conv which is [out, in, H, W]
    
    # Try to get same results from tconv
    nconv_pyt = nn.Conv2d(
        in_channels=in_ch, out_channels=out_ch,
        kernel_size=(9, 9),
        stride=(1,1),
        padding=(0,0) # manually pad ourselves
    )
    
    # Make sure normal conv and transposed conv have the same weights
    with torch.no_grad():
        flipped_weights = torch.flip(tconv_weight_tensor, dims=[2, 3])
        flipped_weights = flipped_weights.permute(1, 0, 2, 3)
        # nconv_pyt.weight.data.copy_(tconv_weight_tensor)
        # need to flip the weights because transposed and math idfk
        nconv_pyt.weight.data.copy_(flipped_weights)
        nconv_pyt.bias.data.copy_(tconv_bias_tensor)
    
    stimulus = torch.randn((1, in_ch, 28, 28))
    
    if(input is not None):
        stimulus = input
    
    # Step 1: Expand height by inserting rows of zeros
    expanded_h = torch.zeros((1, in_ch, 28 * 2 - 1, 28))
    expanded_h[:, :, ::2, :] = stimulus  # Copy original rows to even indices

    # Step 2: Expand width by inserting columns of zeros
    expanded_hw = torch.zeros((1, in_ch, 28 * 2 - 1, 28 * 2 - 1))
    expanded_hw[:, :, :, ::2] = expanded_h  # Copy original columns to even indices

    # Pad with zeros
    # (left, right, top, bottom)
    # Shape is 1, 1, 64, 64
    padded_stimulus = F.pad(expanded_hw, (4, 5, 4, 5), mode='constant', value=0)
    
    # np_stim = padded_stimulus.detach().squeeze(0).numpy()
    np_stim = stimulus.detach().squeeze(0).numpy()
    
    ideal_tconv = tconv_pyt(stimulus).detach().squeeze(0).numpy()
    nconv_result = nconv_pyt(padded_stimulus).detach().squeeze(0).numpy()
    
    print(ideal_tconv.shape)
    print(nconv_result.shape)
    max_err = np.max(ideal_tconv - nconv_result)
    if(max_err < 0.0001):
        print("INFO [compare_tconv_conv_fmod] nn.Transposed and nn.Conv2d match")
    
    # diffs = torch.abs(ideal_tconv - nconv_result)
    # print(torch.max(diffs).detach())
    # deconv_weight = tconv_weight_tensor.numpy()
    deconv_bias   = tconv_bias_tensor.numpy()
    
    input_channels = []
    for i in range(44):
        input_channels.append([])
        
    for ch in range(44):
        for row in range(28):
            for col in range(28):
                input_channels[ch].append(np_stim[ch][row][col])
    # print(input_channels)
    
    print("INFO [compare_tconv_conv_fmod] Emulating low level deconvolution")
    resulting_maps = low_level_deconv(input_channels, flipped_weights, deconv_bias)

    print("INFO [compare_tconv_conv_fmod] Comparing results between fmod and pytorch...")

    num_wrong = 0
    out_ch = 3
    for ch in range(out_ch):
        for row in range(56):
            for col in range(56):
                dut = resulting_maps[ch][row*56 + col]
                ideal = ideal_tconv[ch, row, col]
                
                if(ch == 0 and row < 28):
                    print(f"DUT: {dut:>10.6f} Ideal: {ideal:>10.6f}. Error: {abs(ideal-dut):.8f}")
                
                if(not float_compare(dut, ideal, 0.0001)):
                    num_wrong += 1
                    print(f"Mismatch at [{ch}, {row:>2}, {col:>2}] DUT: {dut:>10.6f} Ideal: {ideal:>10.6f}. Error: {abs(ideal-dut):.8f}")
    
    print("INFO [compare_tconv_conv_fmod] Total differences:", num_wrong)
    
    
if __name__ == '__main__':
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
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
    
    input_channels = [[], [], []]
    for ch in range(3):
        for row in range(28):
            for col in range(28):
                input_channels[ch].append(image_tile_2828[ch][row][col])
    # resulting_maps = low_level_extraction(input_channels, conv_weights, conv_bias, prelu_weight)
             
    # To check that the input is stored correctly:     
    # for ch in range(3):
    #     print(f"Channel {ch}: ", end='')
    #     print(input_channels[ch][:5])
    
    inference = model.feature_extraction(low_res_coin.to(device))
    inference = model.shrink(inference)
    inference = model.map(inference)
    pre_deconv = model.expand(inference)
    inference = model.deconv(pre_deconv)
    inference = inference.squeeze(0).cpu().detach().numpy()
    
    # compare_hls_pytorch(inference, 56*56)
    
    compare_tconv_conv_fmod(input=pre_deconv, fsrcnn=model)
    print(inference[0,0])
    