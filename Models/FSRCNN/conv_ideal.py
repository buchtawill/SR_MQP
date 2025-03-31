import torch
import numpy as np

WEIGHTS = np.array(
    [
        [
        [-0.11433014, -0.06428213,  0.04952151, -0.02263108,  0.02250793],
        [-0.03798941, -0.00633621, -0.07827316,  0.03095409,  0.01443817],
        [ 0.03689848, -0.09081642,  0.05860689,  0.06232064,  0.00874583],
        [ 0.00830404, -0.09028250,  0.04836075,  0.01574024, -0.05474688],
        [ 0.03897108, -0.06225232,  0.01753724,  0.02418811, -0.03822224]
        ],
        [
        [ 0.00477085, -0.05933009, -0.04227761,  0.01911220, -0.06199792 ],
        [-0.04209856, -0.06560600, -0.05108242,  0.08851463, -0.02541809 ],
        [ 0.03671201, -0.03489959, -0.01189760, -0.06215083,  0.01050324 ],
        [-0.00799513,  0.00076000,  0.01511470, -0.10405967,  0.01873976 ],
        [-0.04879151,  0.01140088,  0.04946414,  0.01681182, -0.00011618 ]
        ],
    [
        [-0.03623180,  0.02557710, -0.01484760, -0.01260940,  0.00872839 ],
        [-0.00589560,  0.02435622, -0.06638044,  0.00577510,  0.00808473 ],
        [ 0.01797765, -0.00885874,  0.16383554,  0.02077497,  0.00002132 ],
        [-0.00601372, -0.02807183,  0.08771470, -0.03528251,  0.05220633 ],
        [ 0.01051262,  0.00916464, -0.05010103,  0.00487509,  0.01671523 ]
    ]
])

BIAS_1 = 0.0146
PRELU_1 = 0.3086
NUM_PE = 8

WEIGHTS_DECONV = np.array([
    [ -0.01144055, -0.00907180,  0.01926787,  0.02713044,  0.02557711,  0.02497433,  0.01976402, -0.00352580, -0.00125133 ],
    [ -0.01206590, -0.00707813,  0.02569121,  0.03151914,  0.01713638,  0.00322280,  0.00132756, -0.01895992, -0.00421952 ],
    [ -0.00299183,  0.00488416,  0.01854593,  0.00715575, -0.03299165, -0.05967715, -0.03442585, -0.02584003,  0.00703535 ],
    [  0.00526780,  0.00766025,  0.01146465,  0.00246327, -0.03731396, -0.06639615, -0.05517766, -0.04686022, -0.00171279 ],
    [ -0.00257094, -0.02277388, -0.00893931, -0.01222208,  0.00469413, -0.01561134, -0.04827255, -0.06521229, -0.03007165 ],
    [  0.00624669, -0.01110133, -0.00856166, -0.00355367,  0.03721085,  0.03906970, -0.03614255, -0.06374004, -0.02648782 ],
    [  0.01704879,  0.01068135,  0.00061458,  0.00252205,  0.02824421,  0.02898084, -0.02600434, -0.04214757, -0.00802519 ],
    [ -0.00696383,  0.00114733,  0.01185839,  0.00629945, -0.00332416, -0.00192898, -0.02506248, -0.03565328, -0.00143645 ],
    [ -0.01988062, -0.00953092,  0.01441447,  0.01116130, -0.00873910, -0.01072958, -0.00984131, -0.00831538,  0.00630383 ]
])

def get_real_conv_result(start_x, start_y, channel_arr):
    result = np.sum(channel_arr[start_y:start_y+5, start_x:start_x+5] * WEIGHTS[0]) + BIAS_1
    result = np.maximum(0, result) + PRELU_1 * np.minimum(0, result)
    return result


# def conv_5x5_fixed_pe()

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

def make_hls_1x1(name:str, in_ch:int, out_ch:int, in_width_pix:int, num_pe:int):
    func =  f"void conv_{name}(ch_stream_t tile_in[IN_CHN_LAYER_{name.upper()}], ch_stream_t map_out[OUT_CHN_LAYER_{name.upper()}]){{\n"
    func += f"    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py\n"
    func += f"    fixed_4_8_t slider[IN_CHN_LAYER_{name.upper()}];\n"
    func +=  "    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete\n"
    if(num_pe < out_ch):
        func += f"    ch_stream_t inbuf[IN_CHN_LAYER_{name.upper()}];\n\n"
    
    # Calculate how many times need to go thru PE's
    func += f"    int num_pe_loops = OUT_CHN_LAYER_{name.upper()} / NUM_PE_LAYER_{name.upper()};\n"
    func += f"    if((OUT_CHN_LAYER_{name.upper()} % NUM_PE_LAYER_{name.upper()}) != 0) num_pe_loops++;\n\n"
    
    func +=  "    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){\n"
    func += f"        // WARNING: if number fmap % num_pe != 0, utilization explodes!!\n"
    func += f"        int low_filter = (pe_loop*NUM_PE_LAYER_{name.upper()});\n"
    func += f"        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) < OUT_CHN_LAYER_{name.upper()} ? ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) : OUT_CHN_LAYER_{name.upper()};\n"
    func += f"        for(int row = 0; row < {in_width_pix}; row++){{\n\n" # Calculate size of padding
    
    
    func += f"            // Go across the row\n"
    func += f"            for(int col = 0; col < {in_width_pix}; col++){{\n"
    func +=  "                #pragma HLS PIPELINE II=1\n"
    # Read next slider value
    func +=  "                // Read the next value into the slider\n"
    func += f"                for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func +=  "                    #pragma HLS UNROLL\n"
    func += f"                    fixed_4_8_t next_data;\n"
    func += f"                    if(pe_loop == 0) next_data = tile_in[ch].read();\n"
    if(num_pe < out_ch):
        func += f"                    else             next_data = inbuf[ch].read();\n\n"
    func += f"                    slider[ch] = next_data;\n"
    if(num_pe < out_ch):
        func +=  "                    if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);\n"
    func += f"                }}\n\n" # channel loop (line 202)
    
    func += f"                for(int filter = low_filter; filter < high_filter; filter++){{\n"
    func += f"                    fixed_4_8_t mac = 0.0;\n"
    func += f"                    for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func += f"                        #pragma HLS UNROLL\n"
    func += f"                        mac += slider[ch] * weights_layer_{name}[filter][ch];\n"
    func += f"                    }}\n" # Channel loop 
    func += f"                    map_out[filter].write(prelu(conv_{name}_prelu[filter], (mac + conv_{name}_bias[filter])));\n"
    func += f"                }} // For every filter \n " # Filter loop 2 (line 214)    
    func +=  "            } // For every column \n" # column loop (line 190)    
    func +=  "        } // For every row\n" # row loop (line 170)
    func +=  "    } // For number of times thru PE\n" # pe loop (line 166)
    func +=  "}\n" # Function body
    
    defines  = f"#define IN_CHN_LAYER_{name.upper()}    {in_ch}\n"
    defines += f"#define OUT_CHN_LAYER_{name.upper()}   {out_ch}\n"
    defines += f"#define NUM_PE_LAYER_{name.upper()}    {num_pe}\n"
    weight_arr  = f"const fixed_4_8_t conv_{name}_prelu[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t conv_{name}_bias[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t weights_layer_{name}[{out_ch}][{in_ch}];\n"
    
    return func, (defines, weight_arr)

def make_hls_conv_func(name:str, in_ch:int, out_ch:int, kernel_size:int, in_width_pix:int, num_pe:int):
    
    # Note: Number of PE's, input channels, and output channels has to be defined at compile time 
    # and thus must use macros
    
    padding = kernel_size // 2
    if(kernel_size == 1):
        padding = 0
        
    in_padded_size = in_width_pix + 2*padding
    
    func =  f"void conv_{name}(ch_stream_t tile_in[IN_CHN_LAYER_{name.upper()}], ch_stream_t map_out[OUT_CHN_LAYER_{name.upper()}]){{\n"
    func += f"    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py\n"
    func += f"    fixed_4_8_t slider[IN_CHN_LAYER_{name.upper()}][{kernel_size}];\n"
    func +=  "    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete\n"
    if(num_pe < out_ch):
        func += f"    ch_stream_t inbuf[IN_CHN_LAYER_{name.upper()}];\n\n"
    # Declare PE's
    for i in range(kernel_size-1):
        func += f"    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum{i+1}[NUM_PE_LAYER_{name.upper()}][IN_CHN_LAYER_{name.upper()}];\n"

    # Partition and assign to BRAM
    for i in range(kernel_size-1):
        func += f"    #pragma HLS STREAM variable=psum{i+1} depth={in_width_pix}\n"
        func += f"    #pragma HLS RESOURCE variable=psum{i+1} core=FIFO_BRAM\n"
    
    func +='\n'
    # Calculate how many times need to go thru PE's
    func += f"    int num_pe_loops = OUT_CHN_LAYER_{name.upper()} / NUM_PE_LAYER_{name.upper()};\n"
    func += f"    if((OUT_CHN_LAYER_{name.upper()} % NUM_PE_LAYER_{name.upper()}) != 0) num_pe_loops++;\n\n"
    
    func +=  "    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){\n"
    func += f"        // WARNING: if number fmap % num_pe != 0, utilization explodes!!\n"
    func += f"        int low_filter = (pe_loop*NUM_PE_LAYER_{name.upper()});\n"
    func += f"        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) < OUT_CHN_LAYER_{name.upper()} ? ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) : OUT_CHN_LAYER_{name.upper()};\n"
    func += f"        for(int row = 0; row < {in_width_pix + 2*padding}; row++){{\n\n" # Calculate size of padding
    func +=  "            // Prep the slider\n"
    func += f"            for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func +=  "                #pragma HLS UROLL\n"
    func += f"                for(int idx = 0; idx < {kernel_size-1}; idx++){{\n"
    func +=  "                    #pragma HLS PIPELINE II=1\n"
    
    # First two or last two rows, pad with zeros
    func += f"                    if((row < {padding}) || (row >= {in_width_pix + padding*1}) || (idx < {padding})) slider[ch][idx] = 0;\n"
    func += f"                    else{{\n"
    func += f"                        fixed_4_8_t next_data;\n"
    func += f"                        if(pe_loop == 0) next_data = tile_in[ch].read();\n"
    if(num_pe < out_ch):
        func += f"                        else             next_data = inbuf[ch].read();\n\n"
    func += f"                        slider[ch][idx] = next_data;\n"
    if(num_pe < out_ch):
        func +=  "                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);\n"
    func +=  "                    }\n" # else not middle
    func +=  "                }\n" # idx loop
    func +=  "            }\n\n" # ch loop
    
    func += f"            // Go across the row\n"
    func += f"            for(int col = {kernel_size -1}; col < {in_width_pix + 2*padding}; col++){{\n"
    func +=  "                #pragma HLS PIPELINE II=1\n"
    func += f"                fixed_4_8_t final_sum[OUT_CHN_LAYER_{name.upper()}];\n"
    func += f"                #pragma HLS array_partition variable=final_sum dim=0 type=complete\n"
    func +=  "                for(int filter = low_filter; filter < high_filter; filter++){\n"
    func +=  "                    #pragma HLS UNROLL\n"
    func +=  "                    final_sum[filter] = 0.0;\n"
    func +=  "                }\n\n" # Filter loop 1
    
    # Resume at line 297 from conv2d.cpp
    # Read next slider value
    func +=  "                // Read the next value into the slider\n"
    func += f"                for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func +=  "                    #pragma HLS UNROLL\n"
    func += f"                    if((row < {padding}) || (row >= {in_width_pix + padding*1}) || (col >= {in_width_pix + padding*1})) slider[ch][{kernel_size-1}] = 0;\n"
    func += f"                    else{{\n"
    func += f"                        fixed_4_8_t next_data;\n"
    func += f"                        if(pe_loop == 0) next_data = tile_in[ch].read();\n"
    if(num_pe < out_ch):
        func += f"                        else             next_data = inbuf[ch].read();\n\n"
    func += f"                        slider[ch][{kernel_size-1}] = next_data;\n"
    if(num_pe < out_ch):
        func +=  "                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);\n"
    func += f"                    }}\n" # else read data (line 205)
    func += f"                }}\n\n" # channel loop (line 202)
    
    func += f"                for(int filter = low_filter; filter < high_filter; filter++){{\n"
    func += f"                    for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func += f"                        #pragma HLS UNROLL\n"
    func += f"                        fixed_4_8_t "

    for i in range(kernel_size-1):
        func += f"mac{i}, "
    func += f"mac{kernel_size-1};\n"
    
    func += f"                        fixed_4_8_t "

    for i in range(1,kernel_size-1):
        func += f"row{i}_psum, "
    func += f"row{kernel_size-1}_psum;\n"
    
    last_row_conv = in_padded_size - (kernel_size - 1)
    func += f"                        if(row < {last_row_conv}){{\n"
    func += f"                            mac0 = perform_mac{kernel_size}(weights_layer_{name}[filter][ch][0], slider[ch]);\n"
    func += f"                            psum1[filter % NUM_PE_LAYER_{name.upper()}][ch].write(mac0);\n"
    func += f"                        }}\n"
    for i in range(1, kernel_size-1):
        func += f"                        if(row >= {i} && row < {last_row_conv + i}) {{\n"
        func += f"                            row{i}_psum = psum{i}[filter % NUM_PE_LAYER_{name.upper()}][ch].read();\n"
        func += f"                            mac{i} = perform_mac{kernel_size}(weights_layer_{name}[filter][ch][{i}], slider[ch]);\n"
        func += f"                            psum{i+1}[filter % NUM_PE_LAYER_{name.upper()}][ch].write(row{i}_psum + mac{i});\n"
        func += f"                        }}\n"
        
    func += f"                        if(row >= {kernel_size - 1}){{\n"
    func += f"                            row{kernel_size-1}_psum = psum{kernel_size-1}[filter % NUM_PE_LAYER_{name.upper()}][ch].read();\n"
    func += f"                            mac{kernel_size-1} = perform_mac{kernel_size}(weights_layer_{name}[filter][ch][{kernel_size-1}], slider[ch]);\n"
    func += f"                            fixed_4_8_t pre_activation = row{kernel_size-1}_psum + mac{kernel_size-1};\n"
    func += f"                            final_sum[filter] += pre_activation;\n"
    func += f"                        }}\n"
    func += f"                    }}\n" # Channel loop 
    func += f"\n"
    func += f"                    if(row >= {kernel_size-1}) map_out[filter].write(prelu(conv_{name}_prelu[filter], \\\n"
    func += f"                                                            (final_sum[filter] + conv_{name}_bias[filter])));\n"
    func += f"                }} // For every filter \n " # Filter loop 2 (line 214)
    
    func += f"               for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func += f"                   #pragma HLS_UNROLL\n"
    for i in range(kernel_size-1):
        func += f"                   slider[ch][{i}] = slider[ch][{i+1}];\n"
    func += f"                }}\n"
    
    func +=  "            } // For every column \n" # column loop (line 190)
    
    func +=  "        } // For every row\n" # row loop (line 170)
    func +=  "    } // For number of times thru PE\n" # pe loop (line 166)
    func +=  "}\n" # Function body
    
    defines  = f"#define IN_CHN_LAYER_{name.upper()}    {in_ch}\n"
    defines += f"#define OUT_CHN_LAYER_{name.upper()}   {out_ch}\n"
    defines += f"#define NUM_PE_LAYER_{name.upper()}    {num_pe}\n"
    
    weight_arr  = f"const fixed_4_8_t conv_{name}_prelu[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t conv_{name}_bias[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t weights_layer_{name}[{out_ch}][{in_ch}][{kernel_size}][{kernel_size}];\n"
    return func, (defines, weight_arr)

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
    manual_deconv = transposed_convolution_9x9(input_ch, WEIGHTS_DECONV)
    
    diff = np.abs(pytorch_deconv - manual_deconv)

    return diff

if __name__ == '__main__':
    
    image_tile = np.load('../comparisons/images/image_coin_tile.npy')
    image_tile_2828 = image_tile.astype(np.float32) / 256.
    
    image_tile_2828 = np.transpose(image_tile_2828, (2, 0, 1))
    image_tile = np.pad(image_tile_2828, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
    # print(image_tile.shape) # (3, 32, 32)
    # print(image_tile[0,0])  # Red channel, first row
    
    channel = image_tile[0]
    
    # print("\nComputed via FIFOs and partial sums:")
    # channel[row, col]
    # fifod_conv = fifo_psum_conv(channel, WEIGHTS[0], print_slider=False, add_bias_and_prelu=False)
    # print(fifod_conv[0])
    # exit()
    # first_fmap = emulate_pytorch(image_tile, bias_prelu=True)
    # print(first_fmap[0])
    # exit()
    
    # print("Ideal convolution results after bias and prelu:")
    # ideal = np.zeros((28, 28))
    # for row in range(28):
    #     for col in range(28):
    #         ideal[row, col] = get_real_conv_result(col, row, channel)
    # print(ideal[0])
    # exit
    
    # num_wrong = 0
    # for i in range(28):
    #     for j in range(28):
    #         if(not float_compare(ideal[i,j], fifod_conv[i,j])):
    #             num_wrong+=1
    #             print(f"Error at row {i}, col {j}: Ideal: {ideal[i,j]:9.6f}, FIFO: {fifod_conv[i,j]:9.6f}")
    # print("Num correct:     ", 28*28 - num_wrong)
    # print("Number of errors:", num_wrong)
    
    # print(channel[2, 0:5]) # first five columns of the third row
    
    extraction_func, extraction_defines = make_hls_conv_func('feature_extraction0', in_ch=3, out_ch=44, kernel_size=5, in_width_pix=28, num_pe=4)
    # print(extraction_func)
    # print(extraction_defines)
    # exit()
    
    shrink_body, shrink_defines = make_hls_1x1('shrink0', in_ch=44, out_ch=12, in_width_pix=28, num_pe=2)
    
    map0_body, map0_defines = make_hls_conv_func('map0', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    map2_body, map2_defines = make_hls_conv_func('map2', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    map4_body, map4_defines = make_hls_conv_func('map4', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    map6_body, map6_defines = make_hls_conv_func('map6', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    
    expand_body, expand_defines = make_hls_1x1('expand0', in_ch=12, out_ch=44, in_width_pix=28, num_pe=2)
    
    # The defines
    # print(extraction_defines[0])
    # print(shrink_defines[0])
    # print(map0_defines[0])
    # print(map2_defines[0])
    # print(map4_defines[0])
    # print(map6_defines[0])
    # print(expand_defines[0])
    
    # The weight array declarations
    # print(extraction_defines[1])
    # print(shrink_defines[1])
    # print(map0_defines[1])
    # print(map2_defines[1])
    # print(map4_defines[1])
    # print(map6_defines[1])
    # print(expand_defines[1])
    
    print(extraction_func)
    print(shrink_body)
    print(map0_body)
    print(map2_body)
    print(map4_body)
    print(map6_body)
    print(expand_body)