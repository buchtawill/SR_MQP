# import torch
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

def make_hls_conv_func(name:str, in_ch:int, out_ch:int, kernel_size:int, in_width_pix):
    padding = kernel_size // 2
    if(kernel_size == 1):
        padding = 0
    func =  f"void conv_{name}(ch_stream_t tile_in[IN_CHN_LAYER_{name}], ch_stream_t map_out[OUT_CH_LAYER_{name}]){{\n"
    func += f"    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py"
    func += f"    fixed_4_8_t slider[IN_CHN_LAYER_{name}][{kernel_size}];\n"
    func +=  "    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete\n"
    func += f"    ch_stream_t inbuf[IN_CHN_LAYER_{name}];\n"
    # Declare PE's
    for i in range(kernel_size):
        func += f"    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum{i+1}[NUM_PE_LAYER_{name}][IN_CHN_LAYER_{name}];\n"

    # Partition and assign to BRAM
    for i in range(kernel_size):
        func += f"    #pragma HLS STREAM variable=psum{i+1} depth={in_width_pix}\n"
        func += f"    #pragma HLS RESOURCE variable=psum{i+1} core=FIFO_BRAM\n"
        
    # Calculate how many times need to go thru PE's
    func += f"    int num_pe_loops = OUT_CHN_LAYER_{name} / NUM_PE_LAYER_{name};\n"
    func += f"    if((OUT_CHN_LAYER_{name} % NUM_PE_LAYER_{name}) != 0) num_pe_loops++;\n"
    
    func +=  "    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){\n"
    func += f"        // WARNING: if number fmap % num_pe != 0, utilization explodes!!\n"
    func += f"        int low_filter = (pe_loop*NUM_PE_LAYER_{name});\n"
    func += f"        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_{name}) < OUT_CHN_LAYER_{name} ? ((pe_loop+1)*NUM_PE_LAYER_{name}) : OUT_CHN_LAYER_{name};\n"
    func += f"        for(int row = 0; row < {in_width_pix + 2*padding}; row++){{\n\n" # Calculate size of padding
    func +=  "            // Prep the slider\n"
    func += f"            for(int ch = 0; ch < IN_CHN_LAYER_{name}; ch++){{\n"
    func +=  "                #pragma HLS UROLL\n"
    func += f"                for(int idx = 0; idx < {kernel_size-1}; idx++){{\n"
    func +=  "                    #pragma HLS PIPELINE II=1\n"
    
    # First two or last two rows, pad with zeros
    func += f"                    if((row < {padding}) || (row >= {in_width_pix + padding*1}) || (idx < {padding})) slider[ch][idx] = 0;\n"
    func += f"                    else{{\n"
    func += f"                        fixed_4_8_t next_data;\n"
    func += f"						  if(pe_loop == 0) next_data = tile_in[ch].read();\n"
    func += f"						  else             next_data = inbuf[ch].read();\n\n"
    func += f"                        slider[ch][{kernel_size-1}] = next_data;\n"
    func +=  "                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);\n"
    func +=  "                    }\n" # else not middle
    func +=  "                }\n" # idx loop
    func +=  "            }\n\n" # ch loop
    
    func += f"            // Go across the row\n"
    func += f"            for(int col = {kernel_size -1}; col < {in_width_pix + 2*padding}; col++){{\n"
    func +=  "                #pragma HLS PIPELINE II=1\n"
    func += f"                fixed_4_8_t final_sum[OUT_CHN_LAYER_{name}];\n"
    func += f"                #pragma HLS array_partition variable=final_sum dim=0 type=complete\n"
    func +=  "                for(int filter = low_filter; filter < high_filter; filter++){\n"
    func +=  "                    #pragma HLS UNROLL\n"
    func +=  "                    final_sum[filter] = 0.0;\n"
    func +=  "                }\n" # Filter loop 1
    
    # Resume at line 297 from conv2d.cpp
    
    
    func +=  "            }\n" # column loop
    
    func +=  "        }\n" # row loop
    func +=  "    }\n" # pe loop
    func +=  "}\n" # Function body
    

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