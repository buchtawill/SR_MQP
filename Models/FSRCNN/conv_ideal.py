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

psums = [[], [], [], []]

def get_real_conv_result(start_x, start_y, channel_arr):
    result = np.sum(channel_arr[start_y:start_y+5, start_x:start_x+5] * WEIGHTS[0]) + BIAS_1
    result = np.maximum(0, result) + PRELU_1 * np.minimum(0, result)
    return result

def fifo_psum_conv(channel_arr):
    
    ############################## Do the first 4 rows ##############################
    # print("First row slider")
    for i in range(28):
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
    outputs = np.zeros((28,28))
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
            outputs[row-4, i] = output
    print(len(psums[0]))
    print(len(psums[1]))
    print(len(psums[2]))
    print(len(psums[3]))
    
    ############################ Drain the pipeline ############################
    # for i in range(28):
    #     slider = channel_arr[28, i:i+5]
        # print(slider)
        
    
    return outputs

def float_compare(a, b, epsilon=0.000001):
    return abs(a - b) < epsilon

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
    fifod_conv = fifo_psum_conv(channel)
    # print("\nOutputs:")
    # print(outputs)
    
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