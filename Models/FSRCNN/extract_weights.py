from FSRCNN import *
import sys
import time
import torch
import torchinfo
import numpy as np

COLOR_SPACE = 'rgb'

def sec_to_human(seconds):
    """Return a number of seconds to hours, minutes, and seconds"""
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)

def print_weights_as_c_array(state_dict, tensor_name="feature_extraction.0.weight"):
    if tensor_name not in state_dict:
        print(f"Tensor {tensor_name} not found in state_dict.")
        return
    
    weights = state_dict[tensor_name]  # Shape: [44, 3, 5, 5]
    weights = weights.numpy()  # Convert to NumPy array
    shape = weights.shape
    
    strsplit = tensor_name.split('.')
    layer_num = int(strsplit[1])
    layer_name = f"{strsplit[0]}"
    
    # Bias or prelu
    if("bias" in tensor_name):
        print(f"const fixed_4_8_t conv_{layer_name}{layer_num}_bias[{weights.shape[0]}] = {{")
        for i in range(len(weights)):
            if(i != len(weights) - 1):
                print(f"    {weights[i]:11.8f}, ")
            else:
                print(f"    {weights[-1]:11.8f}")
        
        print("};\n")
        
    elif(len(shape) == 1):
        print(f"const fixed_4_8_t conv_{layer_name}{layer_num-1}_prelu[{weights.shape[0]}] = {{")
        for i in range(len(weights)):
            if(i != len(weights) - 1):
                print(f"    {weights[i]:11.8f}, ")
            else:
                print(f"    {weights[-1]:11.8f}")
        print("};\n")
    
    # convolution layer
    elif(len(shape) == 4):
        
        # 1x1 kernel
        if(shape[2] == 1 and shape[3] == 1):
            # 12 filters, each has 44 elements
            print(f"const fixed_4_8_t weights_layer_{layer_name}{layer_num}[{shape[0]}][{shape[1]}] = {{")
            for i in range(shape[0]):
                print("    {", end='')
                for j in range(shape[1]):                    
                    if((j) % 10 == 0):
                        print("\n        ", end='')                        
                        
                    # Input channel
                    if(j != (shape[1] - 1)):
                        print(f"{weights[i][j][0][0]:11.8f}, ", end='')
                    else:
                        print(f"{weights[i][j][0][0]:11.8f} ")
                        
                        
                if(i != (shape[0] - 1)):
                    print("    },")
                else:
                    print("    }")
                    
                    
            print("};\n")
            
        # Not a 1x1 kernel
        else:
            print(f"const fixed_4_8_t weights_layer_{layer_name}{layer_num}[{shape[0]}][{shape[1]}][{shape[2]}][{shape[3]}] = {{")
            for i in range(shape[0]):
            # for i in range(1):
                print("    // Filter", i)
                print("    {")
                for j in range(shape[1]):
                    print("        {   // Channel", j)
                    for k in range(shape[2]):
                        row_values = ", ".join(f"{weights[i, j, l, k]:11.8f}" for l in range(shape[3]))
                        print(f"            {{ {row_values} }},")
                    print("        },")
                print("    },")
            print("};\n")
    else:
        print("ERROR")
    
def print_model_summary(model, batch_size, in_channels, height, width):
    torchinfo.summary(model, input_size=(batch_size, in_channels, height, width))

def find_extreme_weights(state_dict):
    global_min = float('inf')
    global_max = float('-inf')
    
    # Iterate over all parameters in the state_dict
    for param_name, param_tensor in state_dict.items():
        # if 'weight' in param_name:  # Ensure we're only looking at weight tensors
        param_min = param_tensor.min().item()
        param_max = param_tensor.max().item()
        
        if param_min < global_min:
            global_min = param_min
            min_param_name = param_name
            
        if param_max > global_max:
            global_max = param_max
            max_param_name = param_name

    return global_min, min_param_name, global_max, max_param_name
    
if __name__ == '__main__':
    tstart = time.time()
    
    # Set up device, model
    device = torch.device('cpu')
    model = FSRCNN(upscale_factor=2, color_space=COLOR_SPACE).to(device)
    model.load_state_dict(torch.load('./saved_weights/example_vitis_hls_weights_44.pth', weights_only=True))
    
    state_dict = model.state_dict()
    
    # print(state_dict['feature_extraction.0.weight'].shape) # Conv2D weights
    # print(state_dict['feature_extraction.0.bias'].shape)   # Conv2D bias
    # print(state_dict['feature_extraction.1.weight'])       # PReLU
    
    # weights_0 = state_dict['feature_extraction.0.weight'][0]
    # print(weights_0)
    # print_c_arr(weights_0)
    # exit()
    
    # print("Bias from feature extraction filter0: ")
    # print(state_dict['feature_extraction.0.bias'])
    # print(state_dict['feature_extraction.0.weight'][0]) # Conv2D weights

    # print_weights_as_c_array(state_dict)
    # exit()
    # for param_name, param_tensor in state_dict.items():
        # if("deconv" not in param_name):
            # print_weights_as_c_array(state_dict, param_name)
    """
    feature_extraction.0.weight
    feature_extraction.0.bias
    feature_extraction.1.weight
    shrink.0.weight
    shrink.0.bias
    shrink.1.weight
    map.0.weight
    map.0.bias
    map.1.weight
    map.2.weight
    map.2.bias
    map.3.weight
    map.4.weight
    map.4.bias
    map.5.weight
    map.6.weight
    map.6.bias
    map.7.weight
    expand.0.weight
    expand.0.bias
    expand.1.weight
    deconv.weight
    deconv.bias
    """
    
    print_weights_as_c_array(state_dict, 'feature_extraction.0.weight')
    
    exit()
    conv_weights = state_dict['feature_extraction.0.weight'].detach().numpy()
    conv_bias    = state_dict['feature_extraction.0.bias'].detach().numpy()
    prelu_weight = state_dict['feature_extraction.1.weight'].detach().numpy()
    
    np.save('./saved_weights/extraction_conv_44w', conv_weights)
    np.save('./saved_weights/extraction_conv_44b', conv_bias)
    np.save('./saved_weights/extraction_conv_44pre', prelu_weight)
       
    # print(state_dict['shrink.0.weight'].shape) # Conv2D weights
    # print_weights_as_c_array(state_dict, tensor_name="deconv.weight")
    # print(state_dict['deconv.weight'].shape)
    # print(state_dict['feature_extraction.0.weight'].shape) # Conv2D weights
    

    # global_min, min_param_name, global_max, max_param_name = find_extreme_weights(state_dict)
    # print(f"Global min: {global_min}, Parameter: {min_param_name}")
    # print(f"Global max: {global_max}, Parameter: {max_param_name}")

    
    